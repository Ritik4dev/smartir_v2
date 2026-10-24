"""Guided, memory-bounded learning of a user-confirmed remote LED."""

import cv2
import numpy as np

from detector import find_pc_spot


class LightLearning:
    def __init__(self, background, selected, min_delta=18, max_area=250):
        if background.ndim != 2 or background.dtype != np.uint8:
            raise ValueError("Learning background must be an 8-bit grayscale image")
        self.background = background
        self.selected = tuple(float(v) for v in selected)
        self.min_delta = min_delta
        self.max_area = max_area
        self.radius = max(12, min(background.shape) * .06)
        self.hits = []
        self.chromas = []
        self.deltas = []
        self.frames_seen = 0

    def observe(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape != self.background.shape:
            raise ValueError("Camera resolution changed during remote learning")
        self.frames_seen += 1
        spot, _, _ = find_pc_spot(gray, self.background,
                                  min_delta=self.min_delta,
                                  max_area=self.max_area,
                                  preferred=self.selected,
                                  max_jump=self.radius)
        if spot is None:
            return None
        x, y = (round(v) for v in spot)
        x0, x1 = max(0, x-2), min(gray.shape[1], x+3)
        y0, y1 = max(0, y-2), min(gray.shape[0], y+3)
        local_change = cv2.subtract(gray[y0:y1, x0:x1],
                                    self.background[y0:y1, x0:x1])
        py, px = np.unravel_index(np.argmax(local_change),
                                  (y1-y0, x1-x0))
        yy, xx = y0+py, x0+px
        pixel = frame_bgr[yy, xx].astype(np.float32)
        self.hits.append(spot)
        self.chromas.append(pixel / max(1.0, float(pixel.sum())))
        self.deltas.append(max(0, int(gray[yy, xx]) - int(self.background[yy, xx])))
        return spot

    def finish(self):
        if len(self.hits) < 8 or self.frames_seen < 20:
            raise ValueError("Not enough repeated LED frames; hold the remote button and retry teaching")
        positions = np.asarray(self.hits)
        center = np.median(positions, axis=0)
        spread = np.linalg.norm(positions - center, axis=1)
        if np.percentile(spread, 90) > max(6, self.radius * .5):
            raise ValueError("Light moved too much during teaching; keep the remote still and retry")
        if np.median(self.deltas) < self.min_delta:
            raise ValueError("Learned light is too dim; lower the threshold or aim the remote at the camera")
        chromas = np.asarray(self.chromas)
        median = np.median(chromas, axis=0)
        variation = np.median(np.abs(chromas - median), axis=0)
        distinctive = float(np.ptp(median)) >= .11
        reliable = distinctive and float(variation.sum()) < .10
        return {"version": 1,
                "bgr_chroma": [float(v) for v in median],
                "chroma_tolerance": float(min(.6, max(.22, 4*variation.sum()))),
                "color_reliable": bool(reliable),
                "median_delta": float(np.median(self.deltas)),
                "sample_count": len(self.hits)}


class CameraLock:
    """Require two nearby observations before accepting a new light."""

    def __init__(self, acquire_seconds=.4, hold_seconds=.3, acquire_radius=35):
        self.acquire_seconds = acquire_seconds
        self.hold_seconds = hold_seconds
        self.acquire_radius = acquire_radius
        self.pending = None
        self.pending_at = None
        self.point = None
        self.last_seen = None

    @property
    def preferred(self):
        return self.point if self.point is not None else self.pending

    def update(self, raw, now):
        """Return (confirmed observation, held position, state)."""
        if self.point is not None and now - self.last_seen > self.hold_seconds:
            self.point = None
            self.last_seen = None
        if self.pending is not None and now - self.pending_at > self.acquire_seconds:
            self.pending = None
            self.pending_at = None
        if raw is None:
            return None, self.point, "held" if self.point is not None else "searching"
        raw = tuple(float(v) for v in raw)
        if self.point is not None:
            self.point = raw
            self.last_seen = now
            return raw, None, "detected"
        if self.pending is not None and np.linalg.norm(np.subtract(raw, self.pending)) <= self.acquire_radius:
            self.point = raw
            self.last_seen = now
            self.pending = None
            self.pending_at = None
            return raw, None, "detected"
        self.pending = raw
        self.pending_at = now
        return None, None, "verifying"
