"""Small bright-point detector for ordinary webcams (OpenCV + NumPy only).

This identifies a compact *changing* light, not the infrared wavelength.
"""

import cv2
import numpy as np


def find_pc_spot(frame_gray, background, min_delta=18, max_area=250,
                 preferred=None, max_jump=None, frame_bgr=None, signature=None):
    """Return (camera point, candidate mask, status) for a pulsed LED.

    A temporal difference rejects fixed lamps; local contrast rejects broad
    exposure/screen changes. One-pixel components are allowed because a remote
    LED can occupy less than a pixel after downsampling or at a distance.
    """
    if frame_gray.shape != background.shape or frame_gray.ndim != 2:
        raise ValueError("Camera frame and background must be matching grayscale images")
    if frame_gray.dtype != np.uint8 or background.dtype != np.uint8:
        raise ValueError("Camera frame and background must be uint8")
    if signature is not None and (frame_bgr is None or
                                  frame_bgr.shape[:2] != frame_gray.shape):
        raise ValueError("A learned light signature requires the matching color frame")
    if max_jump is None:
        max_jump = max(25, min(frame_gray.shape) * .25)
    change = cv2.subtract(frame_gray, background)
    global_lift = int(np.median(change))
    if global_lift:
        change = cv2.subtract(change, global_lift)
    local = cv2.subtract(frame_gray, cv2.GaussianBlur(frame_gray, (5, 5), 0))
    local_min = max(5, round(min_delta * .25))
    mask = np.uint8((change >= min_delta) & (local >= local_min) &
                    (frame_gray >= 35)) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area > max_area or max(w, h) > 32:
            continue
        roundness = min(w, h) / max(w, h)
        # A one-pixel emitter has no meaningful contour. For larger blobs,
        # reject streaks/edges and favour a compact, roughly round footprint.
        if area >= 4 and roundness < .45:
            continue
        region = labels[y:y+h, x:x+w] == label
        if area >= 8:
            yy, xx = np.nonzero(region)
            spread = np.linalg.eigvalsh(np.cov(np.stack((xx, yy))))
            if spread[-1] > 0 and spread[0] / spread[-1] < .12:
                continue
        temporal_peak = int(change[y:y+h, x:x+w][region].max())
        local_peak = int(local[y:y+h, x:x+w][region].max())
        score = temporal_peak + .5 * local_peak + 2 * min(area, 12) + 20 * roundness
        if signature is not None and signature.get("color_reliable"):
            peak_y, peak_x = np.unravel_index(
                np.argmax(np.where(region, change[y:y+h, x:x+w], 0)), region.shape)
            pixel = frame_bgr[y+peak_y, x+peak_x].astype(np.float32)
            chroma = pixel / max(1.0, float(pixel.sum()))
            if np.abs(chroma - np.asarray(signature["bgr_chroma"])).sum() > signature["chroma_tolerance"]:
                continue
        candidates.append((score, tuple(float(v) for v in centroids[label])))
    if not candidates:
        return None, mask, "no matching compact light"
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if preferred is not None:
        nearby = [(score, point, float(np.linalg.norm(np.subtract(point, preferred))))
                  for score, point in candidates
                  if np.linalg.norm(np.subtract(point, preferred)) <= max_jump]
        if not nearby:
            return None, mask, "tracking: waiting for same light"
        # A known moving emitter is more credible than a brighter new lamp.
        chosen = max(nearby, key=lambda item: item[0] - .4 * item[2])
        return chosen[1], mask, "tracked compact spot"
    if len(candidates) > 1 and candidates[1][0] >= .8 * candidates[0][0]:
        return None, mask, "ambiguous: multiple lights"
    return candidates[0][1], mask, "small bright spot"
