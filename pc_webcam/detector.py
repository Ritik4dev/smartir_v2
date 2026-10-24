"""Small bright-point detector for ordinary webcams (OpenCV + NumPy only).

This identifies a compact *changing* light, not the infrared wavelength.
"""

import cv2
import numpy as np


def find_pc_spot(frame_gray, background, min_delta=18, max_area=250):
    """Return (camera point, candidate mask, status) for a pulsed LED.

    A temporal difference rejects fixed lamps; local contrast rejects broad
    exposure/screen changes. One-pixel components are allowed because a remote
    LED can occupy less than a pixel after downsampling or at a distance.
    """
    if frame_gray.shape != background.shape or frame_gray.ndim != 2:
        raise ValueError("Camera frame and background must be matching grayscale images")
    if frame_gray.dtype != np.uint8 or background.dtype != np.uint8:
        raise ValueError("Camera frame and background must be uint8")
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
        if area > max_area or max(w, h) > 32 or max(w, h) > max(3, 3 * min(w, h)):
            continue
        region = labels[y:y+h, x:x+w] == label
        temporal_peak = int(change[y:y+h, x:x+w][region].max())
        local_peak = int(local[y:y+h, x:x+w][region].max())
        score = temporal_peak + .5 * local_peak + 2 * min(area, 12)
        candidates.append((score, tuple(float(v) for v in centroids[label])))
    if not candidates:
        return None, mask, "no small changing light"
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] >= .8 * candidates[0][0]:
        return None, mask, "ambiguous: multiple lights"
    return candidates[0][1], mask, "small bright spot"
