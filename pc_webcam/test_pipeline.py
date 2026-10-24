"""Synthetic detector/calibration checks that do not require a webcam."""

import unittest

import cv2
import numpy as np

from smartir_pc import find_spot, fit_calibration, map_point, target_positions, PulseTracker


class PipelineTests(unittest.TestCase):
    def test_single_pixel_remote_led(self):
        background = np.full((120, 160), 25, np.uint8)
        current = background.copy()
        current[45, 73] = 85
        spot, _, reason = find_spot(current, background)
        self.assertEqual(reason, "small bright spot")
        self.assertEqual(spot, (73.0, 45.0))

    def test_dim_compact_remote_led(self):
        background = np.full((120, 160), 40, np.uint8)
        current = background.copy()
        cv2.circle(current, (80, 50), 2, 70, -1)
        spot, _, reason = find_spot(current, background)
        self.assertEqual(reason, "small bright spot")
        self.assertLess(np.linalg.norm(np.asarray(spot) - (80, 50)), 2)

    def test_global_exposure_change_is_not_a_led(self):
        background = np.full((120, 160), 45, np.uint8)
        current = np.full_like(background, 80)
        spot, _, _ = find_spot(current, background)
        self.assertIsNone(spot)

    def test_two_equally_bright_leds_are_ambiguous(self):
        background = np.full((120, 160), 20, np.uint8)
        current = background.copy()
        cv2.circle(current, (30, 40), 2, 120, -1)
        cv2.circle(current, (100, 70), 2, 120, -1)
        spot, _, reason = find_spot(current, background)
        self.assertIsNone(spot)
        self.assertIn("ambiguous", reason)

    def test_new_spot_not_static_bright_patch(self):
        background = np.zeros((120, 160), np.uint8)
        cv2.circle(background, (45, 50), 4, 180, -1)
        current = background.copy()
        cv2.circle(current, (80, 60), 3, 245, -1)
        spot, _, reason = find_spot(current, background)
        self.assertEqual(reason, "small bright spot")
        self.assertLess(np.linalg.norm(np.asarray(spot) - (80, 60)), 2)

    def test_corner_calibration_maps_center(self):
        display = (1920, 1080)
        targets = target_positions(*display)
        camera = [(x / 3 + 15, y / 3 + 10) for x, y in targets]
        profile = fit_calibration(list(zip(camera, targets)), (800, 500), display)
        mapped = map_point(np.asarray(profile["homography"]), camera[-1], display)
        self.assertLess(np.linalg.norm(np.asarray(mapped) - targets[-1]), 1)

    def test_remote_pulse_dropout(self):
        tracker = PulseTracker(release_ms=170)
        self.assertEqual(tracker.update((100, 100), 0)[0], "down")
        self.assertEqual(tracker.update(None, .08)[0], "hold")
        self.assertEqual(tracker.update((103, 101), .10)[0], "move")
        self.assertEqual(tracker.update(None, .28)[0], "up")


if __name__ == "__main__":
    unittest.main()
