"""Synthetic checks for guided remote learning and conservative tracking."""

import unittest

import cv2
import numpy as np

from detector import find_pc_spot
from learning import CameraLock, LightLearning


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.off = np.full((90, 130, 3), 20, np.uint8)
        self.background = cv2.cvtColor(self.off, cv2.COLOR_BGR2GRAY)

    def test_learned_purple_led_rejects_green_round_light(self):
        learner = LightLearning(self.background, (50, 40))
        for index in range(40):
            frame = self.off.copy()
            if index % 4 != 0:
                cv2.circle(frame, (50, 40), 2, (170, 20, 170), -1)
            learner.observe(frame)
        profile = learner.finish()
        self.assertTrue(profile["color_reliable"])
        self.assertGreaterEqual(profile["sample_count"], 8)
        purple = self.off.copy()
        cv2.circle(purple, (70, 40), 2, (170, 20, 170), -1)
        point, _, _ = find_pc_spot(cv2.cvtColor(purple, cv2.COLOR_BGR2GRAY),
                                    self.background, frame_bgr=purple,
                                    signature=profile)
        self.assertIsNotNone(point)
        green = self.off.copy()
        cv2.circle(green, (70, 40), 2, (20, 220, 20), -1)
        point, _, _ = find_pc_spot(cv2.cvtColor(green, cv2.COLOR_BGR2GRAY),
                                    self.background, frame_bgr=green,
                                    signature=profile)
        self.assertIsNone(point)

    def test_learning_rejects_missing_led(self):
        learner = LightLearning(self.background, (50, 40))
        for _ in range(40):
            learner.observe(self.off)
        with self.assertRaisesRegex(ValueError, "Not enough repeated"):
            learner.finish()

    def test_neutral_led_does_not_claim_color_discrimination(self):
        learner = LightLearning(self.background, (50, 40))
        for _ in range(30):
            frame = self.off.copy()
            cv2.circle(frame, (50, 40), 2, (125, 125, 125), -1)
            learner.observe(frame)
        profile = learner.finish()
        self.assertFalse(profile["color_reliable"])

    def test_camera_lock_requires_repeated_flashes_and_holds_short_dropout(self):
        lock = CameraLock(acquire_seconds=.4, hold_seconds=.3)
        self.assertEqual(lock.update((20, 20), 0)[2], "verifying")
        self.assertEqual(lock.update(None, .05)[2], "searching")
        seen, _, state = lock.update((21, 20), .12)
        self.assertEqual(state, "detected")
        self.assertEqual(seen, (21.0, 20.0))
        seen, held, state = lock.update(None, .2)
        self.assertIsNone(seen)
        self.assertEqual(held, (21.0, 20.0))
        self.assertEqual(state, "held")
        self.assertIsNone(lock.update(None, .5)[1])


if __name__ == "__main__":
    unittest.main()
