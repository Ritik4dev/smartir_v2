"""Hardware-free tests for the PC webcam application."""

import unittest

import cv2
import numpy as np

from smartir_pc import (CalibrationHold, WindowsMouse, annotated_camera,
                        calibration_screen, camera_inset, gray_bgr)


class FakeUser32:
    def __init__(self):
        self.positions = []
        self.buttons = 0

    def SetCursorPos(self, x, y):
        self.positions.append((x, y))
        return 1

    def SendInput(self, count, pointer, size):
        self.buttons += count
        return count


class PcTests(unittest.TestCase):
    def test_calibration_hold_requires_stable_repeated_hits(self):
        hold = CalibrationHold(required=3)
        self.assertFalse(hold.update((20, 20), 0.0))
        self.assertFalse(hold.update(None, 0.05))
        self.assertFalse(hold.update((21, 20), 0.1))
        self.assertTrue(hold.update((20, 21), 0.2))
        self.assertEqual(hold.progress, 1)
        self.assertAlmostEqual(hold.point[0], 20)

    def test_calibration_hold_resets_on_jump_or_long_gap(self):
        hold = CalibrationHold(required=3)
        hold.update((20, 20), 0)
        hold.update((21, 20), .1)
        self.assertFalse(hold.update((100, 100), .2))
        self.assertEqual(len(hold.samples), 1)
        self.assertFalse(hold.update((101, 100), 1.0))
        self.assertEqual(len(hold.samples), 1)

    def test_camera_annotation_and_calibration_inset(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        self.assertEqual(annotated_camera(frame, (60, 50), "spot").shape, frame.shape)
        canvas = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(camera_inset(canvas.copy(), frame, (60, 50), "spot").shape,
                         canvas.shape)
        hold = CalibrationHold()
        hold.update((60, 50), 0)
        screen = calibration_screen((1280, 720), (120, 72), 0, frame,
                                    (60, 50), "spot", hold, "capture")
        self.assertEqual(screen.shape, canvas.shape)
        self.assertGreater(np.count_nonzero(screen), 0)

    def test_bgr_grayscale(self):
        frame = np.zeros((4, 4, 3), np.uint8)
        frame[:, :, 2] = 255
        self.assertTrue(np.array_equal(gray_bgr(frame),
                                       cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

    def test_mouse_down_move_release(self):
        api = FakeUser32()
        mouse = WindowsMouse(user32=api)
        mouse.emit("down", (10.4, 20.6))
        mouse.emit("move", (12.2, 23.8))
        mouse.close()
        self.assertEqual(api.positions, [(10, 21), (12, 24)])
        self.assertEqual(api.buttons, 2)
        self.assertFalse(mouse.down)


if __name__ == "__main__":
    unittest.main()
