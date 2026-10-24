"""Hardware-free tests for the PC webcam application."""

import unittest

import cv2
import numpy as np

from smartir_pc import WindowsMouse, gray_bgr


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
