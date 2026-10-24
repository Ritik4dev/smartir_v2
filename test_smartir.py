"""Launcher tests; no Pi hardware or package installation required."""

import io
import unittest
from unittest.mock import patch

import smartir


class LauncherTests(unittest.TestCase):
    def test_package_manifest_is_safe(self):
        self.assertEqual(smartir.apt_packages(), [
            "python3-picamera2", "python3-opencv", "python3-numpy",
            "python3-tk", "python3-evdev", "libjsoncpp26",
        ])

    def test_no_install_when_imports_work(self):
        with patch.object(smartir, "on_raspberry_pi", return_value=True), \
             patch.object(smartir, "missing_imports", return_value=None), \
             patch.object(smartir.subprocess, "run") as run:
            smartir.ensure_dependencies("/usr/bin/python3")
            run.assert_not_called()

    def test_missing_jsoncpp_installs_manifest_and_rechecks(self):
        with patch.object(smartir, "on_raspberry_pi", return_value=True), \
             patch.object(smartir, "missing_imports", side_effect=[
                 "ImportError: libjsoncpp.so.26: cannot open shared object file", None,
             ]) as imports, \
             patch.object(smartir.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="y"), \
             patch.object(smartir.subprocess, "run") as run, \
             patch("sys.stderr", new_callable=io.StringIO):
            smartir.ensure_dependencies("/usr/bin/python3")
            self.assertEqual(imports.call_count, 2)
            self.assertIn("libjsoncpp26", run.call_args_list[-1].args[0])

    def test_noninteractive_missing_packages_does_not_install(self):
        with patch.object(smartir, "on_raspberry_pi", return_value=True), \
             patch.object(smartir, "missing_imports", return_value="missing cv2"), \
             patch.object(smartir.sys.stdin, "isatty", return_value=False), \
             patch.object(smartir.subprocess, "run") as run, \
             patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, "Run interactively"):
                smartir.ensure_dependencies("/usr/bin/python3")
            run.assert_not_called()

    def test_windows_routes_to_pc_webcam_app(self):
        with patch.object(smartir.platform, "system", return_value="Windows"), \
             patch.object(smartir, "missing_pc_imports", return_value=None), \
             patch.object(smartir.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(smartir.main([]), 0)
            self.assertEqual(run.call_args.args[0][-2:], [str(smartir.PC_CORE), "gui"])


if __name__ == "__main__":
    unittest.main()
