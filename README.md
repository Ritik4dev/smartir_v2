# SMARTIR on Raspberry Pi OS

On a Raspberry Pi OS desktop, run from this folder:

```bash
python smartir.py
```

If `python` is not available, use `python3 smartir.py` instead. On first run,
the launcher checks the OS Python for Picamera2, OpenCV, NumPy, Tk, and evdev.
If packages are missing, it asks before running `sudo apt-get update` and
`sudo apt-get install`. A password and internet connection may be required.
It does not silently change the OS. Later runs go straight to the GUI.

Picamera2 and libcamera are Raspberry Pi OS packages: they **cannot be
installed only in this Git folder**. `dependencies.txt` is an apt package
list, not a pip requirements file. To install manually:

```bash
sudo apt update && xargs -a dependencies.txt sudo apt install -y
```

Other commands use the same launcher:

```bash
python smartir.py doctor --camera-test
python smartir.py probe
python smartir.py calibrate
python smartir.py run --output canvas --preview
python smartir.py run --output touch --preview
```

Raspberry Pi OS Lite has no desktop, so the no-argument command runs camera
diagnostics rather than opening a GUI. For calibration and control of other
desktop applications, boot Raspberry Pi OS with Desktop on the connected HDMI
display. System-wide touch additionally needs `/dev/uinput` access, and
reliable visible-light rejection needs a physical IR-pass filter.
