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

The package list targets current Raspberry Pi OS Trixie. It explicitly
includes `libjsoncpp26`, which supplies `libjsoncpp.so.26` needed by OpenCV
on Trixie. If an earlier run reported that library missing, the immediate
repair is:

```bash
sudo apt update && sudo apt install -y libjsoncpp26
/usr/bin/python3 -c 'import cv2; print(cv2.__version__)'
```

Run the GUI from the signed-in desktop user's terminal, **not** from a root
shell. If your prompt begins with `root@`, type `exit` and run
`python3 smartir.py` as that desktop user. Root may not have access to the
desktop's Wayland/X11 session; it is also unnecessary for camera capture.

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
