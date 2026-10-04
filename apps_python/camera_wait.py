#!/usr/bin/env python3
"""
Wait for the IMX219 camera before the tracker starts.

The sensor is probed once at boot. If the ribbon cable is not plugged in
then, the driver gives up (chip id read fails with -121) and the app dies on
"/dev/video-imx219-cam0 doesn't exist", over and over. Instead this script
puts a warning on every connected display, asks the kernel to probe the
sensor again every POLL seconds and returns as soon as the driver binds, so
the tracker starts without a reboot once the camera is connected.

Exits 0 right away if the camera is already there. Never blocks the boot:
it runs inside object-tracker.service, which is Type=simple.
"""

import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# I2C client of the sensor, from the csi0-imx219 overlay (bus 5, address 0x10).
SENSOR = os.environ.get("CAMERA_I2C", "5-0010")
DRIVER = "/sys/bus/i2c/drivers/imx219"
POLL = 2.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def bound():
    return os.path.exists("/sys/bus/i2c/devices/%s/driver" % SENSOR)


def try_bind():
    # Fails with ENODEV/EREMOTEIO while nothing answers on the bus.
    try:
        with open(os.path.join(DRIVER, "bind"), "w") as f:
            f.write(SENSOR)
    except OSError:
        pass
    return bound()


def warning_frame(w, h, seconds):
    img = Image.new("RGB", (w, h), (90, 0, 0))
    d = ImageDraw.Draw(img)
    big = ImageFont.truetype(FONT, max(18, h // 9))
    small = ImageFont.truetype(FONT, max(12, h // 20))
    lines = [
        (big, "KAMERA BAĞLI DEĞİL"),
        (small, "Kamera kablosunu takın,"),
        (small, "takip kendiliğinden başlayacak."),
        (small, "Bekleniyor: %d sn" % seconds),
    ]
    heights = [d.textbbox((0, 0), t, font=f)[3] for f, t in lines]
    y = (h - sum(heights) - 20 * (len(lines) - 1)) // 2
    for (f, t), lh in zip(lines, heights):
        tw = d.textlength(t, font=f)
        d.text(((w - tw) // 2, y), t, font=f, fill=(255, 255, 255))
        y += lh + 20
    return np.asarray(img)


def open_display():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import yaml

        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../configs/tracking.yaml")
        with open(cfg) as f:
            model = next(iter(yaml.safe_load(f)["models"].values()))
        device = str(model.get("display_fb", "/dev/dri/card0"))
        rotate = int(model.get("display_rotate", 0))
        if not device.startswith("/dev/dri/"):
            return None
        from kms_view import KmsView

        return KmsView(device=device, rotate=rotate)
    except Exception as exc:
        print("[WARNING] No display for the camera warning: %s" % exc, flush=True)
        return None


def main():
    if bound() or try_bind():
        return 0
    print("[WARNING] Camera %s not found, waiting for it" % SENSOR, flush=True)
    view = open_display()
    start = time.monotonic()
    try:
        while not try_bind():
            if view is not None:
                view.publish(warning_frame(view.width, view.height,
                                           time.monotonic() - start))
            time.sleep(POLL)
    finally:
        if view is not None:
            view.close()
    print("[INFO] Camera %s connected" % SENSOR, flush=True)
    # Let the CSI receiver register its video nodes before setup_cameras.sh.
    time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
