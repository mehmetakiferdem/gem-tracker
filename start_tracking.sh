#!/bin/bash
# Edge AI ortamını yükle
source /opt/t3-edgeai-env

# Headless OpenCV, installed for this app only:
#   pip3 install --no-deps --target /opt/tracker-pylibs opencv-contrib-python-headless==4.10.0.84
# The system cv2 links 355 libraries and took ~9 s of a cold start just to
# load; this one ~1 s. Only drawing and JPEG encoding use cv2 here.
export PYTHONPATH=/opt/tracker-pylibs${PYTHONPATH:+:$PYTHONPATH}

# Kamerayı tanıt
bash /opt/edgeai-gst-apps/scripts/setup_cameras.sh

# Uygulamayı başlat
cd /home/gemstone/object-tracker-o1/apps_python
# exec: python becomes the service main process and gets the stop signal
# (SIGINT, see the unit) itself, so it always runs APP: Deinit.
exec ./app_edgeai.py ../configs/tracking.yaml -n
