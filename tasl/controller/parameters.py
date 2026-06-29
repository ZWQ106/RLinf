"""Override of DROID's parameters.py with TASL bench IPs.

Mounted into container at /app/droid/misc/parameters.py via docker-compose.
This is the ONE file that must be mounted — the conf/*.yaml are already
byte-identical inside the image; this overrides the image's DROID-default IPs
(which are the robot/nuc addresses the other way round).
"""
import os
from cv2 import aruco

# Robot Params — TASL bench
nuc_ip = "172.16.0.2"       # NUC1's direct-eth side
robot_ip = "172.16.0.1"     # our FR3 (override DROID default 172.16.0.2)
laptop_ip = "100.79.65.37"  # Desktop (Tailscale)
# Secret kept out of the repo. Container runs as root, so this is unused on our
# path; set SUDO_PASSWORD in the environment if a tool ever needs it.
sudo_password = os.environ.get("SUDO_PASSWORD", "")
robot_type = "fr3"
robot_serial_number = "placeholder"  # not used by polymetis driver

# Cameras — handled by Desktop dashboard, not container
hand_camera_id = ""
varied_camera_1_id = ""
varied_camera_2_id = ""

# Charuco — not used for online deploy
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

ubuntu_pro_token = ""

droid_version = "1.3"
