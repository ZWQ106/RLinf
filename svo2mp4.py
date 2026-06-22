import sys
import pyzed.sl as sl
import cv2
svo, out = sys.argv[1], sys.argv[2]
init = sl.InitParameters()
init.set_from_svo_file(svo)
init.svo_real_time_mode = False
init.depth_mode = sl.DEPTH_MODE.NONE
cam = sl.Camera()
if cam.open(init) != sl.ERROR_CODE.SUCCESS:
    sys.exit(f"open fail {svo}")
info = cam.get_camera_information().camera_configuration.resolution
w, h = info.width, info.height
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
img = sl.Mat()
n = 0
while True:
    e = cam.grab()
    if e == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(img, sl.VIEW.LEFT)
        vw.write(img.get_data()[:, :, :3].copy())
        n += 1
    else:
        break
vw.release(); cam.close()
print(f"{out}: {n} frames @ {w}x{h}")
