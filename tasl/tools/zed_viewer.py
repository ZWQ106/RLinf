"""Dual ZED 2i live MJPEG viewer. Open http://<host>:8000 in browser."""
import pyzed.sl as sl
import threading
import time
import io
import subprocess
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

PORT = 8002


def host_ip() -> str:
    """Laptop-facing host IP: Tailscale if up, else the robot-net (172.16.0.x)."""
    try:
        ip = subprocess.check_output(["tailscale", "ip", "-4"], text=True,
                                     timeout=3).splitlines()[0].strip()
        if ip:
            return ip
    except Exception:
        pass
    try:
        ips = subprocess.check_output(["hostname", "-I"], text=True, timeout=3).split()
        for ip in ips:
            if ip.startswith("172.16.0."):
                return ip
        return ips[0] if ips else "<host>"
    except Exception:
        return "<host>"
RESOLUTION = sl.RESOLUTION.HD720
FPS = 15
JPEG_QUALITY = 70

latest_jpeg = {}  # SN -> bytes
running = True

def cam_loop(sn):
    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(sn)
    init.camera_resolution = RESOLUTION
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE  # no depth needed for viewer
    init.coordinate_units = sl.UNIT.METER
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"SN {sn}: open failed {err}")
        return
    print(f"SN {sn}: opened")
    rt = sl.RuntimeParameters()
    img = sl.Mat()
    interval = 1.0 / FPS
    while running:
        t0 = time.time()
        if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(img, sl.VIEW.LEFT)
            arr = img.get_data()  # BGRA
            # to RGB
            rgb = arr[:, :, [2, 1, 0]]
            pil = Image.fromarray(rgb, mode="RGB")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=JPEG_QUALITY)
            latest_jpeg[sn] = buf.getvalue()
        sleep_for = interval - (time.time() - t0)
        if sleep_for > 0:
            time.sleep(sleep_for)
    cam.close()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass  # silence logs

    def do_GET(self):
        if self.path == "/":
            self.send_html()
        elif self.path.startswith("/cam/"):
            try:
                sn = int(self.path.split("/")[2].split(".")[0])
                self.stream_mjpeg(sn)
            except (ValueError, IndexError):
                self.send_error(404)
        else:
            self.send_error(404)

    def send_html(self):
        sns = sorted(latest_jpeg.keys())
        cells = "".join(f"""<div style="flex:1; min-width:0;">
            <h3 style="margin:4px 8px;color:#aaa;font-family:monospace;font-weight:normal;">SN {sn}</h3>
            <img src="/cam/{sn}.mjpg" style="width:100%; display:block;"/>
        </div>""" for sn in sns)
        html = f"""<!DOCTYPE html><html><head><title>ZED dual viewer</title>
            <style>body{{margin:0;background:#111;}} .row{{display:flex;flex-direction:row;}}</style>
            </head><body><div class="row">{cells}</div></body></html>"""
        b = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def stream_mjpeg(self, sn):
        if sn not in latest_jpeg:
            # wait briefly for first frame
            for _ in range(20):
                if sn in latest_jpeg: break
                time.sleep(0.1)
            else:
                self.send_error(503, f"camera {sn} not streaming")
                return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        try:
            while running:
                jpeg = latest_jpeg.get(sn)
                if jpeg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass

class ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def main():
    global running
    devs = sl.Camera.get_device_list()
    if not devs:
        print("no ZED cameras found"); return
    print(f"starting {len(devs)} cam threads...")
    threads = [threading.Thread(target=cam_loop, args=(d.serial_number,), daemon=True) for d in devs]
    for t in threads: t.start()
    time.sleep(2)  # let first frames land
    print(f"serving on http://0.0.0.0:{PORT}")
    print(f"  open http://{host_ip()}:{PORT} in your browser")
    srv = ThreadingServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    running = False

if __name__ == "__main__":
    main()
