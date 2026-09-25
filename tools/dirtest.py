# Which way does the picture move when each servo axis moves? Read only.
# Compares the frame just before an angle change with one 0.5 s later.
import time, urllib.request, numpy as np, cv2
PAN, TILT = "/sys/class/pwm/pwmchip4/pwm1/duty_cycle", "/sys/class/pwm/pwmchip4/pwm0/duty_cycle"
ang = lambda p: (int(open(p).read()) / 1000 - 1500) * 90 / 1000
r = urllib.request.urlopen("http://127.0.0.1:8080/stream", timeout=5)
buf = b""; last = None; pending = None; rows = []; t_end = time.time() + 35
while time.time() < t_end:
    buf += r.read(16384)
    a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a)
    if a < 0 or b < 0: continue
    jpg, buf = buf[a:b+2], buf[b+2:]
    g = np.float32(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)[40:, :])
    now, p, t = time.time(), ang(PAN), ang(TILT)
    if pending and now - pending[3] > 0.5:
        g0, p0, t0 = pending[:3]
        if abs(p - pending[4]) < 0.5 and abs(t - pending[5]) < 0.5:   # nothing else moved since
            (sx, sy), resp = cv2.phaseCorrelate(g0, g)
            if resp > 0.1: rows.append((pending[4] - p0, pending[5] - t0, sx, sy, resp))
        pending = None
    if last is not None and pending is None and (abs(p - last[1]) >= 1 or abs(t - last[2]) >= 1):
        pending = (last[0], last[1], last[2], now, p, t)
    last = (g, p, t)
for r_ in rows: print("dpan=%+5.1f dtilt=%+5.1f  shift_x=%+6.1f shift_y=%+6.1f (resp %.2f)" % r_)
pr = [sx / dp for dp, dt, sx, sy, _ in rows if abs(dp) >= 1 and abs(dt) < 0.5]
tr = [sy / dt for dp, dt, sx, sy, _ in rows if abs(dt) >= 1 and abs(dp) < 0.5]
if pr: print("PAN : %+.1f px per +1 deg in a 640 px view (n=%d)" % (np.median(pr), len(pr)))
if tr: print("TILT: %+.1f px per +1 deg in a 360 px view (n=%d)" % (np.median(tr), len(tr)))
