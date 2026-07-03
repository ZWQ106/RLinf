import sys, numpy as np, cv2, pandas as pd
p, col, out = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_parquet(p, columns=[col])
def dec(d):
    b = d.get("bytes") if isinstance(d, dict) else d
    return None if b is None else cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
rows = df[col].tolist()
f0 = dec(rows[0]); h, w = f0.shape[:2]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
n = 0
for d in rows:
    f = dec(d)
    if f is not None: vw.write(f); n += 1
vw.release(); print(f"{out}: {n} frames @ {w}x{h}")
