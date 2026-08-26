#!/usr/bin/env python3
"""Offline RTC smoke test against a running policy server (no robot, no cams).

    PYTHONPATH=$HOME/work/openpi/packages/openpi-client/src \
        python3 ~/RLinf/tasl/rtc/smoke.py [--host 127.0.0.1] [--port 8000] [--s 4] [--d 3] [--n 10]

Checks, with a synthetic DROID observation:
  1. the server is RTC-capable (response carries `actions_model`);
  2. guided sampling really inpaints: the first d actions of the new chunk
     match the previous chunk's actions [s, s+d) much better than an
     independent plain sample does (both measured in executable action space);
  3. the frozen region is invariant to the schedule while the soft region is
     not ("zeros" vs "exp");
  4. latency: plain vs guided inference, median over --n calls.
Exit code 0 when 1-3 hold.
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtc.executor import shift_chunk  # noqa: E402


def synthetic_obs(prompt: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224]
    base = np.stack([xx, yy, (xx + yy) // 2], -1).astype(np.uint8)
    wrist = (255 - base).astype(np.uint8)
    q = rng.uniform(-0.3, 0.3, 7).astype(np.float32) + np.array([0, -0.5, 0, -2.0, 0, 1.6, 0.8], np.float32)
    return {
        "observation/exterior_image_1_left": base,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": q,
        "observation/gripper_position": np.zeros(1, np.float32),
        "prompt": prompt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--s", type=int, default=4, help="execute horizon (actions consumed)")
    ap.add_argument("--d", type=int, default=3, help="inference delay (frozen prefix)")
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=10, help="timing repetitions")
    ap.add_argument("--prompt", default="pick up the red block")
    args = ap.parse_args()

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    pol = WebsocketClientPolicy(host=args.host, port=args.port)
    obs = synthetic_obs(args.prompt)

    def infer(prev=None, d=args.d, s=args.s, schedule="exp"):
        o = dict(obs)
        o["rtc"] = {"enabled": True, "prev_actions": prev, "inference_delay": d,
                    "prefix_attention_horizon": (H - s) if prev is not None else 0,
                    "schedule": schedule, "max_guidance_weight": args.beta}
        t0 = time.perf_counter()
        r = pol.infer(o)
        ms = (time.perf_counter() - t0) * 1000
        return np.asarray(r["actions"]), (np.asarray(r["actions_model"]) if "actions_model" in r else None), ms, r

    print(f"→ plain sample (first chunk) on :{args.port}")
    H = None
    A0, Am0, ms0, r0 = infer()
    if Am0 is None:
        print("✗ server response has no `actions_model` — not RTC-capable. Start it with tasl/rtc/scripts/serve_policy.py")
        return 1
    H = A0.shape[0]
    print(f"  H={H} exec_dim={A0.shape[1]} model_dim={Am0.shape[1]} infer={ms0:.0f} ms  rtc={r0.get('rtc')}")
    s, d = args.s, args.d
    assert d <= H - s, f"need d <= H - s ({d} <= {H - s})"

    prev = shift_chunk(Am0, s)
    target = A0[s:s + d]                       # what the controller executes at ticks [s, s+d)

    print(f"→ guided sample: s={s} d={d} prefix_h={H - s} beta={args.beta}")
    A1, Am1, ms1, r1 = infer(prev)
    print(f"  infer={ms1:.0f} ms (first guided call includes compile if the server did not warm up)  rtc={r1.get('rtc')}")
    A1, Am1, ms1, r1 = infer(prev)
    A_plain2, _, _, _ = infer()

    err_rtc = np.abs(A1[:d] - target).mean()
    err_plain = np.abs(A_plain2[:d] - target).mean()
    print(f"  prefix |A_new[:d] - A_prev[s:s+d]|  guided={err_rtc:.4f}   independent plain={err_plain:.4f}"
          f"   ratio={err_rtc / max(err_plain, 1e-9):.2f}")
    ok_inpaint = err_rtc < 0.5 * err_plain

    A_hard, _, _, _ = infer(prev, schedule="zeros")
    A_ones, _, _, _ = infer(prev, schedule="ones")
    err_hard = np.abs(A_hard[:d] - target).mean()
    soft_exp = np.abs(A1[d:H - s] - A0[s + d:H]).mean()
    soft_hard = np.abs(A_hard[d:H - s] - A0[s + d:H]).mean()
    soft_ones = np.abs(A_ones[d:H - s] - A0[s + d:H]).mean()
    print(f"  schedule=zeros prefix err={err_hard:.4f}; soft-region dev from prev: exp={soft_exp:.4f} zeros={soft_hard:.4f} ones={soft_ones:.4f}")
    ok_sched = err_hard < 0.5 * err_plain and soft_ones <= soft_hard + 1e-6

    print(f"→ timing over n={args.n}")
    t_plain = [infer()[2] for _ in range(args.n)]
    t_rtc = [infer(prev)[2] for _ in range(args.n)]
    print(f"  plain  median={statistics.median(t_plain):.0f} ms  max={max(t_plain):.0f}")
    print(f"  guided median={statistics.median(t_rtc):.0f} ms  max={max(t_rtc):.0f}")
    dt = 1000 / 15
    print(f"  → at 15 Hz the guided call spans ~{statistics.median(t_rtc) / dt:.1f} ticks (plus obs capture); "
          f"suggested delay_init = {int(np.ceil(statistics.median(t_rtc) / dt + 0.5))} (+0.5 tick for obs capture)")

    ok = ok_inpaint and ok_sched
    print("✓ RTC smoke OK" if ok else "✗ RTC smoke FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
