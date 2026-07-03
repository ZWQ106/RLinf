# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-pose GELLO calibration + verification (all-in-one).

A single matched pose is underdetermined per joint (``q = sign*(raw - offset)``
is one equation, two unknowns; the SIGN is unobservable without motion), so a
static check can't catch a wrong sign — it passes when still, then runs away in
teleop. This drives the arm to TWO poses (speed-bounded leash), you match the
leader at each, and solves both from the difference::

    sign_i   = (robotA_i - robotB_i) / (rawA_i - rawB_i)     # slope -> +-1
    offset_i = rawA_i - sign_i * robotA_i                    # intercept -> pi/2

Then (by default) it WRITES the result into ``fr3_gello_config.py`` and VERIFIES
it against the robot (match the leader to the homed arm; per-joint error must be
small). Reads ACTUAL joint positions, so a soft/compliant arm is fine.

SAFETY: the arm MOVES. Keep a hand on the E-stop. Controller must be live (home
from the dashboard first). Prefer the wrapper ``launch/gello-calibrate.sh`` which
also reaps stale port holders. Direct use::

    python -m rlinf.envs.realworld.common.gello.calibrate_2pose            # calibrate+write+verify
    python -m rlinf.envs.realworld.common.gello.calibrate_2pose --verify   # verify current config only
    python -m rlinf.envs.realworld.common.gello.calibrate_2pose --no-write # calibrate, print, don't edit
"""

import argparse
import os
import re
import sys
import time

import numpy as np

ROBOT_ADDR = os.environ.get("ZERORPC_ADDR", "tcp://172.16.0.2:4242")
PORT = "/dev/gello"
BAUD = 57600
IDS = [1, 2, 3, 4, 5, 6, 7]
_PI2 = np.pi / 2
# Pose A = the DROID home anchor. Pose B moves every joint ~0.3-0.4 rad (within
# FR3 limits, arm stays up) so each joint has a clear slope to solve.
POSE_A = [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0]
POSE_B = [0.40, -0.35, 0.40, -2.20, 0.40, 1.50, 0.40]
MATCH_SECONDS = 25
VERIFY_TOL = 0.30
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "fr3_gello_config.py")


def _connect():
    from gello.dynamixel.driver import DynamixelDriver

    from rlinf.envs.realworld.franka.droid_zerorpc_client import DroidZerorpcClient

    print("connecting to leader (/dev/gello) + robot…", flush=True)
    driver = DynamixelDriver(IDS, port=PORT, baudrate=BAUD)
    robot = DroidZerorpcClient(address=ROBOT_ADDR)
    robot.get_robot_state()  # fail fast if controller not live
    return driver, robot


def _read_raw_steady(driver, secs):
    """Sample raw leader joints while the operator matches; return steady mean."""
    end = time.time() + secs
    samples, last = [], 0.0
    while time.time() < end:
        samples.append(np.asarray(driver.get_joints()[:7], dtype=float))
        now = time.time()
        if now - last >= 2.0:
            print(f"    matching… {end-now:4.0f}s left  raw={[round(float(x),2) for x in samples[-1]]}",
                  flush=True)
            last = now
        time.sleep(0.25)
    return np.mean(samples[-max(4, len(samples) // 3):], axis=0)


def _goto(robot, label, pose):
    print(f"\n=== {label}: arm MOVING (speed-bounded) — HAND ON E-STOP ===", flush=True)
    for k in (3, 2, 1):
        print(f"    moving in {k}…", flush=True)
        time.sleep(1.0)
    err = robot.leashed_move_to_joint(pose, gripper_cmd=0.0, timeout_s=25.0)
    print(f"    arm at {label} (max joint err {err:.3f} rad).", flush=True)


def _capture(driver, robot, label, pose):
    _goto(robot, label, pose)
    print(f"    >>> MATCH the leader to the arm and HOLD steady ({MATCH_SECONDS}s) <<<", flush=True)
    raw = _read_raw_steady(driver, MATCH_SECONDS)
    rob = np.asarray(robot.get_joint_positions()[:7], dtype=float)
    print(f"    captured {label}: robot={[round(float(x),3) for x in rob]}", flush=True)
    return raw, rob


def calibrate(driver, robot):
    """Two-pose solve -> (offsets_pi2, signs, ok_flags)."""
    rawA, robA = _capture(driver, robot, "POSE A (home)", POSE_A)
    rawB, robB = _capture(driver, robot, "POSE B", POSE_B)
    draw, drob = rawA - rawB, robA - robB
    offs, signs, oks = [], [], []
    print("\n========== SOLVE ==========")
    print(f"{'J':>3} {'sign':>5} {'off(pi/2)':>10} {'resid':>7} {'slope':>7}  quality")
    for i in range(7):
        if abs(draw[i]) < 0.08:
            print(f"J{i+1}   ?          ?       ?       ?   BARELY MOVED — increase POSE_B delta")
            offs.append(0); signs.append(1); oks.append(False); continue
        slope = drob[i] / draw[i]
        s = 1 if slope >= 0 else -1
        off = rawA[i] - s * robA[i]
        mult = int(np.round(off / _PI2))
        resid = off - mult * _PI2
        good = abs(resid) < 0.35 and abs(abs(slope) - 1.0) < 0.4
        print(f"J{i+1} {s:>5} {mult:>10} {resid:>+7.2f} {slope:>+7.2f}  "
              f"{'ok' if good else 'CHECK: match better / bigger delta'}")
        offs.append(mult); signs.append(s); oks.append(good)
    return offs, signs, oks


def write_config(offs, signs):
    """Rewrite the offset/sign tuples in fr3_gello_config.py in place."""
    text = open(CONFIG_PATH).read()
    text, n1 = re.subn(r"FR3_GELLO_OFFSETS_PI_2 = \([^)]*\)",
                       f"FR3_GELLO_OFFSETS_PI_2 = {tuple(offs)}", text)
    text, n2 = re.subn(r"FR3_GELLO_JOINT_SIGNS = \([^)]*\)",
                       f"FR3_GELLO_JOINT_SIGNS = {tuple(signs)}", text)
    if n1 != 1 or n2 != 1:
        print(f"!! could not update config cleanly (offsets={n1}, signs={n2}); paste manually:")
        print(f"   FR3_GELLO_OFFSETS_PI_2 = {tuple(offs)}")
        print(f"   FR3_GELLO_JOINT_SIGNS  = {tuple(signs)}")
        return False
    open(CONFIG_PATH, "w").write(text)
    print(f"\n✓ wrote {CONFIG_PATH}:")
    print(f"    FR3_GELLO_OFFSETS_PI_2 = {tuple(offs)}")
    print(f"    FR3_GELLO_JOINT_SIGNS  = {tuple(signs)}")
    return True


def verify(driver, robot, offs, signs):
    """Drive home; operator matches leader; check per-joint (q_gello - q_robot)."""
    _goto(robot, "VERIFY @ home", POSE_A)
    print("    >>> MATCH the leader to the homed arm and HOLD; checking (~20s) <<<", flush=True)
    offsets = np.array(offs, dtype=float) * _PI2
    signs = np.array(signs, dtype=float)
    end = time.time() + 20
    last_err = None
    while time.time() < end:
        raw = np.asarray(driver.get_joints()[:7], dtype=float)
        qg = signs * (raw - offsets)                       # apply calibration
        qr = np.asarray(robot.get_joint_positions()[:7], dtype=float)
        last_err = qg - qr
        tags = " ".join(f"J{i+1}:{last_err[i]:+.2f}{'ok' if abs(last_err[i])<VERIFY_TOL else 'OFF'}"
                        for i in range(7))
        print(f"    {tags}", flush=True)
        time.sleep(1.0)
    bad = [i + 1 for i in range(7) if abs(last_err[i]) >= VERIFY_TOL]
    if bad:
        print(f"\n✗ VERIFY: joints {bad} still off (>|{VERIFY_TOL}|). "
              f"A ~±1.5 error = wrong sign/offset; a ~0.3-0.6 = wrist/holding wobble.")
    else:
        print("\n✓ VERIFY PASSED — all joints track the robot. Ready to teleop.")
    return not bad


def main():
    ap = argparse.ArgumentParser(description="GELLO 2-pose calibration + verify")
    ap.add_argument("--verify", action="store_true", help="verify current config only (no calibration)")
    ap.add_argument("--no-write", action="store_true", help="calibrate + print, do not edit the config")
    ap.add_argument("--no-verify", action="store_true", help="skip the post-write verify")
    args = ap.parse_args()

    try:
        driver, robot = _connect()
    except Exception as e:
        print(f"CONNECT FAILED: {e!r}\nHome the arm from the dashboard first; free /dev/gello.")
        return 1

    try:
        if args.verify:
            import importlib

            from rlinf.envs.realworld.common.gello import fr3_gello_config as cfg
            importlib.reload(cfg)
            ok = verify(driver, robot, cfg.FR3_GELLO_OFFSETS_PI_2, cfg.FR3_GELLO_JOINT_SIGNS)
            return 0 if ok else 2

        offs, signs, oks = calibrate(driver, robot)
        if not args.no_write:
            write_config(offs, signs)
        else:
            print(f"\n(--no-write) FR3_GELLO_OFFSETS_PI_2 = {tuple(offs)}")
            print(f"(--no-write) FR3_GELLO_JOINT_SIGNS  = {tuple(signs)}")
        if not args.no_verify:
            ok = verify(driver, robot, offs, signs)
            return 0 if ok else 2
        return 0
    finally:
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
