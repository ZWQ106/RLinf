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

"""Per-joint GELLO calibration + verification (all-in-one).

A single matched pose is underdetermined per joint (``q = sign*(raw - offset)``
is one equation, two unknowns; the SIGN is unobservable without motion), so a
static check can't catch a wrong sign — it passes when still, then runs away in
teleop. We need the arm to MOVE and the operator to match that motion.

Rather than jumping to one pose with all seven joints displaced at once (which
is hard to read — you can't tell which leader joint to twist), this calibrates
**one joint at a time**. First it homes and captures a full reference match.
Then, for each joint i: it drives ONLY joint i away from home (speed-bounded
leash), you twist ONLY that joint on the leader to follow, it captures, solves
that joint's sign + offset from the home→twisted difference::

    sign_i   = (robot_twist_i - robot_home_i) / (raw_twist_i - raw_home_i)   # slope -> +-1
    offset_i = raw_home_i - sign_i * robot_home_i                           # intercept -> pi/2

and returns joint i home before moving to the next. Because each joint moves
alone, it is unambiguous which leader joint to follow.

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
# HOME = the DROID home anchor (the per-joint reference pose). TWIST holds the
# single-joint target each joint is driven to (only that one element differs from
# HOME at any moment). Each joint's displacement is large enough that its slope
# (hence sign) is unambiguous despite operator matching error. The ROLL joints
# (J1/J3/J5/J7) get LARGE deltas (~0.9 rad) — base/wrist rotation is hard to
# match by eye, and a too-small delta is exactly what flipped J1's sign. Pitch
# joints (J2/J4/J6) stay moderate (they carry load / are nearer the table). All
# within FR3 limits, arm stays up.  (POSE_A/POSE_B kept as aliases for back-compat.)
HOME = [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0]
TWIST = [1.00, -0.15, 0.90, -2.00, 0.90, 1.30, 0.90]
POSE_A, POSE_B = HOME, TWIST
MATCH_SECONDS = 10
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


def _capture(driver, robot, label, pose, instr=None):
    _goto(robot, label, pose)
    instr = instr or "MATCH the leader to the arm"
    print(f"    >>> {instr} and HOLD steady ({MATCH_SECONDS}s) <<<", flush=True)
    raw = _read_raw_steady(driver, MATCH_SECONDS)
    rob = np.asarray(robot.get_joint_positions()[:7], dtype=float)
    print(f"    captured {label}: robot={[round(float(x),3) for x in rob]}", flush=True)
    return raw, rob


def _solve_joint(i, draw, drob, raw_home_i, rob_home_i):
    """Solve one joint's (offset_pi2, sign, ok) from its home->twisted deltas."""
    # Unwrap the raw delta into [-pi, pi] — the true per-joint delta is < pi, so a
    # larger |draw| means the servo angle wrapped; fold it back so the slope sign
    # is correct.
    draw = (draw + np.pi) % (2 * np.pi) - np.pi
    if abs(draw) < 0.08:
        print(f"J{i+1}   ?          ?       ?       ?   BARELY MOVED — twist J{i+1} more")
        return 0, 1, False
    slope = drob / draw
    s = 1 if slope >= 0 else -1
    off = raw_home_i - s * rob_home_i
    mult = int(np.round(off / _PI2))
    resid = off - mult * _PI2
    good = abs(resid) < 0.35 and abs(abs(slope) - 1.0) < 0.4
    print(f"J{i+1} {s:>5} {mult:>10} {resid:>+7.2f} {slope:>+7.2f}  "
          f"{'ok' if good else 'CHECK: match better / twist more'}")
    return mult, s, good


def calibrate(driver, robot):
    """Per-joint solve -> (offsets_pi2, signs, ok_flags).

    Home once for a full reference match, then for each joint: twist ONLY that
    joint, solve it, and return home before the next. One joint moves at a time,
    so it is unambiguous which leader joint to follow.
    """
    rawH, robH = _capture(
        driver, robot, "HOME (reference)", HOME,
        instr="MATCH the WHOLE leader to the homed arm",
    )
    offs, signs, oks = [0] * 7, [1] * 7, [False] * 7
    print("\n========== PER-JOINT SOLVE ==========")
    print(f"{'J':>3} {'sign':>5} {'off(pi/2)':>10} {'resid':>7} {'slope':>7}  quality")
    for i in range(7):
        pose = list(HOME)
        pose[i] = TWIST[i]  # move ONLY joint i away from home
        rawJ, robJ = _capture(
            driver, robot, f"JOINT J{i+1}", pose,
            instr=f"twist ONLY joint J{i+1} on the leader to follow (hold the rest)",
        )
        offs[i], signs[i], oks[i] = _solve_joint(
            i, rawJ[i] - rawH[i], robJ[i] - robH[i], rawH[i], robH[i]
        )
        _goto(robot, f"return J{i+1} to home", HOME)  # reset before the next joint
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
