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

"""Two-pose GELLO calibration — recover joint SIGN and OFFSET from the robot.

A single matched pose is underdetermined per joint: ``q = sign*(raw - offset)``
is one equation with two unknowns, and the SIGN (the slope dq/draw) is
unobservable without motion. So a static "hold the leader to match the arm"
check can set ``offset`` only if ``sign`` is already known — and can't catch a
wrong sign (it passes when still, then runs away in teleop).

This tool drives the arm to TWO poses (speed-bounded leash), you match the leader
at each, and it solves both from the DIFFERENCE between poses::

    sign_i   = (robotA_i - robotB_i) / (rawA_i - rawB_i)     # slope -> +-1
    offset_i = rawA_i - sign_i * robotA_i                    # intercept -> snap pi/2

It prints ``FR3_GELLO_OFFSETS_PI_2`` + ``FR3_GELLO_JOINT_SIGNS`` to paste into
``fr3_gello_config.py``.

SAFETY: the arm MOVES to two poses. Keep a hand on the E-stop. The controller
must be live (home the arm from the dashboard first). Run inside rlinf-eval::

    PYTHONPATH=/workspace/rlinf /opt/venv/openpi/bin/python \
        -m rlinf.envs.realworld.common.gello.calibrate_2pose
"""

import sys
import time

import numpy as np

ROBOT_ADDR = "tcp://172.16.0.2:4242"
PORT = "/dev/gello"
BAUD = 57600
IDS = [1, 2, 3, 4, 5, 6, 7]
_PI2 = np.pi / 2
# Pose A = the DROID home anchor. Pose B moves every joint ~0.3-0.4 rad (all
# within FR3 limits, arm stays up) so each joint has a clear slope to solve.
POSE_A = [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0]
POSE_B = [0.40, -0.35, 0.40, -2.20, 0.40, 1.50, 0.40]
MATCH_SECONDS = 25


def _match_and_read_raw(driver, secs):
    """Sample raw leader joints while the operator matches; return steady mean."""
    end = time.time() + secs
    samples = []
    last_print = 0.0
    while time.time() < end:
        q = np.asarray(driver.get_joints()[:7], dtype=float)
        samples.append(q)
        now = time.time()
        if now - last_print >= 2.0:
            print(f"    matching… {end-now:4.0f}s left  raw={[round(float(x),2) for x in q]}",
                  flush=True)
            last_print = now
        time.sleep(0.25)
    tail = samples[-max(4, len(samples) // 3):]  # steadiest last third
    return np.mean(tail, axis=0)


def _capture(driver, robot, label, pose):
    print(f"\n=== POSE {label}: arm MOVING (speed-bounded) — HAND ON E-STOP ===", flush=True)
    for k in (3, 2, 1):
        print(f"    moving in {k}…", flush=True); time.sleep(1.0)
    err = robot.leashed_move_to_joint(pose, gripper_cmd=0.0, timeout_s=25.0)
    print(f"    arm at {label} (max joint err {err:.3f} rad).", flush=True)
    print(f"    >>> MATCH the leader to the arm now and HOLD steady ({MATCH_SECONDS}s) <<<",
          flush=True)
    raw = _match_and_read_raw(driver, MATCH_SECONDS)
    rob = np.asarray(robot.get_joint_positions()[:7], dtype=float)
    print(f"    captured {label}: robot={[round(float(x),3) for x in rob]}", flush=True)
    return raw, rob


def main():
    from gello.dynamixel.driver import DynamixelDriver

    from rlinf.envs.realworld.franka.droid_zerorpc_client import DroidClient

    print("connecting to leader (/dev/gello) + robot…", flush=True)
    driver = DynamixelDriver(IDS, port=PORT, baudrate=BAUD)
    robot = DroidClient(address=ROBOT_ADDR)
    try:
        robot.get_robot_state()  # fail fast if controller not live
    except Exception as e:
        print(f"ROBOT NOT REACHABLE/LIVE: {e!r}\nHome the arm from the dashboard first.")
        return

    rawA, robA = _capture(driver, robot, "A (home)", POSE_A)
    rawB, robB = _capture(driver, robot, "B", POSE_B)
    robot.close()

    draw, drob = rawA - rawB, robA - robB
    offs, signs = [], []
    print("\n========== RESULT ==========")
    print(f"{'J':>3} {'sign':>5} {'off(pi/2)':>10} {'resid':>7} {'slope':>7}  quality")
    for i in range(7):
        if abs(draw[i]) < 0.08:
            print(f"J{i+1}    ?         ?       ?       ?   BARELY MOVED — increase POSE_B delta")
            offs.append(None); signs.append(None); continue
        slope = drob[i] / draw[i]
        s = 1 if slope >= 0 else -1
        off = rawA[i] - s * robA[i]
        mult = int(np.round(off / _PI2))
        resid = off - mult * _PI2
        good = abs(resid) < 0.35 and abs(abs(slope) - 1.0) < 0.4
        print(f"J{i+1} {s:>5} {mult:>10} {resid:>+7.2f} {slope:>+7.2f}  "
              f"{'ok' if good else 'CHECK: match better / bigger delta'}")
        offs.append(mult); signs.append(s)
    print("\nPaste into fr3_gello_config.py:")
    print(f"FR3_GELLO_OFFSETS_PI_2 = {tuple(offs)}")
    print(f"FR3_GELLO_JOINT_SIGNS  = {tuple(signs)}")


if __name__ == "__main__":
    main()
