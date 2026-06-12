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

"""P1 hardware smoke for the polymetis controller backend (MOVES THE ROBOT).

Exit criteria: state sanity, joint reset convergence, 10 EE square steps with
<10mm tracking error each, gripper open/close cycle. Drives DroidZerorpcClient
+ conversions directly (no env/ray) so failures localize to the client layer.

Run (Desktop, inside rlinf-eval container, robot powered + FCI active):
    PYTHONPATH=/workspace/rlinf /opt/venv/openpi/bin/python \
        /workspace/rlinf/examples/embodiment/scripts/p1_polymetis_smoke.py
"""

import time

import numpy as np

from rlinf.envs.realworld.franka.droid_zerorpc_client import DroidZerorpcClient
from rlinf.envs.realworld.franka.polymetis_conversions import (
    droid_state_to_franka_fields,
    rlinf_ee_action_to_droid,
)

HOME = [0.0, -0.6283, 0.0, -2.5133, 0.0, 1.8850, 0.0]

c = DroidZerorpcClient(address="tcp://100.75.6.62:4242")
print("bootstrapping (launch_controller + launch_robot, ~10s)...")
c.bootstrap()

# 1. State sanity
f = droid_state_to_franka_fields(c.get_robot_state())
print("q:", np.round(f["arm_joint_position"], 3))
print("tcp:", np.round(f["tcp_pose"], 3))
assert np.isfinite(f["tcp_pose"]).all(), "non-finite tcp pose"

# 2. Joint reset to DROID home with convergence check
print("joint reset to DROID home...")
deadline = time.monotonic() + 15.0
while True:
    c.stream_joint_position(HOME, duration_s=1.0)
    q = droid_state_to_franka_fields(c.get_robot_state())["arm_joint_position"]
    err = np.abs(np.asarray(q) - HOME).max()
    if err < 0.05:
        break
    assert time.monotonic() < deadline, f"joint reset did not converge: err={err:.4f}"
print(f"reset err: {err:.4f} rad")

# 3. 10-step EE square (2cm sides) via streamed cartesian targets
base = droid_state_to_franka_fields(c.get_robot_state())["tcp_pose"]
deltas = [(0.02, 0, 0), (0, 0.02, 0), (-0.02, 0, 0), (0, -0.02, 0)] * 2 + [
    (0.02, 0, 0),
    (-0.02, 0, 0),
]
for i, (dx, dy, dz) in enumerate(deltas):
    target = base.copy()
    target[0] += dx
    target[1] += dy
    target[2] += dz
    pose6 = rlinf_ee_action_to_droid(target)
    for _ in range(15):  # 1s @ 15 Hz per waypoint
        c.update_cartesian_position(pose6)
        time.sleep(1 / 15)
    cur = droid_state_to_franka_fields(c.get_robot_state())["tcp_pose"]
    track = np.linalg.norm(cur[:3] - target[:3])
    print(f"step {i}: tracking err {track * 1000:.1f} mm")
    assert track < 0.01, f"EE tracking diverged at step {i}: {track * 1000:.1f} mm"
    base = target

# 4. Gripper cycle
print("gripper close/open cycle...")
c.update_gripper(1.0, blocking=True)
time.sleep(0.5)
assert droid_state_to_franka_fields(c.get_robot_state())["gripper_open"] is False, (
    "gripper did not close"
)
c.update_gripper(0.0, blocking=True)
time.sleep(0.5)
assert droid_state_to_franka_fields(c.get_robot_state())["gripper_open"] is True, (
    "gripper did not open"
)
print("P1 SMOKE PASS")
