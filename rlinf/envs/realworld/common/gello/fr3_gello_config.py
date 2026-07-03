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

"""FR3 GELLO leader calibration (OpenRB-150 on ``/dev/gello``).

``gello_software`` ships only its stock FTDI ``PORT_CONFIG_MAP`` entries; our
OpenRB-150 leader is not among them, so ``GelloAgent(port="/dev/gello")`` would
otherwise assert ``Port /dev/gello not in config map``.

This module keeps the leader's ``DynamixelRobotConfig`` **in the repo** and
injects it into ``PORT_CONFIG_MAP`` at runtime, rather than hand-editing the
installed ``gello_software`` (which lives in the container's ephemeral layer and
is lost on a container rebuild — as happened on 2026-07-03, dropping the original
2026-06-16 calibration).

Values re-derived 2026-07-03 with ``gello_software/scripts/gello_get_offset.py``
plus per-joint validation against the start pose ``[0,0,0,-1.571,0,1.571,0]``:
offsets are multiples of pi/2; signs are the standard Franka build; the gripper
open/close are servo degrees (0 => open). Re-verify against the homed robot after
any re-assembly of the leader.
"""

import numpy as np

# udev symlink /dev/gello -> ttyACM0 (OpenRB-150). See tasl SETUP.md.
FR3_GELLO_PORT = "/dev/gello"

_PI_2 = np.pi / 2
# joint_offsets expressed in pi/2 units (the calibration resolution).
FR3_GELLO_OFFSETS_PI_2 = (2, 0, 0, 2, 2, 1, 2)
FR3_GELLO_JOINT_SIGNS = (1, -1, 1, -1, 1, 1, 1)
# (gripper servo id, open position deg, closed position deg).
FR3_GELLO_GRIPPER_CONFIG = (8, 264, 222)


def register_fr3_gello_config(port: str = FR3_GELLO_PORT) -> bool:
    """Register the FR3 leader config into gello's ``PORT_CONFIG_MAP`` if absent.

    Idempotent and safe to call before constructing the GELLO agent. Returns
    True if a config is present for ``port`` afterwards, False if the ``gello``
    package is unavailable.
    """
    try:
        from gello.agents.gello_agent import (
            PORT_CONFIG_MAP,
            DynamixelRobotConfig,
        )
    except Exception:
        return False
    if port not in PORT_CONFIG_MAP:
        PORT_CONFIG_MAP[port] = DynamixelRobotConfig(
            joint_ids=(1, 2, 3, 4, 5, 6, 7),
            joint_offsets=tuple(m * _PI_2 for m in FR3_GELLO_OFFSETS_PI_2),
            joint_signs=FR3_GELLO_JOINT_SIGNS,
            gripper_config=FR3_GELLO_GRIPPER_CONFIG,
        )
    return True
