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

import subprocess
import time
import numpy as np
import pytest
import zerorpc
import importlib.util

# Load droid_zerorpc_client directly to avoid triggering full rlinf/__init__
spec = importlib.util.spec_from_file_location(
    "droid_zerorpc_client",
    "rlinf/envs/realworld/franka/droid_zerorpc_client.py"
)
client_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_module)

DroidZerorpcClient = client_module.DroidZerorpcClient


class FakeDroidServer:
    def __init__(self):
        self.calls = []

    def launch_controller(self):
        self.calls.append("launch_controller")

    def launch_robot(self):
        self.calls.append("launch_robot")

    def get_robot_state(self):
        state = {
            "joint_positions": [0.0] * 7,
            "joint_velocities": [0.0] * 7,
            "joint_torques_computed": [0.0] * 7,
            "motor_torques_measured": [0.0] * 7,
            "cartesian_position": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
            "gripper_position": 0.0,
            "prev_command_successful": True,
            "prev_controller_latency_ms": 1.0,
        }
        ts = {"robot_timestamp_seconds": 1, "robot_timestamp_nanos": 2}
        return [state, ts]

    def update_command(self, action, action_space, gripper_action_space, blocking):
        self.calls.append(("update_command", list(action), action_space,
                           gripper_action_space, blocking))

    def get_calls(self):
        """RPC-callable for tests to fetch recorded calls from subprocess."""
        return self.calls


def _fake_server_entry():
    """Subprocess entry point: run the FakeDroidServer."""
    srv = FakeDroidServer()
    s = zerorpc.Server(srv)
    s.bind("tcp://127.0.0.1:14242")
    s.run()


@pytest.fixture
def server():
    """Subprocess-based fixture to avoid gevent hang."""
    proc = subprocess.Popen(
        ["python3", "-c", (
            "import sys; sys.path.insert(0, '.'); "
            "from tests.unit_tests.realworld.test_droid_zerorpc_client import _fake_server_entry; "
            "_fake_server_entry()"
        )],
        cwd="/Users/wenqianzhang/Desktop/Research/franka_r3/vendor/RLinf"
    )
    # Wait for server to start
    time.sleep(0.5)

    # Client to fetch calls from the server
    calls_client = zerorpc.Client()
    calls_client.connect("tcp://127.0.0.1:14242")

    class CallsProxy:
        def __getitem__(self, idx):
            calls = calls_client.get_calls()
            return calls[idx]
        def __len__(self):
            return len(calls_client.get_calls())

    yield CallsProxy()

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    calls_client.close()


def test_state_unwrap(server):
    c = DroidZerorpcClient(address="tcp://127.0.0.1:14242")
    st = c.get_robot_state()
    assert st["timestamp_seconds"] == 1
    np.testing.assert_allclose(st["cartesian_position"], [0.4, 0, 0.3, 0, 0, 0])


def test_cartesian_command_is_positional(server):
    c = DroidZerorpcClient(address="tcp://127.0.0.1:14242")
    c.update_cartesian_position(np.array([0.4, 0, 0.3, 0, 0, 0]), gripper_cmd=0.2)
    name, action, space, gspace, blocking = server[-1]
    assert name == "update_command"
    assert space == "cartesian_position" and gspace == "position"
    assert action == [0.4, 0, 0.3, 0, 0, 0, 0.2] and blocking is False
