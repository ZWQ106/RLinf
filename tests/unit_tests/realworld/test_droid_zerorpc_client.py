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
import sys
import socket
import importlib.util
from pathlib import Path

import numpy as np
import pytest

zerorpc = pytest.importorskip("zerorpc")

REPO_ROOT = Path(__file__).resolve().parents[3]

# Load droid_zerorpc_client directly to avoid triggering full rlinf/__init__
spec = importlib.util.spec_from_file_location(
    "droid_zerorpc_client",
    str(REPO_ROOT / "rlinf/envs/realworld/franka/droid_zerorpc_client.py")
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


def _fake_server_entry(port):
    """Subprocess entry point: run the FakeDroidServer."""
    srv = FakeDroidServer()
    s = zerorpc.Server(srv)
    s.bind(f"tcp://127.0.0.1:{port}")
    s.run()


@pytest.fixture
def server():
    """Subprocess-based fixture to avoid gevent hang.

    Yields a tuple of (CallsProxy, address_string).
    """
    # Allocate a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    address = f"tcp://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, '.'); "
            "from tests.unit_tests.realworld.test_droid_zerorpc_client import _fake_server_entry; "
            f"_fake_server_entry({port})"
        )],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to be ready with exponential backoff
    calls_client = zerorpc.Client()
    max_wait = 5.0
    waited = 0.0
    interval = 0.1
    ready = False

    while waited < max_wait:
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            pytest.fail(
                f"Server subprocess exited with code {proc.returncode}. "
                f"stderr: {stderr.decode()}"
            )
        try:
            calls_client.connect(address)
            calls_client.get_calls()  # ping
            ready = True
            break
        except Exception:
            time.sleep(interval)
            waited += interval

    if not ready:
        proc.kill()
        pytest.fail(f"Server at {address} failed to become ready within {max_wait}s")

    class CallsProxy:
        def __getitem__(self, idx):
            calls = calls_client.get_calls()
            return calls[idx]
        def __len__(self):
            return len(calls_client.get_calls())

    yield CallsProxy(), address

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
    calls_client.close()


def _make_client(address=None):
    """Helper to create a client; tests close it at the end."""
    if address is None:
        # Determine address from the server fixture; tests override this
        address = "tcp://127.0.0.1:14242"
    return DroidZerorpcClient(address=address)


def test_state_unwrap(server):
    calls_proxy, address = server
    c = DroidZerorpcClient(address=address)
    try:
        st = c.get_robot_state()
        assert st["timestamp_seconds"] == 1
        np.testing.assert_allclose(st["cartesian_position"], [0.4, 0, 0.3, 0, 0, 0])
    finally:
        c.close()


def test_cartesian_command_is_positional(server):
    calls_proxy, address = server
    c = DroidZerorpcClient(address=address)
    try:
        c.update_cartesian_position(np.array([0.4, 0, 0.3, 0, 0, 0]), gripper_cmd=0.2)
        name, action, space, gspace, blocking = calls_proxy[-1]
        assert name == "update_command"
        assert space == "cartesian_position" and gspace == "position"
        assert action == [0.4, 0, 0.3, 0, 0, 0, 0.2] and blocking is False
    finally:
        c.close()


def test_joint_velocity_command_is_positional(server):
    calls_proxy, address = server
    c = DroidZerorpcClient(address=address)
    try:
        c.update_joint_velocity(np.array([0.1]*7 + [0.3]))
        name, action, space, gspace, blocking = calls_proxy[-1]
        assert name == "update_command"
        assert space == "joint_velocity" and gspace == "position"
        assert len(action) == 8 and blocking is False
    finally:
        c.close()
