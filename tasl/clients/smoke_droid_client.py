"""Standalone smoke test for DroidLikeClient (no Flask).

Usage:
    /usr/bin/python3 _smoke_droid_client.py
"""
import time

import numpy as np

from clients.droid_client import DroidLikeClient


def main():
    c = DroidLikeClient()
    print("CONNECTED")

    s = c.get_robot_state()
    print(f"q = {[round(float(x), 3) for x in s['joint_positions']]}")
    print(f"gripper_position = {round(s['gripper_position'], 3)}")

    print("\nsending action=zeros for 1 sec at 15Hz...")
    for _ in range(15):
        c.update_command(np.zeros(8))
        time.sleep(1 / 15)
    s2 = c.get_robot_state()
    dq = float(np.abs(s2["joint_positions"] - s["joint_positions"]).max())
    print(f"|q drift| max: {dq:.5f} (expect <0.005)")
    assert dq < 0.005, "zero action moved the robot — BUG"

    print("\nsending action=0.05 on J1 for 1 sec at 15Hz...")
    before = c.get_robot_state()["joint_positions"][0]
    for _ in range(15):
        a = np.zeros(8); a[0] = 0.05
        c.update_command(a)
        time.sleep(1 / 15)
    after = c.get_robot_state()["joint_positions"][0]
    delta = after - before
    print(f"J1 moved: {delta:.4f} rad (expect 0.03-0.20)")

    print("\nopening gripper (action[7]=0) ...")
    for _ in range(15):
        a = np.zeros(8)  # action[7]=0 -> open
        c.update_command(a)
        time.sleep(1 / 15)
    g = c.get_robot_state()["gripper_position"]
    print(f"gripper_position = {g:.3f} (expect close to 0)")

    print("\nclosing gripper (action[7]=1) ...")
    for _ in range(15):
        a = np.zeros(8); a[7] = 1.0
        c.update_command(a)
        time.sleep(1 / 15)
    g = c.get_robot_state()["gripper_position"]
    print(f"gripper_position = {g:.3f} (expect close to 1)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
