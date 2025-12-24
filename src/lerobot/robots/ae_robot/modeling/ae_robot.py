#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ...robot import Robot
from ..config import AERobotConfig

logger = logging.getLogger(__name__)


class AERobot(Robot):
    """
    A class to control a custom robot arm which is controlled by an `ae_server`.
    """

    config_class = AERobotConfig
    name = "ae_robot"

    def __init__(self, config: AERobotConfig):
        super().__init__(config)
        self.config = config
        self.last_gripper_act_time = 0
        self.gripper_sleep_duration = 1.0  # seconds
        self._connected = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _robot_ft(self) -> dict[str, tuple]:
        return {
            "tcp_pose": (7,),  # x, y, z, qx, qy, qz, qw
            "gripper_pos": (1,),
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, tuple]:
        return {**self._robot_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_tcp_pose": (6,),  # dx, dy, dz, d_roll, d_pitch, d_yaw
            "gripper_action": (1,), # binary open/close
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            # Check if server is running
            response = requests.post(f"{self.config.server_url}/getstate")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise DeviceNotConnectedError(f"Failed to connect to ae_server at {self.config.server_url}: {e}")

        # Start impedance control
        try:
            requests.post(f"{self.config.server_url}/startimp").raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not start impedance control: {e}")

        self._connected = True

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # ae_server does not have a calibration concept exposed
        return True

    def calibrate(self) -> None:
        # ae_server does not have a calibration concept exposed, so this is a no-op
        logger.info("AERobot does not require calibration.")

    def configure(self) -> None:
        if self.config.compliance_param:
            try:
                requests.post(f"{self.config.server_url}/update_param", json=self.config.compliance_param).raise_for_status()
                logger.info("Compliance parameters updated.")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to set compliance parameters: {e}")
        
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            response = requests.post(f"{self.config.server_url}/getstate")
            response.raise_for_status()
            state = response.json()
            obs_dict = {
                "tcp_pose": np.array(state["pose"]),
                "gripper_pos": np.array([state["gripper_pos"]]),
            }
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to get observation from ae_server: {e}")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Pose action
        delta_pose = action["delta_tcp_pose"]
        
        # Get current pose
        current_state = self.get_observation()
        current_pose_quat = current_state["tcp_pose"]

        # Apply delta
        target_pose_xyz = current_pose_quat[:3] + delta_pose[:3] * self.config.action_scale[0]

        current_rot = R.from_quat(current_pose_quat[3:])
        delta_rot = R.from_euler("xyz", delta_pose[3:])
        target_rot = delta_rot * current_rot
        target_pose_quat_xyzw = target_rot.as_quat()

        target_pose = np.concatenate([target_pose_xyz, target_pose_quat_xyzw])

        # Clip to workspace
        target_pose = self._clip_to_workspace(target_pose)

        # Send pose command
        try:
            requests.post(f"{self.config.server_url}/pose", json={"arr": target_pose.tolist()}).raise_for_status()
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to send pose to ae_server: {e}")

        # Gripper action
        gripper_action = action["gripper_action"][0]
        current_gripper_pos = current_state["gripper_pos"][0]
        self._send_gripper_command(gripper_action, current_gripper_pos)

        return action
    
    def _send_gripper_command(self, pos: float, current_pos: float):
        """Internal function to send gripper command to the robot."""
        # This is a simplified binary gripper control.
        # pos < -0.5: close, pos > 0.5: open
        # This logic is based on `Ae_DualEnv._send_gripper_command`
        if time.time() - self.last_gripper_act_time < self.gripper_sleep_duration:
            return

        # Assuming gripper fully open is > 0.01 and closed is < 0.01
        # This threshold may need adjustment.
        gripper_open_threshold = 0.01 
        
        if pos < -0.5 and current_pos > gripper_open_threshold:  # Close gripper
            command_url = f"{self.config.server_url}/{self.config.gripper_commands['close']}"
            payload = {"position": self.config.gripper_close_width}
            self.last_gripper_act_time = time.time()
        elif pos > 0.5 and current_pos < gripper_open_threshold: # Open gripper
            command_url = f"{self.config.server_url}/{self.config.gripper_commands['open']}"
            payload = {"position": self.config.gripper_open_width}
            self.last_gripper_act_time = time.time()
        else:
            return

        try:
            requests.post(command_url, json=payload).raise_for_status()
            time.sleep(self.gripper_sleep_duration)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to send gripper command: {e}")

    def _clip_to_workspace(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        low = self.config.workspace_limits["low"]
        high = self.config.workspace_limits["high"]

        pose[:3] = np.clip(pose[:3], low[:3], high[:3])
        
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        
        # Clip euler angles
        euler = np.clip(euler, low[3:], high[3:])
        
        pose[3:] = R.from_euler("xyz", euler).as_quat()

        return pose

    def go_to_rest(self):
        """Send robot to a pre-defined rest pose."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        reset_pose_euler = self.config.reset_pose
        reset_pose_quat = np.concatenate([reset_pose_euler[:3], R.from_euler("xyz", reset_pose_euler[3:]).as_quat()])
        
        try:
            # Maybe interpolate? For now, just a direct command.
             requests.post(f"{self.config.server_url}/pose", json={"arr": reset_pose_quat.tolist()}).raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to go to rest pose: {e}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            requests.post(f"{self.config.server_url}/stopimp").raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not stop impedance control: {e}")

        for cam in self.cameras.values():
            cam.disconnect()

        self._connected = False
        logger.info(f"{self} disconnected.")
