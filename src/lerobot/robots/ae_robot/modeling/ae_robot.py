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
from ..config import AERobotArmConfig, AERobotConfig

logger = logging.getLogger(__name__)


class AERobot(Robot):
    """
    A class to control a custom robot arm which is controlled by an `ae_server`.
    """

    config_class = AERobotConfig
    name = "ae_robot"

    def __init__(self, config: AERobotArmConfig, robot_id: str | None = None):
        # Dynamically create an AERobotConfig (which is a RobotConfig) from AERobotArmConfig
        full_robot_config = AERobotConfig()
        
        # Copy fields from AERobotArmConfig to full_robot_config
        for attr in AERobotArmConfig.__dataclass_fields__:
            if hasattr(config, attr):
                setattr(full_robot_config, attr, getattr(config, attr))
        
        # Set id for the full_robot_config, using passed robot_id or a default
        full_robot_config.id = robot_id if robot_id else "aerobot_component"
        # calibration_dir will be handled by RobotConfig's default or through robot_id if needed
        
        super().__init__(full_robot_config) # Call Robot's __init__ with a proper RobotConfig
        self.config = config # Keep the original AERobotArmConfig for AERobot's specific use.
        self.last_gripper_act_time = 0
        self.gripper_sleep_duration = 1.0  # seconds
        self._connected = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        obs_fts = {}
        if self.config.rotation_format == "quat":
            obs_fts.update(
                {
                    "tcp_pose_x": float,
                    "tcp_pose_y": float,
                    "tcp_pose_z": float,
                    "tcp_pose_qx": float,
                    "tcp_pose_qy": float,
                    "tcp_pose_qz": float,
                    "tcp_pose_qw": float,
                }
            )
        else:
            obs_fts.update(
                {
                    "tcp_pose_x": float,
                    "tcp_pose_y": float,
                    "tcp_pose_z": float,
                    "tcp_pose_roll": float,
                    "tcp_pose_pitch": float,
                    "tcp_pose_yaw": float,
                }
            )
        obs_fts["gripper_pos"] = float

        for cam in self.cameras:
            obs_fts[cam] = (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)

        return obs_fts

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {
            "delta_tcp_pose_x": float,
            "delta_tcp_pose_y": float,
            "delta_tcp_pose_z": float,
            "delta_tcp_pose_roll": float,
            "delta_tcp_pose_pitch": float,
            "delta_tcp_pose_yaw": float,
            "gripper_action": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        last_exception = None
        for attempt in range(3):
            try:
                # Check if server is running
                response = requests.post(f"{self.config.server_url}/getstate")
                response.raise_for_status()
                
                # If we get a successful response, break the loop
                logger.info(f"Successfully connected to ae_server on attempt {attempt + 1}.")
                last_exception = None
                break
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in 1 second...")
                time.sleep(1)
        
        if last_exception is not None:
            raise DeviceNotConnectedError(f"Failed to connect to ae_server at {self.config.server_url} after multiple attempts: {last_exception}")

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

    def _send_pose_command(self, target_pose: np.ndarray, timeout: float = 0) -> None:
        """
        Sends a pose command to the robot server.
        If a timeout is provided, it will wait for that duration after sending the command.
        This simulates an interpolated move if the robot server handles interpolation.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            requests.post(f"{self.config.server_url}/pose", json={"arr": target_pose.tolist()}).raise_for_status()
            if timeout > 0:
                time.sleep(timeout)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send pose command: {e}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            response = requests.post(f"{self.config.server_url}/getstate")
            response.raise_for_status()
            state = response.json()

            tcp_pose_quat = np.array(state["pose"])
            obs_dict = {}

            if self.config.rotation_format == "quat":
                obs_dict["tcp_pose_x"] = tcp_pose_quat[0]
                obs_dict["tcp_pose_y"] = tcp_pose_quat[1]
                obs_dict["tcp_pose_z"] = tcp_pose_quat[2]
                obs_dict["tcp_pose_qx"] = tcp_pose_quat[3]
                obs_dict["tcp_pose_qy"] = tcp_pose_quat[4]
                obs_dict["tcp_pose_qz"] = tcp_pose_quat[5]
                obs_dict["tcp_pose_qw"] = tcp_pose_quat[6]
            else:
                rot = R.from_quat(tcp_pose_quat[3:])
                euler = rot.as_euler(self.config.rotation_format)
                obs_dict["tcp_pose_x"] = tcp_pose_quat[0]
                obs_dict["tcp_pose_y"] = tcp_pose_quat[1]
                obs_dict["tcp_pose_z"] = tcp_pose_quat[2]
                obs_dict["tcp_pose_roll"] = euler[0]
                obs_dict["tcp_pose_pitch"] = euler[1]
                obs_dict["tcp_pose_yaw"] = euler[2]

            obs_dict["gripper_pos"] = state["gripper_pos"]

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
        delta_pose = np.array(
            [
                action["delta_tcp_pose_x"],
                action["delta_tcp_pose_y"],
                action["delta_tcp_pose_z"],
                action["delta_tcp_pose_roll"],
                action["delta_tcp_pose_pitch"],
                action["delta_tcp_pose_yaw"],
            ]
        )

        # Get current pose
        current_obs = self.get_observation()

        current_pose_xyz = np.array([current_obs["tcp_pose_x"], current_obs["tcp_pose_y"], current_obs["tcp_pose_z"]])

        if self.config.rotation_format == "quat":
            current_pose_quat_rot = np.array(
                [
                    current_obs["tcp_pose_qx"],
                    current_obs["tcp_pose_qy"],
                    current_obs["tcp_pose_qz"],
                    current_obs["tcp_pose_qw"],
                ]
            )
            current_rot = R.from_quat(current_pose_quat_rot)
        else:
            current_pose_euler_rot = np.array(
                [
                    current_obs["tcp_pose_roll"],
                    current_obs["tcp_pose_pitch"],
                    current_obs["tcp_pose_yaw"],
                ]
            )
            current_rot = R.from_euler(self.config.rotation_format, current_pose_euler_rot)

        target_pose_xyz = current_pose_xyz + delta_pose[:3] * self.config.action_scale[0]

        delta_rot = R.from_euler("xyz", delta_pose[3:]* self.config.action_scale[1])
        target_rot = delta_rot * current_rot
        # target_rot =  current_rot * delta_rot
        target_pose_quat_xyzw = target_rot.as_quat()

        target_pose = np.concatenate([target_pose_xyz, target_pose_quat_xyzw])

        # Clip to workspace
        target_pose = self._clip_to_workspace(target_pose)

        # Send pose command
        self._send_pose_command(target_pose)

        # Gripper action
        gripper_action = action["gripper_action"]
        current_gripper_pos = current_obs["gripper_pos"]
        self._send_gripper_command(gripper_action, current_gripper_pos)

        return action
    
    def _send_gripper_command(self, pos: float, current_pos: float):
        """Internal function to send gripper command to the robot."""
        # This is a simplified binary gripper control.
        # pos < -0.5: close, pos > 0.5: open
        
        # Ensure minimum time between gripper actions
        if time.time() - self.last_gripper_act_time < self.gripper_sleep_duration:
            return

        # Define gripper open/close thresholds based on config
        # Assuming a smaller value means more closed, larger means more open.
        # This needs to be consistent with the actual gripper feedback.
        gripper_close_threshold = self.config.gripper_close_width + 0.005 # A bit above fully closed
        gripper_open_threshold = self.config.gripper_open_width - 0.005  # A bit below fully open
        
        command_url = None
        payload = None

        if pos < -0.5 and current_pos > gripper_close_threshold:  # Close gripper
            command_url = f"{self.config.server_url}/{self.config.gripper_commands['close']}"
            payload = {"position": self.config.gripper_close_width}
        elif pos > 0.5 and current_pos < gripper_open_threshold: # Open gripper
            command_url = f"{self.config.server_url}/{self.config.gripper_commands['open']}"
            payload = {"position": self.config.gripper_open_width}
        else:
            return # No action needed or not meeting criteria

        if command_url and payload:
            try:
                requests.post(command_url, json=payload).raise_for_status()
                self.last_gripper_act_time = time.time() # Update last action time after successful command
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

    def reset(self) -> None:
        """
        Resets the robot to its initial configuration and state.
        This includes setting precision mode, resetting the gripper,
        moving to the reset pose, and then setting compliance mode.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.last_gripper_act_time = 0 # Reset gripper timer to allow immediate action

        # Set precision parameters for reset movement
        if self.config.precision_param:
            try:
                requests.post(f"{self.config.server_url}/update_param", json=self.config.precision_param).raise_for_status()
                logger.info("Precision parameters set for reset.")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to set precision parameters for reset: {e}")
        
        # Set gripper to reset state
        width_key = "gripper_open_width" if self.config.gripper_reset_command == "open" else "gripper_close_width"
        width = getattr(self.config, width_key, None)
        gripper_command_url = f"{self.config.server_url}/{self.config.gripper_commands[self.config.gripper_reset_command]}"
        try:
            requests.post(gripper_command_url, json={"position": width}).raise_for_status()
            self.last_gripper_act_time = time.time() # Update last action time
            time.sleep(self.gripper_sleep_duration) # Wait for gripper action to complete
            logger.info(f"Gripper reset command '{self.config.gripper_reset_command}' sent.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to send gripper reset command: {e}")

        # Move to reset pose
        reset_pose_euler = self.config.reset_pose
        reset_pose_quat = np.concatenate([reset_pose_euler[:3], R.from_euler("xyz", reset_pose_euler[3:]).as_quat()])
        self._send_pose_command(reset_pose_quat, timeout=1.2) # Use timeout for reset movement
        logger.info("Robot moved to reset pose.")

        # Restore compliance parameters
        self.configure() # This will apply compliance_param if configured
        logger.info("Compliance parameters restored after reset.")

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
