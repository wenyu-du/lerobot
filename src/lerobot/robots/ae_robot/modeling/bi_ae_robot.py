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
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ...robot import Robot
from ..config import AERobotConfig, BiAERobotConfig
from .ae_robot import AERobot # Import the single arm AERobot

logger = logging.getLogger(__name__)


class BiAERobot(Robot):
    """
    A class to control two custom robot arms, each controlled by an `ae_server`.
    """

    config_class = BiAERobotConfig
    name = "bi_ae_robot"

    def __init__(self, config: BiAERobotConfig):
        super().__init__(config)
        self.config = config
        self.left_arm = AERobot(config.left_arm_config)
        self.right_arm = AERobot(config.right_arm_config)
        self._connected = False

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        # Combine observation features from both arms
        obs_features = {}
        for key, shape in self.left_arm.observation_features.items():
            obs_features[f"left/{key}"] = shape
        for key, shape in self.right_arm.observation_features.items():
            obs_features[f"right/{key}"] = shape
        return obs_features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        # Combine action features from both arms
        action_features = {}
        for key, shape in self.left_arm.action_features.items():
            action_features[f"left/{key}"] = shape
        for key, shape in self.right_arm.action_features.items():
            action_features[f"right/{key}"] = shape
        return action_features

    @property
    def is_connected(self) -> bool:
        return self._connected and self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect both arms in separate threads to speed up
        left_thread = threading.Thread(target=self.left_arm.connect, args=(calibrate,))
        right_thread = threading.Thread(target=self.right_arm.connect, args=(calibrate,))

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()

        self._connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        logger.info("BiAERobot does not require calibration (individual arms are calibrated if needed).")

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get observations from both arms in separate threads
        left_obs = {}
        right_obs = {}

        def get_left_obs():
            nonlocal left_obs
            left_obs = self.left_arm.get_observation()

        def get_right_obs():
            nonlocal right_obs
            right_obs = self.right_arm.get_observation()

        left_thread = threading.Thread(target=get_left_obs)
        right_thread = threading.Thread(target=get_right_obs)

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()

        # Combine observations with prefixes
        combined_obs = {}
        for key, val in left_obs.items():
            combined_obs[f"left/{key}"] = val
        for key, val in right_obs.items():
            combined_obs[f"right/{key}"] = val
        return combined_obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Split action for left and right arms
        left_action = {}
        right_action = {}
        for key, val in action.items():
            if key.startswith("left/"):
                left_action[key[len("left/"):]] = val
            elif key.startswith("right/"):
                right_action[key[len("right/"):]] = val
            else:
                logger.warning(f"Unexpected action key: {key} for BiAERobot. Ignoring.")

        # Send actions to both arms in separate threads
        left_thread = threading.Thread(target=self.left_arm.send_action, args=(left_action,))
        right_thread = threading.Thread(target=self.right_arm.send_action, args=(right_action,))
        
        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()

        return action # Return the original combined action

    def go_to_rest(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_thread = threading.Thread(target=self.left_arm.reset)
        right_thread = threading.Thread(target=self.right_arm.reset)

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_thread = threading.Thread(target=self.left_arm.disconnect)
        right_thread = threading.Thread(target=self.right_arm.disconnect)

        left_thread.start()
        right_thread.start()

        left_thread.join()
        right_thread.join()

        self._connected = False
        logger.info(f"{self} disconnected.")
