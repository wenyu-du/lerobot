# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import multiprocessing
from functools import cached_property
from typing import Any, List, Tuple

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .configuration_spacemouse import SingleSpaceMouseConfig, BiSpaceMouseConfig
from . import pyspacemouse

logger = logging.getLogger(__name__)


class SingleSpaceMouseExpert:
    """
    This class provides an interface to a single SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self, device_number: int = 0):
        try:
            self.devices = [pyspacemouse.open(DeviceNumber=device_number)]
            if not self.devices[0]:
                 raise Exception("Could not open SpaceMouse device.")
        except Exception as e:
            if "No device connected/supported" in str(e) or "Could not open SpaceMouse device." in str(e):
                logger.warning(f"SpaceMouse (device {device_number}) not connected or supported. Running without SpaceMouse.")
                self.process = None
                self.manager = None
                self.latest_data = None
                return
            else:
                raise e

        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0] # Simplified to 2 buttons

        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while True:
            sm_state = self.devices[0].read()
            if not sm_state:
                continue
            
            # The raw action from pyspacemouse is (x, y, z, roll, pitch, yaw)
            # The provided example from copied_aerobot_rl_env/spacemouse/spacemouse_expert.py
            # uses (-state[0].y, state[0].x, state[0].z, -state[0].roll, -state[0].pitch, -state[0].yaw)
            action = [
                -sm_state.y, sm_state.x, sm_state.z,
                -sm_state.roll, -sm_state.pitch, -sm_state.yaw
            ]
            
            buttons = [0, 0]
            if len(sm_state.buttons) >= 2:
                buttons = [sm_state.buttons[0], sm_state.buttons[1]]
            elif len(sm_state.buttons) == 1:
                buttons = [sm_state.buttons[0], 0]

            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        if self.latest_data is None: # SpaceMouse not connected
            return np.zeros(6), [0, 0]
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
        if self.manager:
            self.manager.shutdown()
        if self.devices and self.devices[0]:
            self.devices[0].close()


class BiSpaceMouseExpert:
    """
    This class provides an interface to multiple SpaceMice (up to 2).
    It continuously reads their states and provides
    a "get_action" method to get the latest action and button state for each.
    """

    def __init__(self):
        try:
            # pyspacemouse.open() with no DeviceNumber will try to open all.
            # It returns a list of DeviceSpec objects.
            self.devices = pyspacemouse.open()
            if not self.devices:
                raise Exception("No SpaceMouse devices found or opened.")
        except Exception as e:
            if "No device connected/supported" in str(e) or "No SpaceMouse devices found or opened." in str(e):
                logger.warning("No SpaceMice connected or supported. Running without BiSpaceMouse.")
                self.process = None
                self.manager = None
                self.latest_data = None
                return
            else:
                raise e

        self.manager = multiprocessing.Manager()
        # Store a list of actions and buttons for each connected device
        self.latest_data = self.manager.dict()
        self.latest_data["actions"] = [[0.0] * 6 for _ in self.devices]
        self.latest_data["all_buttons"] = [[0, 0] for _ in self.devices]

        self.process = multiprocessing.Process(target=self._read_spacemice)
        self.process.daemon = True
        self.process.start()

    def _read_spacemice(self):
        while True:
            all_sm_states = pyspacemouse.read_all()
            if not all_sm_states:
                continue

            current_actions = []
            current_all_buttons = []

            for sm_state in all_sm_states:
                if not sm_state:
                    current_actions.append([0.0] * 6)
                    current_all_buttons.append([0, 0])
                    continue

                action = [
                    -sm_state.y, sm_state.x, sm_state.z,
                    -sm_state.roll, -sm_state.pitch, -sm_state.yaw
                ]
                
                buttons = [0, 0]
                if len(sm_state.buttons) >= 2:
                    buttons = [sm_state.buttons[0], sm_state.buttons[1]]
                elif len(sm_state.buttons) == 1:
                    buttons = [sm_state.buttons[0], 0]
                
                current_actions.append(action)
                current_all_buttons.append(buttons)

            self.latest_data["actions"] = current_actions
            self.latest_data["all_buttons"] = current_all_buttons

    def get_actions(self) -> List[Tuple[np.ndarray, list]]:
        """Returns the latest actions and button states for all SpaceMice."""
        if self.latest_data is None: # SpaceMouse not connected
            return []
        
        actions = [np.array(a) for a in self.latest_data["actions"]]
        all_buttons = list(self.latest_data["all_buttons"])

        return list(zip(actions, all_buttons))
    
    def close(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
        if self.manager:
            self.manager.shutdown()
        # pyspacemouse.close() will close all devices opened by pyspacemouse.open()
        pyspacemouse.close()


class SingleSpaceMouse(Teleoperator):
    """
    A teleoperator for a single 3Dconnexion SpaceMouse.
    """

    config_class = SingleSpaceMouseConfig
    name = "spacemouse"

    def __init__(self, config: SingleSpaceMouseConfig):
        super().__init__(config)
        self.expert = SingleSpaceMouseExpert()
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {
            "delta_tcp_pose": (6,),  # dx, dy, dz, d_roll, d_pitch, d_yaw
            "gripper_action": (1,),  # binary open/close
        }

    @property
    def is_connected(self) -> bool:
        return self.expert.process is not None and self._is_connected

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.expert.process is None:
            raise DeviceNotConnectedError(f"SpaceMouse not found or connected during initialization.")

        self._is_connected = True
        logger.info(f"{self} connected.")

    def disconnect(self):
        if not self._is_connected:
            if self.expert.process is None:
                logger.info(f"{self} was not connected (or failed to init). No disconnect needed.")
                return
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.expert.close()
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_action(self) -> dict[str, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action_6d, buttons = self.expert.get_action()

        gripper_action = 0.0
        if buttons[0]: # left button: close
            gripper_action = -1.0
        elif buttons[1]: # right button: open
            gripper_action = 1.0

        return {
            "delta_tcp_pose": action_6d,
            "gripper_action": np.array([gripper_action]),
        }

    def get_raw_action(self) -> Tuple[np.ndarray, list]:
        """Returns the raw 6D action and 2-element button list for intervention detection."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.expert.get_action()

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("SpaceMouse does not require calibration.")

    def configure(self) -> None:
        logger.info("SpaceMouse does not require configuration.")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        logger.debug("SpaceMouse does not support feedback.")


class BiSpaceMouse(Teleoperator):
    """
    A teleoperator for dual-arm control using two 3Dconnexion SpaceMice.
    """

    config_class = BiSpaceMouseConfig
    name = "bi_spacemouse"

    def __init__(self, config: BiSpaceMouseConfig):
        super().__init__(config)
        self.expert = BiSpaceMouseExpert()
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {
            "left_delta_tcp_pose": (6,),
            "left_gripper_action": (1,),
            "right_delta_tcp_pose": (6,),
            "right_gripper_action": (1,),
        }

    @property
    def is_connected(self) -> bool:
        return self.expert.process is not None and self._is_connected

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.expert.process is None:
            raise DeviceNotConnectedError(f"No SpaceMice found or connected during initialization for BiSpaceMouse.")

        if len(self.expert.devices) < 2:
            logger.warning(f"Only {len(self.expert.devices)} SpaceMouse(s) found. BiSpaceMouse requires 2 devices. "
                           "The missing arm will receive zero actions.")
        self._is_connected = True
        logger.info(f"{self} connected.")

    def disconnect(self):
        if not self._is_connected:
            if self.expert.process is None:
                logger.info(f"{self} was not connected (or failed to init). No disconnect needed.")
                return
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.expert.close()
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_action(self) -> dict[str, float | np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        all_actions_buttons = self.expert.get_actions()

        action_dict = {}

        left_action_6d = np.zeros(6)
        left_gripper_action = 0.0
        right_action_6d = np.zeros(6)
        right_gripper_action = 0.0

        if len(all_actions_buttons) > 0:
            action_sm0, buttons_sm0 = all_actions_buttons[0]
            left_action_6d = action_sm0
            if buttons_sm0[0]: left_gripper_action = -1.0
            elif buttons_sm0[1]: left_gripper_action = 1.0
        
        if len(all_actions_buttons) > 1:
            action_sm1, buttons_sm1 = all_actions_buttons[1]
            right_action_6d = action_sm1
            if buttons_sm1[0]: right_gripper_action = -1.0
            elif buttons_sm1[1]: right_gripper_action = 1.0

        action_dict["left_delta_tcp_pose_x"] = left_action_6d[0]
        action_dict["left_delta_tcp_pose_y"] = -left_action_6d[2]
        action_dict["left_delta_tcp_pose_z"] = left_action_6d[1]
        action_dict["left_delta_tcp_pose_roll"] = left_action_6d[3]
        action_dict["left_delta_tcp_pose_pitch"] = -left_action_6d[5]
        action_dict["left_delta_tcp_pose_yaw"] = left_action_6d[4]
        action_dict["left_gripper_action"] = float(left_gripper_action)

        action_dict["right_delta_tcp_pose_x"] = right_action_6d[0]
        action_dict["right_delta_tcp_pose_y"] = right_action_6d[2]
        action_dict["right_delta_tcp_pose_z"] = -right_action_6d[1]
        action_dict["right_delta_tcp_pose_roll"] = right_action_6d[3]
        action_dict["right_delta_tcp_pose_pitch"] = right_action_6d[5]
        action_dict["right_delta_tcp_pose_yaw"] = -right_action_6d[5]
        action_dict["right_gripper_action"] = float(right_gripper_action)

        return action_dict

    def get_raw_action(self) -> Tuple[np.ndarray, list]:
        """
        Returns the raw combined 12D action and 4-element button list for intervention detection.
        (action_sm0, buttons_sm0, action_sm1, buttons_sm1)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        all_actions_buttons = self.expert.get_actions()
        
        combined_action_6d = np.zeros(12)
        combined_buttons = [0, 0, 0, 0]

        if len(all_actions_buttons) > 0:
            action_sm0, buttons_sm0 = all_actions_buttons[0]
            combined_action_6d[:6] = action_sm0
            combined_buttons[0:2] = buttons_sm0
        
        if len(all_actions_buttons) > 1:
            action_sm1, buttons_sm1 = all_actions_buttons[1]
            combined_action_6d[6:] = action_sm1
            combined_buttons[2:4] = buttons_sm1

        return combined_action_6d, combined_buttons

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("SpaceMouse does not require calibration.")

    def configure(self) -> None:
        logger.info("SpaceMouse does not require configuration.")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        logger.debug("SpaceMouse does not support feedback.")