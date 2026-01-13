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
import time
from collections import deque
from functools import cached_property
from typing import Any, Tuple, List

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Int32

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.import_utils import is_rclpy_available

from .configuration_meta_quest import SingleMetaQuestConfig, BiMetaQuestConfig

logger = logging.getLogger(__name__)

Button = Int32


class SingleQuestExpertProcess:
    """
    A separate process to handle ROS2 subscriptions for a single Meta Quest controller.
    """

    def __init__(self, shared_action, shared_buttons, position_scale, rotation_scale, action_clip, use_filter, ema_weight):
        rclpy.init(args=None)
        self.node = SingleQuestSubscriber(
            shared_action,
            shared_buttons,
            position_scale,
            rotation_scale,
            action_clip,
            use_filter,
            ema_weight,
        )


class SingleQuestSubscriber(Node):
    """ROS2 Node for subscribing to a Meta Quest left controller"""

    def __init__(self, shared_action, shared_buttons, position_scale, rotation_scale, action_clip, use_filter, ema_weight):
        super().__init__('single_quest_subscriber')
        self.shared_action = shared_action
        self.shared_buttons = shared_buttons
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.action_clip = action_clip
        self.use_filter = use_filter
        if not 0.0 <= ema_weight <= 1.0:
            raise ValueError(f"ema_weight must be between 0.0 and 1.0, but got {ema_weight}")
        self.ema_weight = ema_weight
        self.filtered_delta = None
        self.pose_history = deque(maxlen=5)

    def button_callback(self, msg: Button):
        self.shared_buttons[0] = 0
        self.shared_buttons[1] = 0

        if msg.data == 1:
            self.shared_buttons[0] = 1
        elif msg.data == -1:
            self.shared_buttons[1] = 1

    def pose_callback(self, msg: TransformStamped):
        position_delta = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z
        ]) * self.position_scale

        delta_quat = np.array([
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w
        ])

        euler_delta = Rotation.from_quat(delta_quat).as_euler('xyz') * self.rotation_scale

        pose_delta = np.concatenate([position_delta, euler_delta])

        if self.use_filter:
            self.pose_history.append(pose_delta)
            if len(self.pose_history) == self.pose_history.maxlen:
                history_array = np.array(list(self.pose_history))
                filtered_pose_delta = np.median(history_array, axis=0)
            else:
                filtered_pose_delta = pose_delta

            if self.filtered_delta is None:
                self.filtered_delta = filtered_pose_delta
            else:
                self.filtered_delta = (
                    self.filtered_delta * (1 - self.ema_weight) + filtered_pose_delta * self.ema_weight
                )
            delta_to_clip = self.filtered_delta
        else:
            delta_to_clip = pose_delta

        clipped_delta = np.clip(
            delta_to_clip,
            -self.action_clip,
            self.action_clip
        )

        for i in range(6):
            self.shared_action[i] = clipped_delta[i]


class SingleMetaQuest(Teleoperator):
    """
    A teleoperator for a single Meta Quest controller, using ROS2 for communication.
    It uses the left controller for 6-DOF control and gripper actions.
    """

    config_class = SingleMetaQuestConfig
    name = "meta_quest"

    def __init__(self, config: SingleMetaQuestConfig):
        super().__init__(config)
        self.config = config
        if not is_rclpy_available():
            raise ImportError("rclpy is not installed. Please install ROS2 and `rclpy` to use the MetaQuest teleoperator.")

        self.manager = multiprocessing.Manager()
        self.shared_action = self.manager.list([0.0] * 6)
        self.shared_buttons = self.manager.list([0, 0])

        self.ros_process = None
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {
            "delta_tcp_pose": (6,),  # dx, dy, dz, d_roll, d_pitch, d_yaw
            "gripper_action": (1,),  # binary open/close
        }

    @cached_property
    def feedback_features(self) -> dict[str, Any]:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected and (self.ros_process is not None and self.ros_process.is_alive())

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.ros_process = multiprocessing.Process(
            target=SingleQuestExpertProcess,
            args=(
                self.shared_action,
                self.shared_buttons,
                self.config.position_scale,
                self.config.rotation_scale,
                self.config.action_clip,
                self.config.use_filter,
                self.config.ema_weight,
            ),
            daemon=True,
        )
        self.ros_process.start()
        time.sleep(2) # Give ROS node some time to spin up

        if not self.ros_process.is_alive():
            raise DeviceNotConnectedError("Meta Quest ROS process failed to start.")

        self._is_connected = True
        logger.info(f"{self} connected.")

    def disconnect(self):
        if not self._is_connected:
            logger.info(f"{self} was not connected (or failed to init). No active disconnect needed.")
            return

        if self.ros_process and self.ros_process.is_alive():
            self.ros_process.terminate()
            self.ros_process.join(timeout=1)
            if self.ros_process.is_alive():
                logger.warning("Meta Quest ROS process did not terminate gracefully.")
        if self.manager:
            self.manager.shutdown()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_action(self) -> dict[str, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = np.array(self.shared_action)
        buttons = list(self.shared_buttons)

        # Apply transformations from original QuestExpert.get_action
        pos_delta = action[:3]
        rot_euler = action[3:]
        transformed_pos = np.array([-pos_delta[0], pos_delta[2], pos_delta[1]])
        transformed_rot_euler = np.array([-rot_euler[0], rot_euler[2], rot_euler[1]])
        transformed_action = np.concatenate([transformed_pos, transformed_rot_euler])

        # Apply action scaling
        action_scale = np.array(
            [self.config.pos_action_scale] * 3 + [self.config.rot_action_scale] * 3
        )
        scaled_action = transformed_action * action_scale

        gripper_action = 0.0
        if buttons[0]:
            gripper_action = -1.0
        elif buttons[1]:
            gripper_action = 1.0

        action_dict = {}
        action_dict["delta_tcp_pose_x"] = scaled_action[0]
        action_dict["delta_tcp_pose_y"] = scaled_action[1]
        action_dict["delta_tcp_pose_z"] = scaled_action[2]
        action_dict["delta_tcp_pose_roll"] = scaled_action[3]
        action_dict["delta_tcp_pose_pitch"] = scaled_action[4]
        action_dict["delta_tcp_pose_yaw"] = scaled_action[5]
        action_dict["gripper_action"] = float(gripper_action)
        return action_dict

    def get_raw_action(self) -> Tuple[np.ndarray, list]:
        """Returns the raw 6D action and 2-element button list from the Meta Quest."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = np.array(self.shared_action)
        buttons = list(self.shared_buttons)
        return action, buttons


class BiQuestExpertProcess:
    """
    A separate process to handle ROS2 subscriptions for two Meta Quest controllers.
    """

    def __init__(self, shared_action, shared_buttons, position_scale, rotation_scale, action_clip_left, action_clip_right, use_filter, ema_weight):
        rclpy.init(args=None)
        self.node = BiQuestSubscriber(
            shared_action,
            shared_buttons,
            position_scale,
            rotation_scale,
            action_clip_left,
            action_clip_right,
            use_filter,
            ema_weight,
        )
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()
            rclpy.shutdown()


class BiQuestSubscriber(Node):
    """ROS2 Node for subscribing to two Meta Quest devices"""

    def __init__(self, shared_action, shared_buttons, position_scale, rotation_scale, action_clip_left, action_clip_right, use_filter, ema_weight):
        super().__init__('biquest_subscriber')
        self.shared_action = shared_action # 12D for 2 controllers
        self.shared_buttons = shared_buttons # 4D for 2 controllers
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.action_clip_left = action_clip_left
        self.action_clip_right = action_clip_right
        self.use_filter = use_filter
        if not 0.0 <= ema_weight <= 1.0:
            raise ValueError(f"ema_weight must be between 0.0 and 1.0, but got {ema_weight}")
        self.ema_weight = ema_weight

        self.filtered_deltas = [None, None] # [left, right]
        self.pose_history = [deque(maxlen=5), deque(maxlen=5)] # [left, right]

        pose_topics = ['/left_controller_instant_delta', '/right_controller_instant_delta']
        button_topics = ['/left_controller_command', '/right_controller_command']

        self.pose_subscriptions = [
            self.create_subscription(
                TransformStamped,
                pose_topics[i],
                lambda msg, device_idx=i: self._pose_callback(msg, device_idx),
                10
            ) for i in range(2)
        ]

        self.button_subscriptions = [
            self.create_subscription(
                Button,
                button_topics[i],
                lambda msg, device_idx=i: self._button_callback(msg, device_idx),
                10
            ) for i in range(2)
        ]

        self.get_logger().info('Initialized dual Meta Quest subscribers')

    def _button_callback(self, msg: Button, device_idx: int):
        # Buttons are 4D: [left_btn1, left_btn2, right_btn1, right_btn2]
        base_idx = device_idx * 2
        self.shared_buttons[base_idx] = 0
        self.shared_buttons[base_idx + 1] = 0

        if msg.data == 1:
            self.shared_buttons[base_idx] = 1
        elif msg.data == -1:
            self.shared_buttons[base_idx + 1] = 1

    def _pose_callback(self, msg: TransformStamped, device_idx: int):
        position_delta = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z
        ]) * self.position_scale

        delta_quat = np.array([
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w
        ])

        euler_delta = Rotation.from_quat(delta_quat).as_euler('xyz') * self.rotation_scale

        pose_delta = np.concatenate([position_delta, euler_delta])

        if self.use_filter:
            self.pose_history[device_idx].append(pose_delta)
            if len(self.pose_history[device_idx]) == self.pose_history[device_idx].maxlen:
                history_array = np.array(list(self.pose_history[device_idx]))
                filtered_pose_delta = np.median(history_array, axis=0)
            else:
                filtered_pose_delta = pose_delta

            if self.filtered_deltas[device_idx] is None:
                self.filtered_deltas[device_idx] = filtered_pose_delta
            else:
                self.filtered_deltas[device_idx] = (
                    self.filtered_deltas[device_idx] * (1 - self.ema_weight)
                    + filtered_pose_delta * self.ema_weight
                )
            delta_to_clip = self.filtered_deltas[device_idx]
        else:
            delta_to_clip = pose_delta

        clip_value = self.action_clip_left if device_idx == 0 else self.action_clip_right
        clipped_delta = np.clip(
            delta_to_clip,
            -clip_value,
            clip_value
        )

        start_idx = device_idx * 6
        for i in range(6):
            self.shared_action[start_idx + i] = clipped_delta[i]


class BiMetaQuest(Teleoperator):
    """
    A teleoperator for dual-arm control using two Meta Quest controllers, via ROS2.
    It controls left and right arms of the robot.
    """

    config_class = BiMetaQuestConfig
    name = "bi_meta_quest"

    def __init__(self, config: BiMetaQuestConfig):
        super().__init__(config)
        self.config = config
        if not is_rclpy_available():
            raise ImportError("rclpy is not installed. Please install ROS2 and `rclpy` to use the MetaQuest teleoperator.")

        self.manager = multiprocessing.Manager()
        self.shared_action = self.manager.list([0.0] * 12) # 2 controllers * 6D action
        self.shared_buttons = self.manager.list([0, 0, 0, 0]) # 2 controllers * 2 buttons

        self.ros_process = None
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {
            "left_delta_tcp_pose": (6,),
            "left_gripper_action": (1,),
            "right_delta_tcp_pose": (6,),
            "right_gripper_action": (1,),
        }

    @cached_property
    def feedback_features(self) -> dict[str, Any]:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected and (self.ros_process is not None and self.ros_process.is_alive())

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.ros_process = multiprocessing.Process(
            target=BiQuestExpertProcess,
            args=(
                self.shared_action,
                self.shared_buttons,
                self.config.position_scale,
                self.config.rotation_scale,
                self.config.action_clip_left,
                self.config.action_clip_right,
                self.config.use_filter,
                self.config.ema_weight,
            ),
            daemon=True,
        )
        self.ros_process.start()
        time.sleep(2) # Give ROS node some time to spin up

        if not self.ros_process.is_alive():
            raise DeviceNotConnectedError("Meta Quest ROS process failed to start.")

        self._is_connected = True
        logger.info(f"{self} connected.")

    def disconnect(self):
        if not self._is_connected:
            logger.info(f"{self} was not connected (or failed to init). No active disconnect needed.")
            return

        if self.ros_process and self.ros_process.is_alive():
            self.ros_process.terminate()
            self.ros_process.join(timeout=1)
            if self.ros_process.is_alive():
                logger.warning("Meta Quest ROS process did not terminate gracefully.")
        if self.manager:
            self.manager.shutdown()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_action(self) -> dict[str, np.ndarray]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        actions_12d = np.array(self.shared_action)
        buttons_4d = list(self.shared_buttons)

        # Split actions and buttons for left and right arms
        left_action_raw = actions_12d[6:12] # Assuming right controller is device_idx=1 (index 6:12 in original)
        right_action_raw = actions_12d[0:6] # Assuming left controller is device_idx=0 (index 0:6 in original)

        left_buttons = buttons_4d[2:4]
        right_buttons = buttons_4d[0:2]

        # Apply transformations for left arm (from SingleMetaQuest)
        pos_delta_l = left_action_raw[:3]
        rot_euler_l = left_action_raw[3:]
        transformed_pos_l = np.array([-pos_delta_l[0], pos_delta_l[2], pos_delta_l[1]])
        transformed_rot_euler_l = np.array([-rot_euler_l[0], rot_euler_l[2], rot_euler_l[1]])
        transformed_action_l = np.concatenate([transformed_pos_l, transformed_rot_euler_l])

        action_scale_l = np.array(
            [self.config.pos_action_scale] * 3 + [self.config.rot_action_scale] * 3
        )
        left_action = transformed_action_l * action_scale_l

        # Apply transformations for right arm (from SingleMetaQuest, but for right controller, assuming similar transform)
        pos_delta_r = right_action_raw[:3]
        rot_euler_r = right_action_raw[3:]
        # Original dual_quest_expert.py uses different transform for right arm: [-x, -z, -y]
        # Transformed right controller transformation: [x, y, z] -> [-x, -z, -y]
        transformed_pos_r = np.array([-pos_delta_r[0], -pos_delta_r[2], -pos_delta_r[1]])
        transformed_rot_euler_r = np.array([-rot_euler_r[0], -rot_euler_r[2], -rot_euler_r[1]])
        transformed_action_r = np.concatenate([transformed_pos_r, transformed_rot_euler_r])

        action_scale_r = np.array(
            [self.config.pos_action_scale] * 3 + [self.config.rot_action_scale] * 3
        )
        right_action = transformed_action_r * action_scale_r

        left_gripper_action = 0.0
        if left_buttons[0]:  # Close gripper
            left_gripper_action = -1.0
        elif left_buttons[1]:  # Open gripper
            left_gripper_action = 1.0

        right_gripper_action = 0.0
        if right_buttons[0]:  # Close gripper
            right_gripper_action = -1.0
        elif right_buttons[1]:  # Open gripper
            right_gripper_action = 1.0

        action_dict = {}
        action_dict["left_delta_tcp_pose_x"] = left_action[0]
        action_dict["left_delta_tcp_pose_y"] = -left_action[1]
        action_dict["left_delta_tcp_pose_z"] = -left_action[2]
        action_dict["left_delta_tcp_pose_roll"] = left_action[3]
        action_dict["left_delta_tcp_pose_pitch"] = -left_action[4]
        action_dict["left_delta_tcp_pose_yaw"] = -left_action[5]
        action_dict["left_gripper_action"] = float(left_gripper_action)

        action_dict["right_delta_tcp_pose_x"] = right_action[0]
        action_dict["right_delta_tcp_pose_y"] = -right_action[1]
        action_dict["right_delta_tcp_pose_z"] = -right_action[2]
        action_dict["right_delta_tcp_pose_roll"] = right_action[3]
        action_dict["right_delta_tcp_pose_pitch"] = -right_action[4]
        action_dict["right_delta_tcp_pose_yaw"] = -right_action[5]
        action_dict["right_gripper_action"] = float(right_gripper_action)

        return action_dict

    def get_raw_action(self) -> Tuple[np.ndarray, list]:
        """Returns the raw 12D action and 4-element button list from the two Meta Quest controllers."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        actions_12d = np.array(self.shared_action)
        buttons_4d = list(self.shared_buttons)
        return actions_12d, buttons_4d