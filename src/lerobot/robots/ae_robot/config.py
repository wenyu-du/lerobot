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

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("ae_robot")
@dataclass
class AERobotConfig(RobotConfig):
    # URL of the ae_server
    server_url: str = "http://127.0.0.1:5000"

    # Workspace limits [x_low, y_low, z_low, roll_low, pitch_low, yaw_low] and [x_high, y_high, z_high, roll_high, pitch_high, yaw_high]
    workspace_limits: dict[str, list[float]] = field(
        default_factory=lambda: {
            "low": [0.2, -0.5, 0.0, -3.14, -1.57, -3.14],
            "high": [0.8, 0.5, 0.6, 3.14, 1.57, 3.14],
        }
    )

    # Action scaling factor for delta pose
    action_scale: tuple[float, float, float] = (0.05, 1.0, 1.0) # xyz, rpy, gripper

    # Reset pose in euler angles [x, y, z, roll, pitch, yaw]
    reset_pose: list[float] = field(
        default_factory=lambda: [0.3, 0.0, 0.2, 3.14, 0.0, 0.0]
    )

    # Gripper settings
    gripper_open_width: float = 0.08
    gripper_close_width: float = 0.0
    gripper_commands: dict[str, str] = field(
        default_factory=lambda: {"open": "open_gripper", "close": "close_gripper"}
    )

    # Compliance parameters for the robot controller
    compliance_param: dict[str, Any] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("bi_ae_robot")
@dataclass
class BiAERobotConfig(RobotConfig):
    # Left arm configuration
    left_arm_config: AERobotConfig = field(default_factory=AERobotConfig)
    # Right arm configuration
    right_arm_config: AERobotConfig = field(default_factory=AERobotConfig)
