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

from dataclasses import dataclass, field

from lerobot.teleoperators.teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("meta_quest")
@dataclass
class SingleMetaQuestConfig(TeleoperatorConfig):
    """
    Configuration for a single Meta Quest teleoperator.
    This teleoperator reads data from ROS2 topics.
    """

    position_scale: float = 1.0
    rotation_scale: float = 0.5
    action_clip: float = 1.0
    pos_action_scale: int = 500
    rot_action_scale: int = 80
    use_filter: bool = True
    ema_weight: float = 0.05


@TeleoperatorConfig.register_subclass("bi_meta_quest")
@dataclass
class BiMetaQuestConfig(TeleoperatorConfig):
    """
    Configuration for dual-arm Meta Quest teleoperator.
    This teleoperator reads data from ROS2 topics for two controllers.
    """

    position_scale: float = 1.0
    rotation_scale: float = 0.5
    action_clip_left: float = 3.0
    action_clip_right: float = 3.0
    pos_action_scale: int = 195
    rot_action_scale: int = 120
    use_filter: bool = True
    ema_weight: float = 0.1