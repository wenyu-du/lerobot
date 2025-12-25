#!/usr/bin/env python

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

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from lerobot.scripts.lerobot_record_aerobot import record, RecordConfig, DatasetRecordConfig
from lerobot.robots.ae_robot.config import AERobotConfig
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SingleSpaceMouseConfig
from lerobot.configs.policies import PreTrainedConfig


@pytest.fixture
def mock_robot():
    robot = MagicMock()
    robot.name = "ae_robot"
    robot.robot_type = "ae_robot"
    robot.observation_features = {
        "tcp_pose_x": float,
        "tcp_pose_y": float,
        "tcp_pose_z": float,
        "tcp_pose_roll": float,
        "tcp_pose_pitch": float,
        "tcp_pose_yaw": float,
        "gripper_pos": float,
    }
    robot.action_features = {
        "delta_tcp_pose_x": float,
        "delta_tcp_pose_y": float,
        "delta_tcp_pose_z": float,
        "delta_tcp_pose_roll": float,
        "delta_tcp_pose_pitch": float,
        "delta_tcp_pose_yaw": float,
        "gripper_action": float,
    }
    robot.is_connected = False

    def connect():
        robot.is_connected = True

    def disconnect():
        robot.is_connected = False

    robot.connect.side_effect = connect
    robot.disconnect.side_effect = disconnect

    robot.get_observation.return_value = {
        "tcp_pose_x": 0.0,
        "tcp_pose_y": 0.0,
        "tcp_pose_z": 0.0,
        "tcp_pose_roll": 0.0,
        "tcp_pose_pitch": 0.0,
        "tcp_pose_yaw": 0.0,
        "gripper_pos": 0.0,
    }
    return robot


@pytest.fixture
def mock_teleop():
    teleop = MagicMock()
    teleop.is_connected = False

    def connect():
        teleop.is_connected = True

    def disconnect():
        teleop.is_connected = False

    teleop.connect.side_effect = connect
    teleop.disconnect.side_effect = disconnect

    teleop.get_action.return_value = {
        "delta_tcp_pose_x": 1.0,
        "delta_tcp_pose_y": 1.0,
        "delta_tcp_pose_z": 1.0,
        "delta_tcp_pose_roll": 1.0,
        "delta_tcp_pose_pitch": 1.0,
        "delta_tcp_pose_yaw": 1.0,
        "gripper_action": 1.0,
    }
    teleop.get_raw_action.return_value = (np.array([0.0, 0.0]), np.array([0, 0]))
    return teleop


@pytest.fixture
def mock_dataset():
    dataset = MagicMock()
    dataset.meta.stats = {}
    # This reflects what hw_to_dataset_features will generate
    dataset.features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "tcp_pose_x",
                "tcp_pose_y",
                "tcp_pose_z",
                "tcp_pose_roll",
                "tcp_pose_pitch",
                "tcp_pose_yaw",
                "gripper_pos",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "delta_tcp_pose_x",
                "delta_tcp_pose_y",
                "delta_tcp_pose_z",
                "delta_tcp_pose_roll",
                "delta_tcp_pose_pitch",
                "delta_tcp_pose_yaw",
                "gripper_action",
            ],
        },
    }
    dataset.fps = 10
    return dataset

@pytest.fixture
def mock_keyboard_listener():
    listener = MagicMock()
    events = {"stop_recording": False, "exit_early": False, "rerecord_episode": False}
    return listener, events
    
def test_record_script_teleop_only(tmp_path, mock_robot, mock_teleop, mock_dataset, mock_keyboard_listener):
    """
    Test the record script in a teleoperation-only scenario.
    This test mocks the robot, teleoperator, and dataset to ensure the script's main logic
    for recording episodes works as expected without requiring hardware or external services.
    """
    with patch("lerobot.scripts.lerobot_record_aerobot.make_robot_from_config", return_value=mock_robot), \
         patch("lerobot.scripts.lerobot_record_aerobot.make_teleoperator_from_config", return_value=mock_teleop), \
         patch("lerobot.scripts.lerobot_record_aerobot.LeRobotDataset.create", return_value=mock_dataset), \
         patch("lerobot.scripts.lerobot_record_aerobot.init_keyboard_listener", return_value=mock_keyboard_listener), \
         patch("lerobot.scripts.lerobot_record_aerobot.log_say"), \
         patch("lerobot.scripts.lerobot_record_aerobot.is_headless", return_value=True), \
         patch("lerobot.scripts.lerobot_record_aerobot.VideoEncodingManager"):

        repo_id = "test/test_dataset"
        
        robot_cfg = AERobotConfig()
        teleop_cfg = SingleSpaceMouseConfig()
        dataset_cfg = DatasetRecordConfig(
            repo_id=repo_id,
            root=tmp_path,
            single_task="test task",
            fps=10,
            episode_time_s=0.2, # short episode for 2 frames
            reset_time_s=0.1,
            num_episodes=1,
            video=False,
            push_to_hub=False,
        )
        
        cfg = RecordConfig(
            robot=robot_cfg,
            dataset=dataset_cfg,
            teleop=teleop_cfg,
            display_data=False,
            play_sounds=False,
            resume=False,
        )
        
        # Run the record function
        recorded_dataset = record(cfg)

        # Assertions
        mock_robot.connect.assert_called_once()
        mock_teleop.connect.assert_called_once()
        
        # Check that get_observation and send_action were called multiple times in the loop
        assert mock_robot.get_observation.call_count > 0
        assert mock_robot.send_action.call_count > 0
        
        # Check that get_action from teleop was called
        assert mock_teleop.get_action.call_count > 0
        
        # Check that dataset.add_frame was called
        assert mock_dataset.add_frame.call_count > 0
        
        # Check that dataset.save_episode was called
        mock_dataset.save_episode.assert_called_once()

        mock_robot.disconnect.assert_called_once()
        mock_teleop.disconnect.assert_called_once()
        
        mock_dataset.finalize.assert_called_once()
        
        # Check that push_to_hub was not called
        mock_dataset.push_to_hub.assert_not_called()

        assert recorded_dataset == mock_dataset