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

import numpy as np
import pytest
import requests_mock

from lerobot.robots.ae_robot.modeling.ae_robot import AERobot, AERobotConfig


@pytest.fixture
def robot_config():
    """Returns a default AERobot configuration for tests."""
    return AERobotConfig(server_url="http://localhost:5000")


def test_ae_robot_connection(robot_config):
    """Test connection and disconnection of the AERobot."""
    robot = AERobot(config=robot_config)
    with requests_mock.Mocker() as m:
        # Mock server endpoints
        m.post("http://localhost:5000/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5000/startimp", status_code=200)
        m.post("http://localhost:5000/stopimp", status_code=200)

        assert not robot.is_connected
        robot.connect()
        assert robot.is_connected
        robot.disconnect()
        assert not robot.is_connected


def test_ae_robot_observation(robot_config):
    """Test getting an observation from AERobot."""
    robot = AERobot(config=robot_config)
    with requests_mock.Mocker() as m:
        # Mock server endpoints
        m.post("http://localhost:5000/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5000/startimp", status_code=200)
        
        robot.connect()
        
        obs = robot.get_observation()
        
        print("\n--- Observation ---")
        for key, value in obs.items():
            print(f"{key}: {value}")
        
        assert "tcp_pose" in obs
        assert "gripper_pos" in obs
        assert obs["tcp_pose"].shape == (7,)
        assert obs["gripper_pos"].shape == (1,)


def test_ae_robot_action(robot_config):
    """Test sending an action to AERobot."""
    robot = AERobot(config=robot_config)
    with requests_mock.Mocker() as m:
        # Mock server endpoints for connection and observation
        m.post("http://localhost:5000/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5000/startimp", status_code=200)
        
        # Mock endpoints for sending action
        m.post("http://localhost:5000/pose", status_code=200)
        m.post("http://localhost:5000/open_gripper", status_code=200)
        
        robot.connect()

        # Define a sample action
        action = {
            "delta_tcp_pose": np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "gripper_action": np.array([1.0]),  # Open gripper
        }
        
        print("\n--- Sent Action ---")
        for key, value in action.items():
            print(f"{key}: {value}")
            
        sent_action = robot.send_action(action)
        
        # Verify that the sent action is the same as the input
        assert np.array_equal(sent_action["delta_tcp_pose"], action["delta_tcp_pose"])
        assert np.array_equal(sent_action["gripper_action"], action["gripper_action"])
        
        # Check if the correct requests were made
        assert m.called
        pose_request = m.request_history[-2] # Second to last request
        assert pose_request.method == 'POST'
        assert pose_request.url == 'http://localhost:5000/pose'
        
        gripper_request = m.request_history[-1] # Last request
        assert gripper_request.method == 'POST'
        assert 'open_gripper' in gripper_request.url
        
        print("\n--- Mocked API Calls ---")
        print(f"Pose URL called: {pose_request.url}")
        print(f"Pose JSON payload: {pose_request.json()}")
        print(f"Gripper URL called: {gripper_request.url}")
        print(f"Gripper JSON payload: {gripper_request.json()}")
