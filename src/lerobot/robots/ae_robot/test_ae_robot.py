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
from scipy.spatial.transform import Rotation as R

from lerobot.robots.ae_robot.config import AERobotConfig, BiAERobotConfig
from lerobot.robots.ae_robot.modeling.ae_robot import AERobot
from lerobot.robots.ae_robot.modeling.bi_ae_robot import BiAERobot


@pytest.fixture
def robot_config():
    """Returns a default AERobot configuration for tests."""
    return AERobotConfig(server_url="http://localhost:5000")

@pytest.fixture
def bi_robot_config():
    """Returns a default BiAERobot configuration for tests."""
    return BiAERobotConfig(
        left_arm_config=AERobotConfig(server_url="http://localhost:5000"),
        right_arm_config=AERobotConfig(server_url="http://localhost:5001"),
    )

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

def test_bi_ae_robot_connection(bi_robot_config):
    """Test connection and disconnection of the BiAERobot."""
    robot = BiAERobot(config=bi_robot_config)
    with requests_mock.Mocker() as m:
        # Mock server endpoints for left arm
        m.post("http://localhost:5000/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5000/startimp", status_code=200)
        m.post("http://localhost:5000/stopimp", status_code=200)
        m.post("http://localhost:5000/update_param", status_code=200) # Added for configure call during connect

        # Mock server endpoints for right arm
        m.post("http://localhost:5001/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5001/startimp", status_code=200)
        m.post("http://localhost:5001/stopimp", status_code=200)
        m.post("http://localhost:5001/update_param", status_code=200) # Added for configure call during connect

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

        assert "tcp_pose_x" in obs
        assert "tcp_pose_y" in obs
        assert "tcp_pose_z" in obs
        assert "tcp_pose_roll" in obs
        assert "tcp_pose_pitch" in obs
        assert "tcp_pose_yaw" in obs
        assert "gripper_pos" in obs

        assert isinstance(obs["tcp_pose_x"], float)
        assert isinstance(obs["gripper_pos"], float)

    # Test with quat
    robot_config.rotation_format = "quat"
    robot = AERobot(config=robot_config)
    with requests_mock.Mocker() as m:
        m.post("http://localhost:5000/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5000/startimp", status_code=200)
        robot.connect()
        obs = robot.get_observation()
        assert "tcp_pose_qx" in obs
        assert "tcp_pose_roll" not in obs

def test_ae_robot_reset(robot_config):
    """Test the reset functionality of AERobot."""
    robot_config.precision_param = {"mode": "precision_mode"}
    robot_config.compliance_param = {"mode": "compliance_mode"} # Added for testing
    robot_config.gripper_reset_command = "open"
    robot = AERobot(config=robot_config)

    with requests_mock.Mocker() as m:
        # Mock server endpoints for connection and observation
        m.post("http://localhost:5000/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5000/startimp", status_code=200)
        
        # Mock endpoints for reset
        m.post("http://localhost:5000/update_param", status_code=200) # For precision and compliance
        m.post("http://localhost:5000/open_gripper", status_code=200)
        m.post("http://localhost:5000/pose", status_code=200)
        
        robot.connect()
        robot.reset()

        # Verify that reset calls were made
        # Expect 3 update_param calls:
        # 1. During connect/configure (compliance)
        # 2. During reset (precision)
        # 3. During reset/configure (compliance)
        update_param_requests = [req for req in m.request_history if "update_param" in req.url]
        assert len(update_param_requests) >= 3 
        assert any(req.json() == {"mode": "precision_mode"} for req in update_param_requests)
        assert any(req.json() == {"mode": "compliance_mode"} for req in update_param_requests)
        
        # Check gripper command
        gripper_request = [req for req in m.request_history if "open_gripper" in req.url][-1]
        assert gripper_request.method == 'POST'
        assert gripper_request.json() == {"position": robot_config.gripper_open_width}

        # Check pose command
        pose_request = [req for req in m.request_history if "pose" in req.url][-1]
        assert pose_request.method == 'POST'
        reset_pose_euler = robot_config.reset_pose
        expected_reset_pose_quat = np.concatenate([reset_pose_euler[:3], R.from_euler("xyz", reset_pose_euler[3:]).as_quat()])
        assert np.allclose(pose_request.json()["arr"], expected_reset_pose_quat.tolist())

def test_bi_ae_robot_go_to_rest(bi_robot_config):
    """Test the go_to_rest functionality of BiAERobot."""
    # Configure each arm's reset behavior
    bi_robot_config.left_arm_config.precision_param = {"mode": "precision_mode_left"}
    bi_robot_config.left_arm_config.gripper_reset_command = "close"
    bi_robot_config.right_arm_config.precision_param = {"mode": "precision_mode_right"}
    bi_robot_config.right_arm_config.gripper_reset_command = "open"

    robot = BiAERobot(config=bi_robot_config)

    with requests_mock.Mocker() as m:
        # Mock server endpoints for connection and observation (left arm)
        m.post("http://localhost:5000/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5000/startimp", status_code=200)
        m.post("http://localhost:5000/update_param", status_code=200)
        m.post("http://localhost:5000/close_gripper", status_code=200)
        m.post("http://localhost:5000/open_gripper", status_code=200) # for generic tests
        m.post("http://localhost:5000/pose", status_code=200)

        # Mock server endpoints for connection and observation (right arm)
        m.post("http://localhost:5001/getstate", json={"pose": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "gripper_pos": 0.0})
        m.post("http://localhost:5001/startimp", status_code=200)
        m.post("http://localhost:5001/update_param", status_code=200)
        m.post("http://localhost:5001/open_gripper", status_code=200)
        m.post("http://localhost:5001/close_gripper", status_code=200) # for generic tests
        m.post("http://localhost:5001/pose", status_code=200)
        
        robot.connect()
        robot.go_to_rest()

        # Verify left arm reset calls
        left_update_param_requests = [req for req in m.request_history if "localhost:5000/update_param" in req.url]
        assert any(req.json() == {"mode": "precision_mode_left"} for req in left_update_param_requests)
        left_gripper_request = [req for req in m.request_history if "localhost:5000/close_gripper" in req.url][-1]
        assert left_gripper_request.json() == {"position": bi_robot_config.left_arm_config.gripper_close_width}
        left_pose_request = [req for req in m.request_history if "localhost:5000/pose" in req.url][-1]
        left_reset_pose_euler = bi_robot_config.left_arm_config.reset_pose
        expected_left_reset_pose_quat = np.concatenate([left_reset_pose_euler[:3], R.from_euler("xyz", left_reset_pose_euler[3:]).as_quat()])
        assert np.allclose(left_pose_request.json()["arr"], expected_left_reset_pose_quat.tolist())

        # Verify right arm reset calls
        right_update_param_requests = [req for req in m.request_history if "localhost:5001/update_param" in req.url]
        assert any(req.json() == {"mode": "precision_mode_right"} for req in right_update_param_requests)
        right_gripper_request = [req for req in m.request_history if "localhost:5001/open_gripper" in req.url][-1]
        assert right_gripper_request.json() == {"position": bi_robot_config.right_arm_config.gripper_open_width}
        right_pose_request = [req for req in m.request_history if "localhost:5001/pose" in req.url][-1]
        right_reset_pose_euler = bi_robot_config.right_arm_config.reset_pose
        expected_right_reset_pose_quat = np.concatenate([right_reset_pose_euler[:3], R.from_euler("xyz", right_reset_pose_euler[3:]).as_quat()])
        assert np.allclose(right_pose_request.json()["arr"], expected_right_reset_pose_quat.tolist())


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
            "delta_tcp_pose_x": 0.01,
            "delta_tcp_pose_y": 0.0,
            "delta_tcp_pose_z": 0.0,
            "delta_tcp_pose_roll": 0.0,
            "delta_tcp_pose_pitch": 0.0,
            "delta_tcp_pose_yaw": 0.0,
            "gripper_action": 1.0,  # Open gripper
        }

        print("\n--- Sent Action ---")
        for key, value in action.items():
            print(f"{key}: {value}")

        sent_action = robot.send_action(action)

        # Verify that the sent action is the same as the input
        assert sent_action == action

        # Check if the correct requests were made
        assert m.called
        pose_request = m.request_history[-2]  # Second to last request
        assert pose_request.method == "POST"
        assert pose_request.url == "http://localhost:5000/pose"

        gripper_request = m.request_history[-1]  # Last request
        assert gripper_request.method == "POST"
        assert "open_gripper" in gripper_request.url

        print("\n--- Mocked API Calls ---")
        print(f"Pose URL called: {pose_request.url}")
        print(f"Pose JSON payload: {pose_request.json()}")
        print(f"Gripper URL called: {gripper_request.url}")
        print(f"Gripper JSON payload: {gripper_request.json()}")
