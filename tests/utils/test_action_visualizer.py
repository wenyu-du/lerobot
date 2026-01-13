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

from unittest.mock import MagicMock, patch
import pytest
import numpy as np


@patch("lerobot.utils.action_visualizer.plt")
def test_action_visualizer(mock_plt):
    """Test ActionVisualizer class."""
    # Since matplotlib is optional, we need to handle its absence
    try:
        from lerobot.utils.action_visualizer import ActionVisualizer
    except ImportError:
        pytest.skip("matplotlib not installed, skipping ActionVisualizer test.")

    # Mock plt functions
    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    visualizer = ActionVisualizer()

    # Check that plot is initialized
    mock_plt.subplots.assert_called_once()
    mock_plt.ion.assert_called_once()
    mock_plt.show.assert_called_with(block=False)

    # Decorate a mock function
    mock_send_action = MagicMock()
    decorated_send_action = visualizer.decorate(mock_send_action)

    # Call the decorated function with a sample action
    action = {
        "action_1": 1.0,
        "action_2": np.array([0.5, -0.5]),
        "action_3": np.array([-1.0]),
    }
    decorated_send_action(action)

    # Check that original function was called
    mock_send_action.assert_called_with(action)

    # Check that plot was updated
    mock_ax.clear.assert_called_once()

    # Check bar plot call
    # Note: dictionary iteration order is guaranteed in Python 3.7+
    args, _ = mock_ax.bar.call_args
    assert list(args[0]) == ["action_1", "action_2_0", "action_2_1", "action_3"]
    assert list(args[1]) == [1.0, 0.5, -0.5, -1.0]

    mock_ax.set_ylabel.assert_called_with("Action Values")
    mock_ax.tick_params.assert_called_with(axis="x", rotation=45)
    mock_fig.tight_layout.assert_called_once()
    mock_fig.canvas.draw.assert_called_once()
    mock_fig.canvas.flush_events.assert_called_once()
