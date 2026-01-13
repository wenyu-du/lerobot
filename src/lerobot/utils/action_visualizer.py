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

from functools import wraps

# Matplotlib is an optional dependency
try:
    import matplotlib.pyplot as plt
    import numpy as np

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


class ActionVisualizer:
    def __init__(self):
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is not installed. Please `pip install matplotlib` to use ActionVisualizer.")
        self.fig, self.ax = plt.subplots()
        plt.ion()
        plt.show(block=False)

    def decorate(self, func):
        @wraps(func)
        def wrapper(action, *args, **kwargs):
            if isinstance(action, dict):
                flat_action = {}
                for k, v in action.items():
                    if isinstance(v, (np.ndarray, list)) and np.size(v) > 1:
                        for i, val_i in enumerate(np.ravel(v)):
                            flat_action[f"{k}_{i}"] = val_i
                    else:
                        flat_action[k] = v.item() if hasattr(v, "item") else v

                labels = list(flat_action.keys())
                values = list(flat_action.values())

                self.ax.clear()
                self.ax.bar(labels, values)
                self.ax.set_ylabel("Action Values")
                self.ax.tick_params(axis="x", rotation=45)
                self.fig.tight_layout()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            return func(action, *args, **kwargs)

        return wrapper
