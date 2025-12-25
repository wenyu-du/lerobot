You are right to be cautious about changing framework code. I have reverted the change to `src/lerobot/datasets/utils.py`. As you can see from the test run, it now fails with a `KeyError: 'names'`.

Here is a brief explanation of why this happens:

The function `build_dataset_frame` in `utils.py` prepares the robot's data for saving. The original version assumes that any 1D vector of numbers (like `delta_tcp_pose`) is composed of several individual named values and looks for them in a `names` field. This field doesn't exist for the data coming from `AERobot`, causing the `KeyError`. My previous modification was a bug fix to handle this case.

Since you prefer not to modify this core file, we have a few options:

1.  **Re-apply the fix to `src/lerobot/datasets/utils.py`**. This is the cleanest solution as it fixes the underlying issue in the framework.
2.  **Modify the test script to work around the bug**. This would make the test pass, but it would not fix the actual problem and would make the test more complex.
3.  **Leave `utils.py` as is**, which means we won't have a passing test for this script at this time.

Please let me know how you'd like to proceed.
