### 添加ae_robot 后的record指令

```bash

lerobot-record-aerobot \
    --robot.type=bi_ae_robot \
    --robot.left_arm_config.server_url="http://127.0.0.1:5000" \
    --robot.right_arm_config.server_url="http://127.0.0.1:5001" \
    --policy.path=<path_to_your_policy> \
    --teleop.type=bi_meta_quest \
    --dataset.repo_id="<my_username>/<my_dataset_name>" \
    --dataset.single_task="用两只手臂推动方块"

```

