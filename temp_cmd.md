### 添加ae_robot 后的record指令

```bash

lerobot-record-aerobot \
    --robot.type=bi_ae_robot \
    --config-path my_bi_aerobot_config.yaml \
    --teleop.type=bi_spacemouse \
    --dataset.repo_id="wenyudu/test_ae_data" \
    --dataset.single_task="用两只手臂推动方块"

```

