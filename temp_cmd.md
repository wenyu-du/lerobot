### 添加ae_robot 后的record指令

```bash

lerobot-record-aerobot \
    --robot.type=bi_ae_robot \
    --config_path=/home/ae/project/lerobot/src/lerobot/robots/ae_robot/comprehensive_aerobot_record_config.yaml \
    --teleop.type=bi_spacemouse \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv,index_or_path: 6, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id="wenyudu/test_ae_data" \
    --dataset.single_task="用两只手臂推动方块"

```


