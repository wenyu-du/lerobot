

#### 模拟测试脚本 Send a pose via request

**Right Arm**
curl -X POST http://192.168.2.22:5002/startimp
curl -X POST http://192.168.2.22:5002/pose -H "Content-Type: application/json" -d '{"arr": [0.35469, -0.179459, -0.001, 0.5918833489617783 ,-0.561919465067565, 0.40305117887470404, 0.4140898008979244]}'


**Left Arm**
curl -X POST http://192.168.2.22:5003/startimp
curl -X POST http://192.168.2.22:5003/pose -H "Content-Type: application/json" -d '{"arr": [0.44351, 0.223, 0.434, -0.40998122113813656, -0.3103788911034263, -0.5997600950272846, 0.6130808842920747]}'


**下电**
curl -X POST http://192.168.2.22:5002/stopimp
curl -X POST http://192.168.2.22:5003/stopimp

#### 测试工作空间


```bash
sudo rm -r ~/.cache/huggingface/lerobot/wenyudu/260107_aess
```
### 录数据脚本
```bash
lerobot-record-aerobot     --robot.type=bi_ae_robot     --config_path=/home/ae/project/lerobot/src/lerobot/robots/ae_robot/comprehensive_aerobot_record_config.yaml     --teleop.type=bi_spacemouse --dataset.single_task="pick the black cable, then place into the fixture"  --dataset.num_episodes=20 --dataset.push_to_hub=true --dataset.episode_time_s=60 --dataset.fps=10 --dataset.repo_id="wenyudu/260107_ae_alpha"
```

```bash
lerobot-teleoperate     --robot.type=bi_ae_robot     --config_path=/home/ae/project/lerobot/src/lerobot/robots/ae_robot/comprehensive_aerobot_record_config.yaml     --teleop.type=bi_spacemouse --dataset.single_task="test_ae_robot"  --fps=1
```


### 训练脚本
```bash
lerobot-train     --dataset.repo_id="wenyudu/260107_ae_alpha"      --policy.type=pi05     --output_dir=./outputs/pi05_ae_260107_alpha     --job_name=pi05_training     --policy.repo_id=wenyudu/pi05_policy    --policy.pretrained_path=/home/ae/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e50c659e8a0a6a3625d111044d1566672399e95    --policy.compile_model=true     --policy.gradient_checkpointing=true     --wandb.enable=true     --policy.dtype=bfloat16     --steps=20000     --policy.scheduler_decay_steps=20000     --policy.device=cuda     --batch_size=8     --wandb.enable=true
```

### 推理脚本
```bash
lerobot-record-aerobot     --robot.type=bi_ae_robot      --policy.path=/home/ae/project/lerobot/outputs/pi05_ae_260107_alpha/checkpoints/020000/pretrained_model      --config_path=/home/ae/project/lerobot/src/lerobot/robots/ae_robot/comprehensive_aerobot_record_config.yaml --dataset.single_task="pick the black cable, then place into the fixture"       --dataset.num_episodes=10      --dataset.push_to_hub=false      --dataset.episode_time_s=60      --dataset.fps=10      --dataset.repo_id="wenyudu/eval_pi05_01086" 
```

**wandb api**
local-28270fa613c79ab8a5cbe2790a08485126444a2c
