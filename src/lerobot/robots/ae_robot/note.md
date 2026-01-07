left_server pose
get pose:  0.0.0.0 0.44351 0.223 0.434 -0.40998122113813656 -0.3103788911034263 -0.5997600950272846 0.6130808842920747

right_server pose
get pose:  0.0.0.0 0.35469 -0.179459 -0.001 0.5918833489617783 -0.561919465067565 0.40305117887470404 0.4140898008979244


#### Send a pose via request

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
sudo rm -r ~/.cache/huggingface/lerobot/wenyudu/test_ae_data
```

```bash
lerobot-record-aerobot     --robot.type=bi_ae_robot     --config_path=/home/ae/project/lerobot/src/lerobot/robots/ae_robot/comprehensive_aerobot_record_config.yaml     --teleop.type=bi_spacemouse --dataset.single_task="test_ae_robot"  --dataset.num_episodes=2 --dataset.push_to_hub=false --dataset.episode_time_s=10 --dataset.fps=10 --dataset.repo_id="wenyudu/test_ae_data1ff1"
```
   