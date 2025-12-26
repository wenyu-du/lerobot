 

### 遥操控
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port='/dev/follower_port' \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ fixed: {type: opencv,index_or_path: 6, width: 640, height: 480, fps: 30}, handeye: {type: opencv,index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port='/dev/leader_port' \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```



## 录制数据
```bash
lerobot-record     --robot.type=so101_follower     --robot.port='/dev/follower_port'     --robot.cameras="{ camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv,index_or_path: 6, width: 640, height: 480, fps: 30}}"     --robot.id=my_awesome_follower_arm     --teleop.type=so101_leader     --teleop.port='/dev/leader_port'     --teleop.id=my_awesome_leader_arm      --dataset.repo_id=wenyudu/demo_pi05_20251225     --dataset.num_episodes=30     --dataset.single_task="Pick and place the orange rabbit toy"     --display_data=true

lerobot-record  --robot.type=so101_follower     --robot.port='/dev/follower_port'     --robot.cameras="{ camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv,index_or_path: 6, width: 640, height: 480, fps: 30}}"     --robot.id=my_awesome_follower_arm     --teleop.type=so101_leader     --teleop.port='/dev/leader_port'     --teleop.id=my_awesome_leader_arm           --dataset.single_task="Pick and place the brown rabbit toy"     --display_data=true --dataset.repo_id=wenyudu/demo_pi05_20251225_b3     --dataset.num_episodes=20 --daset.push_to_hub=false
```


### Train 脚本

**pi05**

```bash
lerobot-train \
    --dataset.repo_id=wenyudu/demo_pi05_20251226_merged   \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training_1226_merged \
    --job_name=pi05_training \
    --policy.repo_id=wenyudu/pi05_policy\
    --policy.pretrained_path=/home/ae/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/9e50c659e8a0a6a3625d111044d1566672399e95\
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=120000 \
    --policy.scheduler_decay_steps=20000 \
    --policy.device=cuda \
    --batch_size=8 \
    --wandb.enable=true
```

**smolvla**
```bash
lerobot-train \
    --dataset.repo_id=wenyudu/demo_pi05_20251222   \
    --output_dir=./outputs/smolvla_training \
    --job_name=smolvla_training \
    --policy.repo_id=wenyudu/smolvla_policy\
    --policy.path=lerobot/smolvla_base \
    --steps=20000 \
    --policy.device=cuda \
    --batch_size=64 \
    --wandb.enable=false
```

### 推理脚本
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port='/dev/follower_port' \
    --robot.id=my_awesome_follower_arm  \
    --teleop.type=so101_leader \
    --teleop.port='/dev/leader_port' \
    --teleop.id=my_awesome_leader_arm  \
    --dataset.repo_id=wenyudu/eval_pi05_20251224_brown  \
    --policy.path=/home/ae/project/lerobot/outputs/pi05_training_1224/checkpoints/020000/pretrained_model \
    --dataset.num_episodes=10  \
    --dataset.single_task="Pick and place the brown rabbit toy" \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera2: {type: opencv,index_or_path: 6, width: 640, height: 480, fps: 30}}" \
    --dataset.episode_time_s=17 \
    --dataset.reset_time_s=2 \
    --display_data=true  
```

```bash
wandb login --host=http://10.20.220.229:8080
local-28270fa613c79ab8a5cbe2790a08485126444a2c
wandb login --relogin
```

### Replay 脚本
```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port='/dev/follower_port' \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=wenyudu/demo_pi05_20251222 \
    --dataset.episode=19 
```
#### 可视化本地数据集
```bash
lerobot-dataset-viz \
    --repo-id=wenyudu/demo_pi05_20251225_merged \
    --mode=local \
    --episode-index=0
```

### upload file to hf

```bash
hf upload demo_pi05_20251225_merged ~/.cache/huggingface/lerobot/wenyudu/demo_pi05_20251225_merged --repo-type dataset
hf upload <repo_id> <local_path> --repo-type dataset  
data_id, data_path
```


### merge data

lerobot-edit-dataset \
    --repo_id wenyudu/demo_pi05_20251226_merged \
    --operation.type merge \
    --operation.repo_ids "['wenyudu/demo_pi05_20251225_b1' , 'wenyudu/demo_pi05_20251225_o1' ,   'wenyudu/demo_pi05_20251225_o3' ,'wenyudu/demo_pi05_20251225_b2', 'wenyudu/demo_pi05_20251225_o1_2', 'wenyudu/demo_pi05_20251225_b3' , 'wenyudu/demo_pi05_20251225_o2', 'wenyudu/demo_pi05_20251225_o4', 'wenyudu/demo_pi05_20251225_b2']" 


#### 检查并修复数据集
```python

from lerobot.datasets.lerobot_dataset import LeRobotDataset  
  
# 加载数据集并验证  
dataset = LeRobotDataset("/home/ae/.cache/huggingface/lerobot/wenyudu/demo_pi05_20251225_merged")  
  
# 检查总帧数  
print(f"Total frames: {dataset.num_frames}")  
print(f"Total episodes: {dataset.num_episodes}")  
  
# 检查每个episode的边界  
for ep_idx in range(dataset.num_episodes):  
    ep_meta = dataset.meta.episodes[ep_idx]  
    print(f"Episode {ep_idx}: frames {ep_meta['dataset_from_index']} to {ep_meta['dataset_to_index']}")
```
#### load数据集验证
```python
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("wenyudu/demo_pi05_20251225_merged")
for frame in tqdm(dataset):
    pass
print("All frames loaded successfully.")

```