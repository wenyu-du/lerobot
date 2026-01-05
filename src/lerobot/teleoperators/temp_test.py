from lerobot.teleoperators.config import TeleoperatorConfig  
  
# 查看所有已注册的遥操控类型  
print("所有已注册的遥操控类型:")  
print(TeleoperatorConfig.get_known_choices())  
  
# 获取你的遥操控配置类  
config_cls = TeleoperatorConfig.get_choice_class("spacemouse")  
print(f"配置类: {config_cls}")