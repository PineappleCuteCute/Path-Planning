'''Chương trình kiểm tra'''
import torch
import pylab as pl
from copy import deepcopy
from env import PathPlanning, AlgorithmAdaptation
import pandas as pd
from sac import SAC


pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
pl.close('all')                                          # Đóng tất cả các cửa sổ


'''Cài đặt chế độ''' 
MAX_EPISODE = 1        # Tổng số lần huấn luyện/đánh giá
render = True           # Có hiển thị quá trình huấn luyện/đánh giá không (tốc độ mô phỏng sẽ giảm vài trăm lần)
agentpos = []

'''Cài đặt môi trường và thuật toán'''
env = PathPlanning()
env = AlgorithmAdaptation(env)
agent = SAC(env.observation_space, env.action_space, memory_size=1)
agent.load("Model.pkl")


    
'''Huấn luyện/kiểm tra mô phỏng học tăng cường'''
for episode in range(MAX_EPISODE):
    ## Lấy quan sát ban đầu
    obs = env.reset()
    
    ## Tiến hành một vòng mô phỏng
    for steps in range(env.max_episode_steps):
        # Hiển thị
        if render:
            env.render()
        
        # Quyết định hành động
        act = agent.select_action(obs)

        # Mô phỏng
        next_obs, _, _, info, agent1_pos = env.step(act)
        
        # Kết thúc vòng
        if info["done"]:
            print('Vòng: ', episode,'| Trạng thái: ', info,'| Số bước: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
        agentpos.append(agent1_pos)    
    
    # Kết thúc vòng for
    df = pd.DataFrame(agentpos)
    df.to_excel('agentpos.xlsx', index=False)
# Kết thúc vòng for
