'''Chương trình huấn luyện'''
import pylab as pl
from copy import deepcopy
from env import PathPlanning, AlgorithmAdaptation
import pandas as pd
from sac import SAC


'''Cài đặt chế độ''' 
MAX_EPISODE = 1000        # Tổng số lần huấn luyện/đánh giá
render = False           # Có hiển thị quá trình huấn luyện không


'''Cài đặt môi trường và thuật toán'''
env = PathPlanning(max_search_steps=300)
env = AlgorithmAdaptation(env)
agent = SAC(env.observation_space, env.action_space, memory_size=800000) # Khởi tạo thuật toán học tăng cường


    
'''Huấn luyện/kiểm tra mô phỏng học tăng cường'''
steps_out = []
mean_reward_out = []

for episode in range(MAX_EPISODE):
    ## Đặt lại phần thưởng cho vòng
    ep_reward = 0
    
    ## Lấy quan sát ban đầu
    obs = env.reset()
    
    ## Tiến hành một vòng mô phỏng
    for steps in range(env.max_episode_steps):
        # Hiển thị
        if render:
            env.render()
        
        # Quyết định hành động
        act = agent.select_action(obs)  # Chính sách ngẫu nhiên

        # Mô phỏng
        next_obs, reward, done, info, agent1_pos = env.step(act)
        ep_reward += reward
        
        # Lưu trữ
        agent.store_memory((obs, act, reward, next_obs, done))
        
        # Tối ưu hóa
        agent.learn()
        
        # Kết thúc vòng
        if info["done"]:
            mean_reward = ep_reward / (steps + 1)
            print('Vòng: ', episode,'| Tổng phần thưởng: ', round(ep_reward, 2),'| Phần thưởng trung bình: ', round(mean_reward, 2),'| Trạng thái: ', info,'| Số bước: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    # Kết thúc vòng for
    steps_out.append(steps)
    mean_reward_out.append(mean_reward)
    
# Kết thúc vòng for
df1 = pd.DataFrame(steps_out)
df2 = pd.DataFrame(mean_reward_out)
df1.to_excel('steps_out.xlsx', index=False)
df2.to_excel('mean_reward_out.xlsx', index=False)
agent.save("Model.pkl")
