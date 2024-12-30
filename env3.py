import numpy as np
import gym
import pylab as pl
from gym import spaces
from copy import deepcopy
from collections import deque
from scipy.integrate import odeint
from shapely import geometry as geo
from shapely.plotting import plot_polygon
from shapely.ops import nearest_points
from math import sqrt, pow, acos
import random


class Map:
    size = np.array([[-10.0, -10.0], [10.0, 10.0]])  # Giá trị nhỏ nhất và lớn nhất của x, y
    start_pos = np.array([-7, 7])  # Tọa độ điểm bắt đầu
    end_pos = np.array([10, -10])  # Tọa độ điểm kết thúc
    obstacles = []  # Chướng ngại vật tĩnh

    def __init__(self):
        self.randomize_obstacles()

    def randomize_obstacles(self):
        # Xóa các chướng ngại vật cũ
        self.obstacles = []

        # Tạo các chướng ngại vật tĩnh với vị trí ngẫu nhiên và không đè lên nhau
        for _ in range(10):  # Số lượng chướng ngại vật tĩnh
            while True:
                x = random.uniform(-9, 9)
                y = random.uniform(-9, 9)
                width = random.uniform(1, 3)
                height = random.uniform(1, 3)
                new_obstacle = geo.Polygon([
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height)
                ])
                # Kiểm tra xem chướng ngại vật mới có giao với bất kỳ chướng ngại vật nào đã có không
                overlap = False
                for obstacle in self.obstacles:
                    if new_obstacle.intersects(obstacle):
                        overlap = True
                        break
                if not overlap:
                    self.obstacles.append(new_obstacle)
                    break

    def update(cls, x):
        # Không có chướng ngại vật động
        pass

    @classmethod
    def show(cls):
        pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Sửa lỗi phông chữ
        pl.mpl.rcParams['axes.unicode_minus'] = False  # Đảm bảo hiển thị dấu âm bình thường

        pl.close('all')
        pl.figure('Map')
        pl.clf()
    
        # Vẽ các chướng ngại vật
        for o in cls.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)

        # Vẽ điểm bắt đầu và kết thúc
        pl.scatter(cls.start_pos[0], cls.start_pos[1], s=30, c='k', marker='x')
        pl.scatter(cls.end_pos[0], cls.end_pos[1], s=30, c='k', marker='o')
  
        pl.axis('equal')  # Tạo hệ trục tọa độ đều
        pl.xlabel("x")  # Nhãn trục x
        pl.ylabel("y")  # Nhãn trục y
        pl.xlim(cls.size[0][0], cls.size[1][0])  # Giới hạn trục x
        pl.ylim(cls.size[0][1], cls.size[1][1])  # Giới hạn trục y
        pl.title('Bản đồ')  # Tiêu đề
        pl.grid()  # Hiển thị lưới
        pl.grid(alpha=0.3, ls=':')  # Thay đổi độ trong suốt và kiểu đường lưới
        pl.show(block=True)


# Môi trường tĩnh
class PathPlanning(gym.Env):
    # Cài đặt bản đồ
    MAP = Map()

    def __init__(self, max_search_steps=300, use_old_gym=True):
        self.map = self.MAP
        self.max_episode_steps = max_search_steps
        np_low = np.array([-10, -10, -10, -10])
        np_high = np.array([10, 10, 10, 10])
        self.observation_space = spaces.Box(low=np_low, high=np_high, dtype=pl.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=pl.float32)
        self.__render_flag = True
        self.__reset_flag = True
        self.__old_gym = use_old_gym

    def reset(self):
        self.__reset_flag = False
        self.time_steps = 0 
        self.traj = []
        
        # Random lại vị trí các chướng ngại vật tĩnh
        self.map.randomize_obstacles()
        
        self.agent_pos = np.array(self.map.start_pos)
        
        self.obs = np.array([10, 10, 10, 10])
        self.obs[0] = self.map.end_pos[0] - self.map.start_pos[0]
        self.obs[1] = self.map.end_pos[1] - self.map.start_pos[1]
        self.per_act = np.array([100, 100])

        # Cập nhật trạng thái ban đầu của môi trường
        if self.__old_gym:
            return self.obs
        return self.obs, {}
    
    def step(self, act):
        """
        Mô hình chuyển động 1
        Pos_new = act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new)
        Mô hình chuyển động 2
        Pos_new = Pos_old + act, act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new-Pos_old)
        """
        self.now_act = act
        assert not self.__reset_flag, "Phải gọi reset trước khi gọi step"
        # Chuyển đổi trạng thái
        obs = pl.clip(self.obs, self.observation_space.low, self.observation_space.high)
        self.agent_pos = pl.clip(self.agent_pos + act * 0.3, [-10, -10], [10, 10])
        agent1_pos = self.agent_pos
        obs[0] = self.map.end_pos[0] - self.agent_pos[0]
        obs[1] = self.map.end_pos[1] - self.agent_pos[1]
        min_dis = 100
        for o in self.map.obstacles:
            dis = geo.Point(self.agent_pos).distance(o)
            if dis < min_dis:
                min_dis = dis
                o_min = o
        p1, _ = nearest_points(o_min, geo.Point(self.agent_pos))
        p1 = p1.xy
        obs[2] = p1[0][0] - self.agent_pos[0]
        obs[3] = p1[1][0] - self.agent_pos[1]
        obs = np.array(obs)
        self.time_steps += 1
        # Tính toán phần thưởng
        rew, done, info = self.get_reward(obs)
        # Kết thúc tập
        self.obs = deepcopy(obs)
        truncated = self.time_steps >= self.max_episode_steps
        if truncated or done:
            info["done"] = True
            self.__reset_flag = True
        else:
            info["done"] = False
        # Cập nhật trạng thái
        if (self.time_steps // 20) % 2 == 0:
            x = self.time_steps % 20
        else:
            x = 20 - (self.time_steps % 20)
        self.map.update(0.1 * x)
        self.per_act = np.array(act)
        self.obs = deepcopy(obs)
        if self.__old_gym:
            return obs, rew, done, info, agent1_pos
        return obs, rew, done, truncated, info
    
    def get_reward(self, obs):
        rew = 0
        done = False
        pol_ext = geo.LinearRing(geo.Polygon([(-10, 10), (10, 10), (10, -10), (-10, -10)]).exterior.coords)
        d = pol_ext.project(geo.Point(self.agent_pos))
        p = pol_ext.interpolate(d)
        closest_point_coords = list(p.coords)[0]
        # Kiểm tra biên của bản đồ
        if np.linalg.norm((closest_point_coords[0] - self.agent_pos[0], closest_point_coords[1] - self.agent_pos[1])) < 0.3:
            rew += -5
        # Kiểm tra nếu đi về phía trước
        if np.linalg.norm((obs[0], obs[1])) + 0.25 < np.linalg.norm((self.obs[0], self.obs[1])):
            rew += 1
        elif np.linalg.norm((obs[0], obs[1])) - 0.1 > np.linalg.norm((self.obs[0], self.obs[1])):
            rew += -2
        # Chướng ngại vật
        if 0.3 < np.linalg.norm((obs[2], obs[3])) < 0.8:
            rew += -5
        elif np.linalg.norm((obs[2], obs[3])) < 0.3:
            rew += -1000
        elif 0.8 < np.linalg.norm((obs[2], obs[3])):
            rew += 0.5
        # Xoay góc
        if (self.per_act[0] < 10):
            pi = 3.1415
            vector_prod = self.per_act[0] * self.now_act[0] + self.per_act[1] * self.now_act[1]
            length_prod = sqrt(pow(self.per_act[0], 2) + pow(self.per_act[1], 2)) * sqrt(pow(self.now_act[0], 2) + pow(self.now_act[1], 2))
            cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
            dtheta = (acos(cos) / pi) * 180
            if 90 > dtheta > 45:
                rew += -1
            elif dtheta > 90:
                rew += -5
            elif 0 < dtheta < 45:
                rew += 0.5
                
        # Kiểm tra điểm kết thúc
        if np.linalg.norm((self.map.end_pos - self.agent_pos)) < 0.5:
            rew += 5000
            done = True
        info = {}
        info['done'] = done
        
        return rew, done, info
    
    def render(self, mode="human"):
        """Vẽ đồ họa, phải gọi sau khi step"""
        assert not self.__reset_flag, "Phải gọi reset trước khi gọi render"

        if self.__render_flag:
            self.__render_flag = False
            pl.ion()  # Bật chế độ vẽ đồ họa tương tác, chỉ có thể bật một lần

        # Xóa hình cũ
        pl.clf() 
        # Vẽ các chướng ngại vật
        for o in self.map.obstacles:
            plot_polygon(o, facecolor='c', edgecolor='k', add_points=False)
        plot_polygon(geo.Point(self.agent_pos).buffer(0.8), facecolor='y', edgecolor='y', add_points=False)
        plot_polygon(geo.Point(self.agent_pos).buffer(0.3), facecolor='r', edgecolor='r', add_points=False)
        # Vẽ điểm bắt đầu và kết thúc
        pl.scatter(self.map.start_pos[0], self.map.start_pos[1], s=30, c='k', marker='x')
        pl.scatter(self.map.end_pos[0], self.map.end_pos[1], s=30, c='k', marker='o')
        # Vẽ quỹ đạo
        self.traj.append(self.agent_pos.tolist())
        new_lst = [item for sublist in self.traj for item in sublist]
        pl.plot(new_lst[::2], new_lst[1::2], color='b')

        pl.scatter(self.agent_pos[0], self.agent_pos[1], s=1, c='k')
        pl.axis('equal')  # Tạo hệ trục tọa độ đều
        pl.xlabel("x")  # Nhãn trục x
        pl.ylabel("y")  # Nhãn trục y
        pl.xlim(self.map.size[0][0], self.map.size[1][0])  # Giới hạn trục x
        pl.ylim(self.map.size[0][1], self.map.size[1][1])  # Giới hạn trục y
        pl.title('Lập kế hoạch đường đi')  # Tiêu đề
        pl.grid()  # Hiển thị lưới
        pl.grid(alpha=0.3, ls=':')  # Thay đổi độ trong suốt và kiểu đường lưới
        
        pl.pause(0.1)  # Dừng 0.1 giây
        pl.ioff()  # Tắt chế độ vẽ đồ họa tương tác

    def close(self):
        """Đóng đồ họa"""
        self.__render_flag = True
        pl.close()


# Điều chỉnh môi trường với thuật toán
class AlgorithmAdaptation(gym.ActionWrapper):
    def __init__(self, env):
        super(AlgorithmAdaptation, self).__init__(env)
        assert isinstance(env.action_space, spaces.Box), 'Chỉ sử dụng cho không gian hành động Box'
  
    # Chuyển đổi đầu ra từ mạng nơ-ron thành đầu vào cho gym
    def action(self, action): 
        # Tình huống liên tục scale action [-1, 1] -> [lb, ub]
        lb, ub = self.action_space.low, self.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)
        action = pl.clip(action, lb, ub)
        return action

    # Chuyển đổi đầu vào của gym thành đầu ra cho mạng nơ-ron
    def reverse_action(self, action):
        # Tình huống liên tục, chuyển đổi hành động từ [lb, ub] sang [-1, 1]
        lb, ub = self.action_space.low, self.action_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return pl.clip(action, -1.0, 1.0)
       

if __name__ == '__main__':
    Map.show()