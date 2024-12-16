import gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from copy import deepcopy

LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Tạo mô hình MLP
def build_mlp(layer_shape, activation=nn.ReLU, output_activation=nn.Identity, inplace=True):
    """Tạo mô hình MLP

    Tham số:
    ----------
    layer_shape : Kích thước của MLP, list / tuple
    activation : Hàm kích hoạt của MLP, mặc định là nn.ReLU
    output_activation : Hàm kích hoạt đầu ra của MLP, mặc định là nn.Identity
    inplace : Ví dụ như các hàm kích hoạt như ReLU có đặt inplace hay không, mặc định là True
    """
    def _need_inplace(activation) -> bool:
        return activation == nn.ReLU or\
                activation == nn.ReLU6 or\
                activation == nn.RReLU or\
                activation == nn.LeakyReLU or\
                activation == nn.SiLU or\
                activation == nn.ELU or\
                activation == nn.SELU or\
                activation == nn.CELU or\
                activation == nn.Threshold or\
                activation == nn.Hardsigmoid or\
                activation == nn.Hardswish or\
                activation == nn.Mish
    layers = []
    for j in range(len(layer_shape)-1):
        act = activation if j < len(layer_shape)-2 else output_activation
        if inplace and _need_inplace(act):
            layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act(inplace=True)] # Tăng tốc độ, giảm sử dụng bộ nhớ
        else:
            layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()] # Ví dụ như tanh không có tham số inplace
    return nn.Sequential(*layers)


# Mạng Q
class Q_Critic(nn.Module):
    def __init__(self, num_states:int, num_actions:int, hidden_dims:tuple=(128,64)):
        super(Q_Critic, self).__init__()
        mlp_shape = [num_states + num_actions] + list(hidden_dims) + [1]
        self.Q1_Value = build_mlp(mlp_shape)
        self.Q2_Value = build_mlp(mlp_shape)

    def forward(self, obs, action):
        obs = nn.Flatten()(obs)
        x = th.cat([obs, action], -1)
        Q1 = self.Q1_Value(x)
        Q2 = self.Q2_Value(x)
        return Q1, Q2


# Mạng P (phiên bản OpenAI)
class Actor(nn.Module):
    def __init__(self, num_states:int, num_actions:int, hidden_dims=(128,128)):
        super(Actor, self).__init__()
        layer_shape = [num_states] + list(hidden_dims)
        self.mlp_layer = build_mlp(layer_shape, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(layer_shape[-1], num_actions)
        self.log_std_layer = nn.Linear(layer_shape[-1], num_actions)

        self.LOG_STD_MAX = LOG_STD_MAX
        self.LOG_STD_MIN = LOG_STD_MIN

    def forward(self, obs, deterministic=False, with_logprob=True):
        obs = nn.Flatten()(obs)
        x = self.mlp_layer(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # ???
        std = th.exp(log_std)

        # Phân phối chiến lược
        dist = Normal(mu, std)
        if deterministic: u = mu
        else: u = dist.rsample() # Lấy mẫu theo phương pháp tái tham số hóa (sampling không thể đạo hàm)

        a = th.tanh(u) # Nén phạm vi đầu ra [-1, 1]

        # Tính log xác suất phân phối chuẩn log[P_pi(a|s)] -> (batch, act_dim)
        if with_logprob:
            # Công thức trong bài báo SAC tính log xác suất của a thông qua u:
            ''' logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True) '''
            # Công thức trong bài báo SAC có a = tanh(u), dẫn đến gradient bị mất, công thức của OpenAI:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True) # (batch, 1)
        else:
            logp_pi_a = None

        return a, logp_pi_a # (batch, act_dim) và (batch, 1)

    def act(self, obs, deterministic=False) -> np.ndarray[any, float]:
        self.eval()
        with th.no_grad():
            a, _ = self.forward(obs, deterministic, False)
        self.train()  # Khôi phục lại chế độ huấn luyện
        return a.cpu().numpy().flatten() # (act_dim, ) ndarray
    

# Bộ nhớ ReplayBuffer
class EasyBuffer:

    def __init__(self, memory_size, obs_space, act_space):
        assert not isinstance(obs_space, (gym.spaces.Tuple, gym.spaces.Dict)), "1"
        assert not isinstance(act_space, (gym.spaces.Tuple, gym.spaces.Dict)), "1"
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        # Thuộc tính buffer
        self.ptr = 0    # Con trỏ lưu trữ của buffer
        self.idxs = [0] # Dùng cho PER để nhớ chỉ số mẫu lần trước, list một chiều hoặc ndarray
        self.memory_size = memory_size
        self.current_size = 0 
        # Container buffer
        obs_shape = obs_space.shape or (1, )
        act_shape = act_space.shape or (1, )
        self.buffer = {}
        self.buffer["obs"] = np.empty((memory_size, *obs_shape), dtype=obs_space.dtype) # (size, *obs_shape, ) liên tục (size, 1) rời rạc
        self.buffer["next_obs"] = deepcopy(self.buffer["obs"])                          # (size, *obs_shape, ) liên tục (size, 1) rời rạc
        self.buffer["act"] = np.empty((memory_size, *act_shape), dtype=act_space.dtype) # (size, *act_shape, ) liên tục (size, 1) rời rạc
        self.buffer["rew"] = np.empty((memory_size, 1), dtype=np.float32)               # (size, 1)
        self.buffer["done"] = np.empty((memory_size, 1), dtype=bool)                    # (size, 1)

    def __getitem__(self, position):
        """Truy cập theo chỉ số\n
        Tương tự batch = buffer[position] và batch = buffer.sample(idxs=position) có hiệu ứng giống nhau
        """
        if isinstance(position, int): position = [position]
        return self.sample(idxs=position)

    def __len__(self):
        return self.current_size 

    def reset(self):
        """Xóa bộ nhớ"""
        self.ptr = 0
        self.idxs = [0]
        self.current_size = 0

    def push(self, transition, terminal=None):
        """Lưu trữ"""
        # Thêm một chuyển tiếp vào buffer
        self.buffer["obs"][self.ptr] = transition[0]
        self.buffer["act"][self.ptr] = transition[1]
        self.buffer["rew"][self.ptr] = transition[2]
        self.buffer["next_obs"][self.ptr] = transition[3]
        self.buffer["done"][self.ptr] = transition[4]
        # Cập nhật con trỏ và kích thước
        self.ptr = (self.ptr + 1) % self.memory_size # Cập nhật con trỏ
        self.current_size = min(self.current_size + 1, self.memory_size) # Cập nhật dung lượng

    def sample(self, batch_size = 1, *, idxs = None, rate = None, **kwargs):
        """Lấy mẫu"""
        # Tạo chỉ mục
        if idxs is None:
            assert batch_size <= self.current_size, "batch_size phải nhỏ hơn hoặc bằng dung lượng hiện tại"
            idxs = np.random.choice(self.current_size, size=batch_size, replace=False)
        # Lấy mẫu một batch từ buffer
        batch = {}
        for key in self.buffer:
            if key != "act":
                batch[key] = th.FloatTensor(self.buffer[key][idxs]).to(self.device)
            else:
                batch[key] = th.tensor(self.buffer[key][idxs]).to(self.device)
        # Cập nhật chỉ mục
        self.idxs = idxs
        return batch


        
        

class SAC:
    """Soft Actor-Critic (arXiv: 1812)"""
   
    def __init__( 
        self, 
        observation_space: gym.Space, # Không gian quan sát
        action_space: gym.Space,      # Không gian hành động

        *,
        memory_size: int = 500000,  # Kích thước bộ nhớ
            
        gamma: float = 0.99,        # Hệ số chiết khấu γ
        alpha: float = 0.3,         # Hệ số nhiệt α
        
        batch_size: int = 256,      # Kích thước mẫu
        update_after: int = 10000,   # Bắt đầu huấn luyện, batch_size <= update_after <= memory_size  1000

        lr_decay_period: int = None, # Chu kỳ giảm tốc độ học, None không giảm
        lr_critic: float = 1e-3,     # Tốc độ học của Q
        lr_actor: float = 1e-3,      # Tốc độ học của π
        tau: float = 0.005,          # Hệ số cập nhật mềm cho target Q τ

        q_loss_cls = nn.MSELoss,  # Loại hàm mất mát cho Q (cái này không có tác dụng khi use_per=True)
        
        critic_optim_cls = th.optim.Adam, # Loại tối ưu hóa cho Q
        actor_optim_cls = th.optim.Adam,  # Loại tối ưu hóa cho π
        
        adaptive_alpha: bool = True,       # Có điều chỉnh α tự động không
        target_entropy: float = None,      # Mục tiêu entropy cho hệ số α tự động, mặc định: -dim(A)
        lr_alpha: float = 1e-3,            # Tốc độ học cho α
        alpha_optim_class = th.optim.Adam, # Loại tối ưu hóa cho α

        use_per: bool = False,  # Sử dụng replay buffer có ưu tiên không
        per_alpha: float = 0.6, # Ưu tiên replay α
        per_beta0: float = 0.4, # Ưu tiên replay β

        grad_clip: float = None, # Phạm vi cắt gradient cho mạng Q, None là không cắt

    ):
        assert isinstance(action_space, gym.spaces.Box), 'SAC chỉ sử dụng không gian hành động Box'
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        # Thông số môi trường
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_states = np.prod(observation_space.shape)
        self.num_actions = np.prod(action_space.shape)

        # Khởi tạo các tham số SAC
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.update_after = int(update_after)

        self.lr_decay_period = lr_decay_period
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau

        # Khởi tạo ReplayBuffer
        self.memory_size = int(memory_size)
        self.use_per = use_per
        if use_per:
            raise NotImplementedError("Phiên bản rút gọn không hỗ trợ PER")
        else:
            self.buffer = EasyBuffer(self.memory_size, self.observation_space, self.action_space)

        # Khởi tạo mạng neural
        self.actor = Actor(self.num_states, self.num_actions).to(self.device)
        self.q_critic = Q_Critic(self.num_states, self.num_actions).to(self.device) # Twin Q Critic
        self.target_q_critic = self._build_target(self.q_critic)
        
        # Khởi tạo bộ tối ưu hóa
        self.actor_optimizer = actor_optim_cls(self.actor.parameters(), lr_actor)
        self.q_critic_optimizer = critic_optim_cls(self.q_critic.parameters(), lr_critic)
        
        # Thiết lập hàm mất mát
        self.grad_clip = grad_clip
        self.q_loss = q_loss_cls()
        
        # Có điều chỉnh α tự động không
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            target_entropy = target_entropy or -self.num_actions # Target Entropy = −dim(A)
            self.target_entropy = th.tensor(target_entropy, dtype=float, requires_grad=True, device=self.device)
            self.log_alpha = th.tensor(np.log(alpha), dtype=float, requires_grad=True, device=self.device) # log_alpha không có ràng buộc >0
            self.alpha_optimizer = alpha_optim_class([self.log_alpha], lr = lr_alpha)
            self.lr_alpha = lr_alpha

        # Các tham số khác
        self.learn_counter = 0
 
    
    def set_nn(self, actor: Actor, critic: Q_Critic, *, actor_optim_cls=th.optim.Adam, critic_optim_cls=th.optim.Adam, copy=True):
        """Thay đổi mô hình mạng neural, yêu cầu theo định dạng Actor/Q_Critic"""
        self.actor = deepcopy(actor) if copy else actor
        self.actor.train().to(self.device)
        self.q_critic = deepcopy(critic) if copy else critic
        self.q_critic.train().to(self.device) # Twin Q Critic
        self.target_q_critic = self._build_target(self.q_critic)
        self.actor_optimizer = actor_optim_cls(self.actor.parameters(), self.lr_actor)
        self.q_critic_optimizer = critic_optim_cls(self.q_critic.parameters(), self.lr_critic)

    def set_buffer(self, buffer: EasyBuffer):
        """Thay đổi replay buffer, yêu cầu theo định dạng EasyBuffer"""
        self.buffer = buffer
    
    def store_memory(self, transition, terminal: bool = None):
        """Lưu trữ kinh nghiệm"""
        self.buffer.push(transition, terminal)


    def select_action(self, state, *, deterministic=False, **kwargs) -> np.ndarray:
        """Chọn hành động"""
        state = th.FloatTensor(state).unsqueeze(0).to(self.device) # (1, state_dim) tensor GPU
        return self.actor.act(state, deterministic) # (act_dim, ) ndarray


    def learn(self, *, rate: float = None) -> dict:
        """Học tăng cường

        Tham số
        ----------
        rate : float, tùy chọn
            Dùng để cập nhật tham số PER beta, mặc định None không cập nhật
            rate = train_steps / max_train_steps
            beta = beta0 + (1-beta0) * rate
        """
      
        if len(self.buffer) < self.batch_size or \
            len(self.buffer) < self.update_after:    
            return {'q_loss': None, 'actor_loss': None, 'alpha_loss': None, 'q': None, 'alpha': None}
        
        self.learn_counter += 1
        
        ''' replay kinh nghiệm '''
        samples = self.buffer.sample(self.batch_size, rate=rate) # trả về tensor trên GPU
        state = samples["obs"]           # (m, obs_dim) tensor trên GPU
        action = samples["act"]          # (m, act_dim) tensor trên GPU
        reward = samples["rew"]          # (m, 1) tensor trên GPU
        next_state = samples["next_obs"] # (m, obs_dim) tensor trên GPU
        done = samples["done"]           # (m, 1) tensor trên GPU
        if self.use_per:
            IS_weight = samples["IS_weight"] # (m, 1) tensor trên GPU


        ''' Tối ưu mạng Q Critic '''
        # J(Q) = E_{s_t~D, a_t~D, s_t+1~D, a_t+1~π_t+1}[0.5*[ Q(s_t, a_t) - [r + (1-d)*γ* [ Q_tag(s_t+1,a_t+1) - α*logπ_t+1 ] ]^2 ]
        # Tính toán giá trị Q mục tiêu
        with th.no_grad():
            next_action, next_log_pi = self.actor(next_state)                                 # 
            Q1_next, Q2_next = self.target_q_critic(next_state, next_action)                  # 
            Q_next = th.min(Q1_next, Q2_next)                                                 # 
            Q_targ = reward + (1.0 - done) * self.gamma * (Q_next - self.alpha * next_log_pi) # 4-12

        # Tính toán giá trị Q hiện tại
        Q1_curr, Q2_curr = self.q_critic(state, action) # 4-13

        # Tính toán hàm mất mát
        if self.use_per:
            td_err1, td_err2 = Q1_curr-Q_targ, Q2_curr-Q_targ  # (m, 1) tensor trên GPU với grad
            q_loss = (IS_weight * (td_err1 ** 2)).mean() + (IS_weight * (td_err2 ** 2)).mean() # () Lưu ý: mean phải được tính ở ngoài cùng
            self.buffer.update_priorities(td_err1.detach().cpu().numpy().flatten()) # Cập nhật độ ưu tiên td err: (m, ) ndarray
        else:
            q_loss = self.q_loss(Q1_curr, Q_targ) + self.q_loss(Q2_curr, Q_targ) # ()

        # Tối ưu mạng
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.grad_clip)
        self.q_critic_optimizer.step()


        ''' Tối ưu mạng Actor '''
        # J(π) = E_{s_t~D, a~π_t}[ α*logπ_t(a|π_t) - Q(s_t, a) ]   
        self._freeze_network(self.q_critic)
      
        # Đánh giá chính sách
        new_action, log_pi = self.actor(state)    # (m, act_dim), (m, 1) tensor trên GPU với grad
        Q1, Q2 = self.q_critic(state, new_action) # (m, 1) tensor trên GPU không có grad
        Q = th.min(Q1, Q2)                        # (m, 1) tensor trên GPU không có grad

        # Tối ưu chính sách
        a_loss = (self.alpha * log_pi - Q).mean()
        self._optim_step(self.actor_optimizer, a_loss)

        self._unfreeze_network(self.q_critic)


        ''' Tối ưu hệ số nhiệt alpha '''
        # J(α) = E_{a~π_t}[ -α * ( logπ_t(a|π_t) + H0 ) ]
        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()      # log công thức: hội tụ nhanh hơn, tính toán nhanh hơn
            self._optim_step(self.alpha_optimizer, alpha_loss)
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_scalar = alpha_loss.item()
        else:
            alpha_loss_scalar = None


        ''' Tối ưu mạng target Q '''
        self._soft_update(self.target_q_critic, self.q_critic, self.tau)
        

        ''' Sử dụng giảm tốc độ học '''
        self._lr_decay(self.actor_optimizer)
        self._lr_decay(self.q_critic_optimizer)
        if self.adaptive_alpha:
            self._lr_decay(self.alpha_optimizer)


        ''' Trả về thông tin '''
        return {'q_loss': q_loss.item(), 
                'actor_loss': a_loss.item(), 
                'alpha_loss': alpha_loss_scalar, 
                'q': Q1_curr.mean().item(), 
                'alpha': self.alpha
                }
    

    def save(self, file):
        """Lưu trữ trọng số của mạng Actor"""
        th.save(self.actor.state_dict(), file)
        
    
    def load(self, file):
        """Tải trọng số của mạng Actor"""
        self.actor.load_state_dict(th.load(file, map_location=self.device))

    
    @staticmethod
    def _soft_update(target_network: nn.Module, network: nn.Module, tau: float):
        """
        Cập nhật mềm cho mạng neural mục tiêu\n
        >>> for target_param, param in zip(target_network.parameters(), network.parameters()):
        >>>    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        """
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - tau) + param.data * tau ) # Cập nhật mềm
   
    @staticmethod
    def _hard_update(target_network: nn.Module, network: nn.Module):
        """
        Cập nhật cứng cho mạng neural mục tiêu\n
        >>> target_network.load_state_dict(network.state_dict())
        """
        target_network.load_state_dict(network.state_dict()) # Cập nhật cứng

    @staticmethod
    def _freeze_network(network: nn.Module):
        """
        Đóng băng mạng neural\n
        >>> for p in network.parameters():
        >>>     p.requires_grad = False
        """
        for p in network.parameters():
            p.requires_grad = False
        # requires_grad = False được dùng để kết nối mạng (gradient cần được truyền lại), ví dụ như khi gradient của Q cần truyền lại mạng Actor nhưng không cập nhật Critic
        # with th.no_grad() được dùng cho mạng song song hoặc kết nối mạng (gradient không cần được truyền lại), ví dụ như Actor tính toán next_a, và dùng next_a tính Q nhưng không truyền gradient của Q về Actor

    @staticmethod
    def _unfreeze_network(network: nn.Module):
        """
        Mở băng mạng neural\n
        >>> for p in network.parameters():
        >>>     p.requires_grad = True
        """
        for p in network.parameters():
            p.requires_grad = True

    @staticmethod
    def _build_target(network: nn.Module):
        """
        Sao chép mạng neural mục tiêu\n
        >>> target_network = deepcopy(network).eval()
        >>> for p in target_network.parameters():
        >>>     p.requires_grad = False
        """
        target_network = deepcopy(network).eval()
        for p in target_network.parameters():
            p.requires_grad = False
        return target_network
    
    @staticmethod
    def _set_lr(optimizer: th.optim.Optimizer, lr: float):
        """
        Điều chỉnh tốc độ học của bộ tối ưu hóa\n
        >>> for g in optimizer.param_groups:
        >>>     g['lr'] = lr
        """
        for g in optimizer.param_groups:
            g['lr'] = lr

    def _lr_decay(self, optimizer: th.optim.Optimizer):
        """Giảm tốc độ học (giảm đến 0.1 lần tốc độ học ban đầu trong chu kỳ lr_decay_period, period là None/0 thì không giảm)
        >>> lr = 0.9 * lr_init * max(0, 1 - step / lr_decay_period) + 0.1 * lr_init
        >>> self._set_lr(optimizer, lr)
        """
        if self.lr_decay_period:
            lr_init = optimizer.defaults["lr"] # Lấy tốc độ học ban đầu
            lr = 0.9 * lr_init * max(0, 1 - self.learn_counter / self.lr_decay_period) + 0.1 * lr_init # Cập nhật tốc độ học
            self._set_lr(optimizer, lr) # Thay đổi tốc độ học trong param_groups
            # Lưu ý: thay đổi tốc độ học trong param_groups không thay đổi lr trong defaults

    @staticmethod
    def _optim_step(optimizer: th.optim.Optimizer, loss: th.Tensor):
        """
        Cập nhật trọng số mạng neural\n
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
