import gym
import pfrl
from gym import spaces

import torch
from torch import distributions
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pfrl.nn.lmbda import Lambda

import numpy as np

from scipy.integrate import odeint

class ReactorODEEnv(gym.Env):
    def __init__(self):
        super(ReactorODEEnv, self).__init__()
        
        # Define the continuous action space (fraction of D and Sin)
        self.action_space = spaces.Box(low=np.array([0., 0.]), high=np.array([2., 30.]), shape=(2,), dtype=np.float32)
        
        # Define the state space (S, x1, x2)
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.]), high=np.array([20., 10., 10.]), shape=(3,), dtype=np.float32)
        
        # Initial population of species
        self.S  = np.random.uniform(10., 12.)
        self.x1 = np.random.uniform(1.5, 1.9)
        self.x2 = np.random.uniform(1.1, 1.4)
        
        # Parameters for the ODE model
        self.mu1_max, self.mu2_max = 4., 2.
        self.Km1, self.Km2         = 20., 6.
        self.k1, self.k2           = 1., 1.
        self.x1d, self.x2d         = 2., 1.5
        
        # Time step and time horizon for solving the ODE
        self.dt        = 0.1
        self.t_horizon = 10
        
        # Current time step
        self.current_step = 0
        
    def reset(self):
        # Reset the environment by initializing populations
        self.S  = np.random.uniform(10., 12.)
        self.x1 = np.random.uniform(1.5, 1.9)
        self.x2 = np.random.uniform(1.1, 1.4)
        self.current_step = 0
        return np.array([self.S, self.x1, self.x2])

    def step(self, action):
        
        def ode_func(y, t):
            S, x1, x2 = y
            
            dydt = [
                    action[0]*(action[1] - S) - self.mu1_max*S/(self.Km1 + S)*x1 - self.mu2_max*S/(self.Km2 + S)*x2,
                    (self.k1*self.mu1_max*S/(self.Km1 + S) - action[0])*x1,
                    (self.k2*self.mu2_max*S/(self.Km2 + S) - action[0])*x2
            ]
            
            return dydt
        
        t_eval = np.linspace(0, self.dt, 2)
        y0     = [self.S, self.x1, self.x2]
        
        solution = odeint(ode_func, y0, t_eval)
        
        self.S, self.x1, self.x2 = solution[-1]
        
        self.current_step += 1
        
        # Calculate the reward (for example, based on population sizes)
        reward = -(self.x1 - self.x1d)**2 - (self.x2 - self.x2d)**2

        # Check if the episode is done
        done = self.current_step >= self.t_horizon
        
        return np.array([self.S, self.x1, self.x2]), reward, done, {}
    
    def render(self):
        # Optional method for rendering the environment
        pass

    def close(self):
        # Clean-up method if necessary
        pass
    
# env = ReactorODEEnv()
# obs = env.reset()
# done = False

# while not done:
#     action = np.random.rand(3)
#     obs, reward, done, _ = env.step(action)
#     print(f"Step: {env.current_step}, Population: {obs}, Reward: {reward}")

# env.close()


# Custom environment with a continuous action space
env = ReactorODEEnv()

# Define a neural network for the policy and Q-functions
obs_size    = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

action_space = env.action_space

def squashed_diagonal_gaussian_head(x):
    assert x.shape[-1] == action_size * 2
    mean, log_scale = torch.chunk(x, 2, dim=1)
    log_scale = torch.clamp(log_scale, -20.0, 2.0)
    var = torch.exp(log_scale * 2)
    base_distribution = distributions.Independent(
        distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
    )
    # cache_size=1 is required for numerical stability
    return distributions.transformed_distribution.TransformedDistribution(
        base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
    )

policy = nn.Sequential(
    nn.Linear(obs_size, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, action_size * 2),
    Lambda(squashed_diagonal_gaussian_head),
)
torch.nn.init.xavier_uniform_(policy[0].weight)
torch.nn.init.xavier_uniform_(policy[2].weight)
torch.nn.init.xavier_uniform_(policy[4].weight)
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)    

def make_q_func_with_optimizer():
    q_func = nn.Sequential(
        pfrl.nn.ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    torch.nn.init.xavier_uniform_(q_func[1].weight)
    torch.nn.init.xavier_uniform_(q_func[3].weight)
    torch.nn.init.xavier_uniform_(q_func[5].weight)
    q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
    return q_func, q_func_optimizer

q_func1, q_func1_optimizer = make_q_func_with_optimizer()
q_func2, q_func2_optimizer = make_q_func_with_optimizer()

def burnin_action_func():
    """Select random actions until model is updated one or more times."""
    return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

# Create the SAC agent
agent = pfrl.agents.SoftActorCritic(
    policy,
    q_func1,
    q_func2,
    policy_optimizer,
    q_func1_optimizer,
    q_func2_optimizer,
    replay_buffer=pfrl.replay_buffers.ReplayBuffer(10 ** 6),
    gamma=0.99,
    phi=lambda x: x.astype('float32', copy=False),
    gpu=-1,
    burnin_action_func=burnin_action_func,
    entropy_target=-action_size,
    temperature_optimizer_lr=3e-4,
)


f1 = open('x1_state.txt','w')
f1.write('')
    
f2 = open('x2_state.txt','w')
f2.write('')

with open('episode_reward.txt', 'w') as f:
    f.write('Episode\t Total Reward\n')
    # Training loop
    n_episodes      = 2000
    max_episode_len = 200
    for episode in range(n_episodes):
        obs = env.reset()
        R = 0
        t = 0
        x1_list, x2_list = [str(env.x1)], [str(env.x2)]
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            x1_list.append(str(obs[1]))
            x2_list.append(str(obs[2]))
            R += reward
            t += 1
            reset = t == max_episode_len
            
            agent.observe(obs, reward, done, reset)
            if done or reset:
                # Log or print the total episodic reward
                print(f"Episode {episode}: Total Reward: {R}")
                f.write(str(episode)+'\t'+str(R))
                f1.write('\t'.join(x1_list))
                f2.write('\t'.join(x2_list))
                
                if episode < n_episodes-1:
                    f.write('\n')
                    f1.write('\n')
                    f2.write('\n')
                    
                break
           
f1.close()
f2.close()
env.close()
# Save the trained model
agent.save("sac_continuous_action_model")