import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

BATCH_SIZE = 32  # How many transitions to sample each time experience is replayed.
GAMMA = 0.99  # Discount factor
REPLAY_BUFFER_SIZE = 1000000  # Replay buffer size
LEARNING_STARTS = 50000  # After how many environment steps to start replaying experiences
LEARNING_FREQ = 4  # How many steps of environment to take between every experience replay
FRAME_HISTORY_LEN = 4  # How many past frames to include as input to the model.
TARGER_UPDATE_FREQ = 10000  # How many experience replay rounds (not steps!) to perform between
                            # each update to the target Q network
LEARNING_RATE = 0.00025  # Learning rate of the optimizer of the network
ALPHA = 0.95  # Learning rate of the Q-Learning algorithm
EPS = 0.01  # Trade-off between exploration and exploitation

def main(env, num_timesteps):

    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, task.max_timesteps)
