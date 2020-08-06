from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger

def test_sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1):

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)
    torch.manual_seed(seed)
    np.random.seed(seed)
    from bug_crippled import BugCrippledEnv
    test_env = BugCrippledEnv()
    # Create actor-critic module and target networks
    ac = actor_critic(test_env.observation_space, test_env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load("data/sac_bug_cripple_remote_2/sac_bug_cripple_2/model_ac.pth"))
    ac.eval()

    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            o, r, d, _ = test_env.step(get_action(o, True))
            ep_ret += r
            ep_len += 1
            test_env.render()

        print("Episodic Return", ep_ret)
        print("Episodic Length", ep_len)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    test_sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)