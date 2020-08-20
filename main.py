import os
import random
import torch
import argparse
import numpy as np

from network import core
from misc.run_utils import setup_logger_kwargs


def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set Env
    from gym_env.bug_crippled import BugCrippledEnv
    env = BugCrippledEnv(cripple_prob=1.0)
    env.change_env_every = args.change_env_every

    # Set logs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Start either train or test
    if not args.test_mode:
        from trainer import train

        train(lambda: env, actor_critic=core.MLPActorCritic,
              ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
              gamma=args.gamma, seed=args.seed, epochs=args.epochs,
              logger_kwargs=logger_kwargs)
    else:
        from tester import test

        test(lambda: env, actor_critic=core.MLPActorCritic,
             ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
             logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=600 * 6)
    parser.add_argument('--change-env-every', type=int, default=600)

    # Env
    parser.add_argument('--env', type=str, default='BugCrippledEnv')
    parser.add_argument('--exp_name', type=str, default='sac_bug_cripple')
    parser.add_argument("--test-mode", action="store_true", help="If True, perform test mode")
    # Set log name
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Set log name
    args.log_name = \
        "env::%s_seed::%log" % (
            args.env, args.seed)

    main(args)
