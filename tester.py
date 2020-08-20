import torch
import network.core as core


def test(test_env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),
         num_test_episodes=10, max_ep_len=1000, logger_kwargs=dict()):

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32),
                      deterministic)

    # Create actor-critic module and target networks
    ac = actor_critic(test_env.observation_space, test_env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load("data/" + logger_kwargs["exp_name"] + "/model_ac.pth"))
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

