import numpy as np
import pickle
import torch
import random
from docopt import docopt
from model import ActorCriticModel
from utils import create_env
from environments.navigation_env import MinecraftNav
import os
import imageio
import matplotlib.pyplot as plt

def plot_state_visitation(pos_list, pth):
    for i in range(len(pos_list)-1):
        plt.quiver(pos_list[i][0], pos_list[i][1], pos_list[i+1][0]-pos_list[i][0], pos_list[i+1][1]-pos_list[i][1], 
            angles='xy', scale=1, scale_units='xy')
    plt.savefig(pth)
    plt.cla()

# run a test to find target
def run_test(env, model, target, device, config, test_id=None):
    #input('input something')
    # Run and render episode
    done = False
    episode_rewards = []

    # Init recurrent cell
    hxs, cxs = model.init_recurrent_cell_states(1, device)
    if config["recurrence"]["layer_type"] == "gru":
        recurrent_cell = hxs
    elif config["recurrence"]["layer_type"] == "lstm":
        recurrent_cell = (hxs, cxs)

    obs = env.reset()
    #print(env.obs_env['location_stats']['pos'])
    if test_id is not None:
        imageio.imsave('test/{}_{}_begin.png'.format(target,test_id), np.transpose(env.obs_env['rgb'],[1,2,0]))
    while not done:
        # Render environment
        #env.render()
        # Forward model
        policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)).float(), recurrent_cell, device, 1)
        # Sample action
        action = policy.sample().cpu().numpy()
        # Step environemnt
        obs, reward, done, info = env.step(int(action), target=target)
        episode_rewards.append(reward)
    
    #print(info)
    #input('input something')
    success=False
    if 'dis' in info:
        if test_id is not None:
            imageio.imsave('test/{}_{}_find.png'.format(target,test_id), np.transpose(env.obs_env['rgb'],[1,2,0]))
            if env.reach(target, info):
                if test_id is not None:
                    imageio.imsave('test/{}_{}_reach.png'.format(target,test_id), np.transpose(env.obs_env['rgb'],[1,2,0]))
                success=True
    
    if test_id is not None:
        plot_state_visitation(env.visited_pos, 'test/{}_{}_reward_{}.png'.format(target,test_id,info["reward"]))
    return success, env.env.total_steps

    #print("Explore length: " + str(info["length"]))
    #print("Explore reward: " + str(info["reward"]))


'''
Test models:
./summaries/run/20230104-174338/epoch_200_reward_16.13.pth
./summaries/run/20230104-174338/epoch_1_reward_-124.0.pth
./summaries/run/20230109-001127/epoch_350_reward_5.16.pth
'''
def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              [default: ./summaries/dqn/20230130-235125/epoch_250_reward_4.08.pth].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print('running on device: ', device)

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))
    print('model loaded:', model_path)

    # seed control
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Instantiate environment
    env = MinecraftNav(seed=seed, max_steps=20, usage='test', device=device)

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if not os.path.exists('test'):
        os.mkdir('test')
    
    #run test
    for target in ['wood', 'sheep', 'cow']:
        success = 0
        total_steps = 0
        for i in range(10):
            suc, steps = run_test(env, model, target, device, config, test_id=i)
            if suc:
                success += 1
                total_steps += steps
                print('Find {} success. Steps {}'.format(target, steps))
            else:
                print('Fail. Steps {}'.format(steps))
        print('Find {} success rate: {} Average succeed steps: {}'.format(
            target, success/10., (total_steps)/success))
        env.env.remake_env()



if __name__ == "__main__":
    main()