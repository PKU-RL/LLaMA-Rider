import argparse
import utils
import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
#from spinup_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
#from spinup_utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup_utils.run_utils import setup_logger_kwargs
from spinup_utils.logx import EpochLogger
from PIL import Image
import imageio
#from clip_model import build_model, tokenize_batch
#from torchvision.transforms import Resize 
#from skimage.transform import resize
from mineclip_official import build_pretrain_model, tokenize_batch, torch_normalize
#from minecraft import MinecraftEnv, preprocess_obs, transform_action
from envs.minecraft_nav import MinecraftNavEnv, preprocess_obs, transform_action
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import copy
import pickle
import matplotlib.pyplot as plt


def naive_policy(obs):
    goal_emb = obs['goal_emb']
    yaw = goal_emb[2:4]
    dr = goal_emb[4:6]
    dr = np.array([dr[1], -dr[0]])
    dr /= np.linalg.norm(dr)

    act = [0,0,0,12,12,0,0,0]
    # the direction is correct: forward or jump
    if np.dot(dr, yaw)>=np.cos(np.deg2rad(20)):
        if np.random.rand()<0.8:
            act[0] = 1
        else:
            act[2] = 1
        return act
    # should turn left
    if yaw[1]*dr[0]>=yaw[0]*dr[1]:
        act[4] = 10
    # turn right
    else:
        act[4] = 14
    return act



def test_navigation(args, seed=0, device=None, 
        steps_per_epoch=400, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4, vf_lr=1e-4,  
        train_pi_iters=80, train_v_iters=80, lam=0.95, max_ep_len=1000,
        target_kl=0.01, save_freq=1, logger_kwargs=dict(), save_path='checkpoint', 
        clip_config_path='', clip_model_path='', agent_config_path=''):


    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    #setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())


    # Random seed
    #seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # Instantiate environment
    env = MinecraftNavEnv(
        image_size=(160, 256),
        clip_model= None, 
        device=device,
        seed=seed,
        biome='plains'
    )
    obs_dim = env.observation_size
    env_act_dim = env.action_size
    agent_act_dim = len(args.actor_out_dim)
    print('Navigation env created.')
    #print('Task prompt:', env.task_prompt)
    #logger.log('env: obs {}, act {}'.format(env.observation_space, env.action_space))
   

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch

    start_time = time.time()
    saved_traj_cnt = 0 # counter for the saved experience

    # initialize the clip reward model
    #clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0 # Prepare for interaction with environment
        env.set_goal(pos=o['location_stats']['pos'])
        #clip_reward_model.update_obs(o['rgb_emb']) # preprocess the images embedding
        ep_rewards = []
        ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
        ep_ret_yaw, ep_ret_dis, ep_ret_pitch = 0, 0, 0
        rgb_list, pos_list = [], []
        episode_in_epoch_cnt = 0 # episode id in this epoch

        
        # rollout in the environment
        for t in range(local_steps_per_epoch):
            if args.save_raw_rgb:
                rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
            pos_list.append([o['location_stats']['pos'][0], o['location_stats']['pos'][2]])

            env.add_goal_to_obs(o)

            a_env = naive_policy(o)
            next_o, r, d, _ = env.step(a_env)
            #success = r

            # update the recent 16 frames, compute intrinsic reward
            #clip_reward_model.update_obs(next_o['rgb_emb'])
            #r_clip = clip_reward_model.reward(mode=args.clip_reward_mode)

            #r = r * args.reward_success + args.reward_step # + r_clip * args.reward_clip # weighted sum of different rewards
            ep_rewards.append(r)
            ep_obs = torch.cat((ep_obs, 
                torch_normalize(np.asarray(next_o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)), 1)

            #ep_success += success
            #if ep_success > 1:
            #    ep_success = 1
            #ep_ret_clip += r_clip
            ep_ret += r
            ep_ret_yaw += next_o['reward_yaw']
            ep_ret_dis += next_o['reward_dis']
            ep_ret_pitch += next_o['reward_pitch']
            ep_len += 1
            #print(next_o['reward_dis'], next_o['location_stats']['pos'], env.init_pos, env.goal_pos)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                # check and add to imitation buffer if the trajectory ends
                if terminal:
                    if args.save_raw_rgb:
                        rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
                    rgb_list = np.asarray(rgb_list)
                    #print(rgb_list.shape)
                    #expert_save_dir = os.path.join(args.save_path, 'expert_buffer') if args.save_expert_data else None
                    #imitation_buf.eval_and_store(obs_, act_, ep_ret_clip, int(ep_success), rgb_list, expert_save_dir)

                    # save the gif
                    if args.save_raw_rgb and ((epoch % save_freq == 0) or (epoch == epochs-1)) and (episode_in_epoch_cnt%1==0):
                        pth = os.path.join(args.save_path, 'gif', '{}_{}_ret{}.gif'.format(epoch, episode_in_epoch_cnt, ep_ret))
                        imageio.mimsave(pth, [np.transpose(i_, [1,2,0]) for i_ in rgb_list], duration=0.1)
                    # save visualized paths
                    if ((epoch % save_freq == 0) or (epoch == epochs-1)) and (episode_in_epoch_cnt%1==0):
                        plt.plot([a[0] for a in pos_list], [a[1] for a in pos_list], 'o', c='b')
                        #for i_ in range(len(pos_list)-1):
                        #    plt.quiver(pos_list[i_][0], pos_list[i_][1], pos_list[i_+1][0]-pos_list[i_][0], pos_list[i_+1][1]-pos_list[i_][1], angles='xy', scale=1, scale_units='xy')
                        plt.quiver(pos_list[0][0], pos_list[0][1], pos_list[-1][0]-pos_list[0][0], pos_list[-1][1]-pos_list[0][1], angles='xy', scale=1, scale_units='xy')
                        plt.plot(env.goal_pos[0], env.goal_pos[1], 'o', c='r')
                        pth = os.path.join(args.save_path, 'gif', '{}_{}_ret{}.png'.format(epoch, episode_in_epoch_cnt, ep_ret))
                        plt.savefig(pth)
                        plt.cla()


                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    env.add_goal_to_obs(o)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRetYaw=ep_ret_yaw, EpRetDis=ep_ret_dis, EpRetPitch=ep_ret_pitch)

                env.reset(reset_env=False) # in an epoch,  not reset the agent, change the goal only.
                env.set_goal(pos=o['location_stats']['pos'])
                ep_ret, ep_len = 0, 0
                ep_ret_yaw, ep_ret_dis, ep_ret_pitch = 0, 0, 0
                ep_rewards = []
                ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
                #clip_reward_model.reset() # don't forget to reset the clip images buffer
                #clip_reward_model.update_obs(o['rgb_emb']) # preprocess the images embedding
                rgb_list, pos_list = [], []
                episode_in_epoch_cnt += 1

        episode_in_epoch_cnt = 0

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRetYaw', with_min_and_max=True)
        logger.log_tabular('EpRetPitch', with_min_and_max=True)
        logger.log_tabular('EpRetDis', with_min_and_max=True)
        #logger.log_tabular('EpSuccess', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        #logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        #logger.log_tabular('LossPi', average_only=True)
        #logger.log_tabular('LossV', average_only=True)
        #logger.log_tabular('DeltaLossPi', average_only=True)
        #logger.log_tabular('DeltaLossV', average_only=True)
        #logger.log_tabular('Entropy', average_only=True)
        #logger.log_tabular('KL', average_only=True)
        #logger.log_tabular('ClipFrac', average_only=True)
        #logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        # to avoid destroying too many blocks, remake the environment
        if (epoch % 50 == 0) and epoch>0:
            env.remake_env()
            # save the imitation learning buffer
            #pth = os.path.join(save_path, 'buffer_{}.pth'.format(epoch))
            #pickle.dump(imitation_buf, open(pth, 'wb'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic arguments for PPO
    parser.add_argument('--gamma', type=float, default=0.99) # discount
    parser.add_argument('--target-kl', type=float, default=0.5) # kl upper bound for updating policy
    parser.add_argument('--seed', '-s', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--cpu', type=int, default=1) # number of workers
    parser.add_argument('--gpu', default='0') # -1 if use cpu, otherwise select the gpu id
    parser.add_argument('--steps', type=int, default=1000) # sample steps per PPO epoch (buffer size * workers)
    parser.add_argument('--epochs', type=int, default=2) # PPO epoch number
    parser.add_argument('--save-path', type=str, default='checkpoint') # save dir for model&data. Use /sharefs/baaiembodied/xxx on server
    parser.add_argument('--exp-name', type=str, default='test-nav') # experiment log name

    # CLIP model and agent model config
    parser.add_argument('--clip-config-path', type=str, default='mineclip_official/config.yml')
    parser.add_argument('--clip-model-path', type=str, default='mineclip_official/adjust.pth')
    parser.add_argument('--agent-model', type=str, default='mineagent') # agent architecture: mineagent, cnn
    parser.add_argument('--agent-config-path', type=str, default='mineagent/conf_goal_based_agent.yaml') # for mineagent
    parser.add_argument('--actor-out-dim', type=int, nargs='+', default=[12,3])
    ''' 
    actor output dimensions. mineagent official: [3,3,4,25,25,8]; my initial implement: [56,3]
    mineagent with clipped camera space: [3,3,4,5,3] or [12,3]
    should modify transform_action() in minecraft.py together with this arg
    '''
    
    # arguments for related research works
    parser.add_argument('--save-all-data', type=int, default=0) # save all the collected experience
    parser.add_argument('--save-expert-data', type=int, default=0) # save experience in self-imitation buffer
    parser.add_argument('--save-raw-rgb', type=int, default=1) # save rgb images when save the above data; save gif for debug

    args = parser.parse_args()
    #print(args)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = os.path.join(args.save_path, '{}-seed{}'.format(args.exp_name, args.seed))
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    pth = os.path.join(args.save_path, 'gif')
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = os.path.join(args.save_path, 'model')
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = os.path.join(args.save_path, 'experience_buffer')
    if not os.path.exists(pth):
        os.mkdir(pth)

    #mpi_fork(args.cpu)  # run parallel code with mpi
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # set gpu device
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    print('Using device:', device)

    test_navigation(args,
        gamma=args.gamma, save_path=args.save_path, target_kl=args.target_kl,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=device, 
        clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path,
        agent_config_path=args.agent_config_path)
