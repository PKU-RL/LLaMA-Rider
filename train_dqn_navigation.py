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
from mineclip.utils import build_mlp
import random
import collections


class Qnet(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        action: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        self.preprocess = preprocess_net
        self.net = build_mlp(
            input_dim=preprocess_net.output_dim,
            output_dim=action,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None,
        )
        self._action = action
        self._device = device

    def forward(self, x):
        y, _ = self.preprocess(x)
        return self.net(y)

class DQNbuffer:

    def __init__(self, act_dim, size=10000):
        self.obs_buf = [Batch() for i in range(size)]#np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.next_o_buf = [Batch() for i in range(size)]
        self.max_size = size
        self.full_flag = False
        self.ptr = 0

    def store(self, obs, act, rew, next_o):
        if self.ptr >= self.max_size:
            self.full_flag = True
            self.ptr = 0
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_o_buf[self.ptr] = next_o
        self.ptr += 1

    def sample(self, batch_size):
        if self.full_flag:
            get_id = random.sample(range(0,self.max_size),batch_size)
        else:
            get_id = random.sample(range(0,self.ptr),batch_size)
        data = dict(act=self.act_buf[get_id],rew=self.rew_buf[get_id])
        rtn = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        # rtn['obs'] = self.obs_buf[get_id]
        # rtn['next_o'] = self.next_o_buf[get_id]
        _obs = list()
        _next_obs = list()
        for i in get_id:
            _obs.append(self.obs_buf[i])
            _next_obs.append(self.next_o_buf[i])
        
        rtn['obs'] = Batch.cat(_obs)
        rtn['next_o'] = Batch.cat(_next_obs)

        return rtn

    def size(self):
        if self.full_flag:
            return self.max_size
        else:
            return self.ptr

def DQN(args, seed=0, device=None,
        steps_per_epoch=400, epochs=500, gamma=0.99, qf_lr=1e-4, epsilon=0.05, target_update=5, minimal_size = 105,
        action_dim=36, batch_size=100 ,max_ep_len=1000, save_freq=5, logger_kwargs=dict(), save_path='checkpoint',
        clip_config_path='', clip_model_path='', agent_config_path=''):
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load pretrained mineclip model
    clip_config = utils.get_yaml_data(clip_config_path)
    model_clip = build_pretrain_model(
        image_config=clip_config['image_config'],
        text_config=clip_config['text_config'],
        temporal_config=clip_config['temporal_config'],
        adapter_config=clip_config['adaptor_config'],
        state_dict=torch.load(clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model loaded.')

    env = MinecraftNavEnv(
        image_size=(160, 256),
        clip_model=model_clip if (args.agent_model == 'mineagent') else None, 
        device=device,
        seed=seed,
        biome='plains'
    )
    obs_dim = env.observation_size
    env_act_dim = env.action_size
    agent_act_dim = len(args.actor_out_dim)
    print('Navigation env created.')

    # # Instantiate environment
    # env = MinecraftEnv(
    #     task_id=args.task,
    #     image_size=(160, 256),
    #     max_step=args.horizon,
    #     clip_model=model_clip,
    #     device=device,
    #     seed=seed,
    #     dense_reward=bool(args.use_dense)
    # )
    # obs_dim = env.observation_size
    # env_act_dim = env.action_size
    # agent_act_dim = len(args.actor_out_dim)
    # print('Task prompt:', env.task_prompt)

    # Create DQN agent
    agent_config = utils.get_yaml_data(agent_config_path)
    feature_net_kwargs = agent_config['feature_net_kwargs']
    feature_net = {}
    for k, v in feature_net_kwargs.items():
        v = dict(v)
        cls = v.pop("cls")
        cls = getattr(features, cls)
        feature_net[k] = cls(**v, device=device)
    feature_fusion_kwargs = agent_config['feature_fusion']
    feature_net = SimpleFeatureFusion(
        feature_net, **feature_fusion_kwargs, device=device
    )
    #feature_net_v = copy.deepcopy(feature_net)  # actor and critic do not share
    # #feature_net finish
    dqn = Qnet(
        feature_net,
        action=action_dim,  #[12,3]
        device=device,
        **agent_config['actor'],
    ).to(device)
    dqn.eval()
    target_net = copy.deepcopy(dqn)
    optimizer = torch.optim.Adam(dqn.parameters(), lr= qf_lr)
    # Count variables
    var_counts = (utils.count_vars(dqn), utils.count_vars(model_clip))
    logger.log('\nNumber of parameters: \t dqn: %d, \t mineclip: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = DQNbuffer(agent_act_dim)

    def take_action(obs, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(0,action_dim)
        else:
            action = dqn(obs.obs).argmax().item()
        act = action_process(action)
        return act

    def action_process(act):
        action = torch.zeros((1,2), dtype=int)
        action[0][1] = act % 3
        action[0][0] = act // 3
        return action

    local_target_update = target_update
    local_update_count = 0

    def update(data, update_count):
        obs,act,rew,next_o = data['obs'],data['act'].to(device), \
                            data['rew'].to(device),data['next_o']
        obs.to_torch(device=device)
        next_o.to_torch(device=device)
        act = act.to(torch.int64)
        act = act.to(device=device)
        # print(act.device)
        rew = rew.unsqueeze(1)
        # print(rew.shape)
        _act = (torch.tensor([batch_size,1],dtype=torch.int64)).to(device=device)
        _act = (act[:,0]*3 + act[:,1]).unsqueeze(1)
        #print(torch.index_select(act, 1,torch.tensor([0])).device)
        # print(_act)
        q_values = dqn(obs.obs).gather(1, _act)
        max_next_q_values = target_net(next_o.obs).max(1)[0].view(-1,1)
        # print(max_next_q_values.shape)
        
        q_targets = rew + gamma * max_next_q_values
        # print(q_values)
        # print(q_targets)
        # print(q_targets.shape)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        optimizer.zero_grad()
        dqn_loss.backward()
        optimizer.step()
        update_count += 1
        if update_count % local_target_update == 0:
            target_net.load_state_dict(dqn.state_dict())  # 更新目标网络

    start_time = time.time()

    # initialize the clip reward model
    # clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        # save a video of test
        def test_video():
            pth = os.path.join(save_path, '{}.gif'.format(epoch))
            # if not os.path.exists(pth):
            #    os.mkdir(pth)
            dqn.eval()  # in eval mode, the actor is also stochastic now
            obs = env.reset()
            gameover = False
            # i = 0
            img_list = []
            while True:
                img_list.append(np.transpose(obs['rgb'], [1, 2, 0]).astype(np.uint8))
                if gameover:
                    break
                batch = preprocess_obs(obs, device)
                act = take_action(batch,epsilon)
                act = transform_action(act)
                obs, r, gameover, _ = env.step(act)
                # i += 1
            imageio.mimsave(pth, img_list, duration=0.1)
            print("save success")
            # env.reset()
            # mine_agent.train()

        # Save model and test
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            # test_video()
            # logger.save_state({'env': env}, None)
            pth = os.path.join(save_path, 'model_{}.pth'.format(epoch))
            torch.save(dqn.state_dict(), pth)

        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0 # Prepare for interaction with environment
        env.set_goal(pos=o['location_stats']['pos'])
        #clip_reward_model.update_obs(o['rgb_emb']) # preprocess the images embedding
        ep_rewards = []
        ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
        ep_ret_yaw, ep_ret_dis, ep_ret_pitch = 0, 0, 0
        rgb_list, pos_list = [], []
        episode_in_epoch_cnt = 0 # episode id in this epoch
        
        # o, ep_ret, ep_len = env.reset(), 0, 0  # Prepare for interaction with environment
        # clip_reward_model.update_obs(o['rgb_emb'])  # preprocess the images embedding
        # ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
        # #rgb_list = []



        # rollout in the environment
        dqn.train()  # train mode to sample stochastic actions
        target_net.train()
        for t in range(local_steps_per_epoch):

            pos_list.append([o['location_stats']['pos'][0], o['location_stats']['pos'][2]])

            env.add_goal_to_obs(o)





            #if args.save_raw_rgb:
            #    rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))

            batch_o = preprocess_obs(o, device)
            batch_act = take_action(batch_o,epsilon)
            a = batch_act
            # print('a,v,logp = ', a, v, logp)

            a_env = transform_action(a)
            next_o, r, d, _ = env.step(a_env)
            # success = r
            env.add_goal_to_obs(next_o)

            batch_next_o = preprocess_obs(next_o, device)



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





            # # # update the recent 16 frames, compute intrinsic reward
            # # clip_reward_model.update_obs(next_o['rgb_emb'])
            # # r_clip = clip_reward_model.reward(mode=args.clip_reward_mode)

            # r = r * args.reward_success + r_clip * args.reward_clip + args.reward_step  # weighted sum of different rewards

            # # dense reward
            # if args.use_dense:
            #     r_dense = next_o['dense_reward']
            #     r += r_dense * args.reward_dense
            #     ep_ret_dense += r_dense

            # ep_success += success
            # ep_ret_clip += r_clip
            # ep_ret += r
            # ep_len += 1

            # save and log
            batch_o.to_numpy()  # less gpu mem
            batch_next_o.to_numpy()
            buf.store(batch_o, a[0].cpu(), r, batch_next_o)
            # buf.store(batch_o, a[0].cpu(), r)


            # Update obs (critical!)
            o = next_o

            if buf.size() > minimal_size:
                data = buf.sample(batch_size)
                update(data,local_update_count)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                # check and add to imitation buffer if the trajectory ends
                # if terminal:
                    # obs_ = Batch.cat(buf.obs_buf[buf.path_start_idx: buf.ptr])
                    # act_ = buf.act_buf[buf.path_start_idx: buf.ptr].copy()
                    # rgb_list = np.asarray(rgb_list)
                    # print(rgb_list.shape)
                    # imitation_buf.eval_and_store(obs_, act_, ep_ret_clip, int(ep_success), rgb_list)

                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target


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




                # if terminal:
                #     # only save EpRet / EpLen if trajectory finished
                #     logger.store(EpRet=ep_ret, EpLen=ep_len, EpSuccess=ep_success,
                #                  EpRetDense=ep_ret_dense, EpRetClip=ep_ret_clip)

                # o, ep_ret, ep_len = env.reset(), 0, 0
                # ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
                # clip_reward_model.reset()  # don't forget to reset the clip images buffer
                # clip_reward_model.update_obs(o['rgb_emb'])  # preprocess the images embedding
                # #rgb_list = []


                # save the experience
                if args.save_all_data:
                        pth = os.path.join(args.save_path, 'experience_buffer', 'traj_{}.pth'.format(saved_traj_cnt))
                        pickle.dump([obs_, act_, ep_ret_clip, int(ep_success), rgb_list], open(pth, 'wb'))
                        saved_traj_cnt += 1

                # save the gif
                if args.save_raw_rgb and ((epoch % save_freq == 0) or (epoch == epochs-1)) and (episode_in_epoch_cnt%10==0):
                    pth = os.path.join(args.save_path, 'gif', '{}_{}_ret{}.gif'.format(epoch, episode_in_epoch_cnt, ep_ret))
                    imageio.mimsave(pth, [np.transpose(i_, [1,2,0]) for i_ in rgb_list], duration=0.1)
                # save visualized paths
                # if ((epoch % save_freq == 0) or (epoch == epochs-1)) and (episode_in_epoch_cnt%10==0):
                #     plt.plot([a[0] for a in pos_list], [a[1] for a in pos_list], 'o', c='b')
                #     #for i_ in range(len(pos_list)-1):
                #     #    plt.quiver(pos_list[i_][0], pos_list[i_][1], pos_list[i_+1][0]-pos_list[i_][0], pos_list[i_+1][1]-pos_list[i_][1], angles='xy', scale=1, scale_units='xy')
                #     plt.quiver(pos_list[0][0], pos_list[0][1], pos_list[-1][0]-pos_list[0][0], pos_list[-1][1]-pos_list[0][1], angles='xy', scale=1, scale_units='xy')
                #     plt.plot(env.goal_pos[0], env.goal_pos[1], 'o', c='r')
                #     pth = os.path.join(args.save_path, 'gif', '{}_{}_ret{}.png'.format(epoch, episode_in_epoch_cnt, ep_ret))
                #     plt.savefig(pth)
                #     plt.cla()

            # to avoid destroying too many blocks, remake the environment
            if (epoch % 50 == 0) and epoch > 0:
                env.remake_env()
                # save the imitation learning buffer
                pth = os.path.join(save_path, 'buffer_{}.pth'.format(epoch))

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRetYaw', with_min_and_max=True)
        logger.log_tabular('EpRetPitch', with_min_and_max=True)
        logger.log_tabular('EpRetDis', with_min_and_max=True)
        #logger.log_tabular('EpSuccess', with_min_and_max=True)
        # logger.log_tabular('EpRetClip', with_min_and_max=True)
        # logger.log_tabular('EpSuccess', with_min_and_max=True)
        # logger.log_tabular('EpRetDense', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()






'''
PPO training for goal-based navigation policy
'''
def dqn_navigation(args, seed=0, device=None, 
        steps_per_epoch=400, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4, vf_lr=1e-4,  
        train_pi_iters=80, train_v_iters=80, lam=0.95, max_ep_len=1000,
        target_kl=0.01, save_freq=5, logger_kwargs=dict(), save_path='checkpoint', 
        clip_config_path='', clip_model_path='', agent_config_path=''):

    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        device: cpu or cuda gpu device for training NN

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    #setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())


    # Random seed
    #seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # load pretrained mineclip model
    clip_config = utils.get_yaml_data(clip_config_path)
    model_clip = build_pretrain_model(
        image_config = clip_config['image_config'],
        text_config = clip_config['text_config'],
        temporal_config = clip_config['temporal_config'],
        adapter_config = clip_config['adaptor_config'],
        state_dict = torch.load(clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model loaded.')


    # Instantiate environment
    env = MinecraftNavEnv(
        image_size=(160, 256),
        clip_model=model_clip if (args.agent_model == 'mineagent') else None, 
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


    # Create actor-critic agent
    if args.agent_model == 'mineagent':
        agent_config = utils.get_yaml_data(agent_config_path)
        feature_net_kwargs = agent_config['feature_net_kwargs']
        feature_net = {}
        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(features, cls)
            feature_net[k] = cls(**v, device=device)
        feature_fusion_kwargs = agent_config['feature_fusion']
        feature_net = SimpleFeatureFusion(
            feature_net, **feature_fusion_kwargs, device=device
        )
        feature_net_v = copy.deepcopy(feature_net) # actor and critic do not share
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=args.actor_out_dim, #[12,3]
            device=device,
            **agent_config['actor'],
            activation='tanh',
        )
        critic = Critic(
            feature_net_v,
            action_dim=None,
            device=device,
            **agent_config['actor'],
            activation='tanh'
        )
        mine_agent = MineAgent(
            actor=actor, 
            critic=critic,
            deterministic_eval=False
        ).to(device) # use the same stochastic policy in training and test
        mine_agent.eval()
    else:
        raise NotImplementedError

    # Sync params across processes
    #sync_params(ac)

    # Count variables
    var_counts = (#utils.count_vars(actor), utils.count_vars(critic),
        utils.count_vars(mine_agent), utils.count_vars(model_clip))
    #logger.log('\nNumber of parameters: \t actor: %d, \t critic: %d, \t  agent: %d, \t mineclip: %d\n'%var_counts)
    logger.log('\nNumber of parameters: \t agent: %d, \t mineclip: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPOBuffer(agent_act_dim, local_steps_per_epoch, gamma, lam, args.agent_model, obs_dim)
 

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'].to(device), \
                                data['adv'].to(device), data['logp'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)

        # Policy loss
        pi = mine_agent(obs).dist
        logp = pi.log_prob(act)
        #print('logp, logp_old = ', logp, logp_old)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
            obs_ = obs.obs
        else:
            obs_ = obs.to(device)
        return ((mine_agent.critic(obs_) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(mine_agent.actor.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(mine_agent.critic.parameters(), lr=vf_lr)
    #optimizer = torch.optim.Adam(mine_agent.parameters(), lr=lr)


    # a training epoch
    def update():
        mine_agent.train()

        data = buf.get() # dict

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()


        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            #mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl'] #mpi_avg(pi_info['kl'])
            #logger.log('kl={}'.format(kl))
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl, kl=%f.'%(i, kl))
                break
            loss_pi.backward()
            #mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)


        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    
    start_time = time.time()
    saved_traj_cnt = 0 # counter for the saved experience

    # initialize the clip reward model
    #clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        
        # Save model and test
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            #test_video()
            #logger.save_state({'env': env}, None)
            pth = os.path.join(save_path, 'model', 'model_{}.pth'.format(epoch))
            torch.save(mine_agent.state_dict(), pth)


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
        mine_agent.train() # train mode to sample stochastic actions
        for t in range(local_steps_per_epoch):
            if args.save_raw_rgb:
                rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
            pos_list.append([o['location_stats']['pos'][0], o['location_stats']['pos'][2]])

            env.add_goal_to_obs(o)
            if args.agent_model == 'mineagent':
                batch_o = preprocess_obs(o, device)
            else:
                #batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,*obs_dim)
                #batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)
                raise NotImplementedError
            #print(batch_o)

            with torch.no_grad():
                batch_act = mine_agent.forward_actor_critic(batch_o)
            a, v, logp = batch_act.act, batch_act.val, batch_act.logp
            v = v[0]
            logp = logp[0]
            #print('a,v,logp = ', a, v, logp)

            a_env = transform_action(a)
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

            # save and log
            if args.agent_model == 'mineagent':
                batch_o.to_numpy() # less gpu mem
            else:
                batch_o = batch_o.cpu().numpy()
            buf.store(batch_o, a[0].cpu().numpy(), r, v, logp) # the stored reward will be modified at episode end, if use CLIP reward
            logger.store(VVals=v.detach().cpu().numpy())
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                # check and add to imitation buffer if the trajectory ends
                if terminal:
                    if args.agent_model == 'mineagent':
                        obs_ = Batch.cat(buf.obs_buf[buf.path_start_idx: buf.ptr])
                    else:
                        obs_ = buf.obs_buf[buf.path_start_idx: buf.ptr].copy()
                    act_ = buf.act_buf[buf.path_start_idx: buf.ptr].copy()
                    if args.save_raw_rgb:
                        rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
                    rgb_list = np.asarray(rgb_list)
                    #print(rgb_list.shape)
                    #expert_save_dir = os.path.join(args.save_path, 'expert_buffer') if args.save_expert_data else None
                    #imitation_buf.eval_and_store(obs_, act_, ep_ret_clip, int(ep_success), rgb_list, expert_save_dir)

                    # save the experience
                    if args.save_all_data:
                        pth = os.path.join(args.save_path, 'experience_buffer', 'traj_{}.pth'.format(saved_traj_cnt))
                        pickle.dump([obs_, act_, ep_ret_clip, int(ep_success), rgb_list], open(pth, 'wb'))
                        saved_traj_cnt += 1

                    # save the gif
                    if args.save_raw_rgb and ((epoch % save_freq == 0) or (epoch == epochs-1)) and (episode_in_epoch_cnt%10==0):
                        pth = os.path.join(args.save_path, 'gif', '{}_{}_ret{}.gif'.format(epoch, episode_in_epoch_cnt, ep_ret))
                        imageio.mimsave(pth, [np.transpose(i_, [1,2,0]) for i_ in rgb_list], duration=0.1)
                    # save visualized paths
                    if ((epoch % save_freq == 0) or (epoch == epochs-1)) and (episode_in_epoch_cnt%10==0):
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
                    if args.agent_model == 'mineagent':
                        batch_o = preprocess_obs(o, device)
                    else:
                        batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,*obs_dim)
                        batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        v = mine_agent.forward_actor_critic(batch_o).val
                    v = v[0].cpu().detach().numpy()
                else:
                    v = 0
                buf.finish_path(v)



                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRetYaw=ep_ret_yaw, EpRetDis=ep_ret_dis, EpRetPitch=ep_ret_pitch)

                env.reset(reset_env=False) # in an epoch,  not reset the agent, change the goal only.
                env.set_goal(pos=o['location_stats']['pos'])
                ep_ret, ep_len = 0, 0
                ep_ret_yaw, ep_ret_dis, ep_ret_pitch = 0, 0, 0





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


        # Perform PPO update!
        update()
        episode_in_epoch_cnt = 0

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRetYaw', with_min_and_max=True)
        logger.log_tabular('EpRetPitch', with_min_and_max=True)
        logger.log_tabular('EpRetDis', with_min_and_max=True)
        #logger.log_tabular('EpSuccess', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
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
    parser.add_argument('--epochs', type=int, default=1000) # PPO epoch number
    parser.add_argument('--save-path', type=str, default='checkpoint') # save dir for model&data. Use /sharefs/baaiembodied/xxx on server
    parser.add_argument('--exp-name', type=str, default='ppo-nav') # experiment log name

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

    # DQN(args,
    #     gamma=args.gamma, save_path=args.save_path, target_kl=args.target_kl,
    #     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    #     logger_kwargs=logger_kwargs, device=device, 
    #     clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path,
    #     agent_config_path=args.agent_config_path)
    DQN(args ,gamma=args.gamma, save_path=args.save_path,seed=args.seed,steps_per_epoch=args.steps,
        epochs=args.epochs,logger_kwargs=logger_kwargs, device=device,
        clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path,
        agent_config_path=args.agent_config_path, action_dim=np.prod(args.actor_out_dim))
