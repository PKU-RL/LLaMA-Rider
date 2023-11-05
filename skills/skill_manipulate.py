from envs.minecraft_hard_task import preprocess_obs, transform_action, transform_action_2_5
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import pickle
import utils
import copy
import torch
import numpy as np

class SkillManipulate:
    def __init__(self, device=torch.device('cuda:0'), actor_out_dim=[12,3], agent_config_path='mineagent/conf.yaml'):
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
            action_dim=actor_out_dim,
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
        self.mine_agent = MineAgent(
            actor=actor, 
            critic=critic,
            deterministic_eval=False
        ).to(device) # use the same stochastic policy in training and test
        self.mine_agent.eval()
        self.device=device

    def execute(self, target, model_path, max_steps, env, equip_list, **kwargs):
        state_dict = torch.load(model_path, map_location=self.device)
        self.mine_agent.load_state_dict(state_dict)

        # If equipment list is empty, we do not allow the use action
        allow_use = True if len(equip_list)>0 else False

        # wait for some steps
        for step in range(5):
            act = self.reset_camera(env)
            obs, r, done, _ = env.step(act)
            if done:
                return False, bool(r), done

        obs = env.obs
        for step in range(max_steps):
            batch = preprocess_obs(obs, self.device)
            with torch.no_grad():
                act = self.mine_agent(batch).act
            act = transform_action(act, allow_use)
            obs, r, done, _ = env.step(act)
            # detect skill done
            if env.reward_harvest(obs, target):
                return True, bool(r), done # skill done, task success, task done
            elif done:
                return False, bool(r), done # skill done, task success, task done
        return False, False, False # skill done, task success, task done

    def reset_camera(self, env):
        act = env.base_env.action_space.no_op()
        # correct the pitch direction
        pitch = env.obs["location_stats"]["pitch"]
        if pitch>20:
            act[3] = 10
            return act
        elif pitch<-20:
            act[3] = 14
            return act
        return act


MAX_MINE_STEPS = {'cobblestone':200, 'cobblestone_nearby': 200}
DEFAULT_MAX_MINE_STEPS = 500
MAX_BACK_STEPS = {'cobblestone':400, 'cobblestone_nearby':400}
DEFAULT_MAX_BACK_STEPS = 1000

# skill to mine cobblestones, coals and ores in iron-based tasks
class SkillMine:
    def __init__(self, device=torch.device('cuda:0'), actor_out_dim=[2,5], agent_config_path='mineagent/conf.yaml'):
        self.agent_config_path=agent_config_path
        self.device=device

        self.mine_agent = self._make_nn(actor_out_dim)
        state_dict = torch.load('skills/models/mine.pth', map_location=self.device)
        self.mine_agent.load_state_dict(state_dict)

        self.back_agent = self._make_nn(actor_out_dim)
        state_dict = torch.load('skills/models/back.pth', map_location=self.device)
        self.back_agent.load_state_dict(state_dict)
        print('Policies for mining ores loaded.')

    def _make_nn(self, actor_out_dim):
        agent_config = utils.get_yaml_data(self.agent_config_path)
        feature_net_kwargs = agent_config['feature_net_kwargs']
        feature_net = {}
        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(features, cls)
            feature_net[k] = cls(**v, device=self.device)
        feature_fusion_kwargs = agent_config['feature_fusion']
        feature_net = SimpleFeatureFusion(
            feature_net, **feature_fusion_kwargs, device=self.device
        )
        feature_net_v = copy.deepcopy(feature_net) # actor and critic do not share
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=actor_out_dim,
            device=self.device,
            **agent_config['actor'],
            activation='tanh',
        )
        critic = Critic(
            feature_net_v,
            action_dim=None,
            device=self.device,
            **agent_config['actor'],
            activation='tanh'
        )
        mine_agent = MineAgent(
            actor=actor, 
            critic=critic,
            deterministic_eval=False
        ).to(self.device) # use the same stochastic policy in training and test
        mine_agent.eval()
        return mine_agent

    def execute(self, target, env, **kwargs):
        # equip the best pickaxe
        inventory = env.obs['inventory']['name'].tolist()
        tools = ['diamond pickaxe', 'golden pickaxe', 'iron pickaxe', 'stone pickaxe', 'wooden pickaxe']
        for e in tools:
            if e in inventory:
                idx = inventory.index(e)
                act = env.base_env.action_space.no_op()
                act[5] = 5
                act[7] = idx
                obs, r, done, _ = env.step(act)
                if done:
                    return False, bool(r), done # skill done, task success, task done
                break

        # count target items
        def count_items(o, tgt):
            ret = 0
            names, nums = o['inventory']['name'], o['inventory']['quantity']
            idxs = np.where(names==tgt.replace('_',' '))[0]
            if len(idxs)>0:
                ret = np.sum(nums[idxs])
            return ret

        num_init = count_items(env.obs, target)
        init_height = env.obs['location_stats']['pos'][1]
        #print('init height', init_height)

        
        # move to the block center
        def center(obs):
            act = env.base_env.action_space.no_op()
            yaw = obs["location_stats"]["yaw"]
            # the direction is correctr
            if np.cos(np.deg2rad(yaw))>=np.cos(np.deg2rad(20)):
                pos = obs['location_stats']['pos']
                x, z = pos[0], pos[2]
                x, z = x-np.floor(x), z-np.floor(z)
                if z<0.4:
                    act[0] = 1
                    return act
                elif z>0.6:
                    act[0] = 2
                    return act
                elif x<0.4:
                    act[1] = 1
                    return act 
                elif x>0.6:
                    act[1] = 2
                    return act 
                else:
                    return act # no longer need to adjust
            # should turn left
            if np.sin(np.deg2rad(yaw))>=0:
                act[4] = 10
            # turn right
            else:
                act[4] = 14
            return act
        for step in range(20):
            act = center(env.obs)
            obs, r, done, _ = env.step(act)
            if done:
                return False, bool(r), done # skill done, task success, task done
        #print('position:', env.obs['location_stats']['pos'])


        # mine ores
        max_steps = MAX_MINE_STEPS[target] if target in MAX_MINE_STEPS else DEFAULT_MAX_MINE_STEPS
        obs = env.obs
        for step in range(max_steps):
            batch = preprocess_obs(obs, self.device)
            with torch.no_grad():
                act = self.mine_agent(batch).act
            act = transform_action_2_5(act, attack=True)
            #act = [0,0,0,18,12,3,0,0] # dig down
            obs, r, done, _ = env.step(act)
            if done:
                return bool(r), bool(r), done # skill done, task success, task done
            # stop if y<10
            min_depth = 10 if target=='diamond' else 15
            if env.obs['location_stats']['pos'][1]<min_depth:
                break

        # back to the ground
        max_steps = MAX_BACK_STEPS[target] if target in MAX_BACK_STEPS else DEFAULT_MAX_BACK_STEPS
        #print('position:', env.obs['location_stats']['pos'])
        #print(env.obs['inventory']['name'])
        blocks = ['dirt', 'stone', 'cobblestone']
        for step in range(max_steps):
            # if no blocks in hand, equip some blocks
            inventory = env.obs['inventory']['name'].tolist()
            if inventory[0] not in blocks:
                flag = False
                for e in blocks:
                    if e in inventory:
                        idx = inventory.index(e)
                        act = env.base_env.action_space.no_op()
                        act[5] = 5
                        act[7] = idx
                        obs, r, done, _ = env.step(act)
                        if done:
                            return False, bool(r), done # skill done, task success, task done
                        flag = True
                        break
                if not flag: # no enough blocks to equip
                    break
            
            # execute back skill
            batch = preprocess_obs(env.obs, self.device)
            with torch.no_grad():
                act = self.back_agent(batch).act
            act = transform_action_2_5(act, attack=False)
            #act = [0,0,1,18,12, 1,0,0] # come back
            obs, r, done, _ = env.step(act)
            if done:
                return bool(r), bool(r), done # skill done, task success, task done

            #print('height', env.obs['location_stats']['pos'][1])
            # success
            if env.obs['location_stats']['pos'][1]>=init_height-0.5:
                break

        # randomly walk around
        def walk(obs):
            act = env.base_env.action_space.no_op()
            # correct the pitch direction
            pitch = obs["location_stats"]["pitch"]
            if pitch>20:
                act[3] = 10
                act[4] = np.random.randint(0,13)*2
                return act
            elif pitch<-20:
                act[3] = 14
                act[4] = np.random.randint(0,13)*2
                return act
            # walk and break
            act[0] = 1
            act[2] = 1
            act[5] = 3
            return act

        inventory = env.obs['inventory']['name'].tolist()
        for e in tools:
            if e in inventory:
                idx = inventory.index(e)
                act = env.base_env.action_space.no_op()
                act[5] = 5
                act[7] = idx
                obs, r, done, _ = env.step(act)
                if done:
                    return False, bool(r), done # skill done, task success, task done
                break
        for i in range(0,np.random.randint(50,100)):
            act = walk(env.obs)
            obs, r, done, _ = env.step(act)
            if done:
                return False, bool(r), done # skill done, task success, task done

        # skill finish
        num_end = count_items(env.obs, target)
        if num_end-num_init>0:
            return True, False, False
        else:
            return False, False, False # skill done, task success, task done
