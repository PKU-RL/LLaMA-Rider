import utils
from .skill_manipulate import SkillManipulate, SkillMine
from .skill_craft import SkillCraft
from .skill_find import SkillFind
import numpy as np

class SkillsModel:
    def __init__(self, device, path='skills/load_skills.yaml', underground=False):
        self.device = device
        self.skill_info = utils.get_yaml_data(path)
        self.skill_models = [
            SkillFind(device=device),
            SkillManipulate(device=device),
            SkillCraft()
        ]
        self.underground = underground
        if underground:
            self.skill_names_for_mine = ['iron_ore', 'diamond', 'cobblestone', 'cobblestone_nearby']
            self.skill_mine = SkillMine(device=device)
        #print(self.skill_info)

    def execute(self, skill_name, skill_info, env, next_skill_type=2):
        # mine ores in iron-based tasks
        if self.underground and (skill_name in self.skill_names_for_mine):
            target = skill_name
            if skill_name.endswith('nearby'):
                target = skill_name[:-7]
            return self.skill_mine.execute(target=target, env=env)

        skill_type=skill_info['skill_type']
        equip=skill_info['equip']
        inventory = env.obs['inventory']['name'].tolist()
        # equip tools
        TOOL_LEVELS = ['wooden', 'stone', 'iron', 'golden', 'diamond']
        TOOL_LEVELS.reverse()
        TOOL_NAMES = ['_pickaxe', '_sword', '_axe']
        for e in equip:
            # if e is a tool, equip the most advanced one in inventory
            tool_level, tool_name = None, None
            for tn in TOOL_NAMES:
                if e.endswith(tn):
                    tool_name = tn
                    tool_level = e[:-len(tool_name)]
                    break
            if tool_name is not None:
                possess_tool_name = None
                for l in TOOL_LEVELS:
                    e_ = l+tool_name
                    if e_.replace('_', ' ') in inventory:
                        possess_tool_name = e_ 
                        break
                if possess_tool_name is not None:
                    idx = inventory.index(possess_tool_name.replace('_',' '))
                else:
                    idx = inventory.index(e.replace('_',' '))
                print('Plan to equip {}, now equip {}'.format(e, possess_tool_name))
            else:
                idx = inventory.index(e.replace('_',' '))
            act = env.base_env.action_space.no_op()
            act[5] = 5
            act[7] = idx
            obs, r, done, _ = env.step(act)
            if done:
                return False, bool(r), done # skill done, task success, task done
        '''
        # for manipulation skills with nothing to equip
        if len(equip)==0 and skill_type==1:
            idx = inventory.tolist().index('air')
            act = env.base_env.action_space.no_op()
            act[5] = 5
            act[7] = idx
            obs, r, done, _ = env.step(act)
            if done:
                return False, bool(r), done
        '''

        # execute skill
        if skill_type==0:
            assert skill_name.endswith('_nearby')
            skill_done, task_success, task_done =  self.skill_models[0].execute(target=skill_name[:-7], env=env, **self.skill_info['find'])
            if skill_done or task_done:
                return skill_done, task_success, task_done
            else:
                return self.random_walk(env)
        elif skill_type==1:
            if not (skill_name in self.skill_info):
                print('Warning: skill {} is not in load_skills.yaml'.format(skill_name))
                return False, False, False
            skill_done, task_success, task_done =  self.skill_models[1].execute(target=skill_name, env=env, equip_list=equip, **self.skill_info[skill_name])
            if skill_done or task_done:
                return skill_done, task_success, task_done
            else:
                return self.random_walk(env)
        elif skill_type==2:
            return self.skill_models[2].execute(target=skill_name, env=env, next_skill_type=next_skill_type)
        else:
            raise Exception('Illegal skill_type.')


    def random_walk(self, env, times=10):
        flag=True
        for i in range(times):
            act = env.base_env.action_space.no_op()
            if flag:
                act[4] = np.random.randint(0,13)*2
                flag=False
            act[0] = 1
            act[2] = 1
            obs, r, done, _ = env.step(act)
            if done:
                return False, bool(r), done # skill done, task success, task done
        return False, False, False