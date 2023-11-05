import numpy as np
from minedojo.sim.mc_meta.mc import ALL_CRAFT_SMELT_ITEMS

class SkillCraft:
    def __init__(self):
        self.item2id = {}
        for i, n in enumerate(ALL_CRAFT_SMELT_ITEMS):
            self.item2id[n] = i
        #print(self.item2id)

    def execute(self, target, env, next_skill_type=2):
        act = env.base_env.action_space.no_op()
        if not (target in self.item2id):
            print('Warning: target {} is not in the crafting list'.format(target))
            return False, False, False # skill done, task success, task done

        target_id = self.item2id[target]
        act[5] = 4
        act[6] = target_id
        obs, r, done, _ = env.step(act)

        if env.reward_harvest(obs, target):
            skill_success = True
        else:
            skill_success = False

        if done or (skill_success and (next_skill_type==2)):
            return skill_success, bool(r), done # skill done, task success, task done
        # if crafting fails or the next skill is not craft, recycle the table/furnace
        else:
            return self.recycle(env, skill_success)

    # recycle the crafting table or furnace
    def recycle(self, env, skill_done, times=20):
        if not (int(env.obs['nearby_tools']['furnace']) or int(env.obs['nearby_tools']['table'])):
            return skill_done, False, False # skill done, task success, task done

        print('Recycle table/furnace')

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
                    return skill_done, bool(r), done # skill done, task success, task done
                break

        # recycle the table or furnace
        for i in range(times):
            act = env.base_env.action_space.no_op()
            act[5] = 3
            obs, r, done, _ = env.step(act)
            if done:
                return skill_done, bool(r), done # skill done, task success, task done

        act = env.base_env.action_space.no_op()
        act[4] = 14
        obs, r, done, _ = env.step(act)
        return skill_done, bool(r), done # skill done, task success, task done


if __name__=='__main__':
    s = SkillCraft()
    print(s.item2id)