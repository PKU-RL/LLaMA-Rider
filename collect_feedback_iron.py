import numpy as np
import torch
import random
import os, sys
import imageio
import utils
from utils import n_2_w
import argparse
from mineclip_official import build_pretrain_model
from envs.minecraft_hard_task import MinecraftHardHarvestEnv
from skills import skills, skill_search, SkillsModel, convert_state_to_init_items, LLMPlanner
from minedojo.sim import InventoryItem
import matplotlib.pyplot as plt
import sys
import json
import pickle
from copy import deepcopy

def main(args, task, task_conf, planner):

    # save path
    save_dir = args.save_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, task)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Inference device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print('Running on device: ', device)

    # seed control
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load clip model
    clip_config = utils.get_yaml_data(args.clip_config_path)
    model_clip = build_pretrain_model(
        image_config = clip_config['image_config'],
        text_config = clip_config['text_config'],
        temporal_config = clip_config['temporal_config'],
        adapter_config = clip_config['adaptor_config'],
        state_dict = torch.load(args.clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model loaded from:', args.clip_model_path)

    init_items = {}
    init_items_str = []
    task_conf_dict = deepcopy(task_conf)
    if 'initial_inventory' in task_conf:
        init_items = task_conf['initial_inventory']
        init_items_str = [f"{n_2_w(v)} {k}" for k,v in init_items.items() if v > 0]
        init_inv = [InventoryItem(slot=i, name=k, variant=None, quantity=task_conf['initial_inventory'][k]) 
        for i,k in enumerate(list(task_conf['initial_inventory'].keys()))]
        task_conf['initial_inventory'] = init_inv
    init_inventory = init_items
    init_inventory_str = init_items_str
    init_surrounding = {}
    init_surrounding_str = []

    # ablation for max steps
    if args.shorter_episode:
        task_conf['max_steps'] = task_conf['max_steps']//2
    #task_conf['max_steps'] = task_conf['max_steps']*2

    print('task configs', task_conf)
    # Instantiate environment
    env = MinecraftHardHarvestEnv(
        image_size=(160,256),
        seed=seed,
        clip_model=model_clip,
        device=device,
        save_rgb=args.save_gif,
        **task_conf
        )

    # load skills
    skills_model = SkillsModel(device=device, path=args.skills_model_config_path, underground=True)

    # init LLM
    #planner = LLMPlanner()

    # run test
    target_name = task_conf['target_name']

    task_name = task.split('_with_')[0]
    task_name = task_name.replace('_', ' ')

    task_condition = planner.task_to_condition(task, task_conf)
    skill_sequence = planner.make_plan(task_name, task_condition, init_items_str, init_items, init_inventory_str, init_surrounding_str)
    if not skill_sequence:
        skill_sequence = [task_name]

    skill_success_cnt = np.zeros(len(skill_sequence))
    print('Initial skill sequence: {}, length: {}'.format(skill_sequence, len(skill_sequence)))
    test_success_rate = 0

    all_feedback = []
    success_traj = []
    sub_success_tarj = []
    for ep in range(args.test_episode):
        print('Execute task: {}, episode: {}'.format(task, ep))
        
        env.reset()
        episode_snapshots = [('begin', np.transpose(env.obs['rgb'], [1,2,0]).astype(np.uint8))]
        past_skills = []
        traj_plan = []

        skill_next = skill_sequence[0]
        init_items_next = init_items
        init_items_next_str = init_items_str
        init_inventory_next = init_inventory
        init_inventory_next_str = init_inventory_str
        init_surrounding_next = init_surrounding
        init_surrounding_next_str = init_surrounding_str
        planner.reset_info()
        planner.get_sub_tasks(task_conf_dict)
        
        next_skill_type = 2
        
        while True:
            step_info = {'inventory': init_inventory_next_str, 'surrounding': init_surrounding_next_str, 'skill': skill_next, 'sub_task': planner.subtask_str}
            skill_correct = False
            for revise_turn in range(args.fr_turn):
                skill_next, skill_action, feedback_tuple = planner.convert_skill(skill_next, init_items_next)
                if feedback_tuple:
                    feedback, revision_satisfied, revised_skill, revised_raw = feedback_tuple
                    if revision_satisfied:
                        all_feedback.append(feedback)
                        print('revised skill:', revised_skill)
                        skill_next = revised_skill
                        step_info['skill'] = revised_raw
                        traj_plan.append(step_info)
                        skill_correct = True
                        break
                    else:
                        if revised_raw:
                            skill_next = revised_raw
                            continue
                        else:
                            break
                else:
                    traj_plan.append(step_info)
                    skill_correct = True
                    break
            if skill_correct == False:
                task_success = False
                task_done = False
                break
            print('executing skill:',skill_next)
            past_skills.append(skill_next)
            skill_done, task_success, task_done = skills_model.execute(skill_name=skill_next, skill_info=skills[skill_next], env=env, next_skill_type=next_skill_type)
            if skill_done or task_success:
                episode_snapshots.append((skill_next, np.transpose(env.obs['rgb'], [1,2,0]).astype(np.uint8)))

            if task_done:
                break
            init_items_next, init_inventory_next, init_surrounding_next = convert_state_to_init_items(init_items_next, skill_next, skills[skill_next]['skill_type'],
                skill_done, env.obs['inventory']['name'], env.obs['inventory']['quantity'], env.obs['nearby_tools'])
            sub_task_completion, cur_task_completion = planner.check_sub_tasks(init_items_next)
            if sub_task_completion:
                sub_success_tarj.append(deepcopy(traj_plan))
            if cur_task_completion:
                planner.subtask_str = ""
            init_items_next_str = [f"{n_2_w(v)} {k}" for k,v in init_items_next.items() if v > 0]
            init_inventory_next_str = [f"{n_2_w(v)} {k}" for k,v in init_inventory_next.items() if v > 0]
            init_surrounding_next_str = [f"{n_2_w(v)} {k}" for k,v in init_surrounding_next.items() if v > 0]
            skill_sequence_next = planner.make_plan(task_name, task_condition, init_items_next_str, init_items_next, init_inventory_next_str, init_surrounding_next_str, past_skills, first_plan=False)
            if not skill_sequence_next:
                skill_sequence_next = [task_name]
            skill_next = skill_sequence_next[0]

            next_skill_type = 2
            
            print('recomputed skill sequence:', skill_sequence_next)
        print('task done {}'.format(task_done))
        
        if task_success:
            test_success_rate += 1
            success_traj.append(traj_plan)

        # save gif
        if args.save_gif:
            imageio.mimsave(os.path.join(save_dir,'episode{}_success{}.gif'.format(ep,int(task_success))), env.rgb_list, duration=0.1)
            with open(os.path.join(save_dir,'episode{}_skills.json'.format(ep)), 'w') as f:
                json.dump(traj_plan, f)

        save_dir_snapshots = os.path.join(save_dir, 'episode{}_success{}'.format(ep,int(task_success)))
        if not os.path.exists(save_dir_snapshots):
            os.mkdir(save_dir_snapshots)
        for i, (sk, im) in enumerate(episode_snapshots):
            imageio.imsave(os.path.join(save_dir_snapshots, '{}_{}.png'.format(i,sk)), im)

        if ep%100==0 and ep!=0:
            env.remake_env()

        print()

    # save feedback
    if args.save_feedback:
        with open(os.path.join(save_dir, 'feedback.jsonl'), 'w') as f:
            for feedback in all_feedback:
                f.write(json.dumps(feedback)+'\n')
    
    # save success trajectory
    if args.save_success_tarj:
        with open(os.path.join(save_dir, 'success_traj.jsonl'), 'w') as f:
            for traj in success_traj:
                f.write(json.dumps(traj)+'\n')
        with open(os.path.join(save_dir, 'sub_success_traj.jsonl'), 'w') as f:
            for traj in sub_success_tarj:
                f.write(json.dumps(traj)+'\n')
    
    test_success_rate /= float(args.test_episode)
    print('success rate:', test_success_rate)

    # save success rate
    with open(os.path.join(save_dir, 'success_rate.jsonl'), 'w') as f:
        success_info = {'success_rate': test_success_rate}
        f.write(json.dumps(success_info))

    return test_success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shorter-episode', type=int, default=0) # ablation for using 1/2 episode steps?
    parser.add_argument('--test-episode', type=int, default=30) # number of test episodes per task
    parser.add_argument('--seed', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--save-gif', type=int, default=0) # save whole gifs?
    parser.add_argument('--save-feedback', type=int, default=1) # save feedback
    parser.add_argument('--save-success-tarj', type=int, default=1) # save success trajectory
    parser.add_argument('--save-path', type=str, default='results')
    parser.add_argument('--clip-config-path', type=str, default='mineclip_official/config.yml')
    parser.add_argument('--clip-model-path', type=str, default='mineclip_official/attn.pth')
    parser.add_argument('--task-config-path', type=str, default='envs/hard_task_conf.yaml')
    parser.add_argument('--skills-model-config-path', type=str, default='skills/load_skills.yaml')
    parser.add_argument('--fr-turn', type=int, default=5) # feedback-revision turns
    parser.add_argument('--task-range-st', type=int, default=30)
    parser.add_argument('--task-range-ed', type=int, default=40)
    parser.add_argument('--adapter', type=str, default="nope")
    args = parser.parse_args()

    # load task configs
    tasks = utils.get_yaml_data(args.task_config_path)#[args.task]
    success_rates = {}
    planner = LLMPlanner(adapters_name = args.adapter)

    test_list = []
    for task in tasks:
        test_list.append(task)
    test_list = test_list[args.task_range_st:args.task_range_ed]
    print(test_list)
    
    for task in test_list:
        success_rate = main(args, task, tasks[task], planner)
        success_rates[task] = success_rate
    print(success_rates)