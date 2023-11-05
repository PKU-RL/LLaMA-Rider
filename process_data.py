import yaml
import os
import random
import numpy as np
import json
from tqdm import tqdm
import re

def get_yaml_data(yaml_file):
	file = open(yaml_file, 'r', encoding="utf-8")
	file_data = file.read()
	file.close()
	
	#print(file_data)
	data = yaml.load(file_data, Loader=yaml.FullLoader)
	#print(data)
	return data

skills = get_yaml_data('skills/skills.yaml')

with open("skills/skill_description_dict.json", "r") as f:
	skill_description_dict = json.load(f)
des_to_skill = {v: k for k, v in skill_description_dict.items()}

def task_to_condition(task_conf, target = None):
	if not target:
		target = task_conf['target_name']
	condition = []
	if skills[target]['consume']:
		for item in skills[target]['consume']:
			condition.append(f"{skills[target]['consume'][item]} {item}")
	if skills[target]['require']:
		for item in skills[target]['require']:
			condition.append(f"{skills[target]['require'][item]} {item}")
	return ", ".join(condition)

def traj2prompt(trajectory, task, condition):
	skill_prompt = "[INST]Your goal is to complete a task in Minecraft.\n"
	skill_prompt += "Given your current inventory, surroundings and skills you have already executed before, provide the skill you should execute next."
	skill_prompt += "\nNow the information:\n"
	skill_prompt += "\nTask: {task}"
	skill_prompt += "\nInventory: {inventory}"
	skill_prompt += "\nSurroundings: {surrounding}"
	skill_prompt += "\nLast three skills you have just already executed: {past_skills}"
	#skill_prompt += "\n{subtask}"
	skill_prompt += "\nRecipe: The requirements to {task} in Minecraft is: {condition}"
	skill_prompt += "\nYour output:\n[/INST]"
	
	skill_data = []
	past_skills = []
	for step in trajectory:
		if step['inventory']:
			inventory_str = "; ".join(step['inventory'])
		else:
			inventory_str = "Empty"
		if step['surrounding']:
			surrounding_str = "; ".join(step['surrounding'])
		else:
			surrounding_str = "Nothing"

		if past_skills:
			if len(past_skills) > 3:
				past_skills = past_skills[-3:]
			past_skills_input = [f'{i+1}. {s}' for i, s in enumerate(past_skills)]
			past_skills_input = " ".join(past_skills_input)
		else:
			past_skills_input = "None"


		skill_input = skill_prompt.format(task=task, inventory=inventory_str, surrounding=surrounding_str, past_skills=past_skills_input, condition=condition)
		skill_output = f"Next skill: {step['skill']}\n"
		skill_data.append({'input': skill_input, 'output': skill_output})
		past_skills.append(step['skill'])

		if step['sub_task']:
			subtask = step['sub_task'].split("Your current subtask: ")[-1].strip()
			sub_target = des_to_skill[subtask]
			sub_condition = task_to_condition(1, sub_target)
			sub_skill_input = skill_prompt.format(task=subtask, inventory=inventory_str, surrounding=surrounding_str, past_skills=past_skills_input, condition=sub_condition)
			sub_skill_output = f"Next skill: {step['skill']}\n"
			skill_data.append({'input': sub_skill_input, 'output': sub_skill_output})

	return skill_data

if __name__ == "__main__":
	skill_dataset = []
	save_dir_list = ['results_log', 'results_stone', 'results_stone', 'results_mob']
	tasks_list = get_yaml_data('envs/hard_task_conf.yaml')
	for i, save_dir in enumerate(save_dir_list):
		for task in tasks_list:
			dir_name = os.path.join(save_dir, task)
			if not os.path.exists(os.path.join(dir_name, 'success_traj.jsonl')):
				continue
			print("processing task: " + task)
			
			task_condition = task_to_condition(tasks_list[task])
			
			if re.findall("_with_", task):
				task_name = re.split("_with_", task)[0].strip()
			else:
				task_name = task
			task_name = re.sub("_", " ", task_name)
			print("task name convertion: " + task_name)
			with open(os.path.join(dir_name, "success_traj.jsonl"), "r") as f:
				for line in tqdm(f):
					trajectory = json.loads(line)
					skill_data = traj2prompt(trajectory, task_name, task_condition)
					skill_dataset += skill_data

			with open(os.path.join(dir_name, "sub_success_traj.jsonl"), "r") as f:
				for line in tqdm(f):
					trajectory = json.loads(line)
					skill_data = traj2prompt(trajectory, task_name, task_condition)
					skill_dataset += skill_data

	# deduplicate the dataset
	skill_dataset = [dict(t) for t in set([tuple(d.items()) for d in skill_dataset])]
	
	random.shuffle(skill_dataset)

	# size of all datasets
	print("Size of skill dataset: " + str(len(skill_dataset)))
	
	# save all datasets
	root_dir = "dataset"
	with open(os.path.join(root_dir, "skill_dataset.jsonl"), "w") as f:
		for data in skill_dataset:
			f.write(json.dumps(data) + "\n")

