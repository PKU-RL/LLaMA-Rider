import os
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
import utils
from utils import n_2_w
import torch
import json
import requests

class LLMPlanner:
	def __init__(self, model_name='llama-2-70b-chat', adapters_name="nope", root_dir="/ckpts", use_skill_list=False, use_flash_attn=True):
		self.total_history_input_output = []

		self.model_name = model_name
		self.model_path = os.path.join(root_dir, "llama_huggingface", model_name)
		self.adapters_name = adapters_name

		if use_flash_attn:
			print("Adopt flash attention to accelerate")
			from llama2_flash_attn_patch import replace_llama_attn_with_flash_attn
			replace_llama_attn_with_flash_attn()
		print(f"Starting to load the model {model_name} into memory")
		
		self.model = LlamaForCausalLM.from_pretrained(
			self.model_path,
			#load_in_4bit=True,
			torch_dtype=torch.bfloat16,
			device_map="auto",
			quantization_config=BitsAndBytesConfig(
				#load_in_4bit=True,
				#bnb_4bit_compute_dtype=torch.bfloat16,
				bnb_4bit_use_double_quant=True,
				bnb_4bit_quant_type='nf4'
			),
		)

		if adapters_name!="nope":
			print(f"Loading adapters from {adapters_name}")
			adapters_path = adapters_name
			self.model = PeftModel.from_pretrained(self.model, adapters_path)
			self.model = self.model.merge_and_unload()
		self.model.eval()
		self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
		self.tokenizer.bos_token_id = 1
		print(f"Successfully loaded the model {model_name} into memory")

		self.temperature = 0.7
		self.top_p = 0.3
		self.max_new_tokens = 32
		self.min_new_tokens = 2
		self.repetition_penalty = 1.2

		self.skills = utils.get_yaml_data('skills/skills.yaml')
		self.skill_names = list(self.skills.keys())
		self.item_names = self.get_all_items()

		with open("skills/skill_description_dict.json", "r") as f:
			self.skill_description_dict = json.load(f)
		self.all_skills = list(self.skill_description_dict.values())
		if not use_skill_list:
			self.all_skill_embeddings = self.get_skill_embeddings()
		self.des_to_skill = {v: k for k, v in self.skill_description_dict.items()}

		self.use_skill_list = use_skill_list
		
	def get_sub_tasks(self, task_conf):
		# sub task: meet each requirement to achieve target object
		# task & number
		sub_tasks = {}
		target = task_conf['target_name']
		if self.skills[target]['require']:
			for item in self.skills[target]['require']:
				sub_tasks[item] = self.skills[target]['require'][item]
		if self.skills[target]['consume']:
			for item in self.skills[target]['consume']:
				sub_tasks[item] = self.skills[target]['consume'][item]
		if 'furnace' in sub_tasks:
			if 'cobblestone' not in sub_tasks:
				sub_tasks['cobblestone'] = 8
			if 'crafting_table_nearby' not in sub_tasks:
				sub_tasks['crafting_table_nearby'] = 1
		if 'crafting_table' in sub_tasks:
			if 'planks' not in sub_tasks:
				sub_tasks['planks'] = 4
		if 'cobblestone' in sub_tasks:
			sub_tasks['wooden_pickaxe'] = 1
		if 'iron_ore' in sub_tasks:
			sub_tasks['stone_pickaxe'] = 1
		# exclude initial items
		if 'initial_inventory' in task_conf:
			for item in task_conf['initial_inventory']:
				if item in sub_tasks:
					sub_tasks[item] -= task_conf['initial_inventory'][item]
					if sub_tasks[item] <= 0:
						del sub_tasks[item]
		self.sub_tasks = sub_tasks
		# maintain reqirements achieved
		if 'initial_inventory' in task_conf:
			self.requirements_achieved = {}
			for item in task_conf['initial_inventory']:
				self.requirements_achieved[item] = task_conf['initial_inventory'][item]
		else:
			self.requirements_achieved = {}
		return sub_tasks
		
	def check_sub_tasks(self, init_items):
		if self.subtask_str:
			current_subtask = self.subtask_str.split("Your current subtask: ")[-1].strip()
			current_subtask = self.des_to_skill[current_subtask]
		else:
			current_subtask = ''
		# check whether some sub tasks are satisfied
		sub_completion_flag = False
		current_completion_flag = False
		for item in self.sub_tasks:
			if item in init_items:
				if item not in self.requirements_achieved:
					self.requirements_achieved[item] = init_items[item]
					self.sub_tasks[item] -= init_items[item]
					if item == current_subtask:
						current_completion_flag = True
					sub_completion_flag = True
				else:
					if init_items[item] > self.requirements_achieved[item]:
						self.sub_tasks[item] -= (init_items[item] - self.requirements_achieved[item])
						self.requirements_achieved[item] = init_items[item]
						sub_completion_flag = True
						if item == current_subtask:
							current_completion_flag = True
		for item in list(self.sub_tasks.keys()):
			if self.sub_tasks[item] <= 0:
				del self.sub_tasks[item]
		return sub_completion_flag, current_completion_flag

	def skill_retrieval(self, raw_skill, init_items=None):
		raw_skill = re.sub('\d', '', raw_skill)
		if raw_skill.split(' ')[0] == 'craft' and raw_skill not in self.all_skills:
			raw_skill = raw_skill.replace('craft', 'get')
		if 'wooden planks' in raw_skill:
			raw_skill = raw_skill.replace('wooden planks', 'planks')
		if 'wooden plank' in raw_skill:
			raw_skill = raw_skill.replace('wooden plank', 'planks')

		# compare token embeddings of raw_skill and self.all_skills to find the most similar skill
		encoded_raw = self.tokenizer(raw_skill, return_tensors='pt')
		with torch.no_grad():
			raw_embedding = self.model(**encoded_raw, output_hidden_states=True)
		raw_embedding = raw_embedding[2][-1]
		# shape: (1, num_tokens, embedding_dim)
		raw_embedding = torch.mean(raw_embedding, dim=1)

		# find skills with the same noun
		raw_skill_noun = raw_skill.split(' ')[1:]
		for i, word in enumerate(raw_skill_noun):
			if word == 'wood' or word == 'woods':
				raw_skill_noun[i] = 'log'
			if word == 'tree' or word == 'trees':
				raw_skill_noun[i] = 'planks'
		raw_skill_noun = ' '.join(raw_skill_noun)
		skills_embedding = []
		candidate_skills = []
		for i, skill in enumerate(self.all_skills):
			skill_noun = list(self.skill_description_dict.keys())[i]
			skill_noun = skill_noun.replace('_', ' ').strip().lower()
			skill_noun = skill_noun.split(' ')
			if skill_noun[-1] == 'nearby':
				skill_noun = skill_noun[:-1]
			if raw_skill_noun in skill_noun:
				candidate_skills.append(skill)
				skills_embedding.append(self.all_skill_embeddings[i])
			for noun in skill_noun:
				if noun.endswith('s'):
					noun = noun[:-1]
				if re.search(noun, raw_skill_noun):
					candidate_skills.append(skill)
					skills_embedding.append(self.all_skill_embeddings[i])
		if len(candidate_skills) == 1:
			return candidate_skills[0]
		if candidate_skills:
			skills_embedding = torch.stack(skills_embedding, dim=0).cuda()
		else:
			skills_embedding = self.all_skill_embeddings
			candidate_skills = self.all_skills
		#skills_embedding = self.all_skill_embeddings
		similarity = torch.cosine_similarity(raw_embedding, skills_embedding, dim=-1)
		most_similar = torch.argmax(similarity)
		retrieved_skill = candidate_skills[most_similar]
		if retrieved_skill == 'harvest log' and 'log_nearby' not in init_items:
			retrieved_skill = 'find log nearby'
		if retrieved_skill == 'harvest cobblestone' and 'cobblestone_nearby' not in init_items:
			retrieved_skill = 'find cobblestone nearby'
		return retrieved_skill

	def get_skill_embeddings(self):
		skills = self.all_skills
		embeddings = []
		for skill in skills:
			encoded_input = self.tokenizer(skill, return_tensors='pt')
			with torch.no_grad():
				embedding = self.model(**encoded_input, output_hidden_states=True)
			embedding = embedding[2][-1]
			embedding = torch.mean(embedding, dim=1)[0]
			embeddings.append(embedding)
		embeddings = torch.stack(embeddings, dim=0).cuda()
		return embeddings

	def task_to_condition(self, task, task_conf, target = None):
		if not target:
			target = task_conf['target_name']
		condition = []
		if self.skills[target]['consume']:
			for item in self.skills[target]['consume']:
				condition.append(f"{n_2_w(self.skills[target]['consume'][item])} {item}")
		if self.skills[target]['require']:
			for item in self.skills[target]['require']:
				condition.append(f"{n_2_w(self.skills[target]['require'][item])} {item}")
		return ", ".join(condition)

	def get_all_items(self):
		item_names = set()
		for i, k in enumerate(self.skill_names):
			if self.skills[k]['consume'] is not None:
				for p in list(self.skills[k]['consume'].keys()):
					item_names.add(p)
			if self.skills[k]['require'] is not None:
				for p in list(self.skills[k]['require'].keys()):
					item_names.add(p)
			for p in self.skills[k]['equip']:
				item_names.add(p)
			if self.skills[k]['obtain'] is not None:
				for p in list(self.skills[k]['obtain'].keys()):
					item_names.add(p)
		item_names = list(item_names)
		return item_names

	def model_generate(self, prompt, max_new_tokens = self.max_new_tokens, min_new_tokens = self.min_new_tokens):
		temperature = float(self.temperature)
		if temperature < 1e-2:
			temperature = 1e-2
		top_p = float(self.top_p)
		input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
		input_ids = input_ids.to(self.model.device)
		generate_kwargs = dict(
			input_ids = input_ids,
			temperature=temperature,
			max_new_tokens=max_new_tokens,
			min_new_tokens=min_new_tokens,
			top_p=top_p,
			repetition_penalty=self.repetition_penalty,
			do_sample=temperature > 0.0,
		)
		output = self.model.generate(**generate_kwargs)
		output = self.tokenizer.decode(output[0], skip_special_tokens=True)
		return output

	def ask_brief(self, task, condition, inventory, surrounding, subtask_str, past_skills):
		
		if not subtask_str and task == "get furnace nearby":
			if 'furnace' not in inventory:
				subtask_str = "Your current subtask: craft furnace\n"
		if not subtask_str and task == "place crafting table nearby":
			if 'crafting_table' not in inventory:
				subtask_str = "Your current subtask: craft crafting table\n"
		
		if subtask_str:
			task = subtask_str.split("Your current subtask: ")[-1].strip()
			target = self.des_to_skill[task]
			condition = self.task_to_condition(1,1,target)
		
		ask_prompt = "[INST]Your goal is to complete a task in Minecraft.\n"
		ask_prompt += "Given your current inventory, surroundings and skills you have already executed before, provide the skill you should execute next.\n"
		ask_prompt += "The skill name should be no more than 5 words, in the form of a verb plus a noun.\n"
		ask_prompt += "The verb should be one of the following: harvest, craft, find, get, place, mine.\n"
		
		ask_prompt += "Please provide your output in the following format:\n"
		ask_prompt += "Next skill: skill name\n"
		ask_prompt += "\nNow the information:\n"
		ask_prompt += "\nTask: {task}"
		ask_prompt += "\nInventory: {inventory}"
		ask_prompt += "\nSurroundings: {surrounding}"
		ask_prompt += "\nLast three skills you have just already executed: {past_skills}"
		ask_prompt += "\nRecipe: The requirements to {task} in Minecraft is: {condition}"
		ask_prompt += "\nYour output:\n[/INST]"

		prompt = ask_prompt.format(task=task, condition=condition, inventory=inventory, surrounding=surrounding, past_skills=past_skills)

		output = self.model_generate(prompt)

		print(output)
		output = output.split("[/INST]")[-1].strip()
		output += "\n"
		output = output.lower()
		output = re.findall(r'next skill:(.+?)\n', output)[0].split('(')[0].strip()
		return output, prompt

	def ask_requirement(self, task, condition, inventory, surrounding, subtask_str):
		if not subtask_str and task == "get furnace nearby":
			if 'furnace' not in inventory:
				subtask_str = "Your current subtask: craft furnace\n"
		if not subtask_str and task == "place crafting table nearby":
			if 'crafting_table' not in inventory:
				subtask_str = "Your current subtask: craft crafting table\n"

		if subtask_str:
			task = subtask_str.split("Your current subtask: ")[-1].strip()
			target = self.des_to_skill[task]
			condition = self.task_to_condition(1,1,target)

		ask_prompt = "[INST]Given requirements to achieve a task in Minecraft, answer which requirements are not met yet according to the inventory and surroundings.\n"
		ask_prompt += "Think step by step and object by object. Note that objects end with '_nearby' are required to be in the surroundings while other objects are required to be in the inventory. Here's an example:\n\n"
		
		ask_prompt += "Task: craft furnace\n"
		ask_prompt += "The requirements to craft furnace in Minecraft is: 8.0 cobblestone; 1.0 crafting_table_nearby\n"
		ask_prompt += "Objects and their quantities in the inventory: 2.0 log; 3.0 dirt; 4.0 cobblestone\n"
		ask_prompt += "Objects and their quantities in the surroundings: 1.0 cobblestone_nearby\n"
		ask_prompt += "Which requirements are not met yet?\n\n"

		ask_prompt += "Your output:\n\n"

		ask_prompt += "cobblestone: need 8 in the inventory; already have 4; still require 4\n"
		ask_prompt += "crafting_table_nearby: need 1 in the surroundings; already have none; still require 1\n"
		ask_prompt += "Therefore, these requirements are not met yet: 4 cobblestones; 1 crafting_table_nearby\n\n"

		ask_prompt += "Here's another example:\n\n"

		ask_prompt += "Task: craft furnace\n"
		ask_prompt += "The requirements to craft furnace in Minecraft is: 8.0 cobblestone; 1.0 crafting_table_nearby\n"
		ask_prompt += "Objects and their quantities in the inventory: 2.0 log; 3.0 dirt; 11.0 cobblestone\n"
		ask_prompt += "Objects and their quantities in the surroundings: 1.0 crafting_table_nearby\n"
		ask_prompt += "Which requirements are not met yet?\n\n"

		ask_prompt += "Your output:\n\n"

		ask_prompt += "cobblestone: need 8 in the inventory; already have 11; still require 0\n"
		ask_prompt += "crafting_table_nearby: need 1 in the surroundings; already have 1; still require 0\n"
		ask_prompt += "Therefore, all requirements are met, so one can craft furnace directly.\n\n"

		ask_prompt += "Now is your turn:\n\n"

		ask_prompt += "Task: {task}\n"
		ask_prompt += "The requirements to {task} in Minecraft is: {condition}.\n"
		ask_prompt += "Objects and their quantities in the inventory: {inventory}.\n"
		ask_prompt += "Objects and their quantities in the surroundings: {surrounding}.\n"
		ask_prompt += "Which requirements are not met yet?\n"
		ask_prompt += "Your output:\n[/INST]"

		prompt = ask_prompt.format(task=task, condition=condition, inventory=inventory, surrounding=surrounding)

		output = self.model_generate(prompt, max_new_tokens=256, min_new_tokens=0)

		output = output.split("[/INST]")[-1].strip()

		req_output = output

		ask_prompt = prompt + output
		ask_prompt += "\n[INST]Based on your above analysis, to achieve the task, your next step should be?[/INST]"

		output = self.model_generate(ask_prompt, max_new_tokens=64, min_new_tokens=0)

		output = output.split("[/INST]")[-1].strip()

		ask_prompt = ask_prompt + output
		ask_prompt += "\n[INST]Then please provide a skill name according to the next step.\n"
		ask_prompt += "The skill name should be no more than 5 words, in the form of a verb plus a noun.\n"
		ask_prompt += "The verb should be one of the following: harvest, craft, find, get, place, mine.\n"
		ask_prompt += "Please provide your output in the following format:\n"
		ask_prompt += "Next skill: skill name\n[/INST]"

		output = self.model_generate(ask_prompt, max_new_tokens=32, min_new_tokens=0)

		print(output)
		output = output.split("[/INST]")[-1].strip()
		output += "\n"
		output = re.findall(r'Next skill:(.+?)\n', output)[0].split('(')[0].strip()

		return req_output, output

	def make_plan(self, task, condition="", init_items_str=[], init_items = {}, init_inventory_str=[], init_surrounding_str = [], past_skills = [], first_plan = True):
		if first_plan:
			self.subtask_str = ""
		if past_skills:
			if len(past_skills) > 3:
				past_skills = past_skills[-3:]
			past_skills = [f'{i+1}. {self.skill_description_dict[s]}' for i, s in enumerate(past_skills)]
			past_skills = " ".join(past_skills)
		else:
			past_skills = "None"
		if init_inventory_str:
			init_inventory_str = "; ".join(init_inventory_str)
		else:
			init_inventory_str = "Empty"
		if init_surrounding_str:
			init_surrounding_str = "; ".join(init_surrounding_str)
		else:
			init_surrounding_str = "Nothing"
		if not condition:
			condition = "nothing"

		#requirement_ans, skill_ans = self.ask_requirement(task, condition, init_inventory_str, init_surrounding_str, self.subtask_str)

		skill_ans, simple_prompt = self.ask_brief(task, condition, init_inventory_str, init_surrounding_str, self.subtask_str, past_skills)
		
		initial_prompt_simple = simple_prompt[6:-7]

		if first_plan:
			self.task_i = task
			self.init_items_str_i = str(init_items_str)
			self.init_items_i = init_items
			self.past_skills_i = past_skills
			self.init_inventory_str_i = init_inventory_str
			self.init_surrounding_str_i = init_surrounding_str
			self.initial_prompt_i = initial_prompt_simple
			self.task_condition_i = condition
		self.task = task
		self.init_items_str = str(init_items_str)
		self.init_items = init_items
		self.past_skills = past_skills
		self.init_inventory_str = init_inventory_str
		self.init_surrounding_str = init_surrounding_str
		self.revision_history = []
		self.initial_prompt = initial_prompt_simple
		self.task_condition = condition

		if first_plan:
			self.output_plan_i = skill_ans
		self.output_plan = skill_ans
		self.revision_history.append(skill_ans)

		return [skill_ans]

	def reset_info(self):
		self.task = self.task_i
		self.init_items_str = self.init_items_str_i
		self.past_skills = self.past_skills_i
		self.output_plan = self.output_plan_i
		self.init_items = self.init_items_i
		self.init_inventory_str = self.init_inventory_str_i
		self.init_surrounding_str = self.init_surrounding_str_i
		self.revision_history = [self.output_plan_i]
		self.initial_prompt = self.initial_prompt_i
		self.task_condition = self.task_condition_i
		self.subtask_str = ""

	def skill_satisfied(self, skill, init_items):
		if self.skills[skill]['skill_type'] == 0:
			if skill in init_items:
				return False, f"There's no need to find {skill}."
		if self.skills[skill]['consume'] is not None:
			for p in list(self.skills[skill]['consume'].keys()):
				if p not in init_items or init_items[p] < self.skills[skill]['consume'][p]:
					consume_number = self.skills[skill]['consume'][p]
					consume_number = n_2_w(consume_number)
					reason = f"{self.skill_description_dict[skill]} need to consume {consume_number} {p} but not enough now."
					if skill == 'crafting_table_nearby' or skill == 'furnace_nearby':
						reason += f"\nPlease craft {p} first."
					else:
						reason += f" You should get enough {p} to {self.skill_description_dict[skill]}."
					return False, reason
		if self.skills[skill]['require'] is not None:
			for p in list(self.skills[skill]['require'].keys()):
				if p not in init_items or init_items[p] < self.skills[skill]['require'][p]:
					require_number = self.skills[skill]['require'][p]
					require_number = n_2_w(require_number)
					reason = f"{self.skill_description_dict[skill]} requires {require_number} {p} but not enough now."
					reason += f" You should get enough {p} to {self.skill_description_dict[skill]}."
					return False, reason
		return True, None

	def convert_skill(self, raw_skill, init_items):
		raw_skill = raw_skill.replace('_', ' ').strip().lower()
		if not self.use_skill_list:
			if raw_skill not in self.all_skills:
				retrieved_skill = self.skill_retrieval(raw_skill, init_items)
				print("retrieved_skill: ", retrieved_skill)
				raw_skill = retrieved_skill
			print(self.sub_tasks, raw_skill)
			if self.des_to_skill[raw_skill] in self.sub_tasks:
				if raw_skill == 'place crafting table nearby' or raw_skill == 'place furnace nearby':
					pass
				else:
					self.subtask_str = f"Your current subtask: {raw_skill}\n" 
		satisfy, reason = self.skill_satisfied(self.des_to_skill[raw_skill], init_items)
		skill_action = raw_skill.split(' ')[0]
		skill_action = skill_action.lower()
		if satisfy:
			return self.des_to_skill[raw_skill], skill_action, None
		else:
			return "error", skill_action, self.get_feedback(raw_skill, mode='skill inventory unsatisfied', reason=reason)

	def get_feedback(self, raw_skill, mode=None, reason=''):
		if "There's no need" in reason:
			feedback_prompt = "There's no need to {}."
			feedback_prompt = feedback_prompt.format(raw_skill)
		else:
			feedback_prompt = "Your inventory or surroundings does not meet the requirements to perform the skill '{}'.\nSpeculated reason: {}"
			feedback_prompt = feedback_prompt.format(raw_skill, reason)
		return self.feedback_and_replan(feedback_prompt, raw_skill)

	def parse_first_skill(self, skill_output):
		skill_output += '\n'
		raw_skills = re.findall(r'\d+\. (.+?)\n', skill_output)
		if not raw_skills:
			return skill_output
		skill = re.sub(r'\d+\.', '', raw_skills[0])
		skill = skill.split('(')[0]
		skill = skill.strip()
		return skill

	def feedback_and_replan(self, feedback, raw_skill=None):
		self.revision_str = "\n".join(self.revision_history)

		prompt = self.initial_prompt
		if self.revision_history[-1].startswith("Revised skill: "):
			latest_plan = self.revision_history[-1].split("Revised skill: ")[-1]
		else:
			latest_plan = self.revision_history[-1]
		prompt += latest_plan
		latest_plan = self.parse_first_skill(latest_plan)
		if self.use_skill_list:
			prompt += f"\nOK, your proposed next skill is:\n{latest_plan}\n"
		else:
			prompt += f"\nOK, according to your output, your next skill is:\n{raw_skill}\n"
		prompt += f"But the skill failed.\n"
		prompt += "Please find out the reason why the skill failed, and make a revision.\n"
		prompt += f"Here's your inventory: {self.init_inventory_str}\n"
		prompt += f"Here's your surroundings: {self.init_surrounding_str}\n"
		prompt += f"Here's the feedback from the environment: {feedback}\n"
		prompt += "Based on these information, please output the next skill you need to do.\n"
		if self.use_skill_list:
			prompt += "Note the skill must be in the available skill list.\n"
		prompt += "Revised skill:\n1. "
		#print("Feedback prompt:")
		#print(prompt)

		output = self.model_generate(prompt)
		output = output.split("Revised skill:")[-1].strip()
		print("Revised output: ")
		print(output)
		print("End of revision")

		final_feedback, revision_satisfied, revised_skill, revised_raw = self.process_feedback(prompt, raw_skill, feedback, output)
		return final_feedback, revision_satisfied, revised_skill, revised_raw

	def raw_skill_list_satisfaction(self, raw_skill_list, init_items):
		raw_skill = re.sub(r'\d+\.', '', raw_skill_list[0])
		raw_skill = raw_skill.strip()
		raw_skill = raw_skill.replace('_', ' ').lower()

		if not self.use_skill_list:
			retrieved_skill = self.skill_retrieval(raw_skill, init_items)
			print("retrieved_skill: ", retrieved_skill)
			raw_skill = retrieved_skill

		if raw_skill in self.des_to_skill:
			skill = self.des_to_skill[raw_skill]
			satisfy, reason = self.skill_satisfied(skill, init_items)
			if satisfy:
				return True, skill, raw_skill
		elif raw_skill + ' nearby' in self.des_to_skill:
			skill = self.des_to_skill[raw_skill + ' nearby']
			satisfy, reason = self.skill_satisfied(skill, init_items)
			if satisfy:
				return True, skill, raw_skill
		return False, None, raw_skill

	def process_feedback(self, feedback_prompt, draft_plan, feedback, revision):
		question = self.initial_prompt
		revision += '\n'
		revision = re.findall(r'\d+\. .+?\n', revision)
		if revision:
			revision_satisfied, revised_skill, revised_raw = self.raw_skill_list_satisfaction(revision, self.init_items)
		else:
			revision_satisfied = False
			revised_skill = None
			revised_raw = None
			return {'question': question, 'draft_plan': draft_plan, 'feedback': feedback, 'revised_plan': revision, 'revision_satisfied': revision_satisfied}, revision_satisfied, revised_skill, revised_raw

		revision_solid = []
		for i, r_item in enumerate(revision):
			if r_item.split('.')[0].strip() == str(i+1):
				revision_solid.append(r_item)
			else:
				break
		revision = revision_solid[0].strip()
		self.revision_history.append("Feedback: " + feedback)
		self.revision_history.append("Revised skill: 1. " + revised_raw)
		return {'question': question, 'draft_plan': draft_plan, 'feedback': feedback, 'revised_plan': "1. " + revised_raw, 'revision_satisfied': revision_satisfied}, revision_satisfied, revised_skill, revised_raw

	def initialize_history(self):
		self.total_history_input_output = [{'input': self.initial_prompt_i, 'output': 'Next skill: ' + self.output_plan_i}]

