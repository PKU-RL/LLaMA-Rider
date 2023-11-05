import utils
import json

skills = utils.get_yaml_data('skills/skills.yaml')
skill_names = list(skills.keys())
item_names = set()

output_item = {}
for i, k in enumerate(skill_names):
    s = ""
    
    if skills[k]['skill_type'] == 0:
        s += 'find {}'.format(k.replace('_', ' '))
    if skills[k]['skill_type'] == 1:
        if '_nearby' in k:
            s += 'place {}'.format(k.replace('_', ' '))
        else:
            s += 'get {}'.format(k.replace('_', ' '))
    if skills[k]['skill_type'] == 2:
        s += 'craft {}'.format(k.replace('_', ' '))
    output_item[k] = s

with open("skills/skill_description_dict.json", "w") as f:
    json.dump(output_item, f)