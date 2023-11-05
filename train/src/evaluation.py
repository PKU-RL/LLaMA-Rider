import torch
import transformers
import evaluate
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

IGNORE_INDEX = -100

'''
MCQA_PROMPT_DICT = {
    "default": (
        "### Human: Is Minecraft a sandbox game? yes or no? ### Assistant: yes.\n\n"
        "### Human: Is it possible to train and ride dolphins in Minecraft? yes or no? ### Assistant: no.\n\n"
        "### Human: {input} ### Assistant: "
    ),

}

def extract_mcqa_dataset(example):
    prompt_type = "default"
    prompt_format = MCQA_PROMPT_DICT[prompt_type]
    return {'input': prompt_format.format(**example)}
'''

MCQA_PROMPT_DICT = "### Human: {input} You must answer yes or no without other words.\n### Assistant: "

def extract_mcqa_dataset(example):
    example['input'] = MCQA_PROMPT_DICT.format(input=example['input'])
    return example

'''
RECIPE_PROMPT_DICT = {
    "### Human: How many iron_ingot are needed to craft iron_sword?You must answer a number. ### Assistant: 2.\n\n"
    '### Human: {input} ### Assistant: '
}
'''

RECIPE_PROMPT_DICT = "### Human: {input} You must answer an Arabic numeral without other words.\n### Assistant: "
#RECIPE_PROMPT_DICT = "{input} You must answer an Arabic numeral without other words."

def extract_recipe_dataset(example):
    example['input'] = example['input'][:-25]
    example['input'] = RECIPE_PROMPT_DICT.format(input=example['input'])
    return example
    #return {'input': RECIPE_PROMPT_DICT.format(**example)}


def get_eval_results(data_loader, dataset, loss_mc, preds, refs, accuracy, setname="mc"):
    # Extract results by subject.
    results = {f'{setname}_loss':loss_mc/len(data_loader)}
    
    subject = dataset['subject']
    subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
    
    for s, p, r in zip(subject, preds, refs):
        subjects[s]['preds'].append(p)
        subjects[s]['refs'].append(r)
    
    subject_scores = []

    for subject in subjects:
        subject_score = accuracy.compute(
            references=subjects[subject]['refs'],  # list
            predictions=subjects[subject]['preds']  # list
        )['accuracy']
        
        results[f'{setname}_eval_accuracy_{subject}'] = subject_score
        subject_scores.append(subject_score)
    
    results[f'{setname}_eval_accuracy'] = np.mean(subject_scores)
    
    return results   
    
                
def minecraft_eval(args, trainer, tokenizer):

    accuracy = evaluate.load("src/evaluate/metrics/accuracy/accuracy.py")

    mcqa_dataset = load_dataset("json", data_files={"eval": os.path.join(args.root_dir, "evaluation/mcqa_1k.json")})
    #mcqa_dataset = mcqa_dataset.remove_columns('subject')
    mcqa_dataset = mcqa_dataset["eval"]

    if args.use_prompt_eval:
        mcqa_dataset = mcqa_dataset.map(extract_mcqa_dataset)

    yesno_idx = [
        tokenizer("yes", add_special_tokens=False).input_ids[0],
        tokenizer("no", add_special_tokens=False).input_ids[0]
    ]
    print("yesno")
    print(yesno_idx)
    
    recipe_qa_dataset = load_dataset("json", data_files={"eval": os.path.join(args.root_dir, "evaluation/recipe_qa_2k.json")})
    recipe_qa_dataset = recipe_qa_dataset["eval"]
    
    if args.use_prompt_eval:
        recipe_qa_dataset = recipe_qa_dataset.map(extract_recipe_dataset)

    number_idx = [tokenizer(str(n), add_special_tokens=False).input_ids[-1] for n in range(0, 10)]
    
    class MCEvalCallback(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, model, **kwargs):
            
            # MCQA Evaluation
            data_loader = trainer.get_eval_dataloader(mcqa_dataset) 
            source_max_len = trainer.data_collator.source_max_len
            trainer.data_collator.source_max_len = args.mc_source_max_len
            trainer.model.eval()
            preds, refs = [], []
            loss_mc = 0
            
            for batch in tqdm(data_loader, total=len(data_loader)):
                (loss, logits, labels) = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False)
                #raw_input = [trainer.tokenizer.convert_ids_to_tokens(t) for t in batch['input_ids'][0].tolist()]
                # There are two tokens, the output, and eos token.
                for i, logit in enumerate(logits):
                    label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0] # 0~358
                    logit_ans = logit[label_non_zero_id-1]#[yesno_idx]  
                    if torch.argmax(logit_ans).item() ==3869:
                        preds.append(yesno_idx[0])
                    elif torch.argmax(logit_ans).item() == 1939:
                        preds.append(yesno_idx[1])
                    else:
                        preds.append(torch.argmax(logit_ans).item())
                    #print('preds ',preds[-1])
                    #print('preds ', tokenizer._convert_id_to_token(preds[-1]))
                
                labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                #refs += [yesno_idx.index(label) for label in labels.tolist()]
                refs += labels.tolist()
                #print('refs ',refs[-1])
                loss_mc += loss.item()
            
            results = get_eval_results(data_loader, mcqa_dataset, loss_mc, preds, refs, accuracy, setname="mc")
            
            trainer.log(results)
            trainer.data_collator.source_max_len = source_max_len
            
            # Recipe Evaluation
            data_loader = trainer.get_eval_dataloader(recipe_qa_dataset) 
            source_max_len = trainer.data_collator.source_max_len
            trainer.data_collator.source_max_len = args.mc_source_max_len
            trainer.model.eval()
            preds, refs = [], []
            loss_mc = 0
            
            for batch in tqdm(data_loader, total=len(data_loader)):
                (loss, logits, labels) = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False)
                #print("recipe")
                #print(labels)
                for i, logit in enumerate(logits):
                    label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0] # 0~358
                    logit_ans = logit[label_non_zero_id-1+1]#[number_idx]
                    preds.append(torch.argmax(logit_ans).item())
                    #print('preds ',preds[-1])
                
                labels = labels[labels != IGNORE_INDEX].view(-1, 3)[:,1]
                refs += [number_idx.index(label) for label in labels.tolist()]
                refs += labels.tolist()
                #print('refs ', refs[-1])
                loss_mc += loss.item()
            
            results = get_eval_results(data_loader, recipe_qa_dataset, loss_mc, preds, refs, accuracy, setname="recipe")
            
            trainer.log(results)
            trainer.data_collator.source_max_len = source_max_len
            
    trainer.add_callback(MCEvalCallback)

    return trainer
