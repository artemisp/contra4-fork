import json
import os
import re
import torch
from tqdm import tqdm
import argparse
import pickle
import json
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=128)
parser.add_argument('--data_path',type=str, default='../../data/final_data/test.json', help='data path')
parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='model id')
args = parser.parse_args()
BS = args.bs

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_id = args.model_id

data = json.load(open(args.data_path))  

formatted_inputs = {
    "text": [],
    "index": []
}
for d in data:
    num_examples = len(d["examples"])
    options = ", ".join([f"Option {chr(ord('A') + i)}: " + d['examples'][i]['caption'] for i in range(num_examples)])
    question = d['question'] if 'question' in d else d['questions'][0]
    prompt = f"""<s>Answer the question {question} by selecting one of the following options: {options}. Answer: Option"""
    formatted_inputs["text"].append(prompt)
    formatted_inputs["index"].append(d["id"])

os.makedirs(f'caption_baseline', exist_ok=True)
pickle.dump(formatted_inputs, open(os.path.join(f'caption_baseline', 'engine_input.pkl'), 'wb'))

engine_path = 'python src/vllm_engine.py'
command = [
    engine_path, 
    '--model_name', args.model_id,
    '--half', 'True',
    '--generation_kwargs', f"'{json.dumps({'temperature': 0.3, 'top_p': 0.9, 'top_k': 0, 'max_new_tokens': 30, 'num_return_sequences':1, 'num_beams': 1})}'", 
    '--seed', 42, 
    '--batch_size', 32, 
    '--data_path', os.path.join(f'caption_baseline', 'engine_input.pkl'),
    '--output_dir', f'caption_baseline'
]
os.system(" ".join([str(c) for c in command]))
outputs = []
for out_f in os.listdir(os.path.join(f'caption_baseline', 'engine_output')):
    outputs.extend([json.loads(l.strip()) for l in open(os.path.join(f'caption_baseline', 'engine_output', out_f)).readlines()])
index2output = [{"id":o['index'], "pred_answer":o['generated_text'][0].split('\n')[0].strip()} for o in outputs]

json.dump(index2output, open(f'caption_baseline/results.json', "w"))
