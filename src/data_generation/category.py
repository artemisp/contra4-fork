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
parser.add_argument('--strategy',type=str, default='similarity', help='sampling strategy: random or high_sim')
parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='model id')
parser.add_argument('--split', type=str, default='test', help='split')
args = parser.parse_args()
STRATEGY = args.strategy
BS = args.bs
SPLIT = args.split

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_id = args.model_id

prompt = """<s>You are tasked with categorizing a question that compares or evaluates inputs based on a specific property (e.g., which input is more positive, has more action, etc.).

Example Questions and Outputs:
Question: "Which input is more positive in tone?"
Category: Sentiment Analysis
Reasoning: The question explicitly asks about emotional tone, a sentiment-related property.

Question: "Which video has more action?"
Category: Activity Level
Reasoning: The question focuses on the level of dynamism or activity in the input videos.

Question: "Which object is larger?"
Category: Size Comparison
Reasoning: The question compares a specific property, size, between inputs.

Question: "Which scene is more likely to involve human presence?"
Category: Human Presence
Reasoning: The question asks about the likelihood of human presence

Question: "Which scene involves more unpredictable or sudden changes?"
Category: Dynamic Changes
Reasoning: The question asks about the level of unpredictability or sudden changes in the scene.

Question: {}
Category:"""



data = json.load(open(f'./data/filters_{SPLIT}/unanimous_permute_{STRATEGY}_{SPLIT}_balanced.json'))
formatted_inputs = {
    "text": [prompt.format(d["questions"][0]) for d in data],
    "index": list(range(len(data)))
}

os.makedirs(f'./temps_{STRATEGY}_{SPLIT}', exist_ok=True)
pickle.dump(formatted_inputs, open(os.path.join(f'./temps_{STRATEGY}_{SPLIT}', 'engine_input.pkl'), 'wb'))

engine_path = 'python src/vllm_engine.py'
command = [
    engine_path, 
    '--model_name', args.model_id,
    '--half', 'True',
    '--generation_kwargs', f"'{json.dumps({'temperature': 0.3, 'top_p': 0.9, 'top_k': 0, 'max_new_tokens': 30, 'num_return_sequences':1, 'num_beams': 1})}'", 
    '--seed', 42, 
    '--batch_size', 32, 
    '--data_path', os.path.join(f'./temps_{STRATEGY}_{SPLIT}', 'engine_input.pkl'), 
    '--output_dir', f'./temps_{STRATEGY}_{SPLIT}'
]
os.system(" ".join([str(c) for c in command]))
outputs = []
for out_f in os.listdir(os.path.join(f'./temps_{STRATEGY}_{SPLIT}', 'engine_output')):
    outputs.extend([json.loads(l.strip()) for l in open(os.path.join(f'./temps_{STRATEGY}_{SPLIT}', 'engine_output', out_f)).readlines()])
os.system('rm -r ' + os.path.join(f'./temps_{STRATEGY}_{SPLIT}', 'engine_output'))
index2output = {o['index']: o['generated_text'][0].split('\n')[0].strip() for o in outputs}

for i, example in enumerate(data):
    example["category"] = index2output[i]
os.makedirs(f'data/filters_{SPLIT}/', exist_ok=True)
json.dump(data, open(f'data/filters_{SPLIT}/unanimous_permute_filter_{STRATEGY}_{SPLIT}_balanced.json', "w"))        
