import json
import os
from torch import cuda
import transformers
from torch import bfloat16
from tqdm import tqdm
import argparse
from itertools import permutations
from collections import defaultdict
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='llama', help='llama2 or mistral or flan')
parser.add_argument("--strategy", type=str, default='random', help='random or similarity')
parser.add_argument("--split", type=str, default='test', help='val or test')
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()
STRATEGY = args.strategy
MODEL = args.model
SPLIT = args.split
BATCH_SIZE = args.batch_size
print(f"Model: {MODEL}, Strategy: {STRATEGY}, Split: {SPLIT}")

index2letter = lambda x: chr(ord('A') + x)
prompt_from_example = None
if MODEL == 'llama':
    model_id = 'meta-llama/Llama-3.1-8B-Instruct'
    prompt = "<s>Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n Question: {} Choices:{}\nAnswer:"
    def prompt_from_example(example, permutation=None):
        for ex in example["examples"]:
            if 'captions' in ex:
                ex["caption"] = ex["captions"]
            ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
        if permutation:
            captions = [ex['caption'] for ex in [example["examples"][i] for i in permutation]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        return prompt.format(example["questions"][0], options)
elif MODEL == 'mistral':
    model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    prompt = "Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices:{} Answer:"
    def prompt_from_example(example, permutation=None):
        for ex in example["examples"]:
            if 'captions' in ex:
                ex["caption"] = ex["captions"]
            ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
        if permutation:
            captions = [ex['caption'] for ex in [example["examples"][i] for i in permutation]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        chat = [{"role": "user", "content": prompt.format(example["questions"][0], options)}]
        return tokenizer.apply_chat_template(chat, tokenize=False)
# elif MODEL == 'flan':
#     model_id = 'google/flan-t5-xxl'
#     prompt = "Select which of the inputs best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices: {} Answer:"
#     def prompt_from_example(example, permutation=None):
#         for ex in example["examples"]:
#             if 'captions' in ex:
#                 ex["caption"] = ex["captions"]
#             ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
#         if permutation:
#             captions = [ex['caption'] for ex in [example["examples"][i] for i in permutation]]
#         options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
#         return prompt.format(example["questions"][0], options)
elif 'gemma' in MODEL:
    # os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
    model_id = 'google/gemma-2-9b-it'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    prompt = "Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices:{} Answer:"
    def prompt_from_example(example, permutation=None):
        for ex in example["examples"]:
            if 'captions' in ex:
                ex["caption"] = ex["captions"]
            ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
        if permutation:
            captions = [ex['caption'] for ex in [example["examples"][i] for i in permutation]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        chat = [{"role": "user", "content": prompt.format(example["questions"][0], options)}]
        return tokenizer.apply_chat_template(chat, tokenize=False)
elif 'phi' in MODEL:
    # os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

    model_id = 'microsoft/Phi-3-medium-128k-instruct'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    prompt = "Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices:{} Answer:"
    def prompt_from_example(example, permutation=None):
        for ex in example["examples"]:
            if 'captions' in ex:
                ex["caption"] = ex["captions"]
            ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
        if permutation:
            captions = [ex['caption'] for ex in [example["examples"][i] for i in permutation]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        chat = [{"role": "user", "content": prompt.format(example["questions"][0], options)}]
        return tokenizer.apply_chat_template(chat, tokenize=False)
    
    


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

def answer2choice(text):
    text = text.split('.')[0]
    if "A" in text and not 'B' in text and not 'C' in text and not 'D' in text:
        return "Scene A"
    elif "B" in text and not 'A' in text and not 'C' in text and not 'D' in text:
        return "Scene B"
    elif "C" in text and not 'A' in text and not 'B' in text and not 'D' in text:
        return "Scene C"
    elif "D" in text and not 'A' in text and not 'B' in text and not 'C' in text:
        return "Scene D"
    else:
        return text

step4 = json.load(open(f"data/step2_3/{STRATEGY}_{SPLIT}.json"))

formatted_inputs = {'text': [], 'index': []}
example2perm = defaultdict(list)
example_count = 0
for idx in tqdm(range(len(step4))):
# for idx in tqdm(range(len(step4))):
    example = step4[idx]
    example2perm[idx] = []
    all_permutations = list(permutations(range(len(example["examples"]))))
    all_batch = [example]*len(all_permutations)
    inputs = [prompt_from_example(item, perm) for item,perm in zip(all_batch, all_permutations)]
    prev = 0
    for k in range(len(inputs)):
        formatted_inputs['text'].append(inputs[k])
        formatted_inputs['index'].append(example_count)
        example2perm[idx].append(example_count)
        example_count += 1
        
os.makedirs('./temps', exist_ok=True)
engine_temps_path = f'./temps_step4_{MODEL}_{STRATEGY}_{SPLIT}'
os.makedirs(engine_temps_path, exist_ok=True)
pickle.dump(formatted_inputs, open(os.path.join(engine_temps_path, 'engine_input.pkl'), 'wb'))
engine_path = 'python src/vllm_engine.py'
command = [
    engine_path, 
    '--model_name', model_id,
    '--half', 'True',
    '--generation_kwargs', f"'{json.dumps({'temperature': 0.3, 'top_p': 0.9, 'top_k': 0, 'max_new_tokens': 30, 'num_return_sequences':1, 'num_beams': 1})}'", 
    '--seed', 42, 
    '--batch_size', BATCH_SIZE, 
    '--data_path', os.path.join(engine_temps_path, 'engine_input.pkl'), 
    '--output_dir', engine_temps_path
]
os.system(" ".join([str(c) for c in command]))
outputs = []
for out_f in os.listdir(os.path.join(engine_temps_path, 'engine_output')):
    outputs.extend([json.loads(l.strip()) for l in open(os.path.join(engine_temps_path, 'engine_output', out_f)).readlines()])
os.system('rm -r ' + os.path.join(engine_temps_path, 'engine_output'))
index2output = {o['index']: o['generated_text'][0].split('\n')[0].strip() for o in outputs}
for k in index2output:
    if '</s>' in index2output[k]:
        index2output[k] = index2output[k].split('</s>')[0].strip()
for idx in tqdm(range(len(step4))):
    example = step4[idx]
    step4[idx][f"answer_{MODEL}"]  = [answer2choice(index2output[k]) for k in example2perm[idx]]
        
os.makedirs('data/step4', exist_ok=True)
json.dump(step4, open(f"data/step4/{MODEL}_{STRATEGY}_{SPLIT}.json", 'w'))