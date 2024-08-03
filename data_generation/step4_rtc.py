"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import os
from torch import cuda
import transformers
from torch import bfloat16
from tqdm import tqdm
import argparse
from itertools import permutations

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=16)
parser.add_argument('--random', action='store_true', help='sampling strategy: random or high_sim')
parser.add_argument("--model", type=str, default='llama2', help='llama2 or mistral or flan')
args = parser.parse_args()
BS = args.bs
RANDOM = args.random
MODEL = args.model


if MODEL == 'llama2':
    model_id = 'meta-llama/Llama-2-13b-chat-hf'
    prompt = "<s>[INST] <<SYS>>\nSelect which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n<</SYS>>\n\nQuestion: {} Choices:{}[/INST] Scene"
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
    prompt = "<s>[INST] Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices:{} [/INST] Scene"
    def prompt_from_example(example, permutation=None):
        for ex in example["examples"]:
            if 'captions' in ex:
                ex["caption"] = ex["captions"]
            ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
        if permutation:
            captions = [ex['caption'] for ex in [example["examples"][i] for i in permutation]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        return prompt.format(example["questions"][0], options)
elif MODEL == 'flan':
    model_id = 'google/flan-t5-xxl'
    prompt = "Select which of the inputs best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices: {} Answer: Scene"
    def prompt_from_example(example, permutation=None):
        for ex in example["examples"]:
            if 'captions' in ex:
                ex["caption"] = ex["captions"]
            ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
        if permutation:
            captions = [ex['caption'] for ex in [example["examples"][i] for i in permutation]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        return prompt.format(example["questions"][0], options)


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

if MODEL != 'flan':
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, need auth token for these
    token = os.environ.get("HF_ACCESS_TOKEN")
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=token
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=token,
        attn_implementation="flash_attention_2"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=token,
    )
    
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  
        task='text-generation',
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=5,  
        do_sample=False,
        repetition_penalty=1.1,  # without this output begins repeating
        batch_size=BS
    )

    generate_text.tokenizer.pad_token_id = model.config.eos_token_id
    generate_text.tokenizer.padding_side = "left"
else:
    model = T5ForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=token,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        model_id,
        use_auth_token=token,
    )
    def generate_text(inputs,
        temperature=0.4,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=30,  # mex number of tokens to generate in the output
        repetition_penalty=1.1,  # without this output begins repeating
        batch_size=BS):
        decoded_output = []
        for idx in range(0, len(inputs), batch_size):
            curr_inputs = inputs[idx:idx+batch_size]
            curr_inputs = tokenizer(curr_inputs, return_tensors="pt", padding='longest').to(device)
            outputs = model.generate(**curr_inputs, temperature=temperature, max_length=curr_inputs['input_ids'].shape[1] + max_new_tokens, repetition_penalty=repetition_penalty)
            decoded_output.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return decoded_output
    

model.eval()
print(f"Model loaded.")

index2letter = lambda x: chr(ord('A') + x)
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

step4 = json.load(open(f"data/step2_3/output_{'_random' if args.random else 'high_sim'}.json"))

for idx in tqdm(range(0, len(step4), BS)):
    batch = step4[idx:idx + BS]
    all_permutations = [list(permutations(range(len(example["examples"])))) for example in batch]
    lengths = [len(perm) for perm in all_permutations]
    all_batch = [[batch[i]]*len(all_permutations[i]) for i in range(len(batch))]
    all_permutations = [item for sublist in all_permutations for item in sublist]
    all_batch = [item for sublist in all_batch for item in sublist]
    inputs = [prompt_from_example(item, perm) for item,perm in zip(all_batch, all_permutations)]
    outputs = generate_text(inputs)  
    prev = 0
    for k, length in enumerate(lengths):       
        step4[idx + k][f'{model_id}_permutations'] ={"permutations": all_permutations[prev:prev+length], "answers": []}
        for i, out in enumerate(outputs[prev:prev+length]):
            if 'flan' in model_id:
                step4[idx + k][f'{model_id}_permutations']["answers"].append(answer2choice(out))
            else:
                step4[idx + k][f'{model_id}_permutations']["answers"].append(answer2choice(out[0]["generated_text"]))
        prev += length
json.dump(step4, open(f"data/step2_3/output_{'_random' if args.random else 'high_sim'}.json", 'w'))