"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import os
import re
from torch import cuda
import transformers
from torch import bfloat16
from tqdm import tqdm

data = json.load(open('data/discrn_balanced.json'))

bs=128
model_id = 'meta-llama/Llama-2-13b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


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
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=token,
)
model.eval()
print(f"Model loaded on {device}")


llama_prompt = "<s>[INST] <<SYS>>\n You are given a question and you have to identify its main topic. Examples include counting, object existence, location, properties, emotional response, and other. Respond with at most two words, and only include your choice in the response.\n<</SYS>>\n\nQuestion: \"{}\" What is the question's topic? Be specific. [/INST] Topic:"
inputs = [llama_prompt.format(d["questions"][0]) for d in data]

generate_text.tokenizer.pad_token_id = model.config.eos_token_id
generate_text.tokenizer.padding_side = "left"
outputs = generate_text(inputs, max_new_tokens=5, do_sample=False, batch_size=32)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=5,  # mex number of tokens to generate in the output
    do_sample=False,
    repetition_penalty=1.1,  # without this output begins repeating
    batch_size=bs
)

for i,d in enumerate(data):
    d['category'] = outputs[i][0]['generated_text'].strip()
    if d['category'] == 'Movement':
        d['category'] = 'Motion'
    elif d['category'] == 'Actions':
        d['category'] = 'Action'
    elif d['category'] in ['Animal', 'Person', 'Object Existence', 'Characters']:
        d['category'] = 'Existence'

json.dump(data, open('data/discrn_balanced.json', 'w'))