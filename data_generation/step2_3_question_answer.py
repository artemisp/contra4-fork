"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from transformers import pipeline
import os
from torch import cuda, bfloat16
import transformers
import pickle
import json
import pandas as pd
import torch
from tqdm import tqdm
from lavis.datasets.builders import load_dataset
import random
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=8)
parser.add_argument('--random', action='store_true', help='sampling strategy: random or high_sim')
args = parser.parse_args()
RANDOM = args.random
BS = args.bs

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



model_id = 'meta-llama/Llama-2-13b-hf'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


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
    use_auth_token=token
)
model.eval()
print(f"Model loaded.")

inst_msg_q_gen = """Scene A. "a shattered piece of paper, resembling a broken phone and a flying newspaper"
    Scene B. "tourists walking near a catholic church in Mexico on a sunny summer day"
    Generated Question: Which input evokes a sense of chaos and abandonment?

    Scene A. "Someone is using a rip saw in a carpenter's workshop"
    Scene B. "An elegant bathroom featuring a tub, sink, mirror, and decorations"
    Generated Question: Which input depicts a louder scene?

    Scene A. "The night sky showcasing the Milky Way"
    Scene B. "A bustling city street at midday"
    Scene C. "A serene mountain landscape in the morning"
    Generated Question: Which input is different from the other two?

    Scene A. "A painting depicting a stormy sea"
    Scene B. "A photograph of a calm beach at sunset"
    Scene C. "A digital illustration of a bustling space station"
    Scene D. "A sculpture of a tranquil garden"
    Generated Question: Which input is most different from the other?"""

end_q_prompt = {
    2: """
    Scene A. "{}"
    Scene B. "{}"
    Generated Question:""",#+E_INST,
        3: """
    Scene A. "{}"
    Scene B. "{}"
    Scene C. "{}"
    Generated Question:""",#+E_INST,
    4: """
    Scene A. "{}"
    Scene B. "{}"
    Scene C. "{}"
    Scene D. "{}"
    Generated Question:""",#+E_INST
    }

inst_msg_a_gen = """Scene A. "a shattered piece of paper, resembling a broken phone and a flying newspaper"
    Scene B. "tourists walking near a catholic church in Mexico on a sunny summer day"
    Question: Which scene evokes a sense of chaos and abandonment?
    Answer: Scene A. Scene A evokes feelings of chaos and abandonment, contrasting sharply with the joy and vibrancy of Scene B.

    Scene A. "Someone is using a rip saw in a carpenter's workshop"
    Scene B. "An elegant bathroom featuring a tub, sink, mirror, and decorations"
    Question: Which scene depicts a louder scene?
    Answer: Scene A. Scene A is characterized by the noise and activity of craftsmanship, whereas Scene B offers a serene and luxurious ambiance for relaxation.

    Scene A. "The night sky showcasing the Milky Way"
    Scene B. "A bustling city street at midday"
    Scene C. "A serene mountain landscape in the morning"
    Question: Which scene is different from the other two?
    Answer: Scene B. Scene B, with its bustling city life, differs in its dynamic and urban setting from the tranquil and natural settings of Scenes A and C.

    Scene A. "A painting depicting a stormy sea"
    Scene B. "A photograph of a calm beach at sunset"
    Scene C. "A digital illustration of a bustling space station"
    Scene D. "A sculpture of a tranquil garden"
    Question: Which scene is most different from the other?
    Answer: Scene C. Scene C, a digital illustration of a bustling space station, diverges in its futuristic and technological theme from the natural and serene subjects of the other inputs.
    """

end_prompt_a_gen = {
    2: """
    Scene A. {}
    Scene B. {}
    Question: {}
    Answer:""", #+E_INST,
    3: """
    Scene A. {}
    Scene B. {}
    Scene C. {}
    Question: {}
    Answer:""", #+E_INST,
        4: """
    Scene A. {}
    Scene B. {}
    Scene C. {}
    Scene D. {}
    Question: {}
    Answer:
    """ #+E_INST
    }


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False, 
    task='text-generation',
    temperature=0.4,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=45,  # max number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
    batch_size=BS
)
os.makedirs("data", exist_ok=True)
os.makedirs("data/step2_3", exist_ok=True)
generate_text.tokenizer.pad_token_id = model.config.eos_token_id
generate_text.tokenizer.padding_side = "left"
output = []
step1 = json.load(open(f'data/step1/tuples_{"highsim" if not RANDOM else "random"}.json'))
print("Total Output Length: ", len(step1))
print("Random: ", RANDOM)
for i in tqdm(range(0, len(step1), BS)):
    examples = [step1[i+j] for j in range(BS) if i+j<len(output)]
    input_texts = []
    for example in examples: 
        example['questions'] = []   
        example['answers'] = []
    input_texts = []
    for example in examples:
        m = min(4,len(example["examples"]))
        input_text = inst_msg_q_gen
        input_text += end_q_prompt[m].format(*[x_ for x in zip([e['caption'] for e in example["examples"]]) for x_ in x])
        input_texts.append(input_text + " Which scene")
    with torch.no_grad():
        output_texts = generate_text(input_texts, num_return_sequences=1)
    for j, output_text in enumerate(output_texts):
        output_text = output_text[0]['generated_text'].split("?")[0] + "?"
        examples[j]["questions"].append("Which scene " + output_text)
    input_texts = []
    for example in examples:
        m = min(4,len(example["examples"]))
        input_text = inst_msg_a_gen
        fmt_inps = [e['caption'][0] if isinstance(e['caption'], list) else e['caption']  for e in example["examples"]] +[example['questions'][0]]
        input_text += end_prompt_a_gen[m].format(*fmt_inps)
        input_texts.append(input_text)
    with torch.no_grad():
        output_texts = generate_text(input_texts, num_return_sequences=1,  max_new_tokens=10)
    for j, output_text in enumerate(output_texts):
        output_text = output_text[0]['generated_text'].strip()
        examples[j]["answers"].append(output_text) 
        output.append(examples[j])
        
json.dump(output, open(f"data/step2_3/output_{'_random' if args.random else 'high_sim'}.json", "w"))
  
