from transformers import pipeline
import os
from torch import cuda, bfloat16
import transformers
import pickle
import json
import pandas as pd
import torch
from tqdm import tqdm
import random
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=8)
parser.add_argument('--strategy',type=str, default='random', help='sampling strategy: random or high_sim')
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



inst_msg_q_gen = """
    <s>You are given some scenes described in text. Each scene is represented by a short caption. Your task is to generate a question that compares the scenes based on their content. The generated question should be relevant to the context of the scenes and should require a comparison between them. THere should be only one correct answer. Here are some examples to guide you:

    Scene A. "a shattered piece of paper, resembling a broken phone and a flying newspaper"
    Scene B. "tourists walking near a catholic church in Mexico on a sunny summer day"
    Generated Question: Which scene evokes a sense of chaos and abandonment?

    Scene A. "Someone is using a rip saw in a carpenter's workshop"
    Scene B. "An elegant bathroom featuring a tub, sink, mirror, and decorations"
    Generated Question: Which scene is more likely to involve louder noises?

    Scene A. "The night sky showcasing the Milky Way"
    Scene B. "A bustling city street at midday"
    Scene C. "A serene mountain landscape in the morning"
    Generated Question: Which scene is different from the other two?

    Scene A. "A painting depicting a stormy sea"
    Scene B. "A photograph of a calm beach at sunset"
    Scene C. "A digital illustration of a bustling space station"
    Scene D. "A sculpture of a tranquil garden"
    Generated Question: Which scene is most different from the other three?
    
    Scene A. "A team of firefighters putting out a blaze in a city"
    Scene B. "A family enjoying a picnic in a peaceful park"
    Generated Question: Which scene involves a greater sense of danger and urgency?
    
    Scene A. "A snowy mountain peak illuminated by the golden light of sunrise"
    Scene B. "A tropical beach with crystal-clear water and palm trees swaying in the breeze"
    Scene C. "A bustling city park filled with people enjoying outdoor activities"
    Scene D. "A vast desert under a blazing sun with sand dunes stretching to the horizon"
    Generated Question: Which scene represents a colder and more remote environment?  
    """

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

inst_msg_a_gen = """
<s>You are given some scenes described in text as well as a question about them. Each scene is represented by a short caption. Your task is to provide a clear and concise answer that explains the reasoning behind the correct choice. Here are some examples to guide you:
    Scene A. "a shattered piece of paper, resembling a broken phone and a flying newspaper"
    Scene B. "tourists walking near a catholic church in Mexico on a sunny summer day"
    Question: Which scene evokes a sense of chaos and abandonment?
    Answer: Scene A. Scene A evokes feelings of chaos and abandonment, contrasting sharply with the joy and vibrancy of Scene B.

    Scene A. "Someone is using a rip saw in a carpenter's workshop"
    Scene B. "An elegant bathroom featuring a tub, sink, mirror, and decorations"
    Question: Which scene is more likely to involve louder noises?
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
    Question: Which scene is most different from the other three?
    Answer: Scene C. Scene C, a digital illustration of a bustling space station, diverges in its futuristic and technological theme from the natural and serene subjects of the other inputs.

    Scene A. "A team of firefighters putting out a blaze in a city"
    Scene B. "A family enjoying a picnic in a peaceful park"
    Question: Which scene involves a greater sense of danger and urgency?
    Answer: Scene A. Scene A, with firefighters responding to a blaze, conveys a strong sense of danger and urgency compared to the calm and leisurely atmosphere of Scene B.

    Scene A. "A snowy mountain peak illuminated by the golden light of sunrise"
    Scene B. "A tropical beach with crystal-clear water and palm trees swaying in the breeze"
    Scene C. "A bustling city park filled with people enjoying outdoor activities"
    Scene D. "A vast desert under a blazing sun with sand dunes stretching to the horizon"
    Question: Which scene represents a colder and more remote environment?
    Answer: Scene A. Scene A, featuring a snowy mountain peak, exemplifies a cold and remote environment in contrast to the other settings, which are warmer or more populated.
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


os.makedirs("data", exist_ok=True)
os.makedirs("data/step2_3", exist_ok=True)
step1 = json.load(open(f'data/step1/tuples_{STRATEGY}_{SPLIT}.json'))
print("Total Output Length: ", len(step1))
print("Strategy: ", STRATEGY)
print("Split: ", SPLIT)

formatted_inputs = {"text":[], "index": []}
for i, example in enumerate(step1):
    m = min(4,len(example["examples"]))
    input_text = inst_msg_q_gen
    input_text += end_q_prompt[m].format(*[e['caption'][0] for e in example["examples"]])
    formatted_inputs["text"].append(input_text + " Which scene")
    formatted_inputs["index"].append(i)

os.makedirs('./temps', exist_ok=True)
pickle.dump(formatted_inputs, open(os.path.join('./temps', 'engine_input.pkl'), 'wb'))

engine_path = 'python src/vllm_engine.py'
command = [
    engine_path, 
    '--model_name', 'meta-llama/Meta-Llama-3-8B',
    '--half', 'True',
    '--generation_kwargs', f"'{json.dumps({'temperature': 1.05, 'top_p': 0.9, 'top_k': 0, 'max_new_tokens': 120, 'num_return_sequences':1, 'num_beams': 1})}'", 
    '--seed', 42, 
    '--batch_size', 32, 
    '--data_path', os.path.join('./temps', 'engine_input.pkl'), 
    '--output_dir', './temps'
]
os.system(" ".join([str(c) for c in command]))

outputs = []
for out_f in os.listdir(os.path.join('./temps', 'engine_output')):
    outputs.extend([json.loads(l.strip()) for l in open(os.path.join('./temps', 'engine_output', out_f)).readlines()])
os.system('rm -r ' + os.path.join('./temps', 'engine_output'))
index2output = {o['index']: o['generated_text'][0].split('\n')[0].strip() for o in outputs}

for example in step1:
    example["questions"] = []
    example["answers"] = []
for i, example in enumerate(step1):
    example["questions"].append("Which scene " + index2output[i])
    example['step_2_prompt'] = formatted_inputs["text"][i]
    m = min(4,len(example["examples"]))
    example['q_type'] = f"mc_{m}"  
    example['selection_type'] = STRATEGY


formatted_inputs = {"text":[], "index": []}
for i, example in enumerate(step1):
    m = min(4,len(example["examples"]))
    input_text = inst_msg_a_gen
    form_inp = [e['caption'][0] for e in example["examples"]] + example["questions"]
    input_text += end_prompt_a_gen[m].format(*form_inp)
    formatted_inputs["text"].append(input_text)
    formatted_inputs["index"].append(i)
os.makedirs('data/step2/', exist_ok=True)
json.dump(step1, open(f"data/step2/{STRATEGY}_{SPLIT}.json", "w"))        

os.makedirs('./temps', exist_ok=True)
pickle.dump(formatted_inputs, open(os.path.join('./temps', 'engine_input.pkl'), 'wb'))

engine_path = 'python src/vllm_engine.py'
command = [
    engine_path, 
    '--model_name', 'meta-llama/Meta-Llama-3-8B',
    '--half', 'True',
    '--generation_kwargs', f"'{json.dumps({'temperature': 0.3, 'top_p': 0.9, 'top_k': 0, 'max_new_tokens': 120, 'num_return_sequences':1, 'num_beams': 1})}'", 
    '--seed', 42, 
    '--batch_size', 32, 
    '--data_path', os.path.join('./temps', 'engine_input.pkl'), 
    '--output_dir', './temps'
]
os.system(" ".join([str(c) for c in command]))
outputs = []
for out_f in os.listdir(os.path.join('./temps', 'engine_output')):
    outputs.extend([json.loads(l.strip()) for l in open(os.path.join('./temps', 'engine_output', out_f)).readlines()])
os.system('rm -r ' + os.path.join('./temps', 'engine_output'))
index2output = {o['index']: o['generated_text'][0].split('\n')[0].strip() for o in outputs}

for i, example in enumerate(step1):
    example["answers"].append(index2output[i])
os.makedirs('data/step2_3/', exist_ok=True)
json.dump(step1, open(f"data/step2_3/{STRATEGY}_{SPLIT}.json", "w"))        

