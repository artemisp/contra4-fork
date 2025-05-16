"""
This example shows how to use Ray Data for running offline batch inference
distributively on a multi-nodes cluster.

Learn more about Ray Data in https://docs.ray.io/en/latest/data/data.html
"""

from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams
from PIL import Image

assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"

import torch
import os
import pandas as pd
import datasets
import json
import argparse
import pickle
from functools import partial   
from transformers import AutoTokenizer
import cv2
import numpy as np
import base64
from io import BytesIO

def extract_frames(video_path, num_frames=4):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    
    # Read the frames from the specified indices
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video position to the specific frame index
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    # Release the video capture object
    cap.release()
    
    return frames

def encode_frame_to_base64(frame):
    # Convert the frame to JPEG format in memory (without saving to a file)
    _, buffer = cv2.imencode('.jpg', frame)
    # Convert the buffer to base64
    encoded_string = base64.b64encode(buffer).decode('utf-8')
    return encoded_string


num_gpus = torch.cuda.device_count()                       
num_cpus = torch.cuda.device_count() * 8

RAY_STOP = ray.shutdown
RAY_INIT = partial(ray.init, address='local', num_cpus=num_cpus, num_gpus=num_gpus, _temp_dir=os.path.join('ray')) #f'ray start --head'

ctx = RAY_INIT()


PROMPT_TEMPLATE_MAP = {
    "llava-hf/llava-1.5-7b-hf":  "USER: <image>\n{}\nASSISTANT:",
    "llava-hf/llava-1.5-13b-hf":  "USER: <image>\n{}\nASSISTANT:",
    "OpenGVLab/InternVL2-8B": "<|im_start|>User\nImage: <image>\n{}\n<|im_end|>\n<|im_start|>Assistant\n",
    "Qwen/Qwen2-VL-7B-Instruct": "{}"
}

STOP_TOKEN_IDS_MAP = {
    "llava-hf/llava-1.5-7b-hf":  None,
    "llava-hf/llava-1.5-13b-hf": None,
    "OpenGVLab/InternVL2-8B": [1, 0, 92543, 92542, 0],
    "Qwen/Qwen2-VL-7B-Instruct": None
}


parser = argparse.ArgumentParser(
    description='Generate llm outputs')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="huggingface model name to use")
parser.add_argument('--half', type=bool, default=True, help="use float 16")
parser.add_argument('--generation_kwargs', type=str, default="{}", help="generation kwargs")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_kwargs', type=str, default="{}", help="model kwargs")
parser.add_argument('--batch_size', type=int, default=4, help="per device batch size")
parser.add_argument('--strategy', type=str, default='random', help='sampling strategy: random or high_sim')
args = parser.parse_args()

model=args.model_name
tensor_parallel_size=torch.cuda.device_count()
seed=args.seed
half=args.half
num_instances=1

msrvtt_dir = os.environ['MSRVTT_DIR']
    
with ctx:
    formatted_inputs = {'text': [], 'index': [], 'image_path': []}
    for d in os.listdir(msrvtt_dir):
        formatted_inputs['text'].append("Describe this set of frames. Consider the frames to be a part of the same video.")
        formatted_inputs['index'].append(d.split('.')[0])
        formatted_inputs['image_path'].append(os.path.join(msrvtt_dir, d))
    dataset = formatted_inputs
    generation_kwargs = json.loads(args.generation_kwargs)
    if generation_kwargs.get('top_k', -1) == 0:
        generation_kwargs['top_k'] = -1

    SAMPLING_PARAMS = SamplingParams(

        temperature=generation_kwargs.get('temperature', 0.9),
        top_p=generation_kwargs.get('top_p', 0.9),
        top_k=generation_kwargs.get('top_k', -1),
        presence_penalty=0.0,
        frequency_penalty=0.0,
        spaces_between_special_tokens=False,
        truncate_prompt_tokens=None,
        min_tokens=10, 
        max_tokens=generation_kwargs.get('max_new_tokens', 120),
        best_of=generation_kwargs.get('num_return_sequences', 1), 
        n=generation_kwargs.get('num_return_sequences', 1),
        stop_token_ids=STOP_TOKEN_IDS_MAP[model],
        ) 

    print("Device count: ", torch.cuda.device_count())
    


    ## https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py

    # Create a class to do batch inference.
    class LLaVAPredictor:

        def __init__(self, 
                    ):
        
            # Create an LLM.
            self.llm = LLM(model=model,
                        tokenizer=model,
                            tensor_parallel_size=tensor_parallel_size,
                            trust_remote_code=True,
                            seed=seed,
                            dtype=torch.float16 if half else torch.float32, 
                            max_num_seqs=64, 
                            distributed_executor_backend='ray',
                            enforce_eager=True,
                            # max_parallel_loading_workers=4
                            )
            # self.sampling_params = sampling_params
            # self.lora_path = None
            # self.text_key = 'text'

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            batch = [dict(zip(batch, t)) for t in zip(*batch.values())]
            
            outputs = self.llm.generate(batch,
                                        SAMPLING_PARAMS,
                                        use_tqdm=False,
                                        )
            
            prompt: List[str] = []
            generated_text: List[str] = []
            index: List[int] = []
            for i,output in enumerate(outputs):
                prompt.append(output.prompt)
                generated_text.append([o.text for o in output.outputs])
                index.append(batch[i]['index'])
            return {
                "index": index,
                "prompt": prompt,
                "generated_text": generated_text,
            }
        
        
    # Create a class to do batch inference.
    class InternVL2_5Predictor:

        def __init__(self, 
                    ):
            
            # Create an LLM.
            self.llm = LLM(model=model,
                            tokenizer=model,
                            tensor_parallel_size=tensor_parallel_size,
                            trust_remote_code=True,
                            limit_mm_per_prompt={"image": 1},
                            # mm_processor_kwargs={"max_dynamic_patch": 4},
                            enable_chunked_prefill=False,
                            max_model_len=4096,
                            seed=seed,
                            dtype=torch.float16 if half else torch.float32, 
                            max_num_seqs=32, 
                            distributed_executor_backend='ray',
                            enforce_eager=True,
                            # disable_mm_preprocessor_cache=True
                            # max_parallel_loading_workers=4
                            )

            # self.sampling_params = sampling_params
            # self.lora_path = None
            # self.text_key = 'text'

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            batch = [dict(zip(batch, t)) for t in zip(*batch.values())]

            outputs = self.llm.generate(batch,
                                        SAMPLING_PARAMS,
                                        use_tqdm=False,
                                        )
            
            prompt: List[str] = []
            generated_text: List[str] = []
            index: List[int] = []
            for i,output in enumerate(outputs):
                prompt.append(output.prompt)
                generated_text.append([o.text for o in output.outputs])
                index.append(batch[i]['index'])
            return {
                "index": index,
                "prompt": prompt,
                "generated_text": generated_text,
            }
    
    # Create a class to do batch inference.
    class QwenVL2Predictor:

        def __init__(self, 
                    ):
            
            # Create an LLM.
            self.llm = LLM(model=model,
                            tokenizer=model,
                            tensor_parallel_size=tensor_parallel_size,
                            trust_remote_code=True,
                            limit_mm_per_prompt={"image": 4},
                            # mm_processor_kwargs={"max_dynamic_patch": 4},
                            enable_chunked_prefill=False,
                            max_model_len=4096,
                            seed=seed,
                            dtype=torch.float16 if half else torch.float32, 
                            max_num_seqs=32, 
                            distributed_executor_backend='ray',
                            enforce_eager=True,
                            # disable_mm_preprocessor_cache=True
                            # max_parallel_loading_workers=4
                            )

            # self.sampling_params = sampling_params
            # self.lora_path = None
            # self.text_key = 'text'

        def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
            # Generate texts from the prompts.
            # The output is a list of RequestOutput objects that contain the prompt,
            # generated text, and other information.
            batch = [dict(zip(batch, t)) for t in zip(*batch.values())]
            
            final_messages = []
            indices = []
            for message in batch:
                video_frames = extract_frames(message['multimodal_data']['video_path'], num_frames=4)
                final_messages.append([{"role": "user", "content": [{"type": "text", "text": message['prompt']}]}])
                for i in range(len(video_frames)):
                        base64_image = encode_frame_to_base64(video_frames[i]) # base64 encoding.
                        new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        final_messages[-1][-1]["content"].append(new_image)
                indices.append(message['index'])

            outputs = self.llm.chat(final_messages,
                                        SAMPLING_PARAMS,
                                        use_tqdm=False,
                                        )
            
            prompt: List[str] = []
            generated_text: List[str] = []
            index: List[int] = []
            for i,output in enumerate(outputs):
                prompt.append(output.prompt)
                generated_text.append([o.text for o in output.outputs])
                index.append(indices[i])
            return {
                "index": index,
                "prompt": prompt,
                "generated_text": generated_text,
            }
        
        
        

    MODEL_MAP = {
        "llava-hf/llava-1.5-7b-hf": LLaVAPredictor,
        "llava-hf/llava-1.5-13b-hf": LLaVAPredictor,
        "OpenGVLab/InternVL2-8B": InternVL2_5Predictor,
        "Qwen/Qwen2-VL-7B-Instruct": QwenVL2Predictor
    }

    devices = list(range(torch.cuda.device_count()))

    if isinstance(dataset, datasets.Dataset):
        ds = ray.data.from_datasets(dataset)
    elif isinstance(dataset, dict):
        ds = ray.data.from_items([{'text': d, 'image_path': img, 'index': i} for d,img,i in zip(dataset['text'], dataset['image_path'], dataset['index'])])
    elif isinstance(dataset, torch.utils.data.Dataset):
        ds = ray.data.from_torch(dataset)
    elif isinstance(dataset, pd.DataFrame):
        ds = ray.data.from_pandas(dataset)
    else:
        ds = ray.data.from_items(dataset)
    
    ## map text to prompt
    ds = ds.map(lambda x: {'prompt': PROMPT_TEMPLATE_MAP[model].format(x['text']), 
                        'multimodal_data': {"video_path": x['image_path']}, 
                        'index': x['index']}
                ).materialize()
        
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{
                "GPU": 1,
                "CPU": 1
            }] * tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))

    resources_kwarg: Dict[str, Any] = {}
    if len(devices) == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn
    # outputs = self.model(ds)
    ds = ds.map_batches(
        MODEL_MAP[model],
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=int(args.batch_size*torch.cuda.device_count()) if args.batch_size else ds.count(),
        **resources_kwarg
    )

    ds.write_json(os.path.join('results', f'train_val_video_captions_{args.strategy}.pkl'))

    RAY_STOP()