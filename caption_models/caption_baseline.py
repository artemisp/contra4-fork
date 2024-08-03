import os
from torch import cuda, bfloat16
import transformers
import json
import pandas as pd
import torch
from tqdm import tqdm
import random
import argparse
import json
import requests
import io
import zipfile
from transformers import T5ForConditionalGeneration, T5Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="lmsys/vicuna-13b-v1.5", help='"lmsys/vicuna-13b-v1.5", "meta-llama/Llama-2-13b-chat-hf", "google/flan-t5-xxl"')
parser.add_argument("--tokenizer_id", type=str, default="", help='for OneLLM model.')
parser.add_argument("--config", type=str, default="", help='for OneLLM model.')
parser.add_argument("--type", type=str, default="predicted", help="[no_input, predicted, oracle, random]")
parser.add_argument("--bs", type=int, default=16)   
args = parser.parse_args()
model_id = args.model_id
bs = args.bs

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

index2letter = lambda x: chr(ord('A') + x)

if args.type == 'predicted':
    data2pred_captions = {}
    data2pred_captions["coco_val"] = {f'val2014/COCO_val2014_{d["image_id"]:012d}.jpg': d["caption"] for d in json.load(open('caption_prediction/predictions/results_coco_val_instructblip.json'))}
    data2pred_captions["densecap_valid"] = {d["id"]: d["caption"] for d in json.load(open('caption_prediction/predictions/results_dci_val_instructblip.json'))}
    data2pred_captions["densecap_test"] ={d["id"]: d["caption"] for d in json.load(open('caption_prediction/predictions/results_dci_test_instructblip.json'))}
    data2pred_captions["msrvtt_test"] = {d["image_id"]: d["caption"] for d in json.load(open('caption_prediction/predictions/results_msrvtt_test_instructblip.json'))}
    data2pred_captions["charades_v1_test"] = {d["id"]: d["caption"] for d in json.load(open('caption_prediction/predictions/results_charades_test_instructblip.json'))}
    data2pred_captions["clothov1_instruct_val"] = {d["id"]: d["caption"] for d in json.load(open('/export/home/Pengi/results_clothov1.json'))}
    data2pred_captions["clothov2_instruct_val"] = {d["id"]: d["caption"] for d in json.load(open('caption_prediction/predictions/results_clothov2_pengi.json'))}
    data2pred_captions["audiocaps_mm_caption_val"] = {d["id"]: d["caption"] for d in json.load(open('/export/home/Pengi/results_audiocaps_gen.json'))}
    data2pred_captions["objaverse_pointllm_val"] = {d["image_id"]: d["caption"] for d in json.load(open('/export/home/LAVIS-xgen_mm/lavis/output/xinstructblip/eval/vicuna7b/pc/objaverse_captioning_pointllm/20240513174/result/val_epoch0.json'))}
elif args.type == 'random':
    print("Loading audio data....")
    audiocaps_dataset_val = [row for i,row in pd.read_csv('https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv').iterrows()]
    clothov1_dataset_val = [row for i,row in pd.read_csv('https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv').iterrows()]
    clothov2_dataset_val = [row for i,row in pd.read_csv('https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv').iterrows()]
    audio_data = [{'source': "audiocaps_val", "id":x["audiocap_id"], "caption": [x["caption"]]} for x in tqdm(audiocaps_dataset_val)]
    audio_data.extend([{'source': "clothov2_val", "id":d["file_name"], "caption": [d["caption_1"], d["caption_2"], d["caption_3"], d["caption_4"], d["caption_5"]]} for d in tqdm(clothov2_dataset_val)])
    audio_data.extend([{'source': "clothov1_val", "id":d["file_name"], "caption": [d["caption_1"], d["caption_2"], d["caption_3"], d["caption_4"], d["caption_5"]]} for d in tqdm(clothov1_dataset_val)])
    print("Audio Data Size : ", len(audio_data))

    print("Loading 3D data....")
    url = 'https://huggingface.co/datasets/RunsenXu/PointLLM/resolve/main/val_object_ids_3000.txt'
    response = requests.get(url)
    # Ensuring the request was successful
    if response.status_code == 200:
        # Accessing the content of the file
        val_ids = response.text.split("\n")
    else:
        print("Failed to retrieve the Objaverse val ids. Status code:", response.status_code)
    objaverse_dataset_val = [row for i,row in pd.read_csv('https://huggingface.co/datasets/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_no3Dword.csv', names=['id', 'caption'], header=None).iterrows() if row['id'] in val_ids]
    pc_data = [{'source': "3DCap_pointllm_val", "id":x["id"], "caption": x['caption']} for x in tqdm(objaverse_dataset_val)]
    print("3D Data Size : ", len(pc_data))

    print("Loading image data....")
    url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json'
    response = requests.get(url)
    if response.status_code == 200:
        coco_caption_val = json.loads(response.text)
    else:
        print("Failed to retrieve the COCO val. Status code:", response.status_code)
    image_descriptions = [{"source":"coco_val", "id":item['image'], "caption":random.choice(item['caption'])} for item in coco_caption_val]
    import densely_captioned_images.dataset.impl as impl
    test_ds = impl.get_complete_dataset_with_settings('test', load_subcaptions=False, load_base_image=True)
    valid_ds = impl.get_complete_dataset_with_settings('valid', load_subcaptions=False, load_base_image=True)
    image_descriptions.extend([{"source":"densecap_test", "id":x[0]["image"].item(), "caption":' '.join(x[0]["caption"].split('\n')[:2])} for x in tqdm(test_ds)])
    image_descriptions.extend([{"source":"densecap_valid", "id":x[0]["image"].item(), "caption":' '.join(x[0]["caption"].split('\n')[:2])} for x in tqdm(valid_ds)])
    print("Image Data Size : ", len(image_descriptions))
   
    print("Loading video data....")
    response = requests.get('https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip')
    if response.status_code == 200:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        if 'Charades/Charades_v1_test.csv' in zip_file.namelist():
            zip_file.extract('Charades/Charades_v1_test.csv')
            import pandas as pd
            charades_data = [row for i,row in pd.read_csv('Charades/Charades_v1_test.csv').iterrows()]
        else:
            print(f"The file {'Charades_v1_test.csv'} does not exist in the ZIP archive.")
        zip_file.close()
    else:
        print("Failed to download the file. Status code:", response.status_code)
    video_descriptions = [{"source":"charades_v1_test", "id": r['id'],"caption": f"This scene takes place in a {r['descriptions']}. " + r['scene']} for r in tqdm(charades_data)]
    url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/cap_test.json'
    response = requests.get(url)
    if response.status_code == 200:
        msrvtt_test = json.loads(response.text)
    else:
        print("Failed to retrieve the MSRVTT test. Status code:", response.status_code)
    video_descriptions.extend([{"source": "msrvtt_test", "id": item["video"], "caption": item["caption"]} for item in msrvtt_test])
    print("Video Data Size : ", len(video_descriptions))
    
    captions_dict = {
    "video": [d['caption'] if isinstance(d['caption'], str) else d['caption'][0] for d in video_descriptions],
    "audio": [d['caption'] if isinstance(d['caption'], str) else d['caption'][0] for d in audio_data],
    "image":[d['caption'] if isinstance(d['caption'], str) else d['caption'][0] for d in image_descriptions],
    "pc": [d['caption'] if isinstance(d['caption'], str) else d['caption'][0] for d in pc_data]
    }
    all_captions = []
    for k,v in captions_dict.items():
        all_captions.extend(v)

discrn_data_f = json.load(open('../data/discrn_balanced.json'))

if "vicuna" in model_id:
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {} Choices: {}\n\nASSISTANT:"
    def prompt_from_example(example):
        if args.type == 'predicted':
            try:
                for ex in example["examples"]:
                    ex["caption"] = data2pred_captions[ex["source"]][str(ex['id'])]
                    ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
            except:
                print(ex["source"], ex['id'])
                return ""
        elif args.type == 'no_input':
            for ex in example["examples"]:
                ex["caption"] = ""
        elif args.type == 'random':
            for ex in example["examples"]:
                ex["caption"] = random.choice(all_captions)
            
        captions = [ex['caption'] for ex in example["examples"]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        return prompt.format(example["questions"][0], options)
elif "llama" in model_id:
    prompt = "<s>[INST]Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices: {} Answer:[/INST]"
    def prompt_from_example(example):
        if args.type == 'predicted':
            try:
                for ex in example["examples"]:
                    ex["caption"] = data2pred_captions[ex["source"]][str(ex['id'])]
                    ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
            except:
                print(ex["source"], ex['id'])
                return ""
        elif args.type == 'no_input':
            for ex in example["examples"]:
                ex["caption"] = ""
        elif args.type == 'random':
            for ex in example["examples"]:
                ex["caption"] = random.choice(all_captions)
            
        captions = [ex['caption'] for ex in example["examples"]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        return prompt.format(example["questions"][0], options)
elif "OneLLM" in model_id:
    prompt = "<s>[INST]Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices: {} Answer:[/INST]"
    def prompt_from_example(example):
        if args.type == 'predicted':
            try:
                for ex in example["examples"]:
                    ex["caption"] = data2pred_captions[ex["source"]][str(ex['id'])]
                    ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
            except:
                print(ex["source"], ex['id'])
                return ""
        elif args.type == 'no_input':
            for ex in example["examples"]:
                ex["caption"] = ""
        elif args.type == 'random':
            for ex in example["examples"]:
                ex["caption"] = random.choice(all_captions)
            
        captions = [ex['caption'] for ex in example["examples"]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        return prompt.format(example["questions"][0], options)

elif "flan" in model_id:
    prompt = "Select which of the scenes best answers the question. Respond with brevity, and only include your choice in the response.\n\nQuestion: {} Choices: {} Answer:"
    def prompt_from_example(example):
        if args.type == 'predicted':
            try:
                for ex in example["examples"]:
                    ex["caption"] = data2pred_captions[ex["source"]][str(ex['id'])]
                    ex["caption"] = ex["caption"][0] if isinstance(ex["caption"], list) else ex["caption"]
            except:
                print(ex["source"], ex['id'])
                return ""
        elif args.type == 'no_input':
            for ex in example["examples"]:
                ex["caption"] = ""
        elif args.type == 'random':
            for ex in example["examples"]:
                ex["caption"] = random.choice(all_captions)
            
        captions = [ex['caption'] for ex in example["examples"]]
        options = "\n".join([f'Scene {index2letter(i)}. "{c}"' for i,c in enumerate(captions)])
        return prompt.format(example["questions"][0], options)
    



device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


if 'flan' not in model_id and 'onellm' not in model_id.lower():
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    token = os.environ.get("HF_ACCESS_TOKEN")
    model_config = transformers.AutoConfig.from_pretrained(
        model_id if not args.config else args.config,
        use_auth_token=token
    )
    
   
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=token,
        cache_dir='/content/drive/MyDrive/transformers_cache',
        attn_implementation="flash_attention_2"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id if not args.tokenizer_id else args.tokenizer_id,
        use_auth_token=token,
        cache_dir='/content/drive/MyDrive/transformers_cache'
    )

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # langchain expects the full text
        task='text-generation',
        do_sample=False,
        max_new_tokens=30,  # mex number of tokens to generate in the output
        repetition_penalty=1.1, # without this output begins repeating
        batch_size=bs
    )
    generate_text.tokenizer.pad_token_id = model.config.eos_token_id
    generate_text.tokenizer.padding_side = 'left'
elif "onellm" in model_id.lower():
    import sys
    sys.path.append('../cross_modal_baselines/OneLLM')
    from util.misc import default_tensor_type
    import numpy as np
    from model.meta import MetaModel
    import torch
    import torch.distributed as dist
    import multiprocessing as mp
    from fairscale.nn.model_parallel import initialize as fs_init
    from util.misc import default_tensor_type
    from util.misc import setup_for_distributed
    from data.conversation_lib import conv_templates
    
    # mp.set_start_method("spawn")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:5432")
    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    # set the print behavior.
    setup_for_distributed(True)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }['fp16']
    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel("onellm", args.config, None, args.tokenizer_id)
    
    print("Loading pretrained weights ...")
    checkpoint = torch.load(args.model_id, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    model.llma.config =json.load(open(args.config))
    
    def generate_text(inputs,
        temperature=0.4,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=30,  # mex number of tokens to generate in the output
        repetition_penalty=1.1,  # without this output begins repeating
        batch_size=bs): 

        prompts = []
        for inp in inputs:
            conv = conv_templates["v1"].copy()        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())
        outputs = []
        for idx in range(0,len(prompts),bs):
            curr_prompts = prompts[idx:idx+bs]
            with torch.cuda.amp.autocast(dtype=target_dtype):
                responses = model.generate(curr_prompts, None, 5, temperature=temperature, top_p=0.9, modal="")
                for response, prompt in zip(responses, curr_prompts):
                    response = response[len(prompt):].split('###')[0]
                    response = response.replace("Assistant:","").strip()
                    outputs.append(response)
        return outputs
    
else:
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
        batch_size=bs):
        decoded_output = []
        for idx in range(0, len(inputs), batch_size):
            curr_inputs = inputs[idx:idx+batch_size]
            curr_inputs = tokenizer(curr_inputs, return_tensors="pt", padding='longest').to(device)
            outputs = model.generate(**curr_inputs, temperature=temperature, max_length=curr_inputs['input_ids'].shape[1] + max_new_tokens, repetition_penalty=repetition_penalty)
            decoded_output.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return decoded_output
    
model.eval()
print("Model loaded.")

def answer2choice(text):
    if text in ['Scene A', 'Scene B', 'Scene C', 'Scene D']:
        return text
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

predictions = []
for idx in tqdm(range(0, len(discrn_data_f), bs)):
    batch = discrn_data_f[idx:idx + bs]
    inputs = [prompt_from_example(item) for item in batch]
    inputs = [inp for inp in inputs if inp != ""]
    outputs = generate_text(inputs)
    if idx*bs%100 == 0:
        json.dump(predictions, open(f"results/preds_discrn_{model_id.replace('/', '_')}_{args.type}.json", 'w'))
    for i, out in enumerate(outputs):
        if 'flan' in model_id or 'onellm' in model_id.lower():
            predictions.append({"id": batch[i]["id"], "gt_ans": batch[i]["answers"][0], "pred_ans": answer2choice(out)})
        else:
            predictions.append({"id": batch[i]["id"], "gt_ans": batch[i]["answers"][0], "pred_ans": out[0]["generated_text"]})
os.makedirs("results", exist_ok=True)
json.dump(predictions, open(f"results/preds_discrn_{model_id.replace('/', '_')}_{args.type}.json", 'w'))


