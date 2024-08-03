import torch
from PIL import Image
import os
from tqdm import tqdm
import json
from lavis.models.blip2_models.blip2_vicuna_xinstruct import Blip2VicunaXInstruct
from lavis.processors.ulip_processors import ULIPPCProcessor
import requests
import pandas as pd

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

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


import objaverse
annotations = objaverse.load_annotations()
licenses = [annotations[d['id']]['license'] for d in pc_data]
from pdb import set_trace; set_trace()

# model = Blip2VicunaXInstruct.from_pretrained(model_type='vicuna7b')
# preprocess = ULIPPCProcessor()
# data_root = "/export/einstein-vision/3d_vision/objaverse/objaverse_pc_parallel"
# test_pointllm = {}
# for im in tqdm(pc_data):
#     img = os.path.join(data_root, im["id"]+'_8192.npy')['arr_0']
#     image = preprocess(img).unsqueeze(0).to(device)
#     out = model.generate({"image": image, "prompt": "Describe the 3d model."})
#     test_dci[im["id"]] = out
# json.dump(test_pointllm, open("test_objaverse_pointllm.json", "w"))
