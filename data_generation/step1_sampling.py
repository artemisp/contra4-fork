"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
import random
import itertools
import zipfile
import io
import gc
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate tuples for the MC task')
parser.add_argument('--random', action='store_true', help='sampling strategy: random or high_sim')
parser.add_argument('--n_samples', type=int, default=30000, help='Number of samples to generate per modality combination')
args = parser.parse_args()
RANDOM = args.random
N_SAMPLES = args.n_samples


seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
from densely_captioned_images.dataset.impl import get_complete_dataset_with_settings, DenseCaptionedDataset
test_ds = get_complete_dataset_with_settings('test', load_subcaptions=False, load_base_image=True)
valid_ds = get_complete_dataset_with_settings('valid', load_subcaptions=False, load_base_image=True)
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


print("Encoding descriptions....")
model = SentenceTransformer("all-MiniLM-L6-v2")
audio_encodings = model.encode([x['caption'][0] for x in audio_data], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
pc_encodings = model.encode([x['caption'] for x in pc_data], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
image_encodings = model.encode([x['caption'] for x in image_descriptions], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
video_encodings = model.encode([x['caption'] for x in video_descriptions], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
    
gc.collect()
torch.cuda.empty_cache()
del model


def find_top_k_similar_vectors_dot(input_vector, encodings, top_k=5):
    scores = util.dot_score(input_vector, encodings)[0].cpu()
    top_k_indices = torch.argsort(scores, descending=True)[:top_k]
    top_k_ids = [(index, scores[index]) for index in top_k_indices]
    return top_k_ids


modality_to_data = {
    "audio": audio_data,
    "pc": pc_data,
    "image": image_descriptions,
    "video": video_descriptions
}
modality_to_encodings = {
    "audio": audio_encodings,
    "pc": pc_encodings,
    "image": image_encodings,
    "video": video_encodings
}

os.makedirs("data", exist_ok=True)
os.makedirs("data/step1", exist_ok=True)

example_index = 0
MODALITIES = ["audio", "pc", "image", "video"]
seen_sets = set()
output = []
for m in range(2, 5):
    modality_comb = itertools.combinations(MODALITIES, m)
    for modalities_to_sample in modality_comb:
        modalities_to_sample = list(modalities_to_sample)
        for i in tqdm(range(N_SAMPLES)):
            example = {"id": f"{'n' if not RANDOM else 'r'}{example_index}", "selection_type": "high_sim" if not RANDOM else 'random', "q_type": f"mc_{m}", "examples": [],"modalities": []}
            random.shuffle(modalities_to_sample)
            data_to_sample = modality_to_data[modalities_to_sample[0]]
            index = random.randint(0, len(data_to_sample)-1)
            example["examples"].append(data_to_sample[index])
            example["modalities"].append(modalities_to_sample[0])
            query_encoding = modality_to_encodings[modalities_to_sample[0]][index]
            flag_added = False
            while not flag_added:
                for modality in modalities_to_sample[1:]:
                    example["modalities"].append(modality)
                    data_to_sample = modality_to_data[modality]
                    if not RANDOM:
                        index = random.choice(find_top_k_similar_vectors_dot(query_encoding, modality_to_encodings[modality], top_k=30))[0].item()
                    else:
                        index = random.choice(range(len(modality_to_encodings[modality])))
                    example["examples"].append(data_to_sample[index])  
                if frozenset([d_['source'] + str(d_['id']) for d_  in example["examples"]]) not in seen_sets:
                    flag_added = True
                    example_index += 1
                    output.append(example)
                    seen_sets.add(frozenset([d_['source'] + str(d_['id']) for d_  in example["examples"]])) 
json.dump(output, open(f"data/step1/tuples_{'highsim' if not RANDOM else 'random'}.json", "w"))        
print("Number of example tuples:", len(output))