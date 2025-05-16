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
import requests

parser = argparse.ArgumentParser(description='Generate tuples for the MC task')
parser.add_argument('--strategy',type=str, default='similarity', help='sampling strategy: random or high_sim')
parser.add_argument('--n_samples', type=int, default=30000, help='Number of samples to generate per modality combination')
parser.add_argument('--split', type=str, default='test', help='Split to sample from')
args = parser.parse_args()
STRATEGY = args.strategy
N_SAMPLES = args.n_samples
SPLIT = args.split

seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

raw_data_dirpath = 'raw_data'

print("Loading audio data....")
# audiocap_id,youtube_id,start_time,caption
audiocaps_dataset_train = [dict(row) for i,row in pd.read_csv('https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv').iterrows()]
audiocaps_dataset_test = [dict(row) for i,row in pd.read_csv('https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv').iterrows()]
# 'file_name', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5'
clotho_dataset_train = [dict(row) for i,row in pd.read_csv('https://zenodo.org/record/4783391/files/clotho_captions_development.csv').iterrows()]
clotho_dataset_test = [dict(row) for i,row in pd.read_csv('https://zenodo.org/record/4783391/files/clotho_captions_validation.csv').iterrows()]
clotho_dataset_test += [dict(row) for i,row in pd.read_csv('https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv').iterrows()]
audio_data_train = []
for x in tqdm(audiocaps_dataset_train):
    audio_data_train.append({'source': "audiocaps", 
                             "id":x["audiocap_id"], 
                             "url": f"https://www.youtube.com/watch?v={x['youtube_id']}", 
                             "meta": {
                                 "start_time": x['start_time'], 
                                 "end_time": x['start_time']+10},
                             "caption": [x["caption"]]})
for x in tqdm(clotho_dataset_train):
    audio_data_train.append({'source': "clotho", 
                             "id":x["file_name"], 
                             "url": None,
                             "meta": {},
                             "caption": [x["caption_1"], x["caption_2"], x["caption_3"], x["caption_4"], x["caption_5"]]})

audio_data_test = []
for x in tqdm(audiocaps_dataset_test):
    audio_data_test.append({'source': "audiocaps", 
                             "id":x["audiocap_id"], 
                             "url": f"https://www.youtube.com/watch?v={x['youtube_id']}", 
                             "meta": {
                                 "start_time": x['start_time'], 
                                 "end_time": x['start_time']+10},
                             "caption": [x["caption"]]
                             })
for x in tqdm(clotho_dataset_test):
    audio_data_test.append({'source': "clotho", 
                             "id":x["file_name"], 
                             "url": None,
                             "meta": {},
                             "caption": [x["caption_1"], x["caption_2"], x["caption_3"], x["caption_4"], x["caption_5"]]
                             })
print("Audio Data Size : ", len(audio_data_train), len(audio_data_test))


print("Loading 3D data....")
pointllm_train = 'https://huggingface.co/datasets/RunsenXu/PointLLM/resolve/main/PointLLM_brief_description_660K_filtered.json?download=true'
pointllm_test1 = 'https://huggingface.co/datasets/RunsenXu/PointLLM/resolve/main/PointLLM_brief_description_val_200_GT.json?download=true'
pointllm_test2 = 'https://huggingface.co/datasets/RunsenXu/PointLLM/resolve/main/PointLLM_brief_description_val_3000_GT.json?download=true'
def load_json_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        out = json.loads(response.text)
        return out
    else:
        print("Failed to retrieve the Objaverse val ids. Status code:", response.status_code)
import objaverse
def convert_pointllm_sample(sample, annotations):
    caption = sample['conversations'][-1]['value']
    url = annotations[sample['object_id']]['embedUrl']
    return {'source': 'pointllm', 'id': sample['object_id'], 'meta': {}, 'url': url, 'caption': [caption]}
train_data = random.sample(load_json_from_url(pointllm_train), 60000)
test_data = load_json_from_url(pointllm_test1)+load_json_from_url(pointllm_test2)
all_sample_ids = [sample['object_id'] for sample in train_data+test_data]
annotations = objaverse.load_annotations(all_sample_ids)
point_llm_train = [convert_pointllm_sample(sample, annotations) for sample in tqdm(train_data)]
point_llm_test = [convert_pointllm_sample(sample, annotations) for sample in tqdm(test_data)]
print("3D Data Size : ", len(point_llm_train), len(point_llm_test))

# load parquet file
print("Loading video data....")
msrvtt_train = [{"source": "msrvtt_train", "id": row["video_id"], "caption": [row["caption"]], "url": row['url'], 'meta': {'start_time': row['start time'], 'end_time': row['end time']}} for i,row in pd.read_parquet(os.path.join(raw_data_dirpath, 'video/msrvtt/train-00000-of-00001-60e50ff5fbbd1bb5.parquet')).iterrows()]
msrvtt_test = [{"source": "msrvtt_test", "id": row["video_id"], "caption": [row["caption"]], "url": row['url'], 'meta': {'start_time': row['start time'], 'end_time': row['end time']}} for i,row in pd.read_parquet(os.path.join(raw_data_dirpath, 'video/msrvtt/val-00000-of-00001-01bacdd7064306bc.parquet')).iterrows()]
print("Video Data Size : ", len(msrvtt_train), len(msrvtt_test))

print("Loading image data....")
id2image = {row['id']: row for row in json.load(open(os.path.join(raw_data_dirpath, 'image/mscoco/captions_train2017.json')))['images']}
coco_caption_train = [{"source": "coco_train", "id": row["image_id"],"url":id2image[row["image_id"]]['coco_url'], "caption":[row["caption"]]} for row in json.load(open(os.path.join(raw_data_dirpath, 'image/mscoco/captions_train2017.json')))['annotations']]
id2image = {row['id']: row for row in json.load(open(os.path.join(raw_data_dirpath, 'image/mscoco/captions_val2017.json')))['images']}
coco_caption_val = [{"source": "coco_val", "id": row["image_id"],"url":id2image[row["image_id"]]['coco_url'], "caption":[row["caption"]]} for row in json.load(open(os.path.join(raw_data_dirpath, 'image/mscoco/captions_val2017.json')))['annotations']]
print("Image Data Size : ", len(coco_caption_train), len(coco_caption_val))


print("Encoding descriptions....")
model = SentenceTransformer("all-MiniLM-L6-v2")
if SPLIT == 'train':
    audio_data= audio_data_train
    pc_data = point_llm_train
    image_data = coco_caption_train
    video_data = msrvtt_train
else:
    audio_data= audio_data_test
    pc_data = point_llm_test
    image_data = coco_caption_val
    video_data = msrvtt_test

audio_encodings = model.encode([x['caption'][0] for x in audio_data], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
pc_encodings = model.encode([x['caption'][0] for x in pc_data], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
image_encodings = model.encode([x['caption'][0] for x in image_data], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
video_encodings = model.encode([x['caption'][0] for x in video_data], convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).cpu()
    
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
    "image": image_data,
    "video": video_data
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
            example = {"id": f"{STRATEGY}_{example_index}", "selection_type":STRATEGY, "q_type": f"mc_{m}", "examples": [],"modalities": []}
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
                    if STRATEGY!='random':
                        index = random.choice(find_top_k_similar_vectors_dot(query_encoding, modality_to_encodings[modality], top_k=30))[0].item()
                    else:
                        index = random.choice(range(len(modality_to_encodings[modality])))
                    example["examples"].append(data_to_sample[index])  
                if frozenset([d_['source'] + str(d_['id']) for d_  in example["examples"]]) not in seen_sets:
                    flag_added = True
                    example_index += 1
                    output.append(example)
                    seen_sets.add(frozenset([d_['source'] + str(d_['id']) for d_  in example["examples"]])) 
json.dump(output, open(f"data/step1/tuples_{STRATEGY}_{SPLIT}.json", "w"))
print("Number of example tuples:", len(output))