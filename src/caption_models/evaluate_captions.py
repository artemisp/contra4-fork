import json
import os
import re
import torch
from tqdm import tqdm
import argparse
import pickle
import json
import os
import random
import evaluate

meteor = evaluate.load('meteor')

def load_json_from_dirs(dirs):
    data = []
    for d in dirs:
        for f in os.listdir(d):
            data.extend([json.loads(l) for l in open(os.path.join(d, f)).readlines()])
    data = {d['index']: d['generated_text'][0] for d in data}
    return data
modality2predcap = {
    'image': load_json_from_dirs(['results/image_captions.pkl']),
    'video': load_json_from_dirs(['results/train_val_video_captions_random.pkl']),
    'audio': load_json_from_dirs(['results/audiocaps_captions_random.pkl', 'results/clotho_captions_random.pkl']),
    'pc': {d['image_id']:d['caption'] for d in json.load(open('results/3d_captions_pointllm/objaverse_test.json'))}
    
}
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str, default='../../data/final_data/test.json', help='data path')
args = parser.parse_args()

data = json.load(open(args.data_path))

for modality in modality2predcap:
    preds = []
    gts = []
    for d in tqdm(data):
        modalities = d['modalities']
        for i, example in enumerate(d['examples']):
            if modalities[i] != modality:
                continue
            try:
                pred_caption = modality2predcap[modality][example['id']]
            except:
                # try:
                #     pred_caption = modality2predcap[modality][str(example['id'])]
                # except:
                try:
                    pred_caption = modality2predcap[modality][str(example['id']).replace('.wav', '')]
                    
                except:
                    print(f'Could not find caption for {example["id"]}')
                    continue
            gt_caption = example['caption']
            preds.append(pred_caption)
            gts.append(gt_caption)
    results = meteor.compute(predictions=preds, references=gts)
    print(f'Modality: {modality}, METEOR: {results["meteor"]}')