from wrapper import PengiWrapper as Pengi
from tqdm import tqdm
import pandas as pd
import json
from torch.utils.data import DataLoader
import os
import evaluate

# Load the meteor metric
meteor = evaluate.load('meteor')


os.makedirs("predictions", exist_ok=True)

AUDIOCAPS_AUDIO_ROOT = "/export/einstein-vision/audio_datasets/audiocaps/AUDIOCAPS_32000Hz/audio"
CLOTHOV1_AUDIO_ROOT = "/export/einstein-vision/audio_datasets/clothov2/CLOTHO_v2.1/clotho_audio_files/validation"
CLOTHOV2_AUDIO_ROOT = "/export/einstein-vision/audio_datasets/clothov2/CLOTHO_v2.1/clotho_audio_files/evaluation"

def get_audiocaps_path(self, ann):
    if 'end_seconds' not in ann:
        ann['start_seconds'] = float(ann['start_time'])
        ann['end_seconds'] = ann['start_seconds'] + 10.0
    return os.path.join(AUDIOCAPS_AUDIO_ROOT, ann['youtube_id'] + '_{}.flac'.format(int(ann['start_seconds'])))

def get_clothov1_path(self, ann):
    return os.path.join(CLOTHOV1_AUDIO_ROOT, ann['file_name'])

def get_clothov2_path(self, ann):
    return os.path.join(CLOTHOV2_AUDIO_ROOT, ann['file_name'])

print("Loading audio data....")
audiocaps_dataset_val = [row for i,row in pd.read_csv('https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv').iterrows()]
clothov1_dataset_val = [row for i,row in pd.read_csv('https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv').iterrows()]
clothov2_dataset_val = [row for i,row in pd.read_csv('https://zenodo.org/record/4783391/files/clotho_captions_validation.csv').iterrows()]


pengi = Pengi(config="base")
id2gt = {d['audiocap_id']: [d["caption"]] for d in audiocaps_dataset_val}
res = []
dl = DataLoader(audiocaps_dataset_val, batch_size=4, shuffle=False)

for batch in tqdm(dl):
    generated_response = pengi.describe(
                                        audio_paths=[get_audiocaps_path(b) for b in batch],
                                        max_len=30, 
                                        beam_size=3, 
                                        temperature=1.0, 
                                        stop_token=' <|endoftext|>',
                                        )
    print(generated_response)
    for i in range(len(generated_response)):
        res.append({"id": str(batch["audiocap_id"][i]), "caption": generated_response[i], "gt_caption": id2gt[batch["audiocap_id"][i]]})
with open("predictions/results_audiocaps_pengi.json", "w") as f:
    json.dump(res, f)

res = json.load( open("predictions/results_audiocaps_pengi.json"))
score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[int(r['id'])] for r in res])
print(f"Audiocaps METEOR score: {score['meteor']}")

id2gt = {d['file_name']: [d["caption_1"], d["caption_2"], d["caption_3"], d["caption_4"], d["caption_5"]] for d in clothov1_dataset_val}
res = []
dl = DataLoader(clothov1_dataset_val, batch_size=4, shuffle=False)

for batch in tqdm(dl):
    generated_response = pengi.describe(
                                        audio_paths=[get_clothov1_path(b) for b in batch],
                                        max_len=30, 
                                        beam_size=3, 
                                        temperature=1.0, 
                                        stop_token=' <|endoftext|>',
                                        )
    print(generated_response)
    for i in range(len(generated_response)):
        res.append({"id": str(batch["file_name"][i]), "caption": generated_response[i], "gt_caption": id2gt[batch["sound_id"][i]]})
with open("predictions/results_clothov1_pengi.json", "w") as f:
    json.dump(res, f)
res = json.load( open("predictions/results_clothov1_pengi.json"))
score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[r['id']] for r in res])
print(f"Clotho (v1) METEOR score: {score['meteor']}")


id2gt = {d['file_name']: [d["caption_1"], d["caption_2"], d["caption_3"], d["caption_4"], d["caption_5"]] for d in clothov2_dataset_val}
res = []
dl = DataLoader(clothov2_dataset_val, batch_size=4, shuffle=False)

for batch in tqdm(dl):
    generated_response = pengi.describe(
                                        audio_paths=[get_clothov1_path(b) for b in batch],
                                        max_len=30, 
                                        beam_size=3, 
                                        temperature=1.0, 
                                        stop_token=' <|endoftext|>',
                                        )
    print(generated_response)
    for i in range(len(generated_response)):
        res.append({"id": str(batch["file_name"][i]), "caption": generated_response[i], "gt_caption": id2gt[batch["sound_id"][i]]})
with open("predictions/results_clothov2.json", "w") as f:
    json.dump(res, f)
res = json.load( open("predictions/results_clothov2_pengi.json"))
score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[r['id']]  for r in res])
print(f"Clotho (v2) METEOR score: {score['meteor']}")
