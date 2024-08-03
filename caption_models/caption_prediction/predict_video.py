import torch
from PIL import Image
import os
from tqdm import tqdm
import json
import evaluate
from lavis.models import load_model_and_preprocess
from densely_captioned_images.dataset.impl import get_complete_dataset_with_settings, DenseCaptionedDataset
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
import requests
import zipfile
import io
os.makedirs("predictions", exist_ok=True)

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# loads InstructBLIP model
model, _, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
vis_processors = AlproVideoEvalProcessor(n_frms=4, image_size=224)


MSRVTT_ROOT = "/export/share/datasets/vision_language/msrvtt/videos"
CHARADES_ROOT = "/export/video-language-dataset/data/charade/videos"

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
charades_test = [{"source":"charades_v1_test", "id": r['id'],"caption": f"This scene takes place in a {r['descriptions']}. " + '.'.join(r['scene'].split('.')[:2])} for r in tqdm(charades_data)]
url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/cap_test.json'
response = requests.get(url)
if response.status_code == 200:
    msrvtt_test = json.loads(response.text)
else:
    print("Failed to retrieve the MSRVTT test. Status code:", response.status_code)
msrvtt_test = [{"source": "msrvtt_test", "id": item["video"], "caption": item["caption"]} for item in msrvtt_test]



# Load the meteor metric
meteor = evaluate.load('meteor')


id2gt = {d['id']: d["caption"] for d in charades_test}
test_charades = []
for im in tqdm(charades_test):
    img = os.path.join(CHARADES_ROOT, im['id']+'.mp4')
    image = vis_processors(img).unsqueeze(0).to(device)
    out = model.generate({"image": image, "prompt": "Describe the video."})
    test_charades.append({"id": im["id"], "caption": out[0]})
json.dump(test_charades, open("predictions/results_charades_test_instructblip.json", "w"))

res = json.load(open("predictions/results_charades_test_instructblip.json", "r"))

score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[r['id']] for r in res])
print(f"Charades Test METEOR score: {score['meteor']}")

test_msrvtt = []
id2gt ={d['id'].item(): d["caption"] for d  in msrvtt_tests}
for im in tqdm(msrvtt_test):
    img = os.path.join(MSRVTT_ROOT, im['id']+'.mp4')
    image = vis_processors(img).unsqueeze(0).to(device)
    out = model.generate({"image": image, "prompt": "Describe the video."})
    test_msrvtt.append({"id": im['id'], "caption": out[0]})
json.dump(test_msrvtt, open("predictions/results_msrvtt_instructblip.json", "w"))
res = json.load(open("predictions/results_msrvtt_instructblip.json", "r"))

score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[r['id']] for r in res])
print(f"MSRVTT Test METEOR score: {score['meteor']}")

