import torch
from PIL import Image
import os
from tqdm import tqdm
import json
import evaluate
from lavis.models import load_model_and_preprocess
from densely_captioned_images.dataset.impl import get_complete_dataset_with_settings, DenseCaptionedDataset

os.makedirs("predictions", exist_ok=True)

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)


DENSE_CAPTION_IMAGE_ROOT = "/export/einstein-vision/DCI/data/densely_captioned_images/photos"
COCO_ROOT = "/export/share/datasets/vision/coco/images"

# Load the meteor metric
meteor = evaluate.load('meteor')


test_ds = get_complete_dataset_with_settings('test', load_subcaptions=False, load_base_image=True)
valid_ds = get_complete_dataset_with_settings('valid', load_subcaptions=False, load_base_image=True)
# from pdb import set_trace; set_trace()

id2gt = {d[0]['image'].item(): d[0]["caption"].split('\n') for d in test_ds}
test_dci = []
for im in tqdm(test_ds):
    img = Image.open(os.path.join(DENSE_CAPTION_IMAGE_ROOT, im[0]["image"].item()))
    image = vis_processors["eval"](img).unsqueeze(0).to(device)
    out = model.generate({"image": image, "prompt": "Describe the image in detail."})
    test_dci.append({"id": im[0]["image"].item(), "caption": out[0]})
json.dump(test_dci, open("predictions/results_dci_test_instructblip.json", "w"))

res = json.load(open("predictions/results_dci_test_instructblip.json", "r"))
score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[r['id']][:2] for r in res])
print(f"DCI Test METEOR score: {score['meteor']}")

val_dci = []
id2gt = {d[0]['image'].item(): d[0]["caption"].split('\n') for d in valid_ds}
for im in tqdm(valid_ds):
    img = Image.open(os.path.join(DENSE_CAPTION_IMAGE_ROOT, im[0]["image"].item()))
    image = vis_processors["eval"](img).unsqueeze(0).to(device)
    out = model.generate({"image": image, "prompt": "Describe the image in detail."})
    val_dci.append({"id": im[0]["image"].item(), "caption": out[0]})
json.dump(val_dci, open("predictions/results_dci_val_instructblip.json", "w"))
res = json.load(open("predictions/results_dci_val_instructblip.json", "r"))

score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[r['id']][:2] for r in res])
print(f"DCI Val METEOR score: {score['meteor']}")

url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json'
response = requests.get(url)
if response.status_code == 200:
    coco_caption_val = json.loads(response.text)
else:
    print("Failed to retrieve the COCO val. Status code:", response.status_code)
coco_ds = [{"source":"coco_val", "id":item['image'], "caption":item['caption']} for item in coco_caption_val]
id2gt = {d["id"]: d['caption'] for d in coco_ds}

val_coco = []
for im in tqdm(coco_ds):
    img = Image.open(os.path.join(COCO_ROOT, f'val2014/COCO_val2014_{im["id"]:012d}.jpg'))
    image = vis_processors["eval"](img).unsqueeze(0).to(device)
    out = model.generate({"image": image, "prompt": "Describe the image in detail."})
    val_coco.append({"id": im["id"], "caption": out[0]})
json.dump(val_coco, open("predictions/results_coco_val_instructblip.json", "w"))
res = json.load(open("predictions/results_coco_val_instructblip.json", "r"))

score = meteor.compute(predictions=[r['caption'] for r in res], references=[id2gt[r['id']] for r in res])
print(f"COCO Val METEOR score: {score['meteor']}")