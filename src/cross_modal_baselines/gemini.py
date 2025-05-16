import base64
from google import genai
from google.genai import types
import json
from torch.utils.data import Dataset
import os

# Function to read and encode media files
def encode_media(file_path):
    with open(file_path, "rb") as file:
        encoded_content = base64.b64encode(file.read()).decode("utf-8")
    return encoded_content

encode_modality = lambda x: base64.b64encode(open(x, "rb").read()).decode("utf-8")


i2option = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

class CaptionDataset(Dataset):
    def __init__(self, path=None) -> None:
        super().__init__()
        if  path is None:
            self.data = json.load(open('../../data/final_data/test.json'))
        else:
            self.data = json.load(open(path))
       
        self.coco_dir = os.environ['COCO_DIR']
        self.audiocaps_dir = os.environ['AUDIOCAPS_DIR']
        self.msrvtt_dir = os.environ['MSRVTT_DIR']
        self.clotho_dir = os.environ['CLOTHO_DIR']
        self.objaverse_dir = os.environ['OBJAVERSE_DIR']
        
        for item in self.data:
            # e.g. ['audio', 'audio'] or ['image', 'video'] or ['pc'] ...
            # store it so you can access it quickly later.
            item["group_key"] = tuple(item["modalities"])
        # only keep audio,video, image
        self.data = [item for item in self.data if 'pc' not in item["group_key"]]


    def __len__(self):
        return len(self.data)

    def get_group_key(self, index):
        return self.data[index]["group_key"]

    def __getitem__(self, index):
        data = self.data[index]
        image = {}
        try:
            for modality,example in zip(data["modalities"],data["examples"]):
                if modality == "audio":
                    if example["source"] == "audiocaps":
                        image['audio'] = encode_modality(os.path.join(self.audiocaps_dir, f'{example["id"]}.wav'))
                        # image.append(load_audio(os.path.join(self.audiocaps_dir, f'{example["id"]}.wav')))
                    elif example["source"] == "clotho":
                        image['audio'] = encode_modality(os.path.join(self.clotho_dir, f'{example["id"]}'))
                        # image.append(load_audio(os.path.join(self.clotho_dir, f'{example["id"]}')))
                elif modality == "pc":
                    print("Error!!")
                    # image.append(load_pc(os.path.join(self.objaverse_dir, f'{example["id"]}_8192.npz')))
                elif modality == "image":
                    image['image'] = encode_modality(os.path.join(self.coco_dir, f'{str(example["id"]).zfill(12)}.jpg'))
                    # image.append(load_image(os.path.join(self.coco_dir, f'{str(example["id"]).zfill(12)}.jpg')))
                elif modality == "video":
                    image['video'] = encode_modality(os.path.join(self.msrvtt_dir, f'{example["id"]}.mp4'))
                    # image.append(load_video(os.path.join(self.msrvtt_dir, f'{example["id"]}.mp4')))        
            question_id = data['id']
            question = data['question'] if 'question' in data else data['questions'][0]
            question += " Choose from: " + ", ".join([f'Scene {i2option[i]}' for i,m in enumerate(image.keys())])
            # question+= " Choose from: " + ", ".join([f'{m}' for i,m in enumerate(image.keys())])
            answer = data['answer'] if 'answer' in data else data['answers'][0]
        except Exception as e:
            print("Error loading data", data['id'], e)
            dl_fail.append(data['id'])
            return None
        print(data["modalities"])
        return image, data["modalities"], question, question_id, answer


dataset = CaptionDataset()
outputs = []
exist = set([d['question_id'] for d in outputs])
model = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
import time
from tqdm import tqdm
for i in tqdm(range(len(dataset))):
    image, modalities, question, question_id, answer = dataset[i]
    if question_id in exist:
        continue
    inputs = {
        "prompt": question,
    }
    for modality in modalities:
        if modality == "audio":
            # inputs[modality] = encode_media(image[modality], "audio/wav")
            inputs[modality] = types.Part.from_bytes(
                data=image[modality],
                mime_type='audio/wav',
            )
        elif modality == "video":
            # inputs[modality] = encode_media(image[modality], "video/mp4")
            inputs[modality] = types.Part.from_bytes(
                data=image[modality],
                mime_type='video/mp4',
            )
        elif modality == "image":
            # inputs[modality] = encode_media(image[modality], "image/jpeg")
            inputs[modality] = types.Part.from_bytes(
                data=image[modality],
                mime_type='image/jpeg',
            )
    
    contents=[] 
    for j, m in enumerate(inputs):
        contents.extend([f"Scene {i2option[j]}: ", inputs[m]])
    contents+=[question, 'Answer:']
    response = model.models.generate_content(model='gemini-2.0-flash-exp', contents=contents)
    outputs.append({
        "question_id": question_id,
        "pred_ans": response.text,
        "gt_ans": answer,})
    print(response)
    time.sleep(10)
    if i%50 == 0:
        json.dump(outputs, open('results/gemini20exp2.json', 'w'))
json.dump(outputs, open('results/gemini20exp2.json', 'w'))
    
