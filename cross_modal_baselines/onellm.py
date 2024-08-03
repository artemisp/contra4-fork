"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import sys
sys.path.append('OneLLM')
import os
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import numpy as np
from model.meta import MetaModel
from data.conversation_lib import conv_templates
from data import video_utils
from data.data_utils import pc_norm
from data.conversation_lib import conv_templates
from data.data_utils import make_audio_features
import torchvision.transforms as transforms
from PIL import Image

def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None] #[1, 128, 1024]
    return fbank

def load_pc(pc_path):
    pc = np.load(pc_path)['arr_0']
    pc = torch.tensor(pc)
    pc = pc.repeat(2,2)
    pc = pc_norm(pc)
    return pc

T_resized_center_crop = transforms.Compose([
transforms.Resize(
    224, interpolation=transforms.InterpolationMode.BICUBIC
),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = T_resized_center_crop(image)
    return image


index2letter = lambda i: chr(ord('A') + i)

class CaptionDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = json.load(open('../data/discrn_balanced.json'))
        self.key2path = json.load(open('../data/data2path.json'))
        self.key2path["objaverse_pointllm_val"] = self.key2path["objaverse_val"]
        self.key2path["audiocaps_mm_caption_val"] = self.key2path["audiocaps_val"]
              

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        image = []
        for modality,example in zip(data["modalities"],data["examples"]):
            if modality == "audio":
                image.append(load_audio(os.path.join(self.key2path[example["source"]]["directory"], self.key2path[example["source"]]["key2path"][str(example["id"])])))
            elif modality == "pc":
                image.append(load_pc(os.path.join(self.key2path[example["source"]]["directory"], self.key2path[example["source"]]["key2path"][example["id"]])))
            elif modality == "image":
                image.append(load_image(os.path.join(self.key2path[example["source"]]["directory"], self.key2path[example["source"]]["key2path"][example["id"]])))
            elif modality == "video":
                image.append(load_video(os.path.join(self.key2path[example["source"]]["directory"], self.key2path[example["source"]]["key2path"][example["id"]])))
        
        question_id = data['id']
        question = data['questions'][0]
        question+= "Choose from: " + ", ".join([f'Scene {index2letter(i)}' for i,c in enumerate(example)])
        answer = data['answers'][0]
        data['modalities'] = [m if m !='pc' else 'point' for m in data['modalities']]
        print(data["modalities"])
        return image, data["modalities"], question, question_id, answer

import model
from model.LLM.onellm import Transformer
from model.tokenizer import Tokenizer

class DisCRnTransformer(Transformer):
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None, modal='image', tokenizer=Tokenizer(model_path='/export/home/OneLLM/config/llama2/tokenizer.model')):
            _bsz, seqlen = tokens.shape
            if start_pos == 0:
                # kv cache will not re-allocate if size is unchanged
                self._allocate_kv_cache(_bsz)
            h = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            if image is not None:
                if isinstance(image, list):
                    h_bos, h_caption = h[:, :1], h[:, 1:]
                    for i,im in enumerate(image):
                        image_tokens = self.encode_image(im, modal[i][0])
                        input_tokens =  self.tok_embeddings(torch.tensor(tokenizer.encode(f"Scene {chr(ord('A') + i)}: ", bos=False, eos=False)).unsqueeze(0).to(h.device))
                        self.cache_image_words = image_tokens.shape[1]
                        h_bos = torch.cat((h_bos, input_tokens, self.start_tag[modal[i][0]].repeat(_bsz, 1, 1), image_tokens, self.end_tag[modal[i][0]].repeat(_bsz, 1, 1)), dim=1)
                    h = torch.cat((h_bos, h_caption), dim=1)
                    seqlen = h.shape[1]
                    freqs_cis = self.freqs_cis[0: seqlen]

                else:
                    modal = modal[0]
                    h_bos, h_caption = h[:, :1], h[:, 1:]
                    image_tokens = self.encode_image(image, modal)
                    self.cache_image_words = image_tokens.shape[1]
                    h = torch.cat((h_bos, self.start_tag[modal].repeat(_bsz, 1, 1), image_tokens, self.end_tag[modal].repeat(_bsz, 1, 1), h_caption), dim=1)
                    seqlen = h.shape[1]
                    freqs_cis = self.freqs_cis[0: seqlen]
            else:
                if start_pos == 0:
                    self.cache_image_words = 0
                    freqs_cis = self.freqs_cis[0: seqlen]
                else:
                    # if image was not None when start_pos=0,
                    # the offset should be added to start_pos within later forward_inference calls
                    start_pos = start_pos + self.cache_image_words
                    freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

            # freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
            h = self.norm(h)
            output = self.output(h[:, -1, :])  # only compute last logits
            return output.float()
Transformer = DisCRnTransformer
    
if __name__ == "__main__":
    pretrained_path = "/export/home/OneLLM/OneLLM-7B/consolidated.00-of-01.pth"
    answer_path = "results/onellm.json"
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)    
    
    mp.set_start_method("spawn")
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:23563")
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
        model.meta.LLM.__dict__["onellm"].Transformer = DisCRnTransformer
        model = MetaModel("onellm", "/export/home/OneLLM/config/llama2/7B.json", None, "/export/home/OneLLM/config/llama2/tokenizer.model")
    
    print("Loading pretrained weights ...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")

    def multi_modal_generate(images, inps, modal): 
        images = [im.cuda().to(target_dtype) for im in images]           

        prompts = []
        for inp in inps:
            conv = conv_templates["v1"].copy()        
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())

        with torch.cuda.amp.autocast(dtype=target_dtype):

            responses = model.generate(prompts, images, 5, temperature=0.0, top_p=0.9, modal=modal)
            outputs = []
            for response, prompt in zip(responses, prompts):
                response = response[len(prompt):].split('###')[0]
                response = response.replace("Assistant:","").strip()
                print(response)
                outputs.append(response)
        return outputs

    result = {}
    print("Starting...")
    dataset = CaptionDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    predictions = []
    correct = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, modalities, questions, question_ids, answers = data
            preds = multi_modal_generate(images, questions,modal=modalities)
            for question, pred, question_id, answer in zip(questions, preds, question_ids, answers):
                predictions.append({'question_id': question_id, 'answer': pred, 'gt_answer': answer})
                pred = pred.strip().lower()
                answer = answer.strip().lower()
                if (pred in answer) or (answer in pred):
                    correct += 1
    
    acc = float(correct) / len(dataset)
    print('Accuracy:', acc) 

    with open(answer_path, 'w') as f:
        json.dump(predictions, f)  