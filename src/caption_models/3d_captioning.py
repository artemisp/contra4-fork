import sys
sys.path.append('../cross_modal_baselines/LAVIS')
import os
import json
from tqdm import tqdm
import torch
import argparse
import os
import random
import requests
from io import BytesIO

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

import torchvision.transforms as transforms
from PIL import Image

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.blip_processors import BlipImageEvalProcessor
from lavis.processors.alpro_processors import AlproVideoEvalProcessor
from lavis.processors.ulip_processors import ULIPPCProcessor
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_vicuna_xinstruct import Blip2VicunaXInstruct

from omegaconf import OmegaConf
import copy

selection_type = 'random'


os.environ['CUDA_HOME'] ="/usr/local/cuda/"
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CUDA_HOME']}/lib"
os.system('alias ld="ld -l $LD_LIBRARY_PATH"')
index2letter = lambda i: chr(ord('A')+i)

class PCDataset(BaseDataset):        
    def __init__(self, **kwargs):
        super().__init__(None, None,None,[])
        self.modalities = ['pc']
        self.pc_processor = ULIPPCProcessor()
        self.data = json.load(open('../../data/final_data/test.json'))
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
        test_data = load_json_from_url(pointllm_test1)+load_json_from_url(pointllm_test2)
        all_sample_ids = [sample['object_id'] for sample in test_data]
        self.data = all_sample_ids
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        image = []
        modality = 'pc'
        out = {}
        point_path = f'https://storage.googleapis.com/sfr-ulip-code-release-research/ULIP-Objaverse_triplets/objaverse_pc_parallel/{data}/{data}_8192.npz'
        response = requests.get(point_path)
        response.raise_for_status()  # raise an error if download failed

        # 2. Wrap the content in a BytesIO object
        file_obj = BytesIO(response.content)
        # from pdb import set_trace; set_trace()
        points = np.load(file_obj)['arr_0']

        mod_feats = getattr(self, f"{modality if 'image' not in modality else 'vis'}_processor")(points)
        out['pc'] = mod_feats
        
        out["text_input"] = "Describe the 3D model"    
        out["answer"] = "Some caption"
        out["sample_id"] = data
        out["image_id"] = data
        return out

def collater_fn_with_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return []
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch

class DiscrnXinstruct(Blip2VicunaXInstruct):
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
        ):
        if samples == None or samples == {}:
            return None

        # get batch size
        bs = None
        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]
        if len(set(curr_modalities).intersection(set(samples.keys()))) != len(set(curr_modalities)):
            return ""
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)   
            else:
                bs = len(data)     
            break

        if "text_input" not in samples:
            samples["text_input"] = self.prompt
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]] * bs
        text_input = samples['text_input']

        if not prompt and self.prompt:
            prompt=self.prompt
        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            samples["prompt"] = text_input

        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            self.llm_tokenizer.padding_side = "left"

            text_input = samples['text_input'] if 'prompt' not in samples else samples['prompt']
            if self.postfix:
                text_input = [f'{t}{self.postfix}' for t in text_input]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {text_input[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {text_input[i]}' for i in range(bs)]
            llm_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=self.llm_model.get_input_embeddings()(llm_tokens.input_ids),
                    attention_mask=llm_tokens.attention_mask,
                    do_sample=False,
                    num_beams=num_beams,
                    max_length=max_len,
                    min_length=min_len,
                    repetition_penalty=1.5,
                    # eos_token_id=self.eos_token_id,
                    length_penalty=length_penalty,
                )
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(output_text)
            return output_text

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)
        
        #vizwiz
        output_text = [o if o != "" else "unanswerable" for o in output_text]

        return output_text
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        special_qformer_input_prompt=False
        ):
        self.llm_tokenizer.padding_side = "left"

        if samples == None or samples == {}:
            print("empty samples")
            return 

        if 'modalities' in samples:
            curr_modalities = samples['modalities'][0] if isinstance(samples['modalities'][0], list) else  samples['modalities']
        elif self.joint_video_audio:
            curr_modalities = ["video", "audio"]
        else:
            curr_modalities = [modality for modality in self.modalities if modality in samples]

        
        if len(curr_modalities) == 0:
            print("Model modalities do not match sample modalities.")
            return
        
        if len(set(curr_modalities).intersection(set(samples.keys()))) != len(set(curr_modalities)):
            return ""
            
        # get batch size
        bs = None
        for modality in curr_modalities:
            data = samples[modality]
            if isinstance(data, torch.Tensor):
                bs = data.size(0)
            else:
                bs = len(data)
            break
        
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        elif self.prompt and 'text_input' in samples and '{}' in self.prompt:
            prompt = [self.prompt.format(t) for t in samples["text_input"]]
        elif "text_input" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."            

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]


        if 'discrn' in samples and self.use_caption: ## discriminatory reasoning
            if self.postfix:
                prompt = [f'{t}{self.postfix}' for t in prompt]
            if self.enumerate_inputs:
                prompt = [f'{self.prefix}(a){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]] if self.use_cues else " "}{samples["baseline_captions"][i][0]} (b){Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            else:
                prompt = [f'{self.prefix}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][0]]}{samples["baseline_captions"][i][0] if self.use_cues else " "}{Blip2VicunaXInstruct.MODALITY_TO_CUE[samples["modalities"][i][1]] if self.use_cues else " "}{samples["baseline_captions"][i][1]} {prompt[i]}' for i in range(bs)]
            llm_tokens = self.llm_tokenizer(
                prompt,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        
            with self.maybe_autocast():
                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=llm_tokens.attention_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )
        
            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [o.strip() for o in output_text]
            # print(output)
            return output_text

        query_tokens = {}
        for modality in curr_modalities:
            if not self.projection_only and not getattr(self, f"projection_only_{modality}"):
                query_tokens[modality] = getattr(self, f"{modality}_query_tokens").expand(bs, -1, -1)
        if self.qformer_text_input:
            if self.special_qformer_input_prompt or special_qformer_input_prompt:  
                qformer_prompt = special_qformer_input_prompt if special_qformer_input_prompt else self.special_qformer_input_prompt
                qformer_prompt = [qformer_prompt] * len(prompt)
                if "text_input" in samples.keys():
                    if type(samples["text_input"][0]) == list:
                        qformer_prompt = [qformer_prompt[i].format(*samples["text_input"][i]) for i in range(len(qformer_prompt))]
                    else:
                        qformer_prompt = [qformer_prompt[i].format(samples["text_input"][i]) for i in range(len(qformer_prompt))]
                text_Qformer = self.tokenizer(
                    qformer_prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            elif self.use_describe:
                modality2prompt = {
                    "video": "a short description of the video",
                    "audio": "an audio that shows",
                    "image": "a short image caption",
                    "pc": "a 3d model of"
                }
                qformer_prompt = [modality2prompt[modality] for _ in samples['text_input']]

                text_Qformer = self.tokenizer(
                    qformer_prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            else:
                text_Qformer = self.tokenizer(
                    prompt,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)

            Qformer_atts = {}
            query_atts = {}
            
            for modality in curr_modalities:
                if not  getattr(self, f"projection_only_{modality}"):
                    # B, Token Size
                    query_atts[modality] = torch.ones(query_tokens[modality].size()[:-1], dtype=torch.long).to(self.device)
                    # B, Token Size + Inp Size
                    Qformer_atts[modality] = torch.cat([query_atts[modality],text_Qformer.attention_mask],dim=1)

        embeds = {}
        data_atts = {}
        for modality in curr_modalities:
            data = samples[modality]
            ln = getattr(self, f"{modality}_ln")
            encoder = getattr(self, f"{modality}_encoder")
            if modality == "video" and "clip" in self.video_enc_name:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(2)):
                    this_frame = data[:,:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                        data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
            elif modality == 'audio' and 'beats' in self.audio_enc_name:
                embeds[modality] = []
                data_atts[modality] = []
                for j in range(data.size(1)):
                    this_frame = data[:,j,:,:]
                    with self.maybe_autocast():
                        embeds[modality].append(ln(encoder(this_frame)))
                        if self.shared_qformer:
                            embeds[modality][j] = getattr(self, f"{modality}_encoder_projection")(embeds[modality][j])
                    data_atts[modality].append(torch.ones(embeds[modality][j].size()[:-1], dtype=torch.long).to(self.device))
            else:
                with self.maybe_autocast():
                    embeds[modality] = ln(encoder(data))
                if len(embeds[modality].size()) == 2:
                    embeds[modality] = embeds[modality].unsqueeze(1)
                if self.shared_qformer:
                    with self.maybe_autocast():
                        embeds[modality] = getattr(self, f"{modality}_encoder_projection")(embeds[modality])
                data_atts[modality] = torch.ones(embeds[modality].size()[:-1], dtype=torch.long).to(self.device)
            
        query_outputs = {}
        num = {}
        if self.qformer_text_input:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    with self.maybe_autocast():
                        query_output = getattr(self, f"{modality}_Qformer").bert(
                            text_Qformer.input_ids.repeat(num[modality], 1),
                            attention_mask=Qformer_atts[modality].repeat(num[modality], 1),
                            query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                            encoder_hidden_states=reordered_embeds,
                            encoder_attention_mask=reordered_atts,
                            return_dict=True,
                        )
                    query_outputs[modality] = query_output
                else:
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                        else:
                            with self.maybe_autocast():
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                        continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts[modality],
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32),
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
        else:
            for modality in curr_modalities:
                if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                    num[modality] = len(embeds[modality])
                    bs = embeds[modality][0].shape[0]
                    indices = [j_+r for r,j in enumerate([[i*bs for i in range(num[modality])]]*bs) for j_ in j]
                    reordered_embeds = torch.cat(embeds[modality])[indices]
                    reordered_atts = torch.cat(data_atts[modality])[indices]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        if self.proj_dim != 1:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.mean(1,keepdim=True)).view(bs*num[modality], self.num_query_token, -1)
                        else:
                            query_outputs[modality] = getattr(self, f"{modality}_projection")(reordered_embeds.view(reordered_embeds.shape[0],-1))
                        continue
                    query_output = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality].repeat(num[modality], 1, 1),
                        encoder_hidden_states=reordered_embeds,
                        encoder_attention_mask=reordered_atts,
                        return_dict=True,
                    )
                    query_outputs[modality] = query_output
                else:
                    bs = embeds[modality].shape[0]
                    if self.projection_only or getattr(self, f"projection_only_{modality}"):
                        with self.maybe_autocast():
                            if self.proj_dim != 1:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality].mean(1, keepdim=True)).reshape(bs, self.num_query_token,-1)
                            else:
                                query_outputs[modality] = getattr(self, f"{modality}_projection")(embeds[modality]).reshape(bs, self.num_query_token,-1)
                            continue  
                    query_outputs[modality] = getattr(self, f"{modality}_Qformer").bert(
                        query_embeds=query_tokens[modality],
                        encoder_hidden_states=embeds[modality].to(torch.float32),
                        encoder_attention_mask=data_atts[modality],
                        return_dict=True,
                    )
                    
        inputs_llm = {}
        atts_llm = {}
        enumeration = {}

        for i,modality in enumerate(curr_modalities):
            if modality in Blip2VicunaXInstruct.SEQUENCIAL_MODALITIES and getattr(self, f'{modality}_enc_name') in Blip2VicunaXInstruct.SEQUENCIAL_ENCODERS:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim != 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].unsqueeze(1)).reshape(bs*num[modality], self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs*num, self.num_query_token, -1))
                    inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs, num[modality]*self.num_query_token, -1)
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                # num*bs, num query tokens, llm emb size
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].last_hidden_state[:,:query_tokens[modality].size(1),:]) 
                # bs, num, num query tokens, llm emb size -> bs, num*num query tokens, llm emb size
                inputs_llm[modality] = inputs_llm[modality].reshape(bs, num[modality], self.num_query_token, -1).view(bs,  num[modality]*self.num_query_token, -1)
                atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            else:
                if self.projection_only or getattr(self, f"projection_only_{modality}"):
                    if self.proj_dim == 1:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].mean(-1)).reshape(bs, self.num_query_token, -1)
                    else:
                        inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality].reshape(bs, self.num_query_token, -1))
                    atts_llm[modality] =  torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
                    continue
                inputs_llm[modality] = getattr(self, f"{modality}_llm_proj")(query_outputs[modality]['last_hidden_state'][:,:query_tokens[modality].size(1),:])
                atts_llm[modality] = torch.ones(inputs_llm[modality].size()[:-1], dtype=torch.long).to(self.device)
            if self.enumerate_inputs:
                enumeration[modality] = self.llm_tokenizer(
                [f"{'' if i == 0 else ' '}Scene {chr(ord('A')+i)}." for _ in prompt],
                return_tensors="pt",
                add_special_tokens=False if (i!= 0 or self.prefix) else True
                ).to(self.device)

        ## remove trailing whitespace 
        prompt = [p.strip() for p in prompt]

        if 'dialog' in samples:
            llm_tokens = self.llm_tokenizer(
                [f"{d} {p}" if d else p for d, p in zip(samples['dialog'], prompt)],
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)
        else:
            llm_tokens = self.llm_tokenizer(
                [f"{p}{self.postfix}" for p in prompt] if self.postfix else prompt,
                padding="longest",
                return_tensors="pt",
                add_special_tokens= not self.clean_tokenization
            ).to(self.device)
        bs = llm_tokens.input_ids.shape[0]

        att_list = []
        inp_list = []
        if self.prefix:
            att_list = [self.tokenized_prefix.attention_mask.repeat(bs, 1).to(self.device)]
            inp_list = [self.llm_model.get_input_embeddings()(self.tokenized_prefix.input_ids.to(self.device)).repeat(bs, 1, 1)]            

        if self.joint_video_audio:
            for pos in range(num['video']):
                if self.enumerate_inputs:
                    enumeration_pos = self.llm_tokenizer(
                        [f"{'' if pos == 0 else ' '}({chr(97+pos)}) " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False if (pos!= 0 or self.prefix) else True
                        ).to(self.device)
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration_pos.input_ids)
                    enumeration_atts_llm = enumeration_pos.attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    for modality in ['video', 'audio']:
                        if self.clean_tokenization:
                            if self.prefix or pos > 1 or self.enumerate_inputs or modality == 'audio':
                                att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                                inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
                                continue
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality].view(bs,  num[modality], self.num_query_token)[:, pos, :]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality].view(bs,  num[modality], self.num_query_token, -1)[:, pos, :, :]])
                else:
                    att_list.extend([atts_llm[modality].view(bs, num[modality], self.num_query_token)[:, pos, :]])
                    inp_list.extend([inputs_llm[modality].view(bs, num[modality], self.num_query_token, -1)[:, pos, :, :]])
        else:
            for modality in curr_modalities:
                if self.enumerate_inputs:
                    enumeration_inputs_llm = self.llm_model.get_input_embeddings()(enumeration[modality].input_ids.to(self.device))
                    enumeration_atts_llm = enumeration[modality].attention_mask.to(self.device)
                    inp_list.extend([enumeration_inputs_llm])
                    att_list.extend([enumeration_atts_llm])
                if self.use_cues:
                    if self.clean_tokenization or self.remove_start:
                        if (modality==curr_modalities[0] and not (self.prefix or self.enumerate_inputs)):
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                        else:
                            att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask[:,1:]).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                            inp_list.extend([self.emb_cue[modality][:,1:].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])
                    else:
                        att_list.extend([torch.tensor(self.tokenized_cue[modality].attention_mask).to(self.device).repeat(atts_llm[modality].shape[0], 1), atts_llm[modality]])
                        inp_list.extend([self.emb_cue[modality].to(self.device).repeat(inputs_llm[modality].shape[0], 1, 1), inputs_llm[modality]])

                else:
                    att_list.extend([atts_llm[modality]])
                    inp_list.extend([inputs_llm[modality]])

                if self.add_space:
                    space_tok =  self.llm_tokenizer(
                        [f" " for _ in prompt],
                        return_tensors="pt",
                        add_special_tokens=False
                        )
                    space_inputs_llm = self.llm_model.get_input_embeddings()(space_tok.input_ids.to(self.device))
                    space_atts_llm = space_tok.attention_mask.to(self.device)
                    inp_list.extend([space_inputs_llm])
                    att_list.extend([space_atts_llm])

        att_list.append(llm_tokens.attention_mask)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
        inp_list.append(inputs_embeds)
        attention_mask = torch.cat(att_list, dim=1)
        inputs_embeds = torch.cat(inp_list, dim=1)
        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [o.strip() for o in output_text]
        print(output_text)
        
        return output_text

def setup_seeds(seed):
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class('runner_base')

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    import os

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '54325'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '4'
    job_id = now()
    os.makedirs("results/xinstructblip_annotated", exist_ok=True)
    cfg = { "run":{'task': 'gqa', 'lr_sched': 'linear_warmup_cosine_lr', 'init_lr': 0.0001, 'min_lr': 1e-05, 'warmup_lr': 1e-08, 'warmup_steps': 1000, 'weight_decay': 0.05, 'max_epoch': 100, 'batch_size_train': 1, 'batch_size_eval': 1, 'num_workers': 4, 'accum_grad_iters': 1, 'max_len': 5, 'min_len': 1, 'num_beams': 5, 'inference_method': 'generate', 'seed': 42, 'output_dir': 'results/xinstructblip_annotated', 'amp': True, 'evaluate': True, 'train_splits': ['train'], 'valid_splits': ['val'], 'test_splits': ['test'], 'device': 'cuda', 'world_size': 16, 'dist_url': 'env://', 'distributed': True, 'find_unused_parameters': True}}
    options = {"options": None, "cfg_path": "../cross_modal_baselines/LAVIS/lavis/projects/xinstruct_blip/eval/vicuna7b/pc/objaverse_captioning.yaml"}
    cfg = Config(OmegaConf.merge(options,cfg))
     
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(42)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    task = tasks.setup_task(cfg)

    datasets = {
        "val": {"val": PCDataset()},
    }
    model = DiscrnXinstruct.from_pretrained(model_type='vicuna7b')
    # model.enumerate_inputs = True
    # model.add_space = True
    # model.special_qformer_input_prompt = "a short description"
    # model.postfix = ""
    # model.prefix =  "This is a multiple choice question. Carefully read the question and choose the correct answer from the provided choices.\n"
    # model.clean_tokenization = False
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
