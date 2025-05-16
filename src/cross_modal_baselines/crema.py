import sys
sys.path.append('CREMA')
import os
import json
from tqdm import tqdm
import torch
import argparse
import os
import random

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
from lavis.processors.blip_processors import BlipImageEvalProcessor, BlipVideoBaseProcessor, ToUint8, ToTHWC
from lavis.common.registry import registry
from lavis.models.crema_models.crema import CREMA
import copy
from omegaconf import OmegaConf
from lavis.processors import transforms_video
from lavis.common.registry import registry
from lavis.datasets.data_utils import load_video
from torchvision import transforms



class BlipVideoEvalProcessor(BlipVideoBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, n_frms=4):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )
        self.n_frms = n_frms

    def __call__(self, vpath, clip_proposal=None):
        clip, indices, fps = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="uniform",
            clip_proposal=clip_proposal
        )

        return self.transform(clip), indices, fps

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", 4)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)
    



index2letter = lambda i: chr(ord('A') + i)
class Contra4Dataset(BaseDataset):        
    def __init__(self, **kwargs):
        super().__init__(None, None,None,[])
        self.data = json.load(open('../../data/final_data/test.json'))
        self.coco_dir = os.environ['COCO_DIR']
        self.audiocaps_dir = os.environ['AUDIOCAPS_DIR']
        self.msrvtt_dir = os.environ['MSRVTT_DIR']
        self.clotho_dir = os.environ['CLOTHO_DIR']
        self.objaverse_dir = os.environ['OBJAVERSE_DIR']
        
        self.modalities = ['video', 'audio', 'images', 'pc']
        self.audio_processor = BeatsAudioProcessor(model_name='iter3',  sampling_rate=16000, n_frames=2, frame_length=512, is_eval=False)
        self.pc_processor = lambda x: x
        self.video_processor = BlipVideoEvalProcessor()
        self.vis_processor = BlipImageEvalProcessor(image_size=224)
        self.voxel_root = './3D-LLM/3DLanguage_data/ChatCaptioner_based/gen_features/data/features_blip/points'
        self.pc_feat_root = './3D-LLM/3DLanguage_data/ChatCaptioner_based/gen_features/data/features_blip/features'
              


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        image = []
        for example,modality in zip(data["examples"], data['modalities']):
            if modality == "audio":
                if example["source"] == "audiocaps":
                    file = os.path.join(self.audiocaps_dir, f'{example["id"]}.wav')
                elif example["source"] == "clotho":
                        file = os.path.join(self.clotho_dir, f'{example["id"]}')
            elif modality == "pc":
                    file = os.path.join(self.objaverse_dir, f'{example["id"]}_8192.npz')
            elif modality == "image":
                    file = Image.open(os.path.join(self.coco_dir, f'{str(example["id"]).zfill(12)}.jpg')).convert('RGB')
            elif modality == "video":
                    file = os.path.join(self.msrvtt_dir, f'{example["id"]}.mp4')                 
            mod_feats = getattr(self, f"{modality if 'image' not in modality else 'vis'}_processor")(file)
            if modality  == "video":    
                rgb, indices, fps = mod_feats
                data['video'] = rgb.to(torch.float32)
            elif modality == "image":
                data['rgb'] = mod_feats
            elif modality == "pc":
                pc_feat = torch.load(os.path.join(self.pc_feat_root, f"{str(example['id'])}_outside.pt"), map_location="cpu")  # [N, 1408]
                if isinstance(pc_feat, np.ndarray):
                    pc_feat = torch.tensor(pc_feat).float()
                pc = np.load(os.path.join(self.voxel_root, f"{str(example['id'])}_outside.npy"))
                pc = torch.tensor(pc).float().cpu()
                # sample 10000 points: [N, 1408] -> [10000, 1408]
                if pc_feat.shape[0] > 5000:
                    idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:5000])[1]
                    pc_feat = pc_feat[idxes]
                    pc = pc[idxes]
                else:
                    pc_feat = torch.cat([pc_feat, torch.zeros(5000 - pc_feat.shape[0], pc_feat.shape[-1])], dim=0)
                    # pc_feat = torch.cat([pc_feat, torch.zeros(5000 - pc_feat.shape[0], 1408)], dim=0)
                    pc = pc_feat
                    # pc = torch.cat([pc.mean(0), torch.zeros(5000 - pc.shape[0], 3)], dim=0)

                data["pc_feat"] = pc_feat
                data['pc'] = pc
            elif modality == "audio":
                data["audio"] = mod_feats.to(torch.float32)
        data["text_input"] = data['question'] if 'question' in data else data["questions"][0]
        data["text_input"]+= " Choose from: " + ", ".join([f'Scene {index2letter(i)}' for i,c in enumerate(data['modalities'])])
        data["text_input"] += " Answer:"
        data['qa_input'] = data['text_input']
        data["answer"] = data['answer'] if 'answer' in data else data["answers"][0]
        data["qa_output"] = data["answer"]
        data["direct_answers"] = [data["answer"]]
        data["question_id"] = data["id"]
        return data
    
    def collater(self, samples):
        # Filter out None samples
        # samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        # if not samples:
        #     return {"question_id":"error", "direct_answers": "error"}
        collated_dict = {}
        keys = samples[0].keys() # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            # If the value type for the key is torch.Tensor, stack them else return list
            collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        return collated_dict

class Contra4CREMA(CREMA):
    def __init__( self, img_size=224, drop_path_rate=0,
        use_grad_checkpoint=False, vit_precision="fp16", freeze_vit=True,
        num_query_token=32, t5_model="google/flan-t5-xl", prompt="",
        max_txt_len=32, frame_num=8, answer_num=5, apply_lemmatizer=False, 
        task='concate',
        modalities='rgb',
        downstream_task='mcqa', # caption / oeqa / mcqa
        lora_rank=64,
        lora_layer=None,
        lora_dropout=0.1):
        modalities = "_".join(["rgb","audio","pc"])
        self.ANS_MAP_INV = {0:'A', 1: 'B', 2:'C', 3: 'D', 4:'E'}
        super().__init__(img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit, num_query_token, t5_model, prompt, max_txt_len, frame_num, answer_num, apply_lemmatizer, task, modalities, downstream_task, lora_rank, lora_layer, lora_dropout)
    def encode_input(self, input, modality, training=True):

        ln = getattr(self, f"ln_{modality if 'video' != modality else 'rgb'}")

        if modality in ['rgb', 'depth', 'flow', 'norm', 'video']:
            modality = 'visual'
        if modality in ['audio']:
            modality = 'audio'
        if modality in ['pc']:
            modality = 'pc'

        encoder = getattr(self, f"{modality}_encoder")

        if modality == 'visual':
            b, t, c, w, h = input.shape     
            input = input.reshape(-1, c, w, h)
            if training:
                image_embeds = ln(encoder(input))
            else:
                with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
                    image_embeds = ln(encoder(input))
            _, n, _ = image_embeds.shape
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(input.device) # bt n c
            return image_embeds, image_atts
        
        if modality == 'audio':
            embeds, atts = [], []
            for j in range(input.size(1)):
                this_frame = input[:,j,:,:]
                if training:
                    embeds.append(encoder(this_frame))
                else:
                    with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu"))):
                        embeds.append(encoder(this_frame))
                atts.append(torch.ones(embeds[j].size()[:-1], dtype=torch.long).to(input.device))
            
            # print('here', len(embeds), embeds[0].shape) # 2, 3, 256, 768
            embeds = torch.stack(embeds, dim=1)
            # print('audio_embeds 1', embeds.shape) # 3, 2, 256, 768
            atts = torch.stack(atts, dim=1)
            embeds = self.projection_audio(embeds) # 3, 2, 256, 1408
            embeds = ln(embeds.reshape(-1, embeds.shape[-2], embeds.shape[-1]))
            atts = atts.reshape(-1, atts.shape[-1])

            return embeds, atts
        
        if modality == 'pc':
            # use pre-extracted features
            pass
            #return embeds, atts
    
    def get_qformer_embedding(self, embeds, atts, device, modality, bs):

        project = getattr(self, f"t5_proj_{modality if modality != 'video' else 'rgb'}")
        query_tokens = getattr(self, f"query_tokens_{modality if modality != 'video' else 'rgb'}")
        query_tokens = query_tokens.expand(embeds.shape[0], -1, -1)
        # if modality =='video':
        #     modality ='rgb'
        
        skip_flag =""
        modality_ = modality + skip_flag

        query_output = self.Qformer.bert(
            query_embeds=query_tokens, encoder_hidden_states=embeds,
            encoder_attention_mask=atts, return_dict=True, modular=modality_ if modality != 'video' else 'rgb')
        
        query = query_output.last_hidden_state.clone()
        inputs_t5 = project(query_output.last_hidden_state)

        if modality in ['rgb', 'depth', 'flow', 'norm', 'video']:
            # inputs_t5 = inputs_t5.reshape(-1, self.frame_num, inputs_t5.shape[-2], inputs_t5.shape[-1])
            inputs_t5 = inputs_t5.reshape(bs, -1, inputs_t5.shape[-2], inputs_t5.shape[-1])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        if modality in ['audio']:
            inputs_t5 = inputs_t5.reshape(bs, -1, inputs_t5.shape[-2], inputs_t5.shape[-1])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        if modality in ['pc']:
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        return inputs_t5, atts_t5, query
    
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
        if 'qa_input' not in samples:
            return ""
        if isinstance(samples["qa_input"], str):
            samples["qa_input"] = [samples["qa_input"]]
        
        text_input = samples["qa_input"]
        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )['output_text']

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)
    
        output_text = [o if o != "" else "unanswerable" for o in output_text]
        return output_text
    
    def generate(self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5, max_length=30,
        min_length=1, top_p=0.9,
        repetition_penalty=1.0, length_penalty=1.0,
        num_captions=1, temperature=1,):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # if "question_id" not in samples:
        #     return {"output_text": "error", "answer": "error", "question_id": "error", "qid": "error"}
        out = {}
        qid = samples['question_id']
        qa_text = samples['qa_input']
        answer = samples['qa_output']
        b = len(qa_text)
        if 'modalities' in samples:
            self.modalities = [m if m!='image' else 'rgb' for m in samples['modalities'][0]] # assumes batch size=1

        input_embed_dict, input_att_dict = {}, {}

        for modal in self.modalities:
            
            input = samples[modal]
            # visual modality pre-process
            if modal in ['rgb', 'depth', 'norm', 'flow', "video"]:
                if input.shape[1] == 3:
                    if len(input.shape) == 4:
                        input = input.unsqueeze(2)
                    input = input.permute(0, 2, 1, 3, 4)
            # 3d: direct load pre-processed features
            if modal == 'pc':
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    if samples["pc_feat"].shape[-1] < 1408:
                        pc_embeds = torch.cat([samples["pc_feat"],samples["pc_feat"][...,:1408-samples["pc_feat"].shape[-1]]], dim=-1)
                    else:
                        pc_embeds = samples["pc_feat"]
                    # print(pc_embeds.shape)
                    pc = samples["pc"].long()
                    all_pcs = torch.zeros(pc_embeds.shape)
                    # try:
                    for j in range(pc.shape[0]):
                        pcs = []
                        for i in range(3):
                            pc_i = pc[j][:5000, i]
                            # clip pc_i to 0-1407
                            pc_i = torch.clamp(pc_i, 0, 255)
                            # print(pc_i)
                            # print(pc_i.shape)
                            # print(self.pos_embedding.shape)
                            pcs.append(self.pos_embedding[pc_i])

                        pcs = torch.cat(pcs, -1)
                        all_pcs[j][:, :1407] = pcs
                    # except:
                    #     print(f"Failed {qid}")
                    #     return  {'output_text': "error", "answer":answer, "qid":qid}
                    all_pcs = all_pcs.cuda()
                # print(pc_embeds.shape)
                pc_embeds = pc_embeds + 0.01 * all_pcs
                atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)
                input_embed_dict[modal], input_att_dict[modal] = pc_embeds, atts
            else:
                input_embed_dict[modal], input_att_dict[modal] = self.encode_input(input, modal, training=False)
        
        device = input_embed_dict[list(input_embed_dict.keys())[0]].device
        fusion_modal = []
        input_text= self.t5_tokenizer(
                qa_text, padding="longest", truncation=True,
                max_length=self.max_txt_len, return_tensors="pt").to(device)
        input_text_embeds = self.t5_model.encoder.embed_tokens(input_text.input_ids) 

        with torch.no_grad():
            
            t5_inputs, t5_atts, t5_query = {}, {}, {}
            for i, modal in enumerate(self.modalities):
                # try:
                t5_inputs[modal], t5_atts[modal], t5_query[modal] = self.get_qformer_embedding(input_embed_dict[modal], input_att_dict[modal], device, modal if modal != 'image' else 'rgb', b)
                # except:
                #     print(f"Failed {qid}")
                #     return {'output_text': "error", "answer":answer, "qid":qid, "question_id":qid}
                order_embed, order_mask =  self.get_prefix_embedding("Scene "+self.ANS_MAP_INV[i]+": ", b, device)
                if 'video' in modal:
                    vid_prefix_embed, vid_prefix_mask = self.get_prefix_embedding(self.vid_prefix, b, device)
                    t5_inputs[modal] = torch.cat([vid_prefix_embed[:,:t5_inputs[modal].shape[1],...], t5_inputs[modal]], dim=2) # b, t, n_word + m, c
                    t5_atts[modal] = torch.cat([vid_prefix_mask[:,:t5_inputs[modal].shape[1],...], t5_atts[modal]], dim=2) # b, t, n_word + m 
                    t5_inputs[modal] = t5_inputs[modal].reshape(b, -1, t5_inputs[modal].shape[-1])
                    t5_atts[modal] = t5_atts[modal].reshape(b, -1)
                    t5_inputs[modal] = torch.cat([order_embed.squeeze(1), t5_inputs[modal]], dim=1) # b, t, n_word + m, c
                    t5_atts[modal] = torch.cat([order_mask.squeeze(1), t5_atts[modal]], dim=1) # b, t, n_word + m
                if 'rgb' in modal:
                    # rgb_prefix_embed, rgb_prefix_mask = self.get_prefix_embedding(self.vid_prefix, b, device)
                    # t5_inputs[modal] = torch.cat([rgb_prefix_embed, t5_inputs[modal]], dim=2) # b, t, n_word + m, c
                    # t5_atts[modal] = torch.cat([rgb_prefix_mask, t5_atts[modal]], dim=2) # b, t, n_word + m 
                    t5_inputs[modal] = t5_inputs[modal].reshape(b, -1, t5_inputs[modal].shape[-1])
                    t5_atts[modal] = t5_atts[modal].reshape(b, -1)
                    t5_inputs[modal] = torch.cat([order_embed.squeeze(1), t5_inputs[modal]], dim=1) # b, t, n_word + m, c
                    t5_atts[modal] = torch.cat([order_mask.squeeze(1), t5_atts[modal]], dim=1)
                if 'audio' in modal:
                    audio_prefix_embed, audio_prefix_mask = self.get_prefix_embedding(self.audio_prefix, b, device) 
                    t5_inputs[modal] = t5_inputs[modal].reshape(b, -1, t5_inputs[modal].shape[-1])
                    t5_atts[modal] = t5_atts[modal].reshape(b, -1)
                    t5_inputs[modal] = torch.cat([order_embed.squeeze(1), audio_prefix_embed.squeeze(1), t5_inputs['audio'].reshape(b, -1, t5_inputs['audio'].shape[-1])], dim=1)
                    t5_atts[modal] = torch.cat([order_mask.squeeze(1), audio_prefix_mask.squeeze(1), t5_atts['audio'].reshape(b, -1)], dim=1)
                if 'pc' in modal:
                    t5_inputs['pc'] = t5_inputs['pc'].reshape(b, -1, t5_inputs['pc'].shape[-1])
                    t5_atts['pc'] = t5_atts['pc'].reshape(b, -1)
                    pc_prefix_embed, pc_prefix_mask = self.get_prefix_embedding(self.pc_prefix, b, device) 
                    t5_inputs['pc'] = torch.cat([order_embed.squeeze(1), pc_prefix_embed.squeeze(1), t5_inputs['pc']], dim=1)
                    t5_atts['pc'] = torch.cat([order_mask.squeeze(1), pc_prefix_mask.squeeze(1), t5_atts['pc']], dim=1)
                
            inputs_t5 = torch.cat(list(t5_inputs.values()), dim=1)
            atts_t5 = torch.cat(list(t5_atts.values()), dim =1 )
            inputs_embeds = torch.cat([inputs_t5, input_text_embeds], dim=1)
            encoder_atts = torch.cat([atts_t5, input_text.attention_mask], dim=1)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                
                if self.downstream_task == 'mcqa':
                    outputs = self.t5_model.generate(
                        inputs_embeds=inputs_embeds, attention_mask=encoder_atts,
                        do_sample=use_nucleus_sampling, top_p=top_p,
                        temperature=temperature, num_beams=1,
                        max_new_tokens=max_length, min_length=min_length,
                        repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                        num_return_sequences=num_captions, return_dict_in_generate=True,
                        output_hidden_states=True, output_scores=True)
                    try:
                        pred_logits = outputs.scores[1]
                    except:
                        pred_logits = outputs.scores[0]
                    pred_logits = pred_logits[:, self.answer_id] # b, 5
                    pred_ans = torch.argmax(pred_logits, dim=-1).cpu().tolist() 

                elif self.downstream_task == 'oeqa':
                    outputs = self.t5_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_atts,
                        do_sample=False,
                        num_beams=num_beams,
                        max_new_tokens=max_length,
                        min_length=min_length,
                        length_penalty=length_penalty,
                        )
                    pred_ans = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        out['output_text'] = pred_ans
        out['answer'] = answer
        out['qid'] = qid
        out['question_id'] = qid
        print(out['output_text'], out['answer'], out['qid'])

        return out

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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '54322'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    job_id = now()
    
    cfg = {"run_cfg":{'task': 'captioning', 'lr_sched': 'linear_warmup_cosine_lr', 'init_lr': 0.0001, 'min_lr': 1e-05, 'warmup_lr': 1e-08, 'warmup_steps': 1000, 'weight_decay': 0.05, 'max_epoch': 100, 'batch_size_train': 1, 'batch_size_eval': 1, 'num_workers': 4, 'accum_grad_iters': 1, 'max_len': 5, 'min_len': 1, 'num_beams': 5, 'inference_method': 'generate', 'seed': 42, 'output_dir': 'results/CREMA', 'amp': True, 'evaluate': True, 'train_splits': ['train'], 'valid_splits': ['val'], 'test_splits': ['test'], 'device': 'cuda', 'world_size': 16, 'dist_url': 'env://', 'distributed': True, 'find_unused_parameters': True}}
    
    options = {"options": None, "cfg_path": "CREMA/lavis/projects/crema/eval/music_avqa_eval.yaml"}
    cfg = Config(OmegaConf.merge(options,cfg))
    cfg.run_cfg.task = "aok_vqa"
    cfg.run_cfg.batch_size_eval = 1
     
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(42)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    task = tasks.setup_task(cfg)
    model = Contra4CREMA.from_pretrained('pretrain_flant5xl')
    # TODO: add the correct path.
    mmqa_ckpt = 'CREMA/crema_initial.pth'
    model.load_mmqa(mmqa_ckpt)
    model.downstream_task  = 'oeqa'

    datasets = {
        "test": {"test": Contra4Dataset()},
    }
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
