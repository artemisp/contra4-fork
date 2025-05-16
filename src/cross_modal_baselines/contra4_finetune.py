
import sys
sys.path.append('OneLLM')
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import functools
import multiprocessing
from typing import Optional
import copy
import torch
from io import BytesIO
import requests
import model
from model.LLM.onellm import Transformer
from model.tokenizer import Tokenizer


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from fairscale.nn.model_parallel import initialize as fs_init
import warnings
try:
    from apex.optimizers import FusedAdam as AdamW
except ImportError:
    warnings.warn("cannot import FusedAdam from apex, use torch AdamW instead")
    from torch.optim import AdamW

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from model.meta import MetaModel
from engine_finetune import train_one_epoch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from PIL import Image
import random
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
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
from torch.utils.data import Sampler
from collections import defaultdict
import random
import requests

import util.misc as misc
import util.lr_sched as lr_sched
import math
import contextlib

from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.encoded_video_decord import EncodedVideoDecord
from torchvision.transforms._transforms_video import NormalizeVideo
from data.video_utils import get_clip_timepoints, SpatialCrop


warnings.filterwarnings("ignore")

from data.finetune_dataset import FinetuneDialogDataset, FinetuneDistSampler

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

def get_args_parser():
    parser = argparse.ArgumentParser('OneLLM Finetuning', add_help=False)
    parser.add_argument('--datasets', type=str, default='image', nargs='+')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='onellm', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument("--llama_ckpt_dir", type=str, default="OneLLM/OneLLM-7B")
    parser.add_argument("--llama_config", type=str, default="OneLLM/config/llama2/7B.json")
    parser.add_argument("--tokenizer_path", type=str, default="OneLLM/config/llama2/tokenizer.model")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=1e-10, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=float, default=1.0, metavar='N',
                        help='epoch to warmup LR')

    parser.add_argument('--clip_grad', type=int, default=2.0,
                        help='grad clipping norm')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--init_from', default='',
                        help='init from checkpoint')
    parser.add_argument('--init_from_image', action='store_true')

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=torch.cuda.device_count(),
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--model_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel', type=str, choices=['ddp', 'sdp', 'fsdp'], default='sdp')
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16')
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--save_consolidated', action="store_true",
                        help="save consolidated model weights along with regular checkpoints "
                             "used to resume training. useful for convenient deployment but "
                             "will occupy some additional disk space.")
    parser.add_argument("--checkpointing", action="store_true")

    parser.add_argument('--max_words', type=int, default=180)
    parser.add_argument('--image_words', type=int, default=30)

    return parser


def load_and_transform_video_data(
    video_file,
    video_path,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
    with_audio=False
):
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    if isinstance(video_file, str):
        video = EncodedVideo.from_path(
            video_file,
            decoder="pyav",
            decode_audio=with_audio,
            # **{"sample_rate": sample_rate},
        )
    else:
        video = EncodedVideoDecord(video_file, video_name=video_path, decode_video=True, decode_audio=with_audio, sample_rate=sample_rate)
    
    all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

    all_video = []
    for clip_timepoints in all_clips_timepoints:
        # Read the clip, get frames
        clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
        if clip is None:
            raise ValueError("No clip found")
        video_clip = frame_sampler(clip["video"])
        video_clip = video_clip / 255.0  # since this is float, need 0-1

        all_video.append(video_clip)

    all_video = [video_transform(clip) for clip in all_video]
    all_video = SpatialCrop(224, num_crops=3)(all_video)

    all_video = torch.stack(all_video, dim=0)

    if not with_audio:
        return all_video
    else:
        return all_video, clip['audio']

def load_video(video_path):
    video_feats = load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
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


def find_sublist(a: list, b: list):
    len_a, len_b = len(a), len(b)
    for i in range(len_a - len_b + 1):
        if a[i:i+len_b] == b:
            return i
    return -1


dl_fail = []

index2letter = lambda i: chr(ord('A') + i)


def answer2choice(text):
    text = text.split('.')[0]
    if text.startswith('Scene A'):
        return "Scene A"
    elif text.startswith('Scene B'):
        return "Scene B"
    elif text.startswith('Scene C'):
        return "Scene C"
    elif text.startswith('Scene D'):
        return "Scene D"
    if "Scene A" in text and not "Scene B" in text and not "Scene C" in text and not "Scene D" in text:
        return "Scene A"
    elif "Scene B" in text and not "Scene A" in text and not "Scene C" in text and not "Scene D" in text:
        return "Scene B"
    elif "Scene C" in text and not "Scene A" in text and not "Scene B" in text and not "Scene D" in text:
        return "Scene C"
    elif "Scene D" in text and not "Scene A" in text and not "Scene B" in text and not "Scene C" in text:
        return "Scene D"
    if "A" in text and not 'B' in text and not 'C' in text and not 'D' in text:
        return "Scene A"
    elif "B" in text and not 'A' in text and not 'C' in text and not 'D' in text:
        return "Scene B"
    elif "C" in text and not 'A' in text and not 'B' in text and not 'D' in text:
        return "Scene C"
    elif "D" in text and not 'A' in text and not 'B' in text and not 'C' in text:
        return "Scene D"
    else:
        return ""

class CaptionDataset(Dataset):
    def __init__(self, 
                  path=[f'../../data/final_data/train_unanimous_permute_filter.json'],
                 max_words=180,
                 image_words=30
                 ) -> None:
        super().__init__()
        self.data = []
        for p in path:
            self.data.extend(json.load(open(p))) 
        new_data = []
        for d in self.data:
            d['answers_formatted'] = [answer2choice(d['answer']) if 'answer' in d else answer2choice(d['answers'][0])]
            if not flag:
                new_data.append(d)
        self.data = new_data
        self.coco_dir = os.environ['COCO_DIR']
        self.audiocaps_dir = os.environ['AUDIOCAPS_DIR']
        self.msrvtt_dir = os.environ['MSRVTT_DIR']
        self.clotho_dir = os.environ['CLOTHO_DIR']
        self.objaverse_dir = os.environ['OBJAVERSE_DIR']
        self.tokenizer = Tokenizer(model_path="OneLLM/config/llama2/tokenizer.model")
        for item in self.data:
            # e.g. ['audio', 'audio'] or ['image', 'video'] or ['pc'] ...
            # store it so you can access it quickly later.
            item["group_key"] = tuple(item["modalities"])
        self.max_words = max_words
        self.image_words = image_words


    def __len__(self):
        return len(self.data)

    def get_group_key(self, index):
        return self.data[index]["group_key"]

    def __getitem__(self, index):
        data = self.data[index]
        image = []
        try:
            for modality,example in zip(data["modalities"],data["examples"]):
                if modality == "audio":
                    if example["source"] == "audiocaps":
                        try:
                            image.append(load_audio(os.path.join(self.audiocaps_dir, f'{example["id"]}.wav')))
                        except:
                            print("None found", example["id"])
                            return None
                    elif example["source"] == "clotho":
                        image.append(load_audio(os.path.join(self.clotho_dir, f'{example["id"]}')))
                elif modality == "pc":
                    # point_path = f'https://storage.googleapis.com/sfr-ulip-code-release-research/ULIP-Objaverse_triplets/objaverse_pc_parallel/{example["id"]}/{example["id"]}_8192.npz'
                    # response = requests.get(point_path)
                    # response.raise_for_status()  # raise an error if download failed

                    # # 2. Wrap the content in a BytesIO object
                    # file_obj = BytesIO(response.content)
                    # # from pdb import set_trace; set_trace()
                    # points = np.load(file_obj)['arr_0']
                    pc_path = os.path.join(self.objaverse_dir, f'{example["id"]}_8192.npz')
                    image.append(load_pc(pc_path))
                elif modality == "image":
                    image.append(load_image(os.path.join(self.coco_dir, f'{str(example["id"]).zfill(12)}.jpg')))
                elif modality == "video":
                    image.append(load_video(os.path.join(self.msrvtt_dir, f'{example["id"]}.mp4')))        
            question_id = data['id']
            question = data['question'] if 'question' in data else data['questions'][0]
            question+= " Choose from: " + ", ".join([f'Scene {index2letter(i)}' for i,c in enumerate(data["modalities"])])
            answer = data['answers_formatted'][0]
            data['modalities'] = [m if m !='pc' else 'point' for m in data['modalities']]
        except Exception as e:
            print("Error loading data", data['id'], e)
            dl_fail.append(data['id'])
            return None
        tokenzed_conversation = self.tokenizer.encode(
            question + " " + answer, bos=True, eos=True)
        labels = [IGNORE_INDEX for _ in tokenzed_conversation]

        check_pos = 0
        for value in [answer]:
            tokenized_value = self.tokenizer.encode(
                value, bos=False, eos=False)
            value_pos = find_sublist(
                tokenzed_conversation[check_pos:], tokenized_value) + check_pos
            if value_pos == -1:
                print(
                    "a sentence mismatches the corresponding piece in the conversation")
                return self[index-1]
            labels[value_pos:value_pos+len(tokenized_value)] = tokenized_value
            assert labels[value_pos:value_pos+len(
                tokenized_value)] == tokenzed_conversation[value_pos:value_pos+len(tokenized_value)]
            check_pos = value_pos+len(tokenized_value)
        
        input2 = torch.tensor(tokenzed_conversation, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_words = self.image_words * len(data['modalities'])
        for i in range(len(data["modalities"])):
            image_words += len(self.tokenizer.encode(f"Scene {chr(ord('A') + i)}: ", bos=False, eos=False))
            
        if image[i] is not None:
            image[i] = image[i].unsqueeze(0)
        if image is not None:
            max_words = self.max_words - image_words
        else:
            max_words = self.max_words
        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat(
                (input2, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat(
                (labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
            labels = labels[:max_words]

        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        modalities = [m if m !='pc' else 'point' for m in data['modalities']]
        

        return image, modalities, input2, question_id, labels

class GroupedDistSampler(Sampler):
    """
    A distributed sampler that groups samples by `dataset.get_group_key(idx)` so that each
    batch (across all replicas) contains samples from only one group. It ensures each replica
    gets an appropriately sized subset of the data, balancing across the group dimension.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        batch_size: Optional[int] = None,
        acc_grad: int = 1,
        drop_last: bool = True
    ) -> None:
        """
        Args:
            dataset: The dataset object, which must implement `get_group_key(idx) -> Hashable`.
            num_replicas (int): Number of distributed processes.
            rank (int): The current process rank.
            shuffle (bool): Whether to shuffle group chunks for each epoch.
            seed (int): Random seed for shuffling.
            batch_size (int): Number of samples per replica's batch.
            acc_grad (int): Gradient accumulation steps. If you do gradient accumulation, the
                            "effective" global batch size is batch_size * num_replicas * acc_grad.
            drop_last (bool): If True, drop leftover samples in a group that can't fill
                              a 'global batch' chunk.
        """
        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid num_replicas ({num_replicas}) or rank ({rank})."
            )
        if batch_size is None:
            raise ValueError("batch_size must be provided.")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.acc_grad = acc_grad
        self.drop_last = drop_last

        self.epoch = 0
        self.start_iter = 0

        group_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            key = dataset.get_group_key(idx)  # e.g., (modality,) or any hashable
            group_to_indices[key].append(idx)
        self.group_indices = list(group_to_indices.values())

        self.global_batch_size = self.batch_size * self.num_replicas * self.acc_grad

    
        trimmed_groups = []
        for g in self.group_indices:
            length = len(g)
            if self.drop_last:
                usable_len = (length // self.global_batch_size) * self.global_batch_size
            else:
                usable_len = length  # keep all
            trimmed_groups.append(g[:usable_len])
        self.group_indices = trimmed_groups

       
        self.group_n_batch = [len(g) // self.batch_size for g in self.group_indices]
        
       
        self.n_total_batch = sum(self.group_n_batch)
        
        self.total_size = self.n_total_batch * self.batch_size
        
        if self.total_size % self.num_replicas != 0:
            # If you do not want partial, you can force an assert:
            #   raise ValueError("Total size not divisible by num_replicas!")
            pass
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            group_indices_copy = copy.deepcopy(self.group_indices)
            for g in group_indices_copy:
                rng.shuffle(g)
                

            global_batched_indices = [
                g[i:i + self.global_batch_size]
                for g in group_indices_copy
                for i in range(0, len(g), self.global_batch_size)
                if len(g[i:i + self.global_batch_size]) == self.global_batch_size 
                  or not self.drop_last
            ]
            rng.shuffle(global_batched_indices)
            indices = [idx for chunk in global_batched_indices for idx in chunk]
        else:
            group_indices_copy = copy.deepcopy(self.group_indices)
            indices = []
            for g in group_indices_copy:
                for i in range(0, len(g), self.global_batch_size):
                    batch_chunk = g[i:i + self.global_batch_size]
                    if self.drop_last and len(batch_chunk) < self.global_batch_size:
                        continue
                    indices += batch_chunk

       
        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]

      
        skip_count = self.start_iter * self.batch_size
        if skip_count >= len(own_indices):
            own_indices = []
        else:
            own_indices = own_indices[skip_count:]

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, start_iter: int = 0) -> None:
        """
        Sets the epoch for this sampler. When shuffle=True, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration
        of this sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            start_iter (int): start iter (in units of *batches*) to resume from.
        """
        self.epoch = epoch
        self.start_iter = start_iter
    
    
def collater_fn_with_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    batch = [[torch.stack([torch.tensor(batch[i][0][j]) for i in range(len(batch))]) for j in range(len(batch[0][0]))], [b[1] for b in batch], torch.stack([b[2] for b in batch]), [b[3] for b in batch], torch.stack([b[4] for b in batch])]
        
    return batch

tokenizer = Tokenizer(model_path="OneLLM/config/llama2/tokenizer.model")


class DisCRnTransformer(Transformer):  
    def encode_image(self, x, modal='image', **kwargs):
        bsz = x.size(0)
        T = 1
        if modal in ['image']:
            # modified from CLIP
            if len(x.shape) == 5:
                x = x.squeeze(1)
            x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        elif modal in ['audio', 'imu']:
            if len(x.shape) == 5:
                x = x.squeeze(1)
            x = self.conv1[modal](x)
        elif modal == 'point':
            # [B, 16384, 6] -> [B, 1024, 1024, 1]
            if len(x.shape) == 4:
                x = x.squeeze(1)
            x = self.conv1[modal](x.float()).to(x.dtype)
        elif modal in ['video', 'rgbd', 'rgbn']:
            # [B, 15, 3, 224, 224]
            if len(x.shape) == 6:
                x = x.squeeze(1)
            B, T = x.shape[:2]
            bsz = B * T
            x = x.reshape(bsz, *x.shape[2:])
            x = self.clip.visual.conv1(x)
        elif modal == 'fmri':
            x = self.conv1[modal](x)
            # [B, 1, 8196] -> [B, 1024, 8]
            x = x.reshape(x.size(0), self.clip.visual.conv1.out_channels, -1)

        image_feats = self.clip_encode_image(x, modal=modal)
        # take mean on time dimension
        # all inputs are reduced to [B, L, D]
        bsz = int(bsz / T)
        image_feats = image_feats.reshape(
            bsz, T, *image_feats.shape[1:]).mean(dim=1)

        image_feats = self.clip_proj1[modal](image_feats)
        image_feats = torch.cat(
            [self.resample_tokens[modal].repeat(bsz, 1, 1), image_feats], dim=1)

        # routing modalites
        # [B, L, D]->[B, L, N]
        routing_weights = self.routers[modal](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        image_feats_experts = []
        for expert_id in range(self.num_experts):
            image_feats_expert = image_feats
            for layer in self.resample_layers[str(expert_id)]:
                image_feats_expert = layer(image_feats_expert, 0, None, None)

            image_feats_expert = image_feats_expert[:, :self.resample_tokens[modal].size(1)]
            routing_weight = routing_weights[:, :self.resample_tokens[modal].size(
                1), expert_id]
            # [B, L, D] * [B, L, 1]
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]

            image_feats_experts.append(image_feats_expert)

        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2[modal](image_feats)

        return image_feats  
    def forward(self, examples, image=None, modal='image'):
        self._destroy_kv_cache()  # training always disables kv cache
        modal = modal[0]
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        start_pos = 0
        prefix_len = 0
        if image is not None:
            if isinstance(image, list):
                for i,im in enumerate(image):
                    h_bos, h_caption = h[:, :1], h[:, 1:]
                    image_tokens = self.encode_image(im, modal[i])
                    input_tokens =  self.tok_embeddings(torch.tensor(tokenizer.encode(f"Scene {chr(ord('A') + i)}: ", bos=False, eos=False)).unsqueeze(0).to(h.device))
                    self.cache_image_words = image_tokens.shape[1]
                    h_bos = torch.cat((h_bos, input_tokens.repeat(_bsz, 1, 1), self.start_tag[modal[i]].repeat(_bsz, 1, 1), image_tokens, self.end_tag[modal[i]].repeat(_bsz, 1, 1)), dim=1)
                h = torch.cat((h_bos, h_caption), dim=1)
                seqlen = h.shape[1]
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                h_bos, h_caption = h[:, :1], h[:, 1:]
                image_tokens = self.encode_image(image, modal)
                h = torch.cat((h_bos, self.start_tag[modal].expand(
                    _bsz, -1, -1), image_tokens, self.end_tag[modal].expand(_bsz, -1, -1), h_caption), dim=1)
                # bos + image token + start_tag[modal], end_tag[modal] is used for caption generation
                prefix_len = image_tokens.shape[1] + 1 + 1
                seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, prefix_len:, :])
        return output

    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None, modal='image', tokenizer=Tokenizer(model_path='OneLLM/OneLLM-7B/tokenizer.model')):
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

class DisCRnMetaModel(MetaModel):
    def forward(self, examples, labels, image=None, modal='image'):
        output = self.llma(examples, image=image, modal=modal)
        output = output[:, :-1, :]
        labels = labels[:, 1:]
        
        ## hack: fix labels
        # find the firs non zero token
        # nonzero_indices = torch.nonzero(x)
        # first_idx = nonzero_indices[0].item()
        pad = torch.zeros((labels.shape[0], output.shape[1] - labels.shape[1])).to(labels.device).to(labels.dtype)
        labels = torch.cat([pad, labels], dim=1).long()

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, 32000), labels.flatten())

        return c_loss


def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, start_iter, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, data_img  in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, start_iter), start=start_iter
    ):
        flag=False
        if data_img is None or len(data_img) == 0:
            flag=True
        if len(data_img) == 5:
            image, modal, examples, _, labels = data_img
            # examples, labels, image, modal = data_img
        elif len(data_img) == 3:
            examples, labels, modal = data_img
            image = None
        else:
            flag=True
            
        if data_iter_step % accum_iter == 0:
            # lr_sched.adjust_learning_rate(optimizer, data_iter_step, args)
            lr_sched.adjust_learning_rate_epoch(optimizer, data_iter_step / len(data_loader) + epoch, args)
        update_grad = (data_iter_step + 1) % accum_iter == 0

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision] 
        backward_ctx = contextlib.nullcontext() if update_grad else model.no_sync()
        with autocast_ctx:
            if flag:
                i_loss = torch.tensor([0]).cuda()
            else:
                i_loss = model(examples, labels, image, modal)
        i_loss_value = i_loss.item()
        if not math.isfinite(i_loss_value):
            print("[Rank {}] i_loss is {}, stopping training".format(dist.get_rank(), i_loss_value), force=True)
            # print(image_paths, force=True)
            sys.exit(1)
        loss_value = i_loss_value
        if flag:
            grad_norm = 0.
        else:
            with backward_ctx:
                grad_norm = loss_scaler(
                    i_loss / accum_iter, optimizer, model,
                    parameters=model.parameters(),
                    update_grad=update_grad,
                    clip_grad=None if args.clip_grad <= 0 else args.clip_grad,
                )
        if update_grad:
            # assert grad_norm is not None
            metric_logger.update(grad_norm=grad_norm)

        if update_grad:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(iloss=i_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # save checkpoint
        if data_iter_step % 1000 == 0 and data_iter_step != 0:
            misc.save_model(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=data_iter_step, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=None)

        if update_grad:
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            i_loss_value_reduce = misc.all_reduce_mean(i_loss_value)
            if update_grad:
                grad_norm_reduce = misc.all_reduce_mean(grad_norm)

        if log_writer is not None and update_grad:
            log_writer.add_scalar('train_loss', loss_value_reduce, data_iter_step)
            log_writer.add_scalar('i_train_loss', i_loss_value_reduce, data_iter_step)
            if update_grad:
                log_writer.add_scalar('grad_norm', grad_norm_reduce, data_iter_step)
            log_writer.add_scalar('lr', lr, data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def main(args):
    import model

    multiprocessing.set_start_method("spawn")
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    global_rank = misc.get_rank()
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_group = fs_init.get_data_parallel_group()
    
    dataset_train = CaptionDataset()

    # # dataset_train = FinetuneDialogDataset(args.datasets, max_words=args.max_words, image_words=args.image_words, tokenizer_path=args.tokenizer_path)
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    model.meta.LLM.__dict__["onellm"].Transformer = DisCRnTransformer
    model = ("onellm", "OneLLM/config/llama2/7B.json", None, "/OneLLM/OneLLM-7B/tokenizer.model")
    
    model = DisCRnMetaModel(args.llama_type, args.llama_config, args.llama_ckpt_dir, args.tokenizer_path)
    model.to(device)
    print("Model = %s" % str(model))
    pretrained_path = "OneLLM/OneLLM-7B/consolidated.00-of-01.pth"
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    print("Model loaded from %s" % pretrained_path)
    if args.init_from:
        print("Init checkpoint from %s" % args.init_from)
        checkpoint = torch.load(os.path.join(args.init_from, f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth"), map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)

    mixed_precision_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
    }[args.precision]
    TransformerBlock = type(model.llma.layers[0])
    model = FSDP(
        model,
        process_group=fs_init.get_data_parallel_group(),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=[TransformerBlock],
        ),
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
        mixed_precision=MixedPrecision(
            param_dtype=mixed_precision_dtype,
            reduce_dtype=mixed_precision_dtype,
            buffer_dtype=mixed_precision_dtype,
        ),
        sharding_strategy={
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "ddp": ShardingStrategy.NO_SHARD,
            "fsdp": ShardingStrategy.FULL_SHARD,
        }[args.data_parallel],
        ignored_states=[param for param in model.parameters() if not param.requires_grad],
    )

    if args.checkpointing:
        print("apply gradient checkpointing")
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    eff_batch_size = args.batch_size * args.accum_iter * fs_init.get_data_parallel_world_size()
    print("effective batch size: %d" % eff_batch_size)
    # following timm: set wd as 0 for bias and norm layers
    #param_groups = misc.add_weight_decay(model, args.weight_decay)
    param_groups = {
        "decay": {"params": [], "weight_decay": args.weight_decay, "lr": args.lr},
        "no_decay": {"params": [], "weight_decay": 0., "lr": args.lr},
        "scratch_decay": {"params": [], "weight_decay": args.weight_decay, "lr": args.lr},
        "scratch_no_decay": {"params": [], "weight_decay": 0., "lr": args.lr},
    }
    print("Making parameter groups ...")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = name.endswith(".bias") or name.endswith("norm.weight")
        scratch = "llma.resample_layers" in name or "llma.resample_tokens" in name
        group_name = ("scratch_" if scratch else "") + ("no_decay" if no_decay else "decay")
        print(f"{name}: in group {group_name}")
        param_groups[group_name]["params"].append(param)
    optimizer = AdamW(
        [param_groups[key] for key in ["decay", "no_decay", "scratch_decay", "scratch_no_decay"]],
        betas=(0.9, 0.95),
    )
    print(optimizer)
    loss_scaler = NativeScaler(args)

    start_epoch = 0
    start_iter = 0
    if args.resume or args.auto_resume:
        start_epoch, start_iter = misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)
    
    sampler_train = GroupedDistSampler(
        dataset_train, 
        num_replicas=dp_world_size, 
        rank=dp_rank, 
        shuffle=True, 
        batch_size=args.batch_size,
        acc_grad=args.accum_iter
        )
    # sampler_train = FinetuneDistSampler(
    #     dataset_train, num_replicas=dp_world_size, rank=dp_rank, shuffle=True, batch_size=args.batch_size,
    #     acc_grad=args.accum_iter
    # )    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        sampler=sampler_train,
        drop_last=True,
        collate_fn=collater_fn_with_none
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch, start_iter)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, epoch, start_iter, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.save_interval == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=0, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=None,
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     **{f'val_{k}': v for k, v in train_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        start_iter = 0

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
