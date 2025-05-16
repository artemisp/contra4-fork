import sys
sys.path.append('OneLLM')
import os
import json
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from util import misc
import functools
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed._shard.api import load_with_process_group

from fairscale.nn.model_parallel import initialize as fs_init

selection_type='similarity'

os.environ['CUDA_HOME'] = os.environ['CONDA_PREFIX']
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['ld'] = f"ld -l {os.environ['LD_LIBRARY_PATH']}"

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

dl_fail = []

index2letter = lambda i: chr(ord('A') + i)

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
                        image.append(load_audio(os.path.join(self.audiocaps_dir, f'{example["id"]}.wav')))
                    elif example["source"] == "clotho":
                        image.append(load_audio(os.path.join(self.clotho_dir, f'{example["id"]}')))
                elif modality == "pc":
                    image.append(load_pc(os.path.join(self.objaverse_dir, f'{example["id"]}_8192.npz')))
                elif modality == "image":
                    image.append(load_image(os.path.join(self.coco_dir, f'{str(example["id"]).zfill(12)}.jpg')))
                elif modality == "video":
                    image.append(load_video(os.path.join(self.msrvtt_dir, f'{example["id"]}.mp4')))        
            question_id = data['id']
            question =data['question'] if 'question' in data else data['questions'][0]
            question+= " Choose from: " + ", ".join([f'Scene {index2letter(i)}' for i,c in enumerate(data["modalities"])])
            answer =data['answer'] if 'answer' in data else data['answers_formatted'][0]
            data['modalities'] = [m if m !='pc' else 'point' for m in data['modalities']]
        except Exception as e:
            print("Error loading data", data['id'], e)
            dl_fail.append(data['id'])
            return None
        print(data["modalities"])
        return image, data["modalities"], question, question_id, answer

class ModalityGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        """
        dataset   : the instance of CaptionDataset (or any dataset) 
                    that implements get_group_key(index).
        batch_size: size of each batch
        shuffle   : whether to shuffle the examples within each group 
                    and/or the order of the groups.
        drop_last : whether to drop the last incomplete batch in each group
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 1. Group indices by their group_key
        self.grouped_indices = defaultdict(list)
        for idx in range(len(dataset)):
            key = dataset.get_group_key(idx)  # e.g. tuple of modalities
            self.grouped_indices[key].append(idx)

        # 2. For each group, shuffle its indices (optional)
        if self.shuffle:
            for key in self.grouped_indices:
                random.shuffle(self.grouped_indices[key])

        # 3. Build a list of batches, each from a single group
        self.batches = []
        for key, indices in self.grouped_indices.items():
            # chunk these indices into batch_size
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue
                self.batches.append(batch_indices)

        # 4. Optionally shuffle the entire list of batches
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        """Yield one batch of indices at a time."""
        for batch_indices in self.batches:
            yield batch_indices

    def __len__(self):
        return len(self.batches)
    
    
def collater_fn_with_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return []
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch

dataset = CaptionDataset()

    

import model
from model.LLM.onellm import Transformer
from model.tokenizer import Tokenizer

class Contra4Transformer(Transformer):
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
Transformer = Contra4Transformer
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument('--world_size', default=torch.cuda.device_count(),
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--model_parallel_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    args = parser.parse_args()

    pretrained_path = args.pretrained_path
    answer_path = args.answer_path
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)    
    
    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }["bf16"]

    mp.set_start_method("spawn")
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = 0 + misc.get_rank()
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
    
    # define the model
    model.meta.LLM.__dict__["onellm"].Transformer = Contra4Transformer
    model = MetaModel("onellm", "OneLLM/config/llama2/7B.json", None, 'OneLLM/OneLLM-7B/tokenizer.model")
    model.to(device)
    print("Model = %s" % str(model))
    original_path = "OneLLM/OneLLM-7B/consolidated.00-of-01.pth"
    checkpoint = torch.load(original_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    mixed_precision_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
    }["bf16"]
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
        }['sdp'],
        ignored_states=[param for param in model.parameters() if not param.requires_grad],
    )
    
    param_groups = {
        "decay": {"params": [], "weight_decay": args.weight_decay, "lr": 0},
        "no_decay": {"params": [], "weight_decay": 0.,  "lr": 0},
        "scratch_decay": {"params": [], "weight_decay": args.weight_decay,  "lr": 0},
        "scratch_no_decay": {"params": [], "weight_decay": 0., "lr": 0},
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
    no_decay = name.endswith(".bias") or name.endswith("norm.weight")
    scratch = "llma.resample_layers" in name or "llma.resample_tokens" in name
    group_name = ("scratch_" if scratch else "") + ("no_decay" if no_decay else "decay")
    print(f"{name}: in group {group_name}")
    param_groups[group_name]["params"].append(param)
    def load_model(path, model):
        local_checkpoint_path = os.path.join(
            path,
            f"checkpoint.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
        )
        with load_with_process_group(fs_init.get_data_parallel_group()):
            checkpoint = torch.load(local_checkpoint_path, map_location='cpu')
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(checkpoint['model'])
    print("Loading pretrained weights ...")
    # check if path is directory
    if os.path.isdir(pretrained_path):
        load_model(pretrained_path, model)
        consolidated_model_save_path = os.path.join(
            pretrained_path,
            f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.pth",
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            save_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "tf32": torch.float32,
            }["bf16"]
            consolidated_model_state_dict = {
                k: v.to(save_dtype) for k, v in model.state_dict().items()
            }
        if fs_init.get_data_parallel_rank() == 0:
            torch.save(consolidated_model_state_dict, consolidated_model_save_path)
        print("Model saved to", consolidated_model_save_path)
        exit(0)
    
    else:   
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
    dataset = CaptionDataset(path=args.data_path)

    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, collate_fn=collater_fn_with_none)
    predictions = []
    correct = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            try:
                images, modalities, questions, question_ids, answers = data
            except:
                continue
            preds = multi_modal_generate(images, questions,modal=modalities)
            for question, pred, question_id, answer in zip(questions, preds, question_ids, answers):
                predictions.append({'question_id': question_id, 'answer': pred, 'gt_answer': answer})
                pred = pred.strip().lower()
                answer = answer.strip().lower()
                if (pred in answer) or (answer in pred):
                    correct += 1
    print(predictions)
    
    
    acc = float(correct) / len(dataset)
    print('Accuracy:', acc) 
    open(f'dl_fail_random_onllm.json', 'w').write(json.dumps(dl_fail))
    with open(answer_path, 'w') as f: 
        json.dump(predictions, f) # this list should be empty. if non-empty some issue with setup.