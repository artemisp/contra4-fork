"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json 
import os

os.makedirs('data', exist_ok=True)
os.makedirs('data/filters', exist_ok=True)

all_discrn = json.load(open('data/step2_3/output_high_sim.json')) + json.load(open('data/step2_3/output_random.json'))

def get_mc(mc, discrn_data):
    return [d for d in discrn_data if d['q_type']==f'mc_{mc}']
def get_sample(hs, discrn_data):
    if hs:
        return [d for d in discrn_data if d['selection_type']!='random'] 
    else:
        return [d for d in discrn_data if d['selection_type']=='random']

def print_row(discrn_data):
    print(f"{len(get_mc(2, discrn_data)):,} & {len(get_mc(3, discrn_data)):,} & {len(get_mc(4, discrn_data)):,} & {len(get_sample(True, discrn_data)):,} & {len(get_sample(False, discrn_data)):,} & {len(discrn_data):,} \\\\")  

discrn_data_f = all_discrn
print_row(discrn_data_f)

filter_words = ["more than one person","more than one object", "elements", "more than one input", "more than one animal", "similar", 'rating', 'score', 'hear', 'detail', "word","animate","movie","action", "text", "verb", "noun", "more objects", "most objects", "most people", "more colors", "most colors", "most elements", "more elements", "quest", "caption", "sound", "descr", "sentenc", "visual", "image", "video", "audio", "3d", "point cloud", "more people"]
discrn_data_f = [d for d in discrn_data_f if not any([k in d['questions'][0].lower() for k in filter_words])]
print_row(discrn_data_f)
json.dump(discrn_data_f, open("data/filters/word_filtering.json", 'w'))

model_list = ['meta-llama/Llama-2-13b-chat-hf', "mistralai/Mistral-7B-Instruct-v0.2",  'google/flan-t5-xxl',]
for k in model_list:
    discrn_data_m = discrn_data_f
    discrn_data_m_ = [d for d in discrn_data_m if d[f"{k}_permutations"]["answers"][0] == d["answers"][0]]
    print_row(discrn_data_m_)
    json.dump(discrn_data_m_, open(f"data/filters/{k.replace('/', '_')}_filter.json", 'w'))
discrn_data_maj = [d for d in discrn_data_f if sum([d[f"{k}_permutations"]["answers"][0] == d["answers"][0] for k in model_list])>=2]
print_row(discrn_data_maj)
json.dump(discrn_data_maj, open(f"data/filters/majority_filter.json", 'w'))
discrn_data_un = [d for d in discrn_data_f if sum([d[f"{k}_permutations"]["answers"][0] == d["answers"][0] for k in model_list])==3]
print_row(discrn_data_un)
json.dump(discrn_data_un, open(f"data/filters/unanimous_filter.json", 'w'))


def permute_answer(permutations, org_answer):
    ans2idx = {"Scene A": 0, "Scene B": 1, "Scene C": 2, "Scene D": 3}
    idx2ans = {v:k for k,v in ans2idx.items()}
    answers = []
    for p,a in zip(permutations["permutations"], [a.strip() for a in permutations["answers"]]):
        if a not in ans2idx:
            continue
        if ans2idx[a] >= len(p):
            return False
        if p[ans2idx[a]] not in idx2ans:
            return False
        answers.append(idx2ans[p[ans2idx[a]]])
    if all([a == org_answer for a in answers]):
        return True
    else:
        return False
    
for k in model_list:
    k = f"{k}_permutations"
    discrn_data_m = discrn_data_f
    discrn_data_m_ = [d for d in discrn_data_m if  all(f"{k}_permutations" in d for k in model_list) and 'answers' in d and 'permutations' in d[k] and permute_answer(d[k],d["answers"][0])]
    print_row(discrn_data_m_)
    json.dump(discrn_data_m_, open(f"data/filters/{k.replace('/', '_')}_permute_filter.json", 'w'))
discrn_data_maj_perm = [d for d in discrn_data_f if  all(f"{k}_permutations" in d for k in model_list) and 'answers' in d and sum([permute_answer(d[f"{k}_permutations"],d["answers"][0]) for k in model_list])>=2]
json.dump(discrn_data_maj_perm, open(f"data/filters/majority_permute_filter.json", 'w'))
print_row(discrn_data_maj_perm)
discrn_data_un_perm = [d for d in discrn_data_f if all(f"{k}_permutations" in d for k in model_list) and 'answers' in d and sum([permute_answer(d[f"{k}_permutations"],d["answers"][0]) for k in model_list])==3]
print_row(discrn_data_un_perm)
json.dump(discrn_data_un_perm, open('data/filters/unanimous_permute.json', 'w'))