import json 
import os
from tqdm import tqdm 
from itertools import permutations

SPLIT = 'test'

os.makedirs('data', exist_ok=True)
os.makedirs(f'data/filters_{SPLIT}', exist_ok=True)

MODEL_LIST = ['llama', 'mistral', 'phi']
SELECTION_LIST = ['random', 'similarity']

def answer2choice(text):
    text = text.split('.')[0]
    text = text.replace('Answer', '') 
       
    if text.startswith('Scene A'):
        return "Scene A"
    elif text.startswith('Scene B'):
        return "Scene B"
    elif text.startswith('Scene C'):
        return "Scene C"
    elif text.startswith('Scene D'):
        return "Scene D"
    if 'Scene A' in text and not 'Scene B' in text and not 'Scene C' in text and not 'Scene D' in text:
        return "Scene A"
    
    elif 'Scene B' in text and not 'Scene A' in text and not 'Scene C' in text and not 'Scene D' in text:
        return "Scene B"
    elif 'Scene C' in text and not 'Scene A' in text and not 'Scene B' in text and not 'Scene D' in text:
        return "Scene C"
    elif 'Scene D' in text and not 'Scene A' in text and not 'Scene B' in text and not 'Scene C' in text:
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
        return text

## merge
for selection_type in SELECTION_LIST:
    all_data = None
    all_data_indices = []
    for model in MODEL_LIST:
        if all_data is None:
            all_data = json.load(open(f'data/step4/{model}_{selection_type}_{SPLIT}.json'))
            for i,d in enumerate(all_data):
                if len(d['modalities']) != len(set(d['modalities'])):
                    # should not happen but double check
                    continue
                all_data_indices.append(i)
            all_data = [all_data[i] for i in all_data_indices]
      

        else:
            data = json.load(open(f'data/step4/{model}_{selection_type}_{SPLIT}.json'))

            data = [data[i] for i in all_data_indices]
            for i in tqdm(range(len(all_data)), desc=f"merging {model}_{selection_type}"):
                all_data[i][f'answer_{model}'] = data[i][f'answer_{model}']
    for i in tqdm(range(len(all_data)), desc="permutations"):
        all_data[i]['permutations'] = list(permutations(range(len(all_data[i]["examples"]))))
        all_data[i]['answers_formatted'] = [answer2choice(a) for a in all_data[i]['answers']]
        for model in MODEL_LIST:
            all_data[i][f'answer_{model}_formatted'] = [answer2choice(a) for a in all_data[i][f'answer_{model}']]
    
    def get_mc(mc, discrn_data):
        return [d for d in discrn_data if d['q_type']==f'mc_{mc}']
    def get_sample(hs, discrn_data):
        if hs:
            return [d for d in discrn_data if d['selection_type']!='random'] 
        else:
            return [d for d in discrn_data if d['selection_type']=='random']
    def print_row(discrn_data):
        print(f"{len(get_mc(2, discrn_data)):,} & {len(get_mc(3, discrn_data)):,} & {len(get_mc(4, discrn_data)):,} & {len(get_sample(True, discrn_data)):,} & {len(get_sample(False, discrn_data)):,} & {len(discrn_data):,} \\\\")  

    discrn_data_f = all_data
    print_row(discrn_data_f)

    filter_words = ["more than one person","more than one object", "elements", "more than one input", 
                    "more than one animal", "similar", 'rating', 'score', 'hear', 'detail', "word",
                    "animate","movie","action", "text", "verb", "noun", "more objects", "most objects",
                    "most people", "more colors", "most colors", "most elements", "more elements", "quest",
                    "caption", "sound", "descr", "sentenc", "visual", "image", "video", "audio",
                    "3d", "point cloud", "more people"]
    discrn_data_f = [d for d in tqdm(discrn_data_f, desc='word filter') if not any([k in d['questions'][0].lower() for k in filter_words])]
    print_row(discrn_data_f)
    json.dump(discrn_data_f, open(f"data/filters_{SPLIT}/word_filtering_{selection_type}_{SPLIT}.json", 'w'))


    
    m2data = {}
    for model in MODEL_LIST:
        discrn_data_m = discrn_data_f
        discrn_data_m_ = [d for d in tqdm(discrn_data_m, desc=f'{model}_filter') if d[f"answer_{model}_formatted"][0] == d["answers_formatted"][0]]
        m2data[model] = discrn_data_m_
        print_row(discrn_data_m_)
        json.dump(discrn_data_m_, open(f"data/filters_{SPLIT}/{model}_filter_{selection_type}_{SPLIT}.json", 'w'))
    m2ids = {model: set([d['id'] for d in discrn_data_m]) for model, discrn_data_m in m2data.items()}
    un_ids = set.intersection(*m2ids.values())
    maj_ids = []
    for comb in permutations(MODEL_LIST, 2):
        maj_ids += set.intersection(m2ids[comb[0]], m2ids[comb[1]])
    maj_ids = set(maj_ids)
    discrn_data_maj = [d for d in tqdm(discrn_data_f, desc='maj_filter') if d['id'] in maj_ids]
    print_row(discrn_data_maj)
    json.dump(discrn_data_maj, open(f"data/filters_{SPLIT}/majority_filter_{selection_type}_{SPLIT}.json", 'w'))
    un_ids = set(un_ids)
    discrn_data_un = [d for d in tqdm(discrn_data_f, desc='un_filter') if d['id'] in un_ids]
    print_row(discrn_data_un)
    json.dump(discrn_data_un, open(f"data/filters_{SPLIT}/unanimous_filter_{selection_type}_{SPLIT}.json", 'w'))


    def permute_answer(permutations, permutation_answers, org_answer):
        ans2idx = {"Scene A": 0, "Scene B": 1, "Scene C": 2, "Scene D": 3}
        idx2ans = {v:k for k,v in ans2idx.items()}
        answers = []
        permutation_answers = [a.strip() for a in permutation_answers]
        for p,a in zip(permutations, permutation_answers):
            if a not in ans2idx:
                return False
            if ans2idx[a] >= len(p):
                return False
            if p[ans2idx[a]] not in idx2ans:
                return False
            answers.append(idx2ans[p[ans2idx[a]]])
        if all([a == org_answer for a in answers]):
            return True
        else:
            return False

    model2data = {}
    for model in MODEL_LIST:
        discrn_data_m = discrn_data_f
        discrn_data_m_ = [d for d in tqdm(discrn_data_f, desc=f'{model}_perm_filter') if permute_answer(d['permutations'],d[f"answer_{model}_formatted"], d['answers_formatted'][0])]
        print_row(discrn_data_m_)
        model2data[model] = discrn_data_m_
        json.dump(discrn_data_m_, open(f"data/filters_{SPLIT}/{model}_permute_filter_{selection_type}_{SPLIT}.json", 'w'))
    maj_ids = []
    for comb in permutations(MODEL_LIST, 2):
        maj_ids += set.intersection(set([d['id'] for d in model2data[comb[0]]]), set([d['id'] for d in model2data[comb[1]]]))
    maj_ids = set(maj_ids)
    discrn_data_maj_perm = [d for d in tqdm(discrn_data_f, desc='maj_perm_filter') if d['id'] in maj_ids]
    json.dump(discrn_data_maj_perm, open(f"data/filters_{SPLIT}/majority_permute_filter_{selection_type}_{SPLIT}.json", 'w'))
    print_row(discrn_data_maj_perm)
    un_ids = set.intersection(*[set([d['id'] for d in model2data[model]]) for model in MODEL_LIST])
    discrn_data_un_perm = [d for d in  tqdm(discrn_data_f, desc='un_perm_filter') if d['id'] in un_ids]
    
    print_row(discrn_data_un_perm)
    json.dump(discrn_data_un_perm, open(f'data/filters_{SPLIT}/unanimous_permute_{selection_type}_{SPLIT}.json', 'w'))