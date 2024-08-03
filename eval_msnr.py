"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--results_file", type=str)
parser.add_argument("--no_input_results_file", type=str)
parser.add_argument("--random_results_file", type=str)
args = parser.parse_args()


results = json.load(open(args.results_file))
no_input_results = json.load(open(args.no_input_results_file))
random_results = json.load(open(args.random_results_file))

gt = json.load(open('data/discrn_balanced.json', 'r'))
id_key = 'question_id' if 'question_id' in results[0] else 'id'
gt_key = 'gt_ans' if 'gt_ans' in results[0] else 'gt_answer'
pred_key = 'pred_ans' if 'pred_ans' in results[0] else 'answer'


id2mc = {d['id']:d['q_type'] for d in gt}
id2sel = {d['id']:d['selection_type'] for d in gt}
id2modalitites = {d['id']:d['modalities'] for d in gt}
id2ex = {d['id']:d for d in gt} 


ans2idx = {"Scene A": 0, "Scene B": 1, "Scene C": 2, "Scene D": 3}

modality2answer_choices = {
    "audio": ["sound", "audio"],
    "video": ["video", "frames", "clip"],
    "pc": ["3d", "point cloud", "model"],
    "image": ["image", "picture", "photo"]
}

index2answer_choices = {
    0: ["1", "first", "one", "scene a"],
    1: ["2", "second", "two", "scene b"],
    2: ["3", "third", "three", "scene c"],
    3: ["4", "fourth", "four", "scene d"]
}

index2letter = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

all_output = defaultdict(dict)


for tp, results in {"mm":results, "rand":random_results, "no_input":no_input_results}.items():
    total = 0
    correct = 0
    id_key = 'question_id' if 'question_id' in results[0] else 'id'
    gt_key = 'gt_ans' if 'gt_ans' in results[0] else 'gt_answer'
    pred_key = 'pred_ans' if 'pred_ans' in results[0] else 'answer'



    id2correctness = defaultdict(int)
    print(len(results))
    for a in results:
        if a[id_key] not in id2mc:
            continue
        
        gt_answer_idx = ans2idx[a[gt_key]]

        # invalid option
        if gt_answer_idx>=len(id2modalitites[a[id_key]]) :
            total+=1
            continue
        

        gt_modality = id2modalitites[a[id_key]][ans2idx[a[gt_key]]]
        
        answers = index2answer_choices[gt_answer_idx] + modality2answer_choices[gt_modality]
        # other_answers = []
        # for idx in range(4):
        #     if idx != gt_answer_idx:
        #         other_answers += index2answer_choices[idx] + modality2answer_choices[id2modalitites[a[id_key]][idx]]
        
        pred = a[pred_key]
        if pred == '':
            pred = 'none'
            
        if any([(v.lower() in pred.lower() or pred.lower() in v.lower()) for v in answers]) or  \
            index2letter[gt_answer_idx] in pred:
                correct+=1
                id2correctness[a[id_key]] = 1
        total+=1

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total}")

    print("High Sim All Accuracy")
    high_sim = [id2correctness[k] for k,v in id2sel.items() if v == 'high_sim']
    high_sim_all = sum(high_sim)/len(high_sim)
    print(sum(high_sim)/len(high_sim))

    print("Random All Accuracy")
    rand = [id2correctness[k] for k,v in id2sel.items() if v == 'random']
    rand_all = sum(rand)/len(rand)
    print(sum(rand)/len(rand))


    print("High Sim MC2 Accuracy")
    high_sim = [id2correctness[k] for k,v in id2sel.items() if v == 'high_sim' and id2mc[k] == 'mc_2']
    high_sim_mc2 = sum(high_sim)/len(high_sim)
    print(sum(high_sim)/len(high_sim))

    print("Random MC2 Accuracy")
    rand = [id2correctness[k] for k,v in id2sel.items() if v == 'random' and id2mc[k] == 'mc_2']
    random_mc2 = sum(rand)/len(rand)
    print(sum(rand)/len(rand))

    print("High Sim MC3 Accuracy")
    high_sim = [id2correctness[k] for k,v in id2sel.items() if v == 'high_sim' and id2mc[k] == 'mc_3']
    high_sim_mc3 = sum(high_sim)/len(high_sim)
    print(sum(high_sim)/len(high_sim))
    print("Random MC3 Accuracy")
    rand = [id2correctness[k] for k,v in id2sel.items() if v == 'random' and id2mc[k] == 'mc_3']
    random_mc3 = sum(rand)/len(rand)
    print(sum(rand)/len(rand))

    print("High Sim MC4 Accuracy")
    high_sim = [id2correctness[k] for k,v in id2sel.items() if v == 'high_sim' and id2mc[k] == 'mc_4']
    high_sim_mc4 = sum(high_sim)/len(high_sim)
    print(sum(high_sim)/len(high_sim))
    print("Random MC4 Accuracy")
    rand = [id2correctness[k] for k,v in id2sel.items() if v == 'random' and id2mc[k] == 'mc_4']
    random_mc4 = sum(rand)/len(rand)
    print(sum(rand)/len(rand))

    print("MC2 All Accuracy")
    mc2 = [id2correctness[k] for k,v in id2mc.items() if v == 'mc_2']
    mc2_all = sum(mc2)/len(mc2)
    print(sum(mc2)/len(mc2))

    print("MC3 All Accuracy")
    mc3 = [id2correctness[k] for k,v in id2mc.items() if v == 'mc_3']
    mc3_all = sum(mc3)/len(mc3)
    print(sum(mc3)/len(mc3))

    print("MC4 All Accuracy")
    mc4 = [id2correctness[k] for k,v in id2mc.items() if v == 'mc_4']
    mc4_all = sum(mc4)/len(mc4)
    print(sum(mc4)/len(mc4))

    all_output[tp] = {
        "total": total,
        "correct": correct,
        "accuracy": correct/total,
        "high_sim_all": high_sim_all,
        "rand_all": rand_all,
        "high_sim_mc2": high_sim_mc2,
        "rand_mc2": random_mc2,
        "high_sim_mc3": high_sim_mc3,
        "rand_mc3": random_mc3,
        "high_sim_mc4": high_sim_mc4,
        "rand_mc4": random_mc4,
        "mc2_all": mc2_all,
        "mc3_all": mc3_all,
        "mc4_all": mc4_all,
        "individual": id2correctness
    }

import numpy as np
def msnr(res, no_input, rand):
    out = {}
    def get_msrn(p_m, p_0, p_r):
        return (p_m-np.mean([p_0, p_r]))/np.mean([p_0, p_r])
    for k,v in res.items():
        if not isinstance(v, float):
            continue
        out[k] = get_msrn(v, no_input[k], rand[k])
    return out
        
all_output["msnr"] = msnr(all_output["mm"], all_output["no_input"], all_output["rand"])

json.dump(all_output, open(f"{args.results_file.replace('.json', '_msrn_results.json')}", 'w'))
print(all_output["msnr"])

