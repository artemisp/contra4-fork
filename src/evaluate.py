import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gt_file", default="../data/final_data/test.json", type=str)
parser.add_argument("--pred_file", type=str)
args = parser.parse_args()

with open(args.pred_file, "r") as f:
    pred = json.load(f)
with open(args.gt_file, "r") as f:
    gt = json.load(f)


gt_dict = {d["id"]: d for d in gt}
if 'question_id' in pred[0]:
    key_id = 'question_id'
elif 'id' in pred[0]:
    key_id = 'id'
elif 'sample_id' in pred[0]:
    key_id = 'sample_id'
else:
    raise ValueError("pred file has no id field")

pred_dict = {d[key_id]: d for d in pred}

pred_key = 'pred_ans'
gt_key = 'gt_answer'

if 'pred_ans' not in pred_dict[list(pred_dict.keys())[0]]:
    pred_key = 'answer'

predid2gt = {d[key_id]:d[gt_key] for d in pred}
pred2acc = {d[key_id]:d[gt_key] in d[pred_key] for d in pred}
if gt_key not in gt_dict[list(gt_dict.keys())[0]]:
    for k in gt_dict:
        if k not in predid2gt:
            print(k)
            continue
        if predid2gt[k] !=  gt_dict[k]['answers'][0]:
            from pdb import set_trace; set_trace()
        gt_dict[k][gt_key] = predid2gt[k]

correct = 0
total = 0
for d in pred:
    if d[key_id] not in gt_dict:
        print(k)
        continue
    if d[gt_key] in d[pred_key]:
        correct += 1
    total += 1

print(f"all: {correct}/{total} ({correct/total:.2f})")

mc2 = [d[key_id] for d in gt if d['q_type'] == 'mc_2']
mc2_hs = [d[key_id] for d in gt if d['q_type'] == 'mc_2' and d['selection_type'] != 'random']
mc2_rand = [d[key_id] for d in gt if d['q_type'] == 'mc_2' and d['selection_type'] == 'random']
mc3 = [d[key_id] for d in gt if d['q_type'] == 'mc_3']
mc3_hs = [d[key_id] for d in gt if d['q_type'] == 'mc_3' and d['selection_type'] != 'random']
mc3_rand = [d[key_id] for d in gt if d['q_type'] == 'mc_3' and d['selection_type'] == 'random']
mc4 = [d[key_id] for d in gt if d['q_type'] == 'mc_4']
mc4_hs = [d[key_id] for d in gt if d['q_type'] == 'mc_4' and d['selection_type'] != 'random']
mc4_rand = [d[key_id] for d in gt if d['q_type'] == 'mc_4' and d['selection_type'] == 'random']
sample_hs = [d[key_id] for d in gt if d['selection_type'] != 'random']
sample_rand = [d[key_id] for d in gt if d['selection_type'] == 'random']
all_ = [d[key_id] for d in gt]
samples_types = {
    "mc2": mc2, 
    "mc2_hs": mc2_hs,
    "mc2_rand": mc2_rand,
    "mc3": mc3,
    "mc3_hs": mc3_hs,
    "mc3_rand": mc3_rand,
    "mc4": mc4,
    "mc4_hs": mc4_hs,
    "mc4_rand": mc4_rand,
    "sample_hs": sample_hs,
    "sample_rand": sample_rand,
    "all": all_
}
out = ""

for k in ["mc2_rand", "mc2_hs", "mc2", "mc3_rand", "mc3_hs", "mc3", "mc4_rand", "mc4_hs", "mc4", "sample_rand", "sample_hs", "all"]:
    v = samples_types[k]
    correct = 0
    total = 0
    for i in v:
        if i not in pred_dict:
            print(i)
        else:

            if pred_dict[i][gt_key] in pred_dict[i][pred_key]:
                correct += 1
            
            total += 1
    if total == 0:
        continue
    print(f"{k}: {correct}/{total} ({correct/total:.2f})")
    if total == 0:
        continue
    out += f"{correct/total:.2f},"
print(out)


modality_groups = list(set([tuple(d['modalities']) for d in gt if len(d['modalities']) == len(set(d['modalities']))]))
for modality_group in modality_groups:
    modality_group = list(modality_group)
    correct = 0
    total = 0
    for d in gt:
        if d['modalities'] == modality_group:
            if d[key_id] not in pred_dict:
                print(d[key_id])
            if d[gt_key][0] in pred_dict[d[key_id]][pred_key]:
                correct += 1
            total += 1
    print(f"{modality_group}: {correct}/{total} ({correct/total:.2f})")
    
