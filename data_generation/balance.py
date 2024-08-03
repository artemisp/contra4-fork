"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
from collections import Counter
from tqdm import tqdm
all_discrn = json.load(open('data/filters/unanimous_permute.json'))
ans2idx = {"Scene A": 0, "Scene B": 1, "Scene C": 2, "Scene D": 3}
mc2 = []
mc3 = []
mc4 = []
rand = []
hs = []
tot = []
for d in tqdm(all_discrn):
    ans_mod = d["answers"][0]
    if d['q_type'] == "mc_2":
        mc2.append(ans_mod)
    if d['q_type'] == "mc_3":
        mc3.append(ans_mod)
    if d['q_type'] == "mc_4":
        mc4.append(ans_mod)
    
    if d['selection_type'] == "random":
        rand.append(ans_mod)
    if d['selection_type'] != "random":
        hs.append(ans_mod)
    tot.append(ans_mod)

print("MC2")
print(Counter(mc2))
print("MC3")
print(Counter(mc3))
print("MC4")
print(Counter(mc4))
print("Random")
print(Counter(rand))
print("High Similarity")
print(Counter(hs))
final_dataset = all_discrn
total_mc2_r= [d for d in final_dataset if d['q_type'] == 'mc_2' and d['selection_type'] == 'random']
total_mc2_hs= [d for d in final_dataset if d['q_type'] == 'mc_2' and d['selection_type'] != 'random']
total_mc3_r= [d for d in final_dataset if d['q_type'] == 'mc_3' and d['selection_type'] == 'random']
total_mc3_hs= [d for d in final_dataset if d['q_type'] == 'mc_3' and d['selection_type'] != 'random']
total_mc4_r= [d for d in final_dataset if d['q_type'] == 'mc_4' and d['selection_type'] == 'random']
total_mc4_hs= [d for d in final_dataset if d['q_type'] == 'mc_4' and d['selection_type'] != 'random']

ans_candidates = {
    "Scene A": 0,
    "Scene B": 1,
    "Scene C": 2,
    "Scene D":3,
}
for i in range(len(total_mc2_r)):
    if i < .5*len(total_mc2_r):
        answer_idx = ans_candidates[total_mc2_r[i]['answers'][0]]
        if answer_idx == 0:
            continue
        else:
            total_mc2_r[i]['modalities'] = [total_mc2_r[i]['modalities'][1], total_mc2_r[i]['modalities'][0]]
            total_mc2_r[i]['examples'] = [total_mc2_r[i]['examples'][1], total_mc2_r[i]['examples'][0]]
            total_mc2_r[i]['answers'][0] = 'Scene A'
    else:
        answer_idx = ans_candidates[total_mc2_r[i]['answers'][0]]
        if answer_idx == 1:
            continue
        else:
            total_mc2_r[i]['modalities'] = [total_mc2_r[i]['modalities'][1], total_mc2_r[i]['modalities'][0]]
            total_mc2_r[i]['examples'] = [total_mc2_r[i]['examples'][1], total_mc2_r[i]['examples'][0]]
            total_mc2_r[i]['answers'][0] = 'Scene B'

for i in range(len(total_mc2_hs)):
    if i < .5*len(total_mc2_hs):
        answer_idx = ans_candidates[total_mc2_hs[i]['answers'][0]]
        if answer_idx == 0:
            continue
        else:
            total_mc2_hs[i]['modalities'] = [total_mc2_hs[i]['modalities'][1], total_mc2_hs[i]['modalities'][0]]
            total_mc2_hs[i]['examples'] = [total_mc2_hs[i]['examples'][1], total_mc2_hs[i]['examples'][0]]
            total_mc2_hs[i]['answers'][0] = 'Scene A'
    else:
        answer_idx = ans_candidates[total_mc2_hs[i]['answers'][0]]
        if answer_idx == 1:
            continue
        else:
            total_mc2_hs[i]['modalities'] = [total_mc2_hs[i]['modalities'][1], total_mc2_hs[i]['modalities'][0]]
            total_mc2_hs[i]['examples'] = [total_mc2_hs[i]['examples'][1], total_mc2_hs[i]['examples'][0]]
            total_mc2_hs[i]['answers'][0] = 'Scene B'


for i in range(len(total_mc3_r)):
    if i < .33*len(total_mc3_r):
        answer_idx = ans_candidates[total_mc3_r[i]['answers'][0]]
        if answer_idx == 0:
            continue
        else:
            other_idx = [j for j in range(3) if j!=answer_idx]
            total_mc3_r[i]['modalities'] = [total_mc3_r[i]['modalities'][answer_idx], total_mc3_r[i]['modalities'][other_idx[0]], total_mc3_r[i]['modalities'][other_idx[1]]]
            total_mc3_r[i]['examples'] = [total_mc3_r[i]['examples'][answer_idx], total_mc3_r[i]['examples'][other_idx[0]], total_mc3_r[i]['examples'][other_idx[1]]]
            total_mc3_r[i]['answers'][0] = 'Scene A'
    elif i < .66*len(total_mc3_r):
        answer_idx = ans_candidates[total_mc3_r[i]['answers'][0]]
        if answer_idx == 1:
            continue
        else:
            other_idx = [j for j in range(3) if j!=answer_idx]
            total_mc3_r[i]['modalities'] = [total_mc3_r[i]['modalities'][other_idx[0]],total_mc3_r[i]['modalities'][answer_idx],  total_mc3_r[i]['modalities'][other_idx[1]]]
            total_mc3_r[i]['examples'] = [ total_mc3_r[i]['examples'][other_idx[0]],total_mc3_r[i]['examples'][answer_idx], total_mc3_r[i]['examples'][other_idx[1]]]
            total_mc3_r[i]['answers'][0] = 'Scene B'
    else:
        answer_idx = ans_candidates[total_mc3_r[i]['answers'][0]]
        if answer_idx == 2:
            continue
        else:
            other_idx = [j for j in range(3) if j!=answer_idx]
            total_mc3_r[i]['modalities'] = [total_mc3_r[i]['modalities'][other_idx[0]],  total_mc3_r[i]['modalities'][other_idx[1]], total_mc3_r[i]['modalities'][answer_idx]]
            total_mc3_r[i]['examples'] = [ total_mc3_r[i]['examples'][other_idx[0]],total_mc3_r[i]['examples'][other_idx[1]], total_mc3_r[i]['examples'][answer_idx]]
            total_mc3_r[i]['answers'][0] = 'Scene C'
        
for i in range(len(total_mc3_hs)):
    if i < .33*len(total_mc3_hs):
        answer_idx = ans_candidates[total_mc3_hs[i]['answers'][0]]
        if answer_idx == 0:
            continue
        else:
            other_idx = [j for j in range(3) if j!=answer_idx]
            total_mc3_hs[i]['modalities'] = [total_mc3_hs[i]['modalities'][answer_idx], total_mc3_hs[i]['modalities'][other_idx[0]], total_mc3_hs[i]['modalities'][other_idx[1]]]
            total_mc3_hs[i]['examples'] = [total_mc3_hs[i]['examples'][answer_idx], total_mc3_hs[i]['examples'][other_idx[0]], total_mc3_hs[i]['examples'][other_idx[1]]]
            total_mc3_hs[i]['answers'][0] = 'Scene A'
    elif i < .66*len(total_mc3_hs):
        answer_idx = ans_candidates[total_mc3_hs[i]['answers'][0]]
        if answer_idx == 1:
            continue
        else:
            other_idx = [j for j in range(3) if j!=answer_idx]
            total_mc3_hs[i]['modalities'] = [total_mc3_hs[i]['modalities'][other_idx[0]],total_mc3_hs[i]['modalities'][answer_idx],  total_mc3_hs[i]['modalities'][other_idx[1]]]
            total_mc3_hs[i]['examples'] = [ total_mc3_hs[i]['examples'][other_idx[0]],total_mc3_hs[i]['examples'][answer_idx], total_mc3_hs[i]['examples'][other_idx[1]]]
            total_mc3_hs[i]['answers'][0] = 'Scene B'
    else:
        answer_idx = ans_candidates[total_mc3_hs[i]['answers'][0]]
        if answer_idx == 2:
            continue
        else:
            other_idx = [j for j in range(3) if j!=answer_idx]
            total_mc3_hs[i]['modalities'] = [total_mc3_hs[i]['modalities'][other_idx[0]],  total_mc3_hs[i]['modalities'][other_idx[1]], total_mc3_hs[i]['modalities'][answer_idx]]
            total_mc3_hs[i]['examples'] = [ total_mc3_hs[i]['examples'][other_idx[0]],total_mc3_hs[i]['examples'][other_idx[1]], total_mc3_hs[i]['examples'][answer_idx]]
            total_mc3_hs[i]['answers'][0] = 'Scene C'
        
for i in range(len(total_mc4_r)):
    if i < .25*len(total_mc4_r):
        answer_idx = ans_candidates[total_mc4_r[i]['answers'][0]]
        if answer_idx == 0:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_r[i]['modalities'] = [total_mc4_r[i]['modalities'][answer_idx], total_mc4_r[i]['modalities'][other_idx[0]], total_mc4_r[i]['modalities'][other_idx[1]], total_mc4_r[i]['modalities'][other_idx[2]]]
            total_mc4_r[i]['examples'] = [total_mc4_r[i]['examples'][answer_idx], total_mc4_r[i]['examples'][other_idx[0]], total_mc4_r[i]['examples'][other_idx[1]], total_mc4_r[i]['examples'][other_idx[2]]]
            total_mc4_r[i]['answers'][0] = 'Scene A'
    elif i < .50*len(total_mc4_r):
        answer_idx = ans_candidates[total_mc4_r[i]['answers'][0]]
        if answer_idx == 1:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_r[i]['modalities'] = [total_mc4_r[i]['modalities'][other_idx[0]],total_mc4_r[i]['modalities'][answer_idx],  total_mc4_r[i]['modalities'][other_idx[1]], total_mc4_r[i]['modalities'][other_idx[2]]]
            total_mc4_r[i]['examples'] = [ total_mc4_r[i]['examples'][other_idx[0]],total_mc4_r[i]['examples'][answer_idx], total_mc4_r[i]['examples'][other_idx[1]], total_mc4_r[i]['examples'][other_idx[2]]]
            total_mc4_r[i]['answers'][0] = 'Scene B'
    elif i < .75*len(total_mc4_r):
        answer_idx = ans_candidates[total_mc4_r[i]['answers'][0]]
        if answer_idx == 2:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_r[i]['modalities'] = [total_mc4_r[i]['modalities'][other_idx[0]],  total_mc4_r[i]['modalities'][other_idx[1]], total_mc4_r[i]['modalities'][answer_idx], total_mc4_r[i]['modalities'][other_idx[2]]]
            total_mc4_r[i]['examples'] = [ total_mc4_r[i]['examples'][other_idx[0]],total_mc4_r[i]['examples'][other_idx[1]], total_mc4_r[i]['examples'][answer_idx], total_mc4_r[i]['examples'][other_idx[2]]]
            total_mc4_r[i]['answers'][0] = 'Scene C'
    else:
        answer_idx = ans_candidates[total_mc4_r[i]['answers'][0]]
        if answer_idx == 2:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_r[i]['modalities'] = [total_mc4_r[i]['modalities'][other_idx[0]],  total_mc4_r[i]['modalities'][other_idx[1]], total_mc4_r[i]['modalities'][other_idx[2]], total_mc4_r[i]['modalities'][answer_idx]]
            total_mc4_r[i]['examples'] = [ total_mc4_r[i]['examples'][other_idx[0]],total_mc4_r[i]['examples'][other_idx[1]], total_mc4_r[i]['examples'][other_idx[2]], total_mc4_r[i]['examples'][answer_idx]]
            total_mc4_r[i]['answers'][0] = 'Scene D'


for i in range(len(total_mc4_hs)):
    if i < .25*len(total_mc4_hs):
        answer_idx = ans_candidates[total_mc4_hs[i]['answers'][0]]
        if answer_idx == 0:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_hs[i]['modalities'] = [total_mc4_hs[i]['modalities'][answer_idx], total_mc4_hs[i]['modalities'][other_idx[0]], total_mc4_hs[i]['modalities'][other_idx[1]], total_mc4_hs[i]['modalities'][other_idx[2]]]
            total_mc4_hs[i]['examples'] = [total_mc4_hs[i]['examples'][answer_idx], total_mc4_hs[i]['examples'][other_idx[0]], total_mc4_hs[i]['examples'][other_idx[1]], total_mc4_hs[i]['examples'][other_idx[2]]]
            total_mc4_hs[i]['answers'][0] = 'Scene A'
    elif i < .50*len(total_mc4_hs):
        answer_idx = ans_candidates[total_mc4_hs[i]['answers'][0]]
        if answer_idx == 1:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_hs[i]['modalities'] = [total_mc4_hs[i]['modalities'][other_idx[0]],total_mc4_hs[i]['modalities'][answer_idx],  total_mc4_hs[i]['modalities'][other_idx[1]], total_mc4_hs[i]['modalities'][other_idx[2]]]
            total_mc4_hs[i]['examples'] = [ total_mc4_hs[i]['examples'][other_idx[0]],total_mc4_hs[i]['examples'][answer_idx], total_mc4_hs[i]['examples'][other_idx[1]], total_mc4_hs[i]['examples'][other_idx[2]]]
            total_mc4_hs[i]['answers'][0] = 'Scene B'
    elif i < .75*len(total_mc4_hs):
        answer_idx = ans_candidates[total_mc4_hs[i]['answers'][0]]
        if answer_idx == 2:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_hs[i]['modalities'] = [total_mc4_hs[i]['modalities'][other_idx[0]],  total_mc4_hs[i]['modalities'][other_idx[1]], total_mc4_hs[i]['modalities'][answer_idx], total_mc4_hs[i]['modalities'][other_idx[2]]]
            total_mc4_hs[i]['examples'] = [ total_mc4_hs[i]['examples'][other_idx[0]],total_mc4_hs[i]['examples'][other_idx[1]], total_mc4_hs[i]['examples'][answer_idx], total_mc4_hs[i]['examples'][other_idx[2]]]
            total_mc4_hs[i]['answers'][0] = 'Scene C'
    else:
        answer_idx = ans_candidates[total_mc4_hs[i]['answers'][0]]
        if answer_idx == 2:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_hs[i]['modalities'] = [total_mc4_hs[i]['modalities'][other_idx[0]],  total_mc4_hs[i]['modalities'][other_idx[1]], total_mc4_hs[i]['modalities'][other_idx[2]], total_mc4_hs[i]['modalities'][answer_idx]]
            total_mc4_hs[i]['examples'] = [ total_mc4_hs[i]['examples'][other_idx[0]],total_mc4_hs[i]['examples'][other_idx[1]], total_mc4_hs[i]['examples'][other_idx[2]], total_mc4_hs[i]['examples'][answer_idx]]
            total_mc4_hs[i]['answers'][0] = 'Scene D'

balanced_discrn = total_mc2_r+total_mc2_hs+total_mc3_r+total_mc3_hs+total_mc4_r+total_mc4_hs
json.dump(balanced_discrn, open('data/discrn_balanced.json', 'w'))



print("BALANCED")
mc2 = []
mc3 = []
mc4 = []
rand = []
hs = []
tot = []
for d in tqdm(balanced_discrn):
    ans_mod = d["answers"][0]
    if d['q_type'] == "mc_2":
        mc2.append(ans_mod)
    if d['q_type'] == "mc_3":
        mc3.append(ans_mod)
    if d['q_type'] == "mc_4":
        mc4.append(ans_mod)
    
    if d['selection_type'] == "random":
        rand.append(ans_mod)
    if d['selection_type'] != "random":
        hs.append(ans_mod)
    tot.append(ans_mod)

print("MC2")
print(Counter(mc2))
print("MC3")
print(Counter(mc3))
print("MC4")
print(Counter(mc4))
print("Random")
print(Counter(rand))
print("High Similarity")
print(Counter(hs))