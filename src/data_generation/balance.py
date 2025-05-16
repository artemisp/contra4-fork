import json
from collections import Counter
from tqdm import tqdm
selection_type = "random"
split = "test"
all_discrn = json.load(open(f'./data/filters_{split}/unanimous_permute_{selection_type}_{split}.json'))

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

new_data = []
for d in all_discrn:
    new_ans = []
    for ans in d['answers']:
        if answer2choice(ans) != "":
            new_ans.append(answer2choice(ans))
    if len(new_ans) > 0:
        d['answers'] = new_ans
        new_data.append(d)
all_discrn = new_data

ans_candidates = {
    "Scene A": 0,
    "Scene B": 1,
    "Scene C": 2,
    "Scene D":3,
}
new_data = []
for d in all_discrn:
    if len(d['modalities'])!=len(d['examples'])!=int(d['q_type'].split('_')[-1]):
        print(d)
    elif ans_candidates[d['answers'][0]] >= len(d['modalities']):
        print(d)
    else:
        new_data.append(d)
all_discrn = new_data
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
    if d['selection_type'] != "similarity":
        hs.append(ans_mod)
    tot.append(ans_mod)

print("MC2")
print(Counter(mc2))
print("MC3")
print(Counter(mc3))
print("MC4")
print(Counter(mc4))
print("similarity")
print(Counter(rand))
print("High similarity")
print(Counter(hs))
final_dataset = all_discrn
total_mc2_r= [d for d in final_dataset if d['q_type'] == 'mc_2' and d['selection_type'] == 'similarity']
total_mc2_hs= [d for d in final_dataset if d['q_type'] == 'mc_2' and d['selection_type'] != 'similarity']
total_mc3_r= [d for d in final_dataset if d['q_type'] == 'mc_3' and d['selection_type'] == 'similarity']
total_mc3_hs= [d for d in final_dataset if d['q_type'] == 'mc_3' and d['selection_type'] != 'similarity']
total_mc4_r= [d for d in final_dataset if d['q_type'] == 'mc_4' and d['selection_type'] == 'similarity']
total_mc4_hs= [d for d in final_dataset if d['q_type'] == 'mc_4' and d['selection_type'] != 'similarity']


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
            try:
                total_mc3_r[i]['modalities'] = [total_mc3_r[i]['modalities'][other_idx[0]],  total_mc3_r[i]['modalities'][other_idx[1]], total_mc3_r[i]['modalities'][answer_idx]]
                total_mc3_r[i]['examples'] = [ total_mc3_r[i]['examples'][other_idx[0]],total_mc3_r[i]['examples'][other_idx[1]], total_mc3_r[i]['examples'][answer_idx]]
                total_mc3_r[i]['answers'][0] = 'Scene C'
            except:
                from pdb import set_trace; set_trace()
        
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
        if answer_idx == 3:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_r[i]['modalities'] = [total_mc4_r[i]['modalities'][other_idx[0]],  total_mc4_r[i]['modalities'][other_idx[1]], total_mc4_r[i]['modalities'][other_idx[2]], total_mc4_r[i]['modalities'][answer_idx]]
            total_mc4_r[i]['examples'] = [ total_mc4_r[i]['examples'][other_idx[0]],total_mc4_r[i]['examples'][other_idx[1]], total_mc4_r[i]['examples'][other_idx[2]], total_mc4_r[i]['examples'][answer_idx]]
            total_mc4_r[i]['answers'][0] = 'Scene D'

print(len(total_mc4_hs))


for i in range(len(total_mc4_hs)):
    if i < len(total_mc4_hs)//4:
        answer_idx = ans_candidates[total_mc4_hs[i]['answers'][0]]
        if answer_idx == 0:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_hs[i]['modalities'] = [total_mc4_hs[i]['modalities'][answer_idx], total_mc4_hs[i]['modalities'][other_idx[0]], total_mc4_hs[i]['modalities'][other_idx[1]], total_mc4_hs[i]['modalities'][other_idx[2]]]
            total_mc4_hs[i]['examples'] = [total_mc4_hs[i]['examples'][answer_idx], total_mc4_hs[i]['examples'][other_idx[0]], total_mc4_hs[i]['examples'][other_idx[1]], total_mc4_hs[i]['examples'][other_idx[2]]]
            total_mc4_hs[i]['answers'][0] = 'Scene A'
    elif i < len(total_mc4_hs)//2:
        answer_idx = ans_candidates[total_mc4_hs[i]['answers'][0]]
        if answer_idx == 1:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_hs[i]['modalities'] = [total_mc4_hs[i]['modalities'][other_idx[0]],total_mc4_hs[i]['modalities'][answer_idx],  total_mc4_hs[i]['modalities'][other_idx[1]], total_mc4_hs[i]['modalities'][other_idx[2]]]
            total_mc4_hs[i]['examples'] = [ total_mc4_hs[i]['examples'][other_idx[0]],total_mc4_hs[i]['examples'][answer_idx], total_mc4_hs[i]['examples'][other_idx[1]], total_mc4_hs[i]['examples'][other_idx[2]]]
            total_mc4_hs[i]['answers'][0] = 'Scene B'
    elif i < 3*len(total_mc4_hs)//4:
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
        if answer_idx == 3:
            continue
        else:
            other_idx = [j for j in range(4) if j!=answer_idx]
            total_mc4_hs[i]['modalities'] = [total_mc4_hs[i]['modalities'][other_idx[0]],  total_mc4_hs[i]['modalities'][other_idx[1]], total_mc4_hs[i]['modalities'][other_idx[2]], total_mc4_hs[i]['modalities'][answer_idx]]
            total_mc4_hs[i]['examples'] = [ total_mc4_hs[i]['examples'][other_idx[0]],total_mc4_hs[i]['examples'][other_idx[1]], total_mc4_hs[i]['examples'][other_idx[2]], total_mc4_hs[i]['examples'][answer_idx]]
            total_mc4_hs[i]['answers'][0] = 'Scene D'

balanced_discrn = total_mc2_r+total_mc2_hs+total_mc3_r+total_mc3_hs+total_mc4_r+total_mc4_hs
json.dump(balanced_discrn, open(f'./data/filters_{split}/unanimous_permute_{selection_type}_{split}_balanced.json', 'w'))



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
    
    if d['selection_type'] != "similarity":
        rand.append(ans_mod)
    else:
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
print("High similarity")
print(Counter(hs))