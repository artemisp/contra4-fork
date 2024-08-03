"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
from collections import defaultdict
import numpy as np

model2results = {
    "X-InstructBLIP":'../../cross_modal_baselines/results/xinstructblip.json',
    "CREMA":'../../cross_modal_baselines/results/crema.json',
    "OneLLM":'../../cross_modal_baselines/results/onellm.json'
}
model2mod2performance = {}
model2ansmod2perfomance = {}
model2anscategory2performance = {}

CATEGORIES = ["Location", "Comparison", "Action", "Counting","Motion","Sports", "Existence", "Location", "Properties", "Emotional Response", "Other"]

for model, results_file in model2results.items():
    results = json.load(open(results_file))
    gt = json.load(open('../../data/discrn_balanced.json', 'r'))
    id_key = 'question_id' if 'question_id' in results[0] else 'id'
    gt_key = 'gt_ans' if 'gt_ans' in results[0] else 'gt_answer'
    pred_key = 'pred_ans' if 'pred_ans' in results[0] else 'answer'


    id2mc = {d['id']:d['q_type'] for d in gt}
    id2category = {d['id']:d['category'] for d in gt}
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


    total = 0
    correct = 0
    mod2performance =  defaultdict(list)
    ansmod2performance =  defaultdict(list)
    anscategory2performance = defaultdict(list)
    id2correctness = defaultdict(int)
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
                mod2performance[frozenset(id2ex[a[id_key]]['modalities'])].append(1) 
                ansmod2performance[gt_modality].append(1)  
                if id2category[a[id_key]] in CATEGORIES:
                    anscategory2performance[id2category[a[id_key]]].append(1)
                else:
                    anscategory2performance['Other'].append(1)
                 
        else:
            mod2performance[frozenset(id2ex[a[id_key]]['modalities'])].append(0)
            ansmod2performance[gt_modality].append(0)   
            if id2category[a[id_key]] in CATEGORIES:
                anscategory2performance[id2category[a[id_key]]].append(0)
            else:
                anscategory2performance['Other'].append(0)
            # anscategory2performance[id2category[a[id_key]]].append(0)
        total+=1
    model2mod2performance[model] = mod2performance
    model2ansmod2perfomance[model] = ansmod2performance
    anscategory2performance['Emotion'] = anscategory2performance['Emotional Response']
    del anscategory2performance['Emotional Response']
    model2anscategory2performance[model] = anscategory2performance
    


import matplotlib.pyplot as plt

models = ['X-InstructBLIP', 'CREMA', 'OneLLM']
all_data = [model2mod2performance[model] for model in models]
all_data_ans = [model2ansmod2perfomance[model] for model in models]
all_data_cat = [model2anscategory2performance[model] for model in models]
all_data = [{k:sum(v)/len(v) for k,v in all_data[0].items()}, {k:sum(v)/len(v) for k,v in all_data[1].items()}, {k:sum(v)/len(v) for k,v in all_data[2].items()}]
all_data_ans = [{k:sum(v)/len(v) for k,v in all_data_ans[0].items()}, {k:sum(v)/len(v) for k,v in all_data_ans[1].items()}, {k:sum(v)/len(v) for k,v in all_data_ans[2].items()}]
all_data_cat = [{k:sum(v)/len(v) for k,v in all_data_cat[0].items()}, {k:sum(v)/len(v) for k,v in all_data_cat[1].items()}, {k:sum(v)/len(v) for k,v in all_data_cat[2].items()}]
xinstructblip = all_data_ans[0]
crema = all_data_ans[1]
onellm = all_data_ans[2]
# set fontsize
plt.rc('font', size=20)

# Labels and data for plotting
labels = list(set(xinstructblip.keys()).union(crema.keys()).union(onellm.keys()))
labels = sorted(labels, key=lambda x: (len(x), x))

xinstructblip_values = [xinstructblip.get(label, 0) for label in labels]
crema_values = [crema.get(label, 0) for label in labels]
onellm_values = [onellm.get(label, 0) for label in labels]
# parse_label = {'audio': 'A', 'image': 'I', 'pc': '3D', 'video': 'V'}
parse_label = {'audio': 'Audio', 'image': 'Image', 'pc': '3D', 'video': 'Video'}
x = range(len(labels))
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 8))
ax.bar(x, xinstructblip_values, width, label='X-InstructBLIP', color=plt.cm.Set3.colors[4])
ax.bar([p + width for p in x], crema_values, width, label='CREMA', color=plt.cm.Set3.colors[3])
ax.bar([p + width * 2 for p in x], onellm_values, width, label='OneLLM', color=plt.cm.Set3.colors[2])

# ax.set_xlabel('Modality Combinations')
ax.set_ylabel('Accuracy')
# ax.set_title('Comparative Bar Plot of Different Sources')
ax.set_xticks([p + width for p in x])
ax.set_xticklabels([parse_label[label] for label in labels], rotation=0)

# ax.set_xticklabels([' + '.join([parse_label[l] for l in label]) for label in labels], rotation=0)
# ax.legend()

plt.savefig('plots/answer_modality_comparison_performance.pdf', bbox_inches='tight')
plt.clf()

xinstructblip_values_ans = xinstructblip_values
crema_values_ans = crema_values
onellm_values_ans = onellm_values
labes_ans = labels


xinstructblip = all_data[0]
crema = all_data[1]
onellm = all_data[2]
# set fontsize
plt.rc('font', size=14)

# Labels and data for plotting
labels = list(set(xinstructblip.keys()).union(crema.keys()).union(onellm.keys()))
labels = sorted(labels, key=lambda x: (len(x), x))

xinstructblip_values = [xinstructblip.get(label, 0) for label in labels]
crema_values = [crema.get(label, 0) for label in labels]
onellm_values = [onellm.get(label, 0) for label in labels]
parse_label = {'audio': 'A', 'image': 'I', 'pc': '3D', 'video': 'V'}
# parse_label = {'audio': 'Audio', 'image': 'Image', 'pc': '3D', 'video': 'Video'}
x = range(len(labels))
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(25, 8))
ax.bar(x, xinstructblip_values, width, label='X-InstructBLIP', color=plt.cm.Set3.colors[4])
ax.bar([p + width for p in x], crema_values, width, label='CREMA', color=plt.cm.Set3.colors[3])
ax.bar([p + width * 2 for p in x], onellm_values, width, label='OneLLM', color=plt.cm.Set3.colors[2])

# ax.set_xlabel('Modality Combinations')
ax.set_ylabel('Accuracy')
# ax.set_title('Comparative Bar Plot of Different Sources')
ax.set_xticks([p + width for p in x])
# ax.set_xticklabels([parse_label[label] for label in labels], rotation=0)

ax.set_xticklabels([' + '.join([parse_label[l] for l in label]) for label in labels], rotation=0)
ax.legend()

plt.savefig('plots/modality_comparison_performance.pdf', bbox_inches='tight')
plt.clf()


# set font size
plt.rc('font', size=16)
fig, axs = plt.subplots(1, 3, figsize=(25, 6), subplot_kw=dict(polar=True))
def plot_radar_chart(ax, labels, values, line_color, label=None):
    # Number of variables we're plotting.
    num_vars = len(labels)

    # Compute angle each bar is centered on:
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    print(len(angles)   , len(values), len(labels))
    # The radar chart is a circle, so it needs to be closed by appending the start value to the end.
    values += values[:1]
    angles += angles[:1]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14)
    # Draw the plot
    ax.set_facecolor('#F6F6F6')  
    ax.set_yticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0],labels=[0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)
    ax.fill(angles, values, color=line_color, alpha=0.25)
    ax.plot(angles, values, color=line_color, label=label)  # Plot the data


parse_label = {'audio': 'Audio', 'image': 'Image', 'pc': '3D', 'video': 'Video'}
labels_ = [parse_label[label] for label in labes_ans]
values = xinstructblip_values_ans

plot_radar_chart(axs[0], labels_, values, line_color=plt.cm.Set3.colors[4], label='X-InstructBLIP')
values = crema_values_ans
plot_radar_chart(axs[0], labels_, values, line_color=plt.cm.Set3.colors[3], label='CREMA')
values = onellm_values_ans
plot_radar_chart(axs[0], labels_, values, line_color=plt.cm.Set3.colors[2], label='OneLLM')
axs[0].set_title('Answer Modality')

parse_label = {'audio': 'A', 'image': 'I', 'pc': '3D', 'video': 'V'}
labels_ = ['+'.join([parse_label[l] for l in label]) for label in labels]
values = xinstructblip_values

plot_radar_chart(axs[1], labels_, values, line_color=plt.cm.Set3.colors[4], label='X-InstructBLIP')
values = crema_values
plot_radar_chart(axs[1], labels_, values, line_color=plt.cm.Set3.colors[3], label='CREMA')
values = onellm_values
plot_radar_chart(axs[1], labels_, values, line_color=plt.cm.Set3.colors[2], label='OneLLM')
axs[1].set_title('Input Modalities')

labels_ = [category for category in all_data_cat[0]]
values = [all_data_cat[0][category] for category in all_data_cat[0]]
plot_radar_chart(axs[2], labels_, values, line_color=plt.cm.Set3.colors[4], label='X-InstructBLIP')
labels_ = [category for category in all_data_cat[1]]
values = [all_data_cat[1][category] for category in all_data_cat[1]]
plot_radar_chart(axs[2], labels_, values, line_color=plt.cm.Set3.colors[3], label='CREMA')
labels_ = [category for category in all_data_cat[2]]
values = [all_data_cat[2][category] for category in all_data_cat[2]]
plot_radar_chart(axs[2], labels_, values, line_color=plt.cm.Set3.colors[2], label='OneLLM')
axs[2].set_title('Category')
plt.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.3))
plt.subplots_adjust(wspace=0.46)  # Adjust this value based on your visual preference
plt.show()
