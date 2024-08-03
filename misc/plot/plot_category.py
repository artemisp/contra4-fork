"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
from collections import Counter
from matplotlib import pyplot as plt

data = json.load(open('data/discrn_balanced.json'))

c = Counter([d['category'] for d in data])
common_count = {k[0]:c[k[0]] for k in c.most_common()[:14]}
common_count['other'] = sum([p[1] for p in c.most_common()[14:]])
common_count['Other'] = common_count['other']; del common_count['other']

plt.figure(figsize=(10,10))
plt.rc('font', size=11)
plt.pie(common_count.values(), labels=common_count.keys(), autopct='%d%%', colors=plt.cm.Set3(range(len(common_count))))
# plt.legend(bbox_to_anchor=(.9, .95), loc='upper left')
plt.savefig('plot/plots/topic_distribution.pdf', bbox_inches='tight')
