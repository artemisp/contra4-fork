"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = json.load(open('data/discrn_balanced.json'))

questions = [d['questions'][0] for d in data]

stop_words = set(stopwords.words('english'))
word_count = Counter()
lengths = []
for q in questions:
    words = word_tokenize(q)
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    lengths.append(len(words))
    word_count.update(words)

print(f"Total questions: {len(set(questions))}")
print(f"Total words: {sum(word_count.values()):,}")
print(f"Unique words: {len(word_count):,}")
print(f"Average word count: {sum(lengths)/len(lengths):.2f}")

del word_count["Which"]
del word_count["scene"]
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set3').generate_from_frequencies(word_count)

# Display the generated image:
plt.figure(figsize=(10, 20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # Turn off axis numbers and ticks
wordcloud.to_file('plot/plots/final_wordcloud.pdf')