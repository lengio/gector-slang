from math import ceil
from typing import List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from gector.gec_model import GecBERTModel
import pandas as pd
from matplotlib import pylab as plt

from sentence_correctness_model import SentenceCorrectnessT5, base_model_name


# %%
def load_model():
    return GecBERTModel(vocab_path='model/vocabulary',
                        model_paths=['model/roberta-large_1_best_10k.th', 'model/xlnet-large_1_best_10k.th'],
                        max_len=50, min_len=3,
                        iterations=5,
                        min_error_probability=0.0,
                        lowercase_tokens=False,
                        model_name='roberta',
                        special_tokens_fix=1,
                        log=False,
                        confidence=0,
                        del_confidence=0,
                        is_ensemble=True,
                        weigths=None)


# %%
def process_batches(texts: List[str], model: GecBERTModel, batch_size=128):
    probabilities = []
    predicted_labels = []
    error_probabilities = []
    batch = []
    for i, sent in enumerate(texts):
        batch.append(sent.split())
        if len(batch) == batch_size or i == len(texts) - 1:
            batch_probabilities, batch_labels, batch_error_probs = model.get_tags(batch)
            probabilities.extend(batch_probabilities)
            predicted_labels.extend(batch_labels)
            error_probabilities.extend(batch_error_probs)
            batch = []
    return probabilities, predicted_labels, error_probabilities


# %%
def dataframe_from_sample(original_f, result_f, expected):
    originals = open(original_f).readlines()
    results = open(result_f).readlines()
    return pd.DataFrame({
        'original': [o.strip() for o in originals],
        'result': [r.strip() for r in results],
        'incorrect': expected,
        'corrected': [o.strip() != r.strip() for o, r in zip(originals, results)]
    })


df_sample = dataframe_from_sample('examples/sample.txt', 'examples/sample_out.txt', False)
df_incorrect = dataframe_from_sample('examples/incorrect.txt', 'examples/incorrect_out.txt', True)

# %%
df = pd.concat([df_sample, df_incorrect])
model = load_model()
# %%
probabilities, predicted_labels, error_probabilities = process_batches(df.original.values, model)
# %%
del model
# %%
df['error_probabilities'] = error_probabilities
df['avg_probability'] = [sum(probs) / len(probs) for probs in probabilities]
max_error_prob = [
    max([0] + [p for p, t in zip(probs, tags) if t != '$KEEP'])
    for probs, tags in zip(probabilities, predicted_labels)
]
df['max_error_prob'] = max_error_prob
# %%
a = 0.365
b = 0.65
colors = ['red' if x else 'blue' for x in df.incorrect.values]
# %%
plt.scatter(df.error_probabilities, df.corrected, c=colors, marker="+")
plt.plot([a, b], [1, 0], '-g')
plt.show()
# %%

df['prediction'] = (df.error_probabilities + (b - a) * df.corrected - b) > 0

print(f'incorrect: {df.incorrect.value_counts()[True]}')
print(f'correct: {df.incorrect.value_counts()[False]}')
print(f'false positives: {(~df.incorrect & df.prediction).value_counts()[True]}')
print(f'false negative: {(df.incorrect & ~df.prediction).value_counts()[True]}')
#%%
df.to_csv('examples/test.csv')

# %%

df = pd.read_csv('examples/test.csv')
#%%
print(f'incorrect: {df.incorrect.value_counts()[True]}')
print(f'correct: {df.incorrect.value_counts()[False]}')
print(f'false positives: {(~df.incorrect & df.corrected).value_counts()[True]}')
print(f'false negative: {(df.incorrect & ~df.corrected).value_counts()[True]}')