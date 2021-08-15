# -*- coding: utf-8 -*-

#%% ------------------------------------------------------------
import re
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

#%% ------------------------------------------------------------
results_dir = "./results"
ensemble_results_file = f"{results_dir}/ensemble_results.csv"

#%% ------------------------------------------------------------
pd_results = pd.DataFrame()
#  for i in range(total_folds):
import glob
fold_results_files = glob.glob(f"{results_dir}/result_fold*.txt")
fold_results_files = sorted(fold_results_files)
total_folds = len(fold_results_files)
for i, results_file in enumerate(fold_results_files):
    print(f"{i}: {results_file}")
    pd_result = pd.read_csv(results_file,
                            names=['ID', 'PREDICTION'],
                            header=None)
    if i == 0:
        pd_results['ID'] = pd_result['ID']
    pd_results[f"P{i}"] = pd_result['PREDICTION']

#%% ------------------------------------------------------------
ensemble_results = []

for i, row in tqdm(pd_results.iterrows()):

    results = [
        re.sub("[^\w\*\-\+\.\(\)\（\）]", "", x)
        if type(x) == str and len(x) < 30 else ""
        for x in [row[f"P{i}"] for i in range(total_folds)]
    ]

    results = [x for x in results if len(x) > 0]
    num_results = len(results)
    if num_results == 0:
        ensemble_results.append([row['ID'], "NaN"])
        continue

    c = Counter(results)
    if len(c) == 1:
        ensemble_results.append([row['ID'], results[0]])
        #  ensemble_results.append([row['ID'], "NaN"])
        continue

    guess_results = [row['ID']]

    #  # ----------------
    # Best 0.821840830144845
    # 选项多于4个的，删除只有1张投票的选项。
    predictions = c.most_common()
    for m in predictions:
        if m[0] == "NaN":
            continue
        if len(m[0]) == 0:
            continue
        if len(predictions) < 4:
            guess_results.append(m[0])
        else:
            if m[1] > 1:
                guess_results.append(m[0])

    ensemble_results.append(guess_results)

#%% ------------------------------------------------------------
with open(ensemble_results_file, 'w') as F:
    for predictions in ensemble_results:
        if len(predictions) <= 1:
            line = f"\"{predictions[0]}\",\"NaN"
        else:
            line = f"\"{predictions[0]}\",\""
            for p in predictions[1:]:
                if type(p) == str and len(p) == 0:
                    break
                line += f"{p},"
            line = line[:-1]
        F.write(line + "\"\n")
