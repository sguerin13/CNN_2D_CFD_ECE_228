# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:35:08 2020

@author: Aleks
"""
import matplotlib.pyplot as plt
import pickle

folder = 'runs_hist'
baseline_file = "runs_hist/baseline.p"
groups = [
    ["runs_hist/16_filters.p",
    "runs_hist/32_filters.p",
    "runs_hist/128_filters.p"],
    
    ["runs_hist/high_learning_rate_(1e-3).p",
    "runs_hist/low_learning_rate_(1e-5).p"],
    
    ["runs_hist/high_weight_decay_(1e-4).p",
    "runs_hist/low_weight_decay_(1e-5).p"],
    
    ["runs_hist/short_model.p",
    "runs_hist/straight_model.p",
    "runs_hist/tiny_model.p"]
]

all_leg = []
all_time = []
new_baseline = True
for group in groups:
    plt.ion()
    plt.figure()
    leg = []
    
    baseline = pickle.load(open(baseline_file, 'rb'))
    plt.semilogy(baseline['val_loss'])
    if new_baseline:
        new_baseline = False
        all_leg.append(baseline['name'])
        all_time.append(baseline['elapsed'])
    leg.append(baseline['name'])
    
    for file in group:
        record = pickle.load(open(file, 'rb'))
        plt.semilogy(record['val_loss'])
        leg.append(record['name'])
        all_leg.append(record['name'])
        all_time.append(record['elapsed'])
    
    plt.legend(leg)
    plt.title("Validation loss vs epoch and hyperparameter")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    
plt.figure()
x_pos = [i for i, _ in enumerate(all_leg)]
plt.barh(x_pos, all_time)
plt.xlabel("Model variation")
plt.ylabel("Training time (s)")
plt.title("Training time vs model type")
plt.yticks(x_pos, all_leg)

plt.subplots_adjust(left=0.3)