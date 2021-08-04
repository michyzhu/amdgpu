"""
FileName:	print_label.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:49:58
"""

import os
import json
import natsort
import data_config


def mapping_types(num_classes):
    if num_classes == 10:
        labels = data_config.class_10
    elif num_classes == 50:
        labels = data_config.class_50
    elif num_classes == 100:
        labels = data_config.class_100
    else:
        raise ValueError("num_classes only support (10, 50, 100)")
    label_to_target = {label: idx for idx, label in enumerate(labels)}
    return label_to_target


num_classes = 10
label_to_target = mapping_types(num_classes)
print(label_to_target)
json_dir = './Datasets/train/json_label'
all_jsons = os.listdir(json_dir)
total_jsons = natsort.natsorted(all_jsons)
# print(total_jsons)
for idx in range(len(total_jsons)):
    path_json = os.path.join(json_dir, total_jsons[idx])
    # print(path_json)
    with open(path_json) as f:
        mrc_json = json.load(f)
    target = label_to_target[mrc_json['name']]
    print(total_jsons[idx], target)
