"""Convert image features from bottom up attention to numpy array"""

# Example
# python convert_data.py --expid ${expid}$ --epoch ${epoch}$

import os
import base64
import csv
import sys
import argparse
import numpy as np

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_name', 'image_w', 'image_h', 'num_boxes', 'boxes', 'pred_scores', 'features', 'fg_feature']

parser = argparse.ArgumentParser()
parser.add_argument("--expid", type=str, required=True, help="experiment name")
parser.add_argument("--epoch", type=int, required=True, help="epoch for evaluation")
parser.add_argument("--eval_type", type=str, default="eval", help="evaluation type")
args = parser.parse_args()

dataset_dir = os.path.join('../result', args.expid, args.eval_type, str(args.epoch))
csv_file = os.path.join(dataset_dir, '{}.csv'.format(args.eval_type))
assert (os.path.exists(csv_file))
csv_data = csv.DictReader(open(csv_file, 'r'))
meta, bboxes, scores, features, fg_feature = [], {}, {}, {}, {}
for i, row in enumerate(csv_data):
    meta.append(i)
    bboxes[i] = None
    scores[i] = None
    features[i] = None
    fg_feature[i] = None

input_file = os.path.join(dataset_dir, "{}_roiinfos.csv".format(args.eval_type))
assert (os.path.exists(input_file))
with open(input_file, "r+") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for item in reader:
        item['image_id'] = int(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['num_boxes'] = int(item['num_boxes'])
        for field in ['boxes', 'pred_scores', 'features']:
            data = item[field]
            buf = base64.b64decode(data[1:])
            temp = np.frombuffer(buf, dtype=np.float32)
            item[field] = temp.reshape((item['num_boxes'], -1))
        for field in ['fg_feature']:
            data = item[field]
            buf = base64.b64decode(data[1:])
            temp = np.frombuffer(buf, dtype=np.float32)
            item[field] = temp.reshape((1, -1))
        idx = np.argsort(-item['boxes'][:, 5])
        item['boxes'] = item['boxes'][idx, :]
        item['pred_scores'] = item['pred_scores'][idx, :]
        item['features'] = item['features'][idx, :]

        if item['image_id'] in bboxes:
            bboxes[item['image_id']] = item['boxes']
            scores[item['image_id']] = item['pred_scores']
            features[item['image_id']] = item['features']
            fg_feature[item['image_id']] = item['fg_feature']

output_dict = {
    "bboxes": bboxes,
    "scores": scores,
    "feats": features,
    "fgfeats": fg_feature
}
for k, v in output_dict.items():
    output_file = os.path.join(dataset_dir, "{}_{}.npy".format(args.eval_type, k))
    data_out = np.stack([v[sid] for sid in meta], axis=0)
    np.save(output_file, data_out)
