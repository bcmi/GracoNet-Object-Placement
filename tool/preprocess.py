import argparse
import csv
import os
import sys
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())
from loader import get_dataset


def preprocess(data_root):
    # calculate the mapping from foreground id to category name
    fg_id_to_catnm = {}
    root = os.path.join(data_root, 'foreground')
    for catnm in os.listdir(root): 
        for filename in os.listdir(os.path.join(root, catnm)):
            if filename.startswith('mask'):
                continue
            id = filename.split('.')[0]
            assert (id not in fg_id_to_catnm)
            fg_id_to_catnm[id] = catnm

    # calculate set of background ids for each category
    bg_catnm_idset = {}
    root = os.path.join(data_root, 'background')
    for catnm in os.listdir(root): 
        for filename in os.listdir(os.path.join(root, catnm)):
            id = filename.split('.')[0]
            try:
                bg_catnm_idset[catnm].add(id)
            except:
                bg_catnm_idset[catnm] = set([id])

    # obtain the transformed training data
    with open(os.path.join(data_root, 'train_data.csv'), 'w') as g:
        g.write('imgID,annID,scID,bbox,scale,catnm,position,label,new_img_path,new_msk_path\n')
        csv_data = csv.DictReader(open(os.path.join(data_root, 'train_set.csv'), 'r'))
        for i, row in enumerate(csv_data):
            assert (row['fg_id'] in fg_id_to_catnm)
            catnm = fg_id_to_catnm[row['fg_id']]
            assert (catnm in bg_catnm_idset and row['bg_id'] in bg_catnm_idset[catnm])
            res = [
                str(i), row['fg_id'], row['bg_id'], '"' + row['position'] + '"', 
                row['scale'], catnm, "-1", 
                row['label'], row['img_name'][8:], row['mask_name'][8:]
            ]
            g.write(','.join(res) + '\n')

    # obtain the transformed test data
    with open(os.path.join(data_root, 'test_data.csv'), 'w') as g:
        g.write('imgID,annID,scID,bbox,scale,catnm,position,label,new_img_path,new_msk_path\n')
        csv_data = csv.DictReader(open(os.path.join(data_root, 'test_set.csv'), 'r'))
        for i, row in enumerate(csv_data):
            assert (row['fg_id'] in fg_id_to_catnm)
            catnm = fg_id_to_catnm[row['fg_id']]
            assert (catnm in bg_catnm_idset and row['bg_id'] in bg_catnm_idset[catnm])
            res = [
                str(i), row['fg_id'], row['bg_id'], '"' + row['position'] + '"', 
                row['scale'], catnm, "-1", 
                row['label'], row['img_name'][8:], row['mask_name'][8:]
            ]
            g.write(','.join(res) + '\n')

    # obtain the training data with only positive labels
    with open(os.path.join(data_root, 'train_data.csv'), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(data_root, 'train_data_pos.csv'), 'w') as g:
        for i, line in enumerate(lines):
            if i != 0 and int(line.split(',')[10]) == 0:
                continue
            g.write(line)

    # obtain the test data with only positive labels
    with open(os.path.join(data_root, 'test_data.csv'), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(data_root, 'test_data_pos.csv'), 'w') as g:
        for i, line in enumerate(lines):
            if i != 0 and int(line.split(',')[10]) == 0:
                continue
            g.write(line)

    # obtain the test data with only positive labels, 
    # and filter out repetitive samples with the same foregroud/background pairs
    with open(os.path.join(data_root, 'test_data_pos.csv'), 'r') as f:
        lines = f.readlines()
    fgbg_set = set([])
    with open(os.path.join(data_root, 'test_data_pos_unique.csv'), 'w') as g:
        for i, line in enumerate(lines):
            if i == 0:
                g.write(line)
                continue
            line_list = line.split(',')
            annID, scID = line_list[1], line_list[2]
            fgbg_mark = annID + '_' + scID
            if fgbg_mark in fgbg_set:
                continue
            fgbg_set.add(fgbg_mark)
            g.write(line)

    # obtain the ground-truth positive composite images that are resized to size 299,
    # which are used to calculate FID scores during evaluation
    output_dir_299 = os.path.join(data_root, "com_pic_testpos299")
    if not os.path.exists(output_dir_299):
        os.makedirs(output_dir_299)
    eval_dataset = get_dataset("OPABasicDataset", image_size=None, mode_type="eval", data_root=data_root)
    for i in range(len(eval_dataset)):
        index_, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk = eval_dataset[i]
        comp_img_299 = transforms.Resize((299, 299), interpolation=Image.BILINEAR)(comp_img)
        comp_img_299.save(os.path.join(output_dir_299, '{}.jpg'.format(index_)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="OPA", help="dataset root")
    opt = parser.parse_args()
    preprocess(opt.data_root)
