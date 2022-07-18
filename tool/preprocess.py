import argparse
import os
import sys
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.append(os.getcwd())
from loader import get_dataset


def preprocess(data_root):
    # obtain the training data with only positive labels
    with open(os.path.join(data_root, 'train_data.csv'), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(data_root, 'train_data_pos.csv'), 'w') as g:
        for i, line in enumerate(lines):
            if i != 0 and int(line.split(',')[10]) == 0:
                continue
            g.write(line)

    # obtain the test data with only negative labels
    with open(os.path.join(data_root, 'test_data.csv'), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(data_root, 'test_data_pos.csv'), 'w') as g:
        for i, line in enumerate(lines):
            if i != 0 and int(line.split(',')[10]) == 0:
                continue
            g.write(line)

    # obtain the test data with only negative labels, 
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
        index_, filename_, imgid, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk = eval_dataset[i]
        comp_img_299 = transforms.Resize((299, 299), interpolation=InterpolationMode.BILINEAR)(comp_img)
        comp_img_299.save(os.path.join(output_dir_299, '{}_{}.jpg'.format(index_, filename_)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="OPA", help="dataset root")
    opt = parser.parse_args()
    preprocess(opt.data_root)
