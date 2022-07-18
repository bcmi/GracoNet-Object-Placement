import argparse
import csv
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for evaluation")
    parser.add_argument("--eval_type", type=str, default="eval", help="evaluation type")
    opt = parser.parse_args()

    data_dir = os.path.join('result', opt.expid, opt.eval_type, str(opt.epoch))
    assert (os.path.exists(data_dir))
    if not os.path.exists(os.path.join(data_dir, 'images299')):
        os.mkdir(os.path.join(data_dir, 'images299'))
    csv_file = os.path.join(data_dir, '{}.csv'.format(opt.eval_type))
    csv_data = csv.DictReader(open(csv_file, 'r'))
    for i, row in tqdm(enumerate(csv_data)):
        img_src = os.path.join(data_dir, row['img_path'])
        img_tar = os.path.join(data_dir, 'images299', row['img_path'].split('/')[-1])
        comp_img = Image.open(img_src).convert('RGB')
        comp_img_299 = transforms.Resize((299, 299), interpolation=InterpolationMode.BILINEAR)(comp_img)
        comp_img_299.save(img_tar)
