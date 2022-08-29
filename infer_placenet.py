import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from loader import dataset_dict, get_loader


def img_pad(img, padding, img_type="RGB"):
    w, h = img.size[0], img.size[1]
    img_template = Image.new(img_type, (w + 2 * padding, h + 2 * padding))
    img_template.paste(img, (padding, padding))
    return img_template

def img_crop(img, padding):
    w, h = img.size[0], img.size[1]
    img_crop = img.crop((padding, padding, w - padding, h - padding))
    return img_crop

def gen_composite_image(bg_img, fg_img, fg_msk, locations):
    bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
    padding = min(bg_w, bg_h)
    xc, yc = bg_w * locations[0], bg_h * locations[1]
    if bg_w / bg_h > fg_w / fg_h:
        w_, h_ = bg_h * locations[2] * fg_w / fg_h, bg_h * locations[2]
    else:
        w_, h_ = bg_w * locations[2], bg_w * locations[2] * fg_h / fg_w
    x, y, w, h = int(round(xc - w_ / 2)), int(round(yc - h_ / 2)), int(max(1, round(w_))), int(max(1, round(h_)))
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > bg_w or y + h > bg_h:
        _w = bg_w - x
        _h = bg_h - y
        r = min(_w / w, _h / h)
        w, h = int(max(1, r * w)), int(max(1, r * h))
    fg_img = fg_img.resize((w, h))
    fg_msk = fg_msk.resize((w, h))
    bg_img_pad = img_pad(bg_img, padding=padding, img_type="RGB")
    bg_img_pad.paste(fg_img, (x + padding, y + padding), fg_msk)
    gen_img = img_crop(bg_img_pad, padding=padding)
    bg_zero_pad = Image.new("L", (bg_w + 2 * padding, bg_h + 2 * padding))
    bg_zero_pad.paste(fg_msk, (x + padding, y + padding))
    gen_msk = img_crop(bg_zero_pad, padding=padding)
    if x>=bg_w or x+w<=0 or y>=bg_h or y+h<=0:
        white_dot_RGB = Image.new("RGB", (1, 1), "white")
        gen_img.paste(white_dot_RGB, (bg_w//2, bg_h//2))
        white_dot_L = Image.new("L", (1, 1), "white")
        gen_msk.paste(white_dot_L, (bg_w//2, bg_h//2))
        bbox = [bg_w//2, bg_h//2, 1, 1]
    else:
        x1, x2, y1, y2 = max(0, x), min(bg_w, x+w), max(0, y), min(bg_h, y+h)
        bbox = [x1, y1, x2-x1, y2-y1]
    assert (gen_img.size == bg_img.size and gen_msk.size == bg_img.size)
    return gen_img, gen_msk, bbox


def infer(eval_loader, opt, model=None, repeat=1):
    def csv_title():
        return 'annID,scID,bbox,catnm,label,img_path,msk_path'
    def csv_str(annid, scid, gen_comp_bbox, catnm, gen_file_name):
        return '{},{},"{}",{},-1,images/{}.jpg,masks/{}.png'.format(annid, scid, gen_comp_bbox, catnm, gen_file_name, gen_file_name)

    assert (repeat >= 1)
    save_dir = os.path.join('result', opt.expid)
    eval_dir = os.path.join(save_dir, opt.eval_type, str(opt.epoch))
    assert (not os.path.exists(eval_dir))
    img_sav_dir = os.path.join(eval_dir, 'images')
    msk_sav_dir = os.path.join(eval_dir, 'masks')
    csv_sav_file = os.path.join(eval_dir, '{}.csv'.format(opt.eval_type))
    os.makedirs(eval_dir)
    os.mkdir(img_sav_dir)
    os.mkdir(msk_sav_dir)

    if model is None:
        from model_placenet import GAN
        model_dir = os.path.join(save_dir, 'models')
        model_path = os.path.join(model_dir, str(opt.epoch) + '.pth')
        assert(os.path.exists(model_path))
        model = GAN(opt)
        loaded = torch.load(model_path)
        assert(opt.epoch == loaded['epoch'])
        model.load_state_dict(loaded['model'], strict=True)
    model.start_eval()

    gen_res = []

    for i, (indices, annids, scids, comp_w, comp_h, bg_img_arrs, fg_img_arrs, fg_msk_arrs, comp_img_arrs, comp_msk_arrs, bg_img_feats, fg_img_feats, fg_msk_feats, comp_img_feats, comp_msk_feats, comp_bboxes, labels, catnms) in enumerate(tqdm(eval_loader)):
        index, annid, scid, comp_w, comp_h, bg_img_arr, fg_img_arr, fg_msk_arr, comp_img_arr, comp_msk_arr, comp_bbox, label, catnm = indices[0], annids[0], scids[0], comp_w[0], comp_h[0], bg_img_arrs[0], fg_img_arrs[0], fg_msk_arrs[0], comp_img_arrs[0], comp_msk_arrs[0], comp_bboxes[0], labels[0], catnms[0]
        for repeat_id in range(repeat):
            bg_img = Image.fromarray(bg_img_arr.numpy().astype(np.uint8)).convert('RGB')
            fg_img = Image.fromarray(fg_img_arr.numpy().astype(np.uint8)).convert('RGB')
            fg_msk = Image.fromarray(fg_msk_arr.numpy().astype(np.uint8)).convert('L')
            pred_locations = model.test_genorator(bg_img_feats, fg_img_feats)
            locations = (pred_locations.cpu().numpy().astype(np.float32)[0]).tolist()
            assert (len(locations) == 3)
            gen_comp_img, gen_comp_msk, gen_comp_bbox = gen_composite_image(bg_img, fg_img, fg_msk, locations)
            if repeat == 1:
                gen_file_name = "{}_{}_{}_{}_{}_{}_{}".format(index, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])
            else:
                gen_file_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(index, repeat_id, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])
            gen_comp_img.save(os.path.join(img_sav_dir, '{}.jpg'.format(gen_file_name)))
            gen_comp_msk.save(os.path.join(msk_sav_dir, '{}.png'.format(gen_file_name)))
            gen_res.append(csv_str(annid, scid, gen_comp_bbox, catnm, gen_file_name))

    with open(csv_sav_file, "w") as f:
        f.write(csv_title() + '\n')
        for line in gen_res:
            f.write(line + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=str, choices=list(dataset_dict.keys()), default="OPADst3", help="dataloder type")
    parser.add_argument("--img_size", type=int, default=256, help="size of images")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="adam: weight decay")
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--data_root", type=str, default="OPA", help="dataset root")
    parser.add_argument("--eval_type", type=str, choices=["train", "trainpos", "sample", "eval", "evaluni"], default="eval", help="evaluation type")
    parser.add_argument("--samp_N", type=int, default=4, help="sampling count of random z during training")
    parser.add_argument("--d_emb", type=int, default=512, help="embedding dimension")
    parser.add_argument("--d_fc_gen", type=int, default=512, help="generator fc dimension")
    parser.add_argument("--d_fc_disc", type=int, default=512, help="discriminator fc dimension")
    parser.add_argument("--margin", type=float, default=1, help="alpha in ndiv loss")
    parser.add_argument("--epoch", type=int, required=True, help="which epoch to evaluate")
    parser.add_argument("--repeat", type=int, default=1, help="number of times to sample different random vectors")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    eval_loader = get_loader(opt.dst, batch_size=1, num_workers=1, image_size=opt.img_size, shuffle=False, mode_type=opt.eval_type, data_root=opt.data_root)
    with torch.no_grad():
        infer(eval_loader, opt, model=None, repeat=opt.repeat)
