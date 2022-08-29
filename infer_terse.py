import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from loader import dataset_dict, get_loader
from loader.utils import gen_composite_image


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
        from model_terse import GAN
        model_dir = os.path.join(save_dir, 'models')
        model_path = os.path.join(model_dir, str(opt.epoch) + '.pth')
        assert(os.path.exists(model_path))
        model = GAN(opt)
        loaded = torch.load(model_path)
        assert(opt.epoch == loaded['epoch'])
        model.load_state_dict(loaded['model'], strict=True)
    model.start_eval()

    gen_res = []

    for i, (indices, annids, scids, bg_img_arrs, fg_img_arrs, fg_msk_arrs, comp_img_arrs, comp_msk_arrs, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, comp_img_feats, comp_msk_feats, comp_crop_feats, labels, trans_labels, catnms) in enumerate(tqdm(eval_loader)):
        index, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, comp_img_arr, comp_msk_arr, label, trans_label, catnm = \
            indices[0], annids[0], scids[0], bg_img_arrs[0], fg_img_arrs[0], fg_msk_arrs[0], comp_img_arrs[0], comp_msk_arrs[0], labels[0], trans_labels[0], catnms[0]
        for repeat_id in range(repeat):
            pred_img_, pred_msk_, pred_trans_ = model.test_genorator(bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes)
            gen_comp_img, gen_comp_msk, gen_comp_bbox = gen_composite_image(
                bg_img=Image.fromarray(bg_img_arr.numpy().astype(np.uint8)).convert('RGB'), 
                fg_img=Image.fromarray(fg_img_arr.numpy().astype(np.uint8)).convert('RGB'), 
                fg_msk=Image.fromarray(fg_msk_arr.numpy().astype(np.uint8)).convert('L'), 
                trans=(pred_trans_.cpu().numpy().astype(np.float32)[0]).tolist(),
                fg_bbox=None
            )
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
    parser.add_argument("--dst", type=str, choices=list(dataset_dict.keys()), default="OPADst1", help="dataloder type")
    parser.add_argument("--img_size", type=int, default=256, help="size of images")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="adam: weight decay")
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--data_root", type=str, default="OPA", help="dataset root")
    parser.add_argument("--eval_type", type=str, choices=["train", "trainpos", "sample", "eval", "evaluni"], default="eval", help="evaluation type")
    parser.add_argument("--dim_fc", type=int, default=1600, help="fc input dimension")
    parser.add_argument("--d_model", type=int, default=512, help="backbone feature dimension")
    parser.add_argument("--d_branch", type=int, default=20, help="branch feature dimension")
    parser.add_argument("--epoch", type=int, required=True, help="which epoch to evaluate")
    parser.add_argument("--repeat", type=int, default=1, help="number of times to sample different random vectors")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    eval_loader = get_loader(opt.dst, batch_size=1, num_workers=1, image_size=opt.img_size, shuffle=False, mode_type=opt.eval_type, data_root=opt.data_root)
    with torch.no_grad():
        infer(eval_loader, opt, model=None, repeat=opt.repeat)
