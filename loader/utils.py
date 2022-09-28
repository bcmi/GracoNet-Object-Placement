import csv
from PIL import Image
import numpy as np
from torchvision import transforms


def obtain_opa_data(csv_file):
    csv_data = csv.DictReader(open(csv_file, 'r'))
    res_data = [
        [
            i, int(row['annID']), int(row['scID']),
            list(map(int, row['bbox'][1:-1].split(','))),
            row['scale'], int(row['label']), row['catnm'],
            row['new_img_path'], row['new_msk_path'],
        ]
        for i, row in enumerate(csv_data)
    ]
    return res_data


def img_crop(x, x_mode, bbox):
    assert (x_mode in ['gray', 'color'])
    h_low, h_high, w_low, w_high = bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]
    y_arr = np.array(x, dtype=np.uint8)
    if x_mode == 'gray':
        y_arr = y_arr[h_low:h_high, w_low:w_high]
    else:
        y_arr = y_arr[h_low:h_high, w_low:w_high, :]
    y = Image.fromarray(y_arr)
    return y


def get_trans_label(bg_img, fg_img, bbox):
    assert (bg_img.size[0] > bbox[2] and bg_img.size[1] > bbox[3])
    bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
    trans_label = np.zeros(3, dtype=np.float32) # [relative_scale, relative_x, relative_y] in (0,1)^3
    if bg_w / bg_h > fg_w / fg_h:
        trans_label[0] = bbox[3] / bg_h
    else:
        trans_label[0] = bbox[2] / bg_w
    trans_label[1] = bbox[0] / (bg_w - bbox[2])
    trans_label[2] = bbox[1] / (bg_h - bbox[3])
    assert (trans_label.min() >= 0 and trans_label.max() <= 1)
    return trans_label


def gen_composite_image(bg_img, fg_img, fg_msk, trans, fg_bbox=None):
    def modify(x, y, w, h):
        if x < 0:
            x = 0
        if x >= bg_img.size[0]:
            x = bg_img.size[0] - 1
        if y < 0:
            y = 0
        if y >= bg_img.size[1]:
            y = bg_img.size[1] - 1
        if w <= 0:
            w = 1
        if h <= 0:
            h = 1
        return x, y, w, h
    if fg_bbox != None:
        fg_img = img_crop(fg_img, 'color', fg_bbox)
        fg_msk = img_crop(fg_msk, 'gray', fg_bbox)
    bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
    relative_scale, relative_x, relative_y = trans[0], trans[1], trans[2]
    if bg_w / bg_h > fg_w / fg_h:
        fg_w_new, fg_h_new = bg_h * relative_scale * fg_w / fg_h, bg_h * relative_scale
    else:
        fg_w_new, fg_h_new = bg_w * relative_scale, bg_w * relative_scale * fg_h / fg_w
    start_x, start_y, width, height = round((bg_w - fg_w_new) * relative_x), round((bg_h - fg_h_new) * relative_y), round(fg_w_new), round(fg_h_new)
    start_x, start_y, width, height = modify(start_x, start_y, width, height)
    resize_func = transforms.Resize((height, width), interpolation=Image.BILINEAR)
    fg_img_new, fg_msk_new = resize_func(fg_img), resize_func(fg_msk)
    comp_img_arr, bg_img_arr, fg_img_arr, fg_msk_arr = np.array(bg_img), np.array(bg_img), np.array(fg_img_new), np.array(fg_msk_new)
    fg_msk_arr_norm = fg_msk_arr[:,:,np.newaxis].repeat(3, axis=2) / 255.0
    comp_img_arr[start_y:start_y+height, start_x:start_x+width, :] = fg_msk_arr_norm * fg_img_arr + (1.0 - fg_msk_arr_norm) * bg_img_arr[start_y:start_y+height, start_x:start_x+width, :]
    comp_img = Image.fromarray(comp_img_arr.astype(np.uint8)).convert('RGB')
    comp_msk_arr = np.zeros(comp_img_arr.shape[:2])
    comp_msk_arr[start_y:start_y+height, start_x:start_x+width] = fg_msk_arr
    comp_msk = Image.fromarray(comp_msk_arr.astype(np.uint8)).convert('L')
    return comp_img, comp_msk, [start_x, start_y, width, height]
