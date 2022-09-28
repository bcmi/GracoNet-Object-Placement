import math
import random
import numpy as np
from PIL import Image
from torchvision import transforms

if __name__ == 'loader.datasets':
    from .base import OPABasicDataset
    from .utils import img_crop, get_trans_label
elif __name__ == '__main__' or __name__ == 'datasets':
    from base import OPABasicDataset
    from utils import img_crop, get_trans_label
else:
    raise NotImplementedError


class OPADst1(OPABasicDataset):
    def __init__(self, size, mode_type, data_root):
        super().__init__(size, mode_type, data_root)

    def __getitem__(self, index):
        index_, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk = super().__getitem__(index)

        bg_img_arr = np.array(bg_img, dtype=np.uint8)
        fg_img_arr = np.array(fg_img, dtype=np.uint8)
        fg_msk_arr = np.array(fg_msk, dtype=np.uint8)
        comp_img_arr = np.array(comp_img, dtype=np.uint8)
        comp_msk_arr = np.array(comp_msk, dtype=np.uint8)

        bg_feat = self.img_trans_bg(bg_img)
        fg_feat = self.img_trans_fg(fg_img, 'color', bg_img, fg_img)
        fg_msk_feat = self.img_trans_fg(fg_msk, 'gray', bg_img, fg_img)
        comp_feat = self.img_trans_bg(comp_img)
        comp_msk_feat = self.img_trans_bg(comp_msk)
        comp_crop_feat = self.img_trans_fg(img_crop(comp_img, 'color', bbox), 'color', bg_img, fg_img)
        fg_bbox = self.get_fg_bbox(bg_img, fg_img)
        trans_label = get_trans_label(bg_img, fg_img, bbox)

        if "eval" in self.mode_type:
            return index_, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, comp_img_arr, comp_msk_arr, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, comp_crop_feat, label, trans_label, catnm
        else:
            return index_, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, comp_crop_feat, label, trans_label, catnm

    def img_trans_bg(self, x):
        y = transforms.Resize((self.size, self.size), interpolation=Image.BILINEAR)(x)
        y = transforms.ToTensor()(y)
        return y

    def img_trans_fg(self, x, x_mode, bg_img, fg_img):
        # assert (math.fabs((x.size[0] * fg_img.size[1]) / (x.size[1] * fg_img.size[0]) - 1.0) < self.error_bar)
        assert (x_mode in ['gray', 'color'])
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y = transforms.Resize((self.size, (self.size * fg_w * bg_h) // (fg_h * bg_w)), interpolation=Image.BILINEAR)(x)
            delta_w = self.size - y.size[0]
            delta_w0, delta_w1 = delta_w // 2, delta_w - (delta_w // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((self.size, delta_w0), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1), dtype=np.uint8)
            else:
                d0 = np.zeros((self.size, delta_w0, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=1)
        else:
            y = transforms.Resize(((self.size * fg_h * bg_w) // (fg_w * bg_h), self.size), interpolation=Image.BILINEAR)(x)
            delta_h = self.size - y.size[1]
            delta_h0, delta_h1 = delta_h // 2, delta_h - (delta_h // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((delta_h0, self.size), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size), dtype=np.uint8)
            else:
                d0 = np.zeros((delta_h0, self.size, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=0)
        y = Image.fromarray(y_arr)
        assert (y.size == (self.size, self.size))
        y = transforms.ToTensor()(y)
        return y

    def get_fg_bbox(self, bg_img, fg_img):
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y_w = (self.size * fg_w * bg_h) // (fg_h * bg_w)
            delta_w0 = (self.size - y_w) // 2
            fg_bbox = np.array([delta_w0, 0, y_w, self.size])
        else:
            y_h = (self.size * fg_h * bg_w) // (fg_w * bg_h)
            delta_h0 = (self.size - y_h) // 2
            fg_bbox = np.array([0, delta_h0, self.size, y_h])
        return fg_bbox


class OPADst3(OPABasicDataset):
    def __init__(self, size, mode_type, data_root):
        super().__init__(size, mode_type, data_root)

    def __getitem__(self, index):
        index_, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk = super().__getitem__(index)

        comp_w, comp_h = comp_img.size[0], comp_img.size[1]

        bg_img_arr = np.array(bg_img, dtype=np.uint8)
        fg_img_arr = np.array(fg_img, dtype=np.uint8)
        fg_msk_arr = np.array(fg_msk, dtype=np.uint8)
        comp_img_arr = np.array(comp_img, dtype=np.uint8)
        comp_msk_arr = np.array(comp_msk, dtype=np.uint8)

        bg_feat = self.img_trans_bg(bg_img)
        fg_feat = self.img_trans_fg(fg_img, 'color', bg_img, fg_img)
        fg_msk_feat = self.img_trans_fg(fg_msk, 'gray', bg_img, fg_img)
        comp_feat = self.img_trans_bg(comp_img)
        comp_msk_feat = self.img_trans_bg(comp_msk)

        comp_bbox = self.get_resized_bbox(bbox, bg_img, fg_img)

        if "eval" in self.mode_type:
            return index_, annid, scid, comp_w, comp_h, bg_img_arr, fg_img_arr, fg_msk_arr, comp_img_arr, comp_msk_arr, bg_feat, fg_feat, fg_msk_feat, comp_feat, comp_msk_feat, comp_bbox, label, catnm
        else:
            return index_, bg_feat, fg_feat, fg_msk_feat, comp_feat, comp_msk_feat, comp_bbox, label, catnm

    def img_trans_bg(self, x):
        y = transforms.Resize((self.size, self.size), interpolation=Image.BILINEAR)(x)
        y = transforms.ToTensor()(y)
        return y

    def img_trans_fg(self, x, x_mode, bg_img, fg_img):
        # assert (math.fabs((x.size[0] * fg_img.size[1]) / (x.size[1] * fg_img.size[0]) - 1.0) < self.error_bar)
        assert (x_mode in ['gray', 'color'])
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y = transforms.Resize((self.size, (self.size * fg_w * bg_h) // (fg_h * bg_w)), interpolation=Image.BILINEAR)(x)
            delta_w = self.size - y.size[0]
            delta_w0, delta_w1 = delta_w // 2, delta_w - (delta_w // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((self.size, delta_w0), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1), dtype=np.uint8)
            else:
                d0 = np.zeros((self.size, delta_w0, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=1)
        else:
            y = transforms.Resize(((self.size * fg_h * bg_w) // (fg_w * bg_h), self.size), interpolation=Image.BILINEAR)(x)
            delta_h = self.size - y.size[1]
            delta_h0, delta_h1 = delta_h // 2, delta_h - (delta_h // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((delta_h0, self.size), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size), dtype=np.uint8)
            else:
                d0 = np.zeros((delta_h0, self.size, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=0)
        y = Image.fromarray(y_arr)
        assert (y.size == (self.size, self.size))
        y = transforms.ToTensor()(y)
        return y

    def get_resized_bbox(self, bbox, bg_img, fg_img):
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        xc = ((x1 + x2) / 2) / bg_w
        yc = ((y1 + y2) / 2) / bg_h
        if bg_w / bg_h > fg_w / fg_h:
            r = h / bg_h
        else:
            r = w / bg_w
        bbox_new = np.array([xc, yc, r], dtype=np.float32)
        return bbox_new
