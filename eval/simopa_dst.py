import os
import torch
import numpy as np
from PIL import Image
import csv
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from simopa_cfg import opt


class ImageDataset(Dataset):
    def __init__(self, istrain=True):
        self.istrain = istrain

        with open(opt.train_data_path if istrain else opt.test_data_path, "r") as f:
            reader = csv.reader(f)
            reader = list(reader)
        title = reader[0]
        annid_index = title.index('annID')
        scid_index = title.index('scID')
        category_index = title.index('catnm')
        label_index = title.index('label')
        image_path_index = title.index('img_path')
        mask_path_index = title.index('msk_path')
        target_box_index = title.index('bbox')

        self.sample_ids = []
        self.labels = []
        self.images_path = []
        self.mask_path = []
        self.target_box = []
        self.dic_name = []
        self.target_class = []

        for row in reader[1:]:
            category = row[category_index]
            label = int(row[label_index])
            image_path = row[image_path_index]
            mask_path = row[mask_path_index]
            target_box = eval(row[target_box_index])
            sample_id = "{}_{}_{}_{}_{}_{}".format(row[annid_index], row[scid_index], target_box[0], target_box[1], target_box[2], target_box[3])
            self.sample_ids.append(sample_id)
            self.labels.append(label)
            self.images_path.append(os.path.join(opt.img_path, image_path))
            self.mask_path.append(os.path.join(opt.mask_path, mask_path))
            self.target_box.append(target_box)
            self.dic_name.append(image_path)
            self.target_class.append(category)

        self.img_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor()
        ])
        self.transforms_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
        ])

        # reference box and depth feature
        if istrain:
            self.refer_box_dic = np.load(opt.box_dic_path, allow_pickle=True)
            self.depth_feats_path = opt.depth_feats_path
            self.target_features = np.load(opt.train_target_feature_path, allow_pickle=True)
            self.refer_features = np.load(opt.train_reference_feature_path)
        else:
            self.refer_box_dic = np.load(opt.test_box_dic_path, allow_pickle=True)
            self.target_features = np.load(opt.test_target_feature_path, allow_pickle=True)
            self.refer_features = np.load(opt.test_reference_feature_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert('RGB')
        w = img.width
        h = img.height
        img = self.img_transform(img)

        mask = Image.open(self.mask_path[index]).convert('L')
        mask = self.img_transform(mask)

        is_flip = False
        if self.istrain and np.random.uniform() < 0.5:
            img = self.transforms_flip(img)
            mask = self.transforms_flip(mask)
            is_flip = True
        img_mask = torch.cat([img, mask], dim=0)

        label = self.labels[index]
        target_box = self.target_box[index]
        x1, y1, bw, bh = target_box
        x2, y2 = x1 + bw, y1 + bh
        if is_flip:
            x1 = w - x1
            x2 = w - x2
            x1, x2 = x2, x1
        target_box = torch.tensor([x1, y1, x2, y2])

        refer_box = self.refer_box_dic[index]
        refer_score = refer_box[:, -1]
        refer_keep = np.argsort(refer_score)[::-1][:opt.refer_num]
        refer_box = refer_box[refer_keep]
        refer_box = torch.from_numpy(refer_box)

        refer_feats = self.refer_features[index][refer_keep]
        target_feats = self.target_features[index]

        if is_flip:
            x1, y1, x2, y2 = refer_box[:, 0], refer_box[:, 1], refer_box[:, 2], refer_box[:, 3]
            x1 = w - x1
            x2 = w - x2
            x1, x2 = x2, x1
            refer_box = torch.cat([x1[:, None], y1[:, None], x2[:, None], y2[:, None],
                                   refer_box[:, 4:5], refer_box[:, 5:]], dim=1)

        # produce binary mask for target/reference boxes
        bm_size = opt.binary_mask_size
        scale_x1 = (target_box[0] / w * bm_size).int()
        scale_y1 = (target_box[1] / h * bm_size).int()
        scale_x2 = (target_box[2] / w * bm_size).int()
        scale_y2 = (target_box[3] / h * bm_size).int()

        target_mask = torch.zeros(1, bm_size, bm_size, dtype=img.dtype)
        target_mask[0, scale_y1: scale_y2, scale_x1: scale_x2] = 1

        refer_mask = torch.zeros(opt.refer_num, bm_size, bm_size, dtype=target_mask.dtype)
        scale_x1 = (refer_box[:, 0] / w * bm_size).int()
        scale_y1 = (refer_box[:, 1] / h * bm_size).int()
        scale_x2 = (refer_box[:, 2] / w * bm_size).int()
        scale_y2 = (refer_box[:, 3] / h * bm_size).int()

        for i in range(opt.refer_num):
            refer_mask[i, scale_y1[i]: scale_y2[i], scale_x1[i]: scale_x2[i]] = 1
        tar_class = self.target_class[index]

        sample_id = self.sample_ids[index]
        return sample_id, img_mask, label, target_box, refer_box, target_feats, refer_feats, target_mask, refer_mask, tar_class, w, h

    def __len__(self):
        return len(self.labels)
