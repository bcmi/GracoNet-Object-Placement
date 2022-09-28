import os
from PIL import Image
from torch.utils.data import Dataset

if __name__ == 'loader.base':
    from .utils import obtain_opa_data
elif __name__ == '__main__' or __name__ == 'base':
    from utils import obtain_opa_data
else:
    raise NotImplementedError


class OPABasicDataset(Dataset):
    def __init__(self, size, mode_type, data_root):
        # self.error_bar = 0.15
        self.size = size
        self.mode_type = mode_type
        self.data_root = data_root
        self.bg_dir = os.path.join(data_root, "background")
        self.fg_dir = os.path.join(data_root, "foreground")
        self.fg_msk_dir = os.path.join(data_root, "foreground")

        if mode_type == "train":
            csv_file = os.path.join(data_root, "train_data.csv")
        elif mode_type == "trainpos":
            csv_file = os.path.join(data_root, "train_data_pos.csv")
        elif mode_type == "sample":
            csv_file = os.path.join(data_root, "test_data.csv")
        elif mode_type == "eval":
            csv_file = os.path.join(data_root, "test_data_pos.csv")
        elif mode_type == "evaluni":
            csv_file = os.path.join(data_root, "test_data_pos_unique.csv")
        else:
            raise NotImplementedError
        self.data = obtain_opa_data(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index_, annid, scid, bbox, scale, label, catnm, img_path, msk_path = self.data[index]

        bg_path = os.path.join(self.bg_dir, catnm, "{}.jpg".format(scid))
        fg_path = os.path.join(self.fg_dir, catnm, "{}.jpg".format(annid))
        fg_mask_path = os.path.join(self.fg_msk_dir, catnm, "mask_{}.jpg".format(annid))
        img_path = os.path.join(self.data_root, img_path)
        msk_path = os.path.join(self.data_root, msk_path)

        bg_img = Image.open(bg_path).convert('RGB')
        fg_img = Image.open(fg_path).convert('RGB')
        fg_msk = Image.open(fg_mask_path).convert('L')
        comp_img = Image.open(img_path).convert('RGB')
        comp_msk = Image.open(msk_path).convert('L')

        assert (bg_img.size == comp_img.size and comp_img.size == comp_msk.size and fg_img.size == fg_msk.size)
        # assert (math.fabs((bbox[2] * fg_img.size[1]) / (bbox[3] * fg_img.size[0]) - 1.0) < self.error_bar)
        assert (bbox[0] + bbox[2] <= bg_img.size[0] and bbox[1] + bbox[3] <= bg_img.size[1])

        return index_, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk
