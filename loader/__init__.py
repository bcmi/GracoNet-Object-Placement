from torch.utils.data import DataLoader

if __name__ == 'loader':
    from .base import OPABasicDataset
    from .datasets import OPADst1, OPADst3
elif __name__ == '__init__':
    from base import OPABasicDataset
    from datasets import OPADst1, OPADst3
else:
    raise NotImplementedError


dataset_dict = {"OPABasicDataset": OPABasicDataset, "OPADst1": OPADst1, "OPADst3": OPADst3}

def get_loader(name, batch_size, num_workers, image_size, shuffle, mode_type, data_root):
    dset = dataset_dict[name](size=image_size, mode_type=mode_type, data_root=data_root)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def get_dataset(name, image_size, mode_type, data_root):
    dset = dataset_dict[name](size=image_size, mode_type=mode_type, data_root=data_root)
    return dset
