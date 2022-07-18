import os
import logging
import datetime
from PIL import Image
import torch


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = log_file
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('logfile = {}'.format(logfile))
    return logger


def make_dirs(save_dir):
    is_old_exp = os.path.exists(save_dir)

    model_dir = os.path.join(save_dir, 'models')
    sample_dir = os.path.join(save_dir, 'sample')
    tblog_dir = os.path.join(save_dir, 'tblog')
    log_path = os.path.join(save_dir, 'log-{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')))

    if not is_old_exp:
        os.makedirs(save_dir)
        os.mkdir(model_dir)
        os.mkdir(sample_dir)
        os.mkdir(tblog_dir)

    return {
        'save_dir': save_dir,
        'model_dir': model_dir,
        'sample_dir': sample_dir,
        'tblog_dir': tblog_dir,
        'log_path': log_path
    }, is_old_exp


def save(model_dir, model, opt, logger=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    sav_path = os.path.join(model_dir, '{}.pth'.format(opt.epoch))
    if logger is None:
        print("=> saving checkpoint to '{}'".format(sav_path))
    else:
        logger.info("=> saving checkpoint to '{}'".format(sav_path))

    torch.save({
        'epoch': opt.epoch,
        'model': model.state_dict(),
        'opt': opt,
        'optimizer': model.optimizer_dict(),
    }, sav_path)


def resume(path, model, resume_list, strict=False, logger=None):
    if path is None:
        return model, 0

    assert (os.path.exists(path))
    if logger is None:
        print("=> loading {} from checkpoint '{}' with strict={}".format(resume_list, path, strict))
    else:
        logger.info("=> loading {} from checkpoint '{}' with strict={}".format(resume_list, path, strict))
    checkpoint = torch.load(path)

    pretrained_model_dict = checkpoint['model']
    model_dict = model.state_dict()
    for k in pretrained_model_dict:
        if k in resume_list:
            model_dict[k].update(pretrained_model_dict[k])
    model.load_state_dict(model_dict, strict=strict)

    pretrained_opt_dict = checkpoint['optimizer']
    opt_dict = model.optimizer_dict()
    for k in pretrained_opt_dict:
        if k in resume_list:
            opt_dict[k].update(pretrained_opt_dict[k])
    model.load_opt_state_dict(opt_dict)

    epoch = checkpoint['epoch']
    return model, epoch
