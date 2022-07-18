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


def validate(model, loader, logger):
    model.start_eval()
    correct = [0, 0, 0]
    size = len(loader.dataset)

    TP = [0, 0, 0]
    TN = [0, 0, 0]
    FP = [0, 0, 0]
    FN = [0, 0, 0]

    fake_pred = torch.zeros(size, )
    real_pred = torch.zeros(size, )
    fake_labels = torch.zeros(size, )
    real_labels = torch.zeros(size, )

    if model.is_cuda:
        fake_pred = fake_pred.cuda()
        real_pred = real_pred.cuda()
        fake_labels = fake_labels.cuda()
        real_labels = real_labels.cuda()

    for i, data in enumerate(loader):
        model.start_eval()

        indices, bg_feats, fg_feats, msk_feats, comp_feats, comp_msk_feats, labels, catnms  = data
        labels = labels.float()
        batch_size = indices.size(0)
        if model.is_cuda:
            labels = labels.cuda()

        discri_sc, real_sc = model.validate_discriminator(bg_feats, fg_feats, msk_feats, comp_feats, comp_msk_feats)

        fake_pred[indices] = discri_sc.ge(0.5).float()
        real_pred[indices] = real_sc.ge(0.5).float()
        real_labels[indices] = labels

    for kk in range(3):
        if kk == 0:
            #####fake pred
            pred = fake_pred
            labels_ = fake_labels
        elif kk == 1:
            #####real pred
            pred = real_pred
            labels_ = real_labels
        else:
            pred = torch.cat((fake_pred, real_pred), dim = 0)
            labels_ = torch.cat((fake_labels, real_labels), dim = 0)
        correct[kk] += pred.eq(labels_.view_as(pred)).sum().item()
        TP[kk] += ((pred.cpu().numpy() == 1) & (labels_.cpu().numpy() == 1)).sum()
        TN[kk] += ((pred.cpu().numpy() == 0) & (labels_.cpu().numpy() == 0)).sum()
        FP[kk] += ((pred.cpu().numpy() == 1) & (labels_.cpu().numpy() == 0)).sum()
        FN[kk] += ((pred.cpu().numpy() == 0) & (labels_.cpu().numpy() == 1)).sum()

    mode = ["fake", "real", "all"]
    for xx in range(3):
        if xx == 0:
            logger.info("Fake output:")
        elif xx == 1:
            logger.info("Real output:")
        elif xx == 2:
            logger.info("All output:")
        acc = 100 * correct[xx] / size
        acc_str = 'Accurancy: {:.4f}%, '.format(acc)

        precision = TP[xx] / (TP[xx] + FP[xx])
        recall = TP[xx] / (TP[xx] + FN[xx])
        fscore = (2 * precision * recall) / (precision + recall)
        fscore_str = 'F-1 Measure: %f, ' % fscore
        pred_neg = TN[xx] / (TN[xx] + FP[xx])
        pred_pos = TP[xx] / (TP[xx] + FN[xx])
        pred_str = 'pred_neg: %f, pred_pos: %f, ' % (pred_neg, pred_pos)
        weighted_acc = (pred_neg + pred_pos) / 2
        weighted_acc_str = 'Weighted acc measure: %f, ' % weighted_acc
        tb_logger.log_value('%s f1 score'%(mode[xx]), fscore, step=model.Eiters)
        tb_logger.log_value('%s Weighted acc'%(mode[xx]), weighted_acc, step=model.Eiters)
        log = acc_str + fscore_str + weighted_acc_str + pred_str + \
              ' TP: %f, TN: %f, FP: %f, FN: %f'%(TP[xx], TN[xx], FP[xx], FN[xx])
        logger.info(log)

    return fscore


def validate_only_real(model, loader, logger):
    """测试数据只有数据库中的数据"""
    model.start_eval()
    correct = 0
    size = len(loader.dataset)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, data in enumerate(loader):
        model.start_eval()

        indices, bg_feats, fg_feats, msk_feats, comp_feats, comp_msk_feats, labels, catnms = data
        labels = labels.float()
        batch_size = indices.size(0)
        if model.is_cuda:
            labels = labels.cuda()

        real_sc = model.validate_discriminator_onlyreal(comp_feats, comp_msk_feats)

        real_pred = real_sc.ge(0.5)

        pred = real_pred
        labels_ = labels

        correct += pred.eq(labels_.view_as(pred)).sum().item()
        TP += ((pred.cpu().numpy() == 1) & (labels_.cpu().numpy() == 1)).sum()
        TN += ((pred.cpu().numpy() == 0) & (labels_.cpu().numpy() == 0)).sum()
        FP += ((pred.cpu().numpy() == 1) & (labels_.cpu().numpy() == 0)).sum()
        FN += ((pred.cpu().numpy() == 0) & (labels_.cpu().numpy() == 1)).sum()


    acc = 100 * correct / size
    acc_str = 'Accurancy: {:.4f}%, '.format(acc)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore_str = 'F-1 Measure: %f, ' % fscore
    pred_neg = TN / (TN + FP)
    pred_pos = TP/ (TP + FN)
    pred_str = 'pred_neg: %f, pred_pos: %f, ' % (pred_neg, pred_pos)
    weighted_acc = (pred_neg + pred_pos) / 2
    weighted_acc_str = 'Weighted acc measure: %f, ' % weighted_acc
    tb_logger.log_value('f1 score', fscore, step=model.Eiters)
    tb_logger.log_value('Weighted acc', weighted_acc, step=model.Eiters)
    log = acc_str + fscore_str + weighted_acc_str + pred_str + \
          ' TP: %f, TN: %f, FP: %f, FN: %f'%(TP, TN, FP, FN)
    logger.info(log)

    return fscore

