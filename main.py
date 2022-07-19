import argparse
import os
import torch
from PIL import Image
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorboard_logger as tb_logger

from tool.utils import make_dirs, save, resume, make_logger, AverageMeter
from loader import dataset_dict, get_loader, get_dataset
from model import GAN
from infer import sample, infer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=str, choices=list(dataset_dict.keys()), default="OPADst1", help="dataloder type")
    parser.add_argument("--n_epochs", type=int, default=11, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=256, help="size of image")
    parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--resume_pth", type=str, default=None, help="specify a .pth path to resume training, or None to train from scratch")
    parser.add_argument("--data_root", type=str, default="OPA", help="dataset root")
    parser.add_argument("--eval_type", type=str, choices=["train", "trainpos", "sample", "eval", "evaluni"], default="eval", help="evaluation type")
    parser.add_argument("--d_noise", type=int, default=1024, help="dimension of random noise/vector")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of features")
    parser.add_argument("--d_k", type=int, default=64, help="dimension of key in multi-head attention")
    parser.add_argument("--d_v", type=int, default=64, help="dimension of value in multi-head attention")
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads in multi-head attention")
    parser.add_argument("--len_k", type=int, default=84, help="number of background nodes")
    parser.add_argument("--with_infer", action='store_true', default=False, help="action to make inference after each training epoch")
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
    save_dir = os.path.join('result', opt.expid)
    dirs, is_old_exp = make_dirs(save_dir)
    model_dir, sample_dir, tblog_dir, log_path = dirs['model_dir'], dirs['sample_dir'], dirs['tblog_dir'], dirs['log_path']
    assert (is_old_exp or opt.resume_pth is None)

    tb_logger.configure(tblog_dir, flush_secs=5)
    logger = make_logger(log_path)
    logger.info(opt)

    train_loader = get_loader(opt.dst, batch_size=opt.batch_size, num_workers=8, image_size=opt.img_size, shuffle=True, mode_type="train", data_root=opt.data_root)
    sample_dataset = get_dataset(opt.dst, image_size=opt.img_size, mode_type="sample", data_root=opt.data_root)
    eval_loader = get_loader(opt.dst, batch_size=1, num_workers=1, image_size=opt.img_size, shuffle=False, mode_type=opt.eval_type, data_root=opt.data_root)

    model = GAN(opt)
    model, start_ep = resume(opt.resume_pth, model, resume_list=['generator', 'discriminator'], strict=True, logger=logger)
    assert (start_ep < opt.n_epochs)
    model.Eiters = start_ep * len(train_loader)

    g_gan_loss1_meter, g_gan_loss2_meter, g_rec_loss_meter, g_kld_loss_meter, d_real_loss_meter, d_fake_loss1_meter, d_fake_loss2_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for epoch in range(start_ep, opt.n_epochs):
        for i, (indices, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, comp_img_feats, comp_msk_feats, comp_crop_feats, labels, trans_labels, catnms) in enumerate(train_loader):
            model.start_train()
            g_gan_loss1, g_gan_loss2, g_rec_loss, g_kld_loss, d_real_loss, d_fake_loss1, d_fake_loss2 = \
                             model.train_disc_gen(bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, comp_img_feats, comp_msk_feats, labels, trans_labels)

            tb_logger.log_value('g_gan_loss1', g_gan_loss1.item(), step=model.Eiters)
            tb_logger.log_value('g_gan_loss2', g_gan_loss2.item(), step=model.Eiters)
            tb_logger.log_value('g_rec_loss', g_rec_loss.item(), step=model.Eiters)
            tb_logger.log_value('g_kld_loss', g_kld_loss.item(), step=model.Eiters)
            tb_logger.log_value('d_real_loss', d_real_loss.item(), step=model.Eiters)
            tb_logger.log_value('d_fake_loss1', d_fake_loss1.item(), step=model.Eiters)
            tb_logger.log_value('d_fake_loss2', d_fake_loss2.item(), step=model.Eiters)

            bs = len(indices)
            g_gan_loss1_meter.update(g_gan_loss1.item(), bs)
            g_gan_loss2_meter.update(g_gan_loss2.item(), bs)
            g_rec_loss_meter.update(g_rec_loss.item(), bs)
            g_kld_loss_meter.update(g_kld_loss.item(), bs)
            d_real_loss_meter.update(d_real_loss.item(), bs)
            d_fake_loss1_meter.update(d_fake_loss1.item(), bs)
            d_fake_loss2_meter.update(d_fake_loss2.item(), bs)

            if (epoch * len(train_loader) + i) % 10 == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [G - gan1: %.3f, gan2: %.3f, rec: %.3f, kld: %.4f] [D - real: %.3f, fake1: %.3f, fake2: %.3f]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(train_loader), g_gan_loss1_meter.avg, g_gan_loss2_meter.avg, g_rec_loss_meter.avg, g_kld_loss_meter.avg, d_real_loss_meter.avg, d_fake_loss1_meter.avg, d_fake_loss2_meter.avg)
                )

            if (epoch * len(train_loader) + i) % opt.sample_interval == 0:
                with torch.no_grad():
                    sample(sample_dataset, model, epoch * len(train_loader) + i, sample_dir)

        opt.epoch = epoch + 1
        if opt.with_infer:
            with torch.no_grad():
                infer(eval_loader, opt, model)

        save(model_dir, model, opt, logger=logger)

        g_gan_loss1_meter.reset()
        g_gan_loss2_meter.reset()
        g_rec_loss_meter.reset()
        g_kld_loss_meter.reset()
        d_real_loss_meter.reset()
        d_fake_loss1_meter.reset()
        d_fake_loss2_meter.reset()


if __name__ == '__main__':
    main()
