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
from model_placenet import GAN
from infer_placenet import infer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=str, choices=list(dataset_dict.keys()), default="OPADst3", help="dataloder type")
    parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=256, help="size of images")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="adam: weight decay")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--resume_pth", type=str, default=None, help="specify a .pth path to resume training, or None to train from scratch")
    parser.add_argument("--data_root", type=str, default="OPA", help="dataset root")
    parser.add_argument("--eval_type", type=str, choices=["train", "trainpos", "sample", "eval", "evaluni"], default="eval", help="evaluation type")
    parser.add_argument("--samp_N", type=int, default=4, help="sampling count of random z during training")
    parser.add_argument("--d_emb", type=int, default=512, help="embedding dimension")
    parser.add_argument("--d_fc_gen", type=int, default=512, help="generator fc dimension")
    parser.add_argument("--d_fc_disc", type=int, default=512, help="discriminator fc dimension")
    parser.add_argument("--margin", type=float, default=1, help="alpha in ndiv loss")
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

    g_gan_loss_meter, g_ndiv_loss_meter, d_real_loss_meter, d_fake_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for epoch in range(start_ep, opt.n_epochs):
        for i, (indices, bg_img_feats, fg_img_feats, fg_msk_feats, comp_img_feats, comp_msk_feats, comp_bboxes, labels, catnms) in enumerate(train_loader):
            model.start_train()
            g_gan_loss, g_ndiv_loss, d_real_loss, d_fake_loss = model.train_disc_gen(bg_img_feats, fg_img_feats, fg_msk_feats, comp_img_feats, comp_msk_feats, comp_bboxes, labels)

            tb_logger.log_value('g_gan_loss', g_gan_loss.item(), step=model.Eiters)
            tb_logger.log_value('g_ndiv_loss', g_ndiv_loss.item(), step=model.Eiters)
            tb_logger.log_value('d_real_loss', d_real_loss.item(), step=model.Eiters)
            tb_logger.log_value('d_fake_loss', d_fake_loss.item(), step=model.Eiters)

            bs = len(indices)
            g_gan_loss_meter.update(g_gan_loss.item(), bs)
            g_ndiv_loss_meter.update(g_ndiv_loss.item(), bs)
            d_real_loss_meter.update(d_real_loss.item(), bs)
            d_fake_loss_meter.update(d_fake_loss.item(), bs)

            if (epoch * len(train_loader) + i) % 10 == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [G - gan: %.3f ndiv: %.5f] [D - real: %.3f, fake1: %.3f]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(train_loader), g_gan_loss_meter.avg, g_ndiv_loss_meter.avg, d_real_loss_meter.avg, d_fake_loss_meter.avg)
                )

        opt.epoch = epoch + 1
        if opt.with_infer:
            with torch.no_grad():
                infer(eval_loader, opt, model)

        save(model_dir, model, opt, logger=logger)

        g_gan_loss_meter.reset()
        g_ndiv_loss_meter.reset()
        d_real_loss_meter.reset()
        d_fake_loss_meter.reset()


if __name__ == '__main__':
    main()
