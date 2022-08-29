import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from network_placenet import Encoder


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img, is_cuda=True):
        assert (img.shape[1] == 3)
        proc_img = torch.zeros(img.shape)
        if is_cuda:
            proc_img = proc_img.cuda()
        proc_img[:, 0, :, :] = (img[:, 0, :, :] - self.mean[0]) / self.std[0]
        proc_img[:, 1, :, :] = (img[:, 1, :, :] - self.mean[1]) / self.std[1]
        proc_img[:, 2, :, :] = (img[:, 2, :, :] - self.mean[2]) / self.std[2]

        return proc_img


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)


class SysNet(nn.Module):
    def __init__(self, opt, init_weight=True):
        super(SysNet, self).__init__()
        self.opt = opt
        self.normalize = Normalize()

        self.fg_encoder = Encoder()
        self.bg_encoder = Encoder()
        self.decoder = nn.Sequential(
            nn.Linear(opt.d_emb * 2 + 3, opt.d_fc_gen),
            nn.BatchNorm1d(opt.d_fc_gen),
            nn.ReLU(inplace=True),
            nn.Linear(opt.d_fc_gen, opt.d_fc_gen),
            nn.BatchNorm1d(opt.d_fc_gen),
            nn.ReLU(inplace=True),
            nn.Linear(opt.d_fc_gen, 3),
            nn.Sigmoid()
        )

        if init_weight:
            self.initialize_weight()

    def initialize_weight(self):
        self.decoder.apply(weights_init_normal)

    def enc(self, bg_img, fg_img, samp_N):
        fg_feats_ = self.fg_encoder(fg_img).unsqueeze(1).repeat(1, samp_N, 1)
        bg_feats_ = self.bg_encoder(bg_img).unsqueeze(1).repeat(1, samp_N, 1)
        fg_feats = fg_feats_.view(-1, fg_feats_.shape[2])
        bg_feats = bg_feats_.view(-1, bg_feats_.shape[2])
        return fg_feats, bg_feats
    
    def dec(self, noises, fg_feats, bg_feats):
        dec_in = torch.cat((fg_feats, noises, bg_feats), dim=1)
        locations = self.decoder(dec_in)
        return locations

    def forward(self, bg_img, fg_img, is_train=True):
        samp_N = self.opt.samp_N if is_train else 1
        batch_size = bg_img.shape[0]
        bg_img_norm = self.normalize(bg_img)
        fg_img_norm = self.normalize(fg_img)
        fg_feats, bg_feats = self.enc(bg_img_norm, fg_img_norm, samp_N=samp_N)
        noises = torch.randn((batch_size * samp_N, 3)).cuda()
        locations = self.dec(noises, fg_feats, bg_feats)
        return noises, locations, fg_feats, bg_feats


class Discriminator(nn.Module):
    def __init__(self, opt, init_weight=True):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.d_emb * 2 + 3, opt.d_fc_disc),
            nn.BatchNorm1d(opt.d_fc_disc),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(opt.d_fc_disc, opt.d_fc_disc),
            nn.BatchNorm1d(opt.d_fc_disc),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(opt.d_fc_disc, 1),
            nn.Sigmoid()
        )

        if init_weight:
            self.initialize_weight()

    def initialize_weight(self):
        self.model.apply(weights_init_normal)

    def forward(self, locations, fg_feats, bg_feats):
        disc_in = torch.cat([fg_feats, locations, bg_feats], dim=1)
        disc_out = self.model(disc_in)
        return disc_out.view(-1)


class GAN(object):
    def __init__(self, opt):
        self.Eiters = 0
        self.opt = opt
        self.generator = SysNet(opt)
        self.discriminator = Discriminator(opt)
        self.to_cuda(multigpus=False)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
        self.discri_loss = torch.nn.BCELoss()

    def start_train(self):
        self.generator.train()
        self.discriminator.train()

    def start_eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def to_cuda(self, multigpus=False):
        if multigpus:
            self.generator = nn.DataParallel(self.generator, device_ids=[0, 1])
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=[0, 1])
        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()

    def state_dict(self):
        model_dict = dict()
        model_dict["generator"] = self.generator.state_dict()
        model_dict["discriminator"] = self.discriminator.state_dict()
        return model_dict

    def optimizer_dict(self):
        optimizer_state_dict = dict()
        optimizer_state_dict["generator"] = self.optimizer_G.state_dict()
        optimizer_state_dict["discriminator"] = self.optimizer_D.state_dict()
        return optimizer_state_dict

    def load_state_dict(self, pretrained_dict, strict=False):
        for k in pretrained_dict:
            if k == "generator":
                self.generator.load_state_dict(pretrained_dict[k], strict=strict)
            elif k == "discriminator":
                self.discriminator.load_state_dict(pretrained_dict[k], strict=strict)

    def load_opt_state_dict(self, pretrained_dict):
        for k in pretrained_dict:
            if k == "generator":
                self.optimizer_G.load_state_dict(pretrained_dict[k])
            elif k == "discriminator":
                self.optimizer_D.load_state_dict(pretrained_dict[k])

    def train_disc_gen(self, bg_img, fg_img, fg_msk, comp_img, comp_msk, comp_bbox, label):
        self.Eiters += 1
        batch_size = len(label)

        bg_img_v = Variable(bg_img, requires_grad=False).cuda()
        fg_img_v = Variable(fg_img, requires_grad=False).cuda()
        comp_bbox_v = Variable(comp_bbox, requires_grad=False).cuda().unsqueeze(1).repeat(1, self.opt.samp_N, 1).view(-1, 3)
        label_v = Variable(label.float(), requires_grad=False).cuda().unsqueeze(1).repeat(1, self.opt.samp_N).view(-1)
        valid = Variable(torch.ones(batch_size * self.opt.samp_N), requires_grad=False).cuda()
        fake = Variable(torch.zeros(batch_size * self.opt.samp_N), requires_grad=False).cuda()

        # forward
        noises, locations, fg_feats, bg_feats = self.generator(bg_img_v, fg_img_v)
        discri_target_gen = self.discriminator(locations, fg_feats, bg_feats)
        discri_target_gen_detach = self.discriminator(locations.detach(), fg_feats.detach(), bg_feats.detach())
        discri_target_real = self.discriminator(comp_bbox_v, fg_feats.detach(), bg_feats.detach())

        # discriminator loss
        d_real_loss = self.discri_loss(discri_target_real, label_v)
        d_fake_loss = self.discri_loss(discri_target_gen_detach, fake)
        d_loss = d_real_loss + d_fake_loss

        # generator loss
        noises_ = noises.view(batch_size, self.opt.samp_N, 3)
        locations_ = locations.view(batch_size, self.opt.samp_N, 3)
        D_z = torch.norm(noises_[:,:,None] - noises_[:,None,], dim=3, p=2)
        D_y = torch.norm(locations_[:,:,None] - locations_[:,None,], dim=3, p=2)
        D_z_avg = D_z / (torch.sum(D_z, dim=2).unsqueeze(2).repeat(1, 1, self.opt.samp_N))
        D_y_avg = D_y / (torch.sum(D_y, dim=2).unsqueeze(2).repeat(1, 1, self.opt.samp_N))
        g_ndiv_loss = torch.sum(F.relu(self.opt.margin * D_z_avg - D_y_avg)) / ((self.opt.samp_N * self.opt.samp_N - self.opt.samp_N) * batch_size)
        g_gan_loss = self.discri_loss(discri_target_gen, valid)
        g_loss = g_gan_loss + g_ndiv_loss * 50

        # generator backward
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # discriminator backward
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()  

        return g_gan_loss, g_ndiv_loss, d_real_loss, d_fake_loss

    def test_genorator(self, bg_img, fg_img):
        bg_img = bg_img.cuda()
        fg_img = fg_img.cuda()
        noises, locations, fg_feats, bg_feats = self.generator(bg_img, fg_img, is_train=False)
        return locations
