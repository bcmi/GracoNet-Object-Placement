import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

from network_terse import split_branch, RegressionFC


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
    def __init__(self, opt, img_size, init_weight=True):
        super(SysNet, self).__init__()
        self.img_size = img_size
        self.normalize = Normalize()

        self.shared_net = nn.Sequential(*list(models.vgg16_bn(pretrained=True).features[:34]))
        self.bg_net = split_branch(opt)
        self.fg_net = split_branch(opt)
        self.regress_net = RegressionFC(opt)

        if init_weight:
            self.initialize_weight()

    def initialize_weight(self):
        for m in [self.bg_net, self.fg_net, self.regress_net]:
            m.apply(weights_init_normal)

    def gen_blend(self, bg_img, fg_img, fg_msk, fg_bbox, trans):
        batch_size = len(trans)
        theta = torch.cat((
            1 / (trans[:,0] + 1e-6), torch.zeros(batch_size).cuda(), (1 - 2 * trans[:,1]) * (1 / (trans[:,0] + 1e-6) - fg_bbox[:,2] / self.img_size),
            torch.zeros(batch_size).cuda(), 1 / (trans[:,0] + 1e-6), (1 - 2 * trans[:,2]) * (1 / (trans[:,0] + 1e-6) - fg_bbox[:,3] / self.img_size)
        ), dim=0).view(2, 3, batch_size).permute(2, 0, 1).contiguous()
        grid = F.affine_grid(theta, fg_img.size(), align_corners=True)
        fg_img_out = F.grid_sample(fg_img, grid, align_corners=True)
        fg_msk_out = F.grid_sample(fg_msk, grid, align_corners=True)
        comp_out = fg_msk_out * fg_img_out + (1 - fg_msk_out) * bg_img

        return comp_out, fg_msk_out

    def forward(self, bg_img, fg_img, fg_msk, fg_bbox):
        bg_img_norm = self.normalize(bg_img)
        fg_img_norm = self.normalize(fg_img)
        fg_feats = self.shared_net(fg_img_norm)
        bg_feats = self.shared_net(bg_img_norm)
        fg_feats = self.fg_net(fg_feats)
        bg_feats = self.bg_net(bg_feats)
        comb_feats = torch.cat((fg_feats, bg_feats), dim=1)
        trans = self.regress_net(comb_feats)
        trans = torch.tanh(trans) / 2.0 + 0.5
        blend_img, blend_msk = self.gen_blend(bg_img, fg_img, fg_msk, fg_bbox, trans)

        return blend_img, blend_msk, trans


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=True, init_weight=True):
        super(Discriminator, self).__init__()
        self.normalize = Normalize()

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

        if init_weight:
            self.initialize_weight()

    def initialize_weight(self):
        self.model.apply(weights_init_normal)

    def forward(self, img, mask):
        img_norm = self.normalize(img)
        out = self.model(torch.cat((img_norm, mask), dim=1))
        out_avg = F.adaptive_avg_pool2d(out, (1, 1))
        return out_avg.view(out_avg.shape[0])


class GAN(object):
    def __init__(self, opt):
        self.Eiters = 0
        self.generator = SysNet(opt, img_size=opt.img_size)
        self.discriminator = Discriminator(input_nc=4)
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

    def train_disc_gen(self, bg_img, fg_img, fg_msk, fg_bbox, comp_img, comp_msk, label):
        self.Eiters += 1
        batch_size = len(label)

        bg_img_v = Variable(bg_img, requires_grad=False).cuda()
        fg_img_v = Variable(fg_img, requires_grad=False).cuda()
        fg_msk_v = Variable(fg_msk, requires_grad=False).cuda()
        fg_bbox_v = Variable(fg_bbox.float(), requires_grad=False).cuda()
        comp_img_v = Variable(comp_img, requires_grad=False).cuda()
        comp_msk_v = Variable(comp_msk, requires_grad=False).cuda()
        label_v = Variable(label.float(), requires_grad=False).cuda()
        valid = Variable(torch.ones(batch_size), requires_grad=False).cuda()
        fake = Variable(torch.zeros(batch_size), requires_grad=False).cuda()

        # forward
        gen_comps, gen_msks, _ = self.generator(bg_img_v, fg_img_v, fg_msk_v, fg_bbox_v)
        discri_target_gen = self.discriminator(gen_comps, gen_msks)
        discri_target_gen_detach = self.discriminator(gen_comps.detach(), gen_msks.detach())
        discri_target_real = self.discriminator(comp_img_v, comp_msk_v)

        # discriminator loss
        d_real_loss = self.discri_loss(discri_target_real, label_v)
        d_fake_loss = self.discri_loss(discri_target_gen_detach, fake)
        d_loss = d_real_loss + d_fake_loss

        # generator loss
        g_gan_loss = self.discri_loss(discri_target_gen, valid)
        g_loss = g_gan_loss

        # generator backward
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # discriminator backward
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()  

        return g_gan_loss, d_real_loss, d_fake_loss

    def test_genorator(self, bg_img, fg_img, fg_msk, fg_bbox):
        bg_img = bg_img.cuda()
        fg_img = fg_img.cuda()
        fg_msk = fg_msk.cuda()
        fg_bbox = fg_bbox.float().cuda()

        return self.generator(bg_img, fg_img, fg_msk, fg_bbox)
