import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from network import vgg16_bn, FgBgAttention, FgBgHead, FgBgRegression, VAEEncoder


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


def get_params(model, key):
    if key == "g1x":
        for m in model.named_modules():
            if "shared_net" in m[0]:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear) or isinstance(m[1], nn.BatchNorm1d) or isinstance(m[1], nn.Embedding) or isinstance(m[1], nn.LayerNorm):
                    for p in m[1].parameters():
                        yield p
    if key == "g10x":
        for m in model.named_modules():
            if "shared_net" not in m[0]:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear) or isinstance(m[1], nn.BatchNorm1d) or isinstance(m[1], nn.Embedding) or isinstance(m[1], nn.LayerNorm):
                    for p in m[1].parameters():
                        yield p
    if key == "d1x":
        for m in model.named_modules():
            if True:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear) or isinstance(m[1], nn.BatchNorm1d) or isinstance(m[1], nn.Embedding) or isinstance(m[1], nn.LayerNorm):
                    for p in m[1].parameters():
                        yield p


class SysNet(nn.Module):
    def __init__(self, opt, img_size, init_weight=True):
        super(SysNet, self).__init__()
        self.opt = opt
        self.img_size = img_size
        self.normalize = Normalize()

        self.shared_net = nn.Sequential(*list(vgg16_bn(pretrained=True).features[:34]))
        self.fg_head = FgBgHead(opt, n_mesh_list=[1])
        self.bg_head = FgBgHead(opt, n_mesh_list=[2, 4, 8])
        self.latent_head = FgBgHead(opt, n_mesh_list=[1])
        self.att = FgBgAttention(opt)
        self.vae_enc = VAEEncoder(opt)
        self.regress_net = FgBgRegression(opt)

        if init_weight:
            self.initialize_weight()

    def initialize_weight(self):
        for m in [self.fg_head, self.bg_head, self.latent_head, self.att, self.vae_enc, self.regress_net]:
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

    def forward(self, bg_img, fg_img, fg_msk, fg_bbox, comp_img=None, comp_msk=None):
        assert (not ((comp_img is None) ^ (comp_msk is None)))

        bg_img_norm = self.normalize(bg_img)
        fg_img_norm = self.normalize(fg_img)
        fg_feats_ = self.shared_net(torch.cat((fg_img_norm, fg_msk), dim=1))
        bg_feats_ = self.shared_net(torch.cat((bg_img_norm, torch.zeros(fg_msk.shape).cuda()), dim=1))
        fg_feats = self.fg_head(fg_feats_)
        bg_feats = self.bg_head(bg_feats_)
        attn_feats_1, attn_1 = self.att(fg_feats, bg_feats)
        randomness = torch.randn((attn_feats_1.shape[0], self.opt.d_noise)).cuda()
        trans_1_ = self.regress_net(torch.cat((attn_feats_1.squeeze(1), randomness), dim=1))
        trans_1 = torch.tanh(trans_1_) / 2.0 + 0.5
        blend_img_1, blend_msk_1 = self.gen_blend(bg_img, fg_img, fg_msk, fg_bbox, trans_1)

        if comp_img is None:
            return blend_img_1, blend_msk_1, trans_1

        comp_img_norm = self.normalize(comp_img)
        comp_feats_ = self.shared_net(torch.cat((comp_img_norm, comp_msk), dim=1))
        latent_feats = self.latent_head(comp_feats_)
        latent_code, mu, logvar = self.vae_enc(latent_feats.squeeze(1))
        trans_2_ = self.regress_net(torch.cat((attn_feats_1.squeeze(1), latent_code), dim=1))
        trans_2 = torch.tanh(trans_2_) / 2.0 + 0.5
        blend_img_2, blend_msk_2 = self.gen_blend(bg_img, fg_img, fg_msk, fg_bbox, trans_2)

        return blend_img_1, blend_msk_1, trans_1, blend_img_2, blend_msk_2, trans_2, mu, logvar


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
        self.to_cuda()
        self.optimizer_G = self.get_G_optimizer(opt)
        self.optimizer_D = self.get_D_optimizer(opt)
        self.discri_loss = torch.nn.BCELoss()
        self.discri_loss_no_reduction = torch.nn.BCELoss(reduction='none')
        self.recons_loss_no_reduction = torch.nn.MSELoss(reduction='none')

    def get_G_optimizer(self, opt):
        return torch.optim.Adam(
            params=[
                {
                    "params": get_params(self.generator, key="g1x"),
                    "lr": 1 * opt.lr,
                    "initial_lr": 1 * opt.lr,
                    "weight_decay": 0.0005,
                },
                {
                    "params": get_params(self.generator, key="g10x"),
                    "lr": 10 * opt.lr,
                    "initial_lr": 10 * opt.lr,
                    "weight_decay": 0,
                },
            ], 
            betas=(opt.b1, opt.b2)
        )

    def get_D_optimizer(self, opt):
        return torch.optim.Adam(
            params=[
                {
                    "params": get_params(self.discriminator, key="d1x"),
                    "lr": 1 * opt.lr,
                    "initial_lr": 1 * opt.lr,
                    "weight_decay": 0,
                }
            ], 
            betas=(opt.b1, opt.b2)
        )

    def start_train(self):
        self.generator.train()
        self.discriminator.train()

    def start_eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def to_cuda(self):
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

    def train_disc_gen(self, bg_img, fg_img, fg_msk, fg_bbox, comp_img, comp_msk, label, trans_label):
        self.Eiters += 1
        batch_size = len(label)

        bg_img_v = Variable(bg_img, requires_grad=False).cuda()
        fg_img_v = Variable(fg_img, requires_grad=False).cuda()
        fg_msk_v = Variable(fg_msk, requires_grad=False).cuda()
        fg_bbox_v = Variable(fg_bbox.float(), requires_grad=False).cuda()
        comp_img_v = Variable(comp_img, requires_grad=False).cuda()
        comp_msk_v = Variable(comp_msk, requires_grad=False).cuda()
        label_v = Variable(label.float(), requires_grad=False).cuda()
        trans_label_v = Variable(trans_label, requires_grad=False).cuda()
        valid = Variable(torch.ones(batch_size), requires_grad=False).cuda()
        fake = Variable(torch.zeros(batch_size), requires_grad=False).cuda()

        # forward
        gen_comps_1, gen_msks_1, trans_1, gen_comps_2, gen_msks_2, trans_2, mu, logvar = \
                self.generator(bg_img_v, fg_img_v, fg_msk_v, fg_bbox_v, comp_img_v, comp_msk_v)
        discri_target_gen1 = self.discriminator(gen_comps_1, gen_msks_1)
        discri_target_gen1_detach = self.discriminator(gen_comps_1.detach(), gen_msks_1.detach())
        discri_target_gen2 = self.discriminator(gen_comps_2, gen_msks_2)
        discri_target_gen2_detach = self.discriminator(gen_comps_2.detach(), gen_msks_2.detach())
        discri_target_real = self.discriminator(comp_img_v, comp_msk_v)

        # discriminator loss
        d_real_loss = self.discri_loss(discri_target_real, label_v)
        d_fake_loss1 = self.discri_loss(discri_target_gen1_detach, fake)
        if label_v.sum() > 0.5:
            d_fake_loss2 = (self.discri_loss_no_reduction(discri_target_gen2_detach, fake) * label_v).sum() / label_v.sum()
        else:
            d_fake_loss2 = torch.tensor(0).cuda()
        d_fake_loss = d_fake_loss1 * 0.25 + d_fake_loss2
        d_loss = d_real_loss + d_fake_loss

        # generator loss
        g_gan_loss1 = self.discri_loss(discri_target_gen1, valid)
        if label_v.sum() > 0.5:
            g_gan_loss2 = (self.discri_loss_no_reduction(discri_target_gen2, valid) * label_v).sum() / label_v.sum()
            g_rec_loss = ((self.recons_loss_no_reduction(trans_2, trans_label_v) * torch.cat((torch.sin(trans_2[:,:1] * np.pi / 2), torch.cos(trans_2[:,:1] * np.pi / 2), torch.cos(trans_2[:,:1] * np.pi / 2)), dim=1).detach()).mean(dim=1) * label_v).sum() / label_v.sum()
            g_kld_loss = -0.5 * ((1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1) * label_v).sum() / label_v.sum()
        else:
            g_gan_loss2 = torch.tensor(0).cuda()
            g_rec_loss = torch.tensor(0).cuda()
            g_kld_loss = torch.tensor(0).cuda()
        g_gan_loss = g_gan_loss1 * 4.0 + g_gan_loss2
        g_loss = g_gan_loss + g_rec_loss * 50.0 + g_kld_loss

        # generator backward
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # discriminator backward
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()  

        return g_gan_loss1, g_gan_loss2, g_rec_loss, g_kld_loss, d_real_loss, d_fake_loss1, d_fake_loss2

    def test_genorator(self, bg_img, fg_img, fg_msk, fg_bbox):
        bg_img = bg_img.cuda()
        fg_img = fg_img.cuda()
        fg_msk = fg_msk.cuda()
        fg_bbox = fg_bbox.float().cuda()

        return self.generator(bg_img, fg_img, fg_msk, fg_bbox)

