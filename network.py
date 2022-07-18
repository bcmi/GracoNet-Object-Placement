import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


BN_MOMENTUM = 0.1


def vgg16_bn(pretrained):
    model = models.vgg16_bn(pretrained=False)
    model.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True)

    if pretrained:
        pretrained_state_dict = models.vgg16_bn(pretrained=True).state_dict()
        conv = pretrained_state_dict['features.0.weight']
        new = torch.zeros(64, 1, 3, 3)
        for i, output_channel in enumerate(conv):
            new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrained_state_dict['features.0.weight'] = torch.cat((conv, new), dim=1)
        model.load_state_dict(pretrained_state_dict)

    return model


class VAEEncoder(nn.Module): 
    def __init__(self, opt):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(opt.d_model, 1024),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(1024, opt.d_noise)
        self.fc3 = nn.Linear(1024, opt.d_noise)
    
    def encode(self, x):
        h = self.fc1(x)
        mu = self.fc2(h)
        logvar = self.fc3(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent_code = self.reparameterize(mu, logvar)
        return latent_code, mu, logvar


class FgBgRegression(nn.Module):
    def __init__(self, opt):
        super(FgBgRegression, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(opt.d_model + opt.d_noise, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3),
        )

    def forward(self, x):
        out = self.regressor(x)
        return out


class FgBgLayer(nn.Module):
    def __init__(self, opt, n_mesh):
        super(FgBgLayer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(opt.d_model, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, opt.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(opt.d_model, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((n_mesh, n_mesh))

    def forward(self, x):
        feats = self.features(x)
        pooled_feats = self.pool(feats)
        nodes = pooled_feats.view(pooled_feats.shape[0], pooled_feats.shape[1], -1).transpose(1, 2).contiguous()
        return nodes


class FgBgHead(nn.Module):
    def __init__(self, opt, n_mesh_list):
        super(FgBgHead, self).__init__()
        self.layers = nn.ModuleList([FgBgLayer(opt, n_mesh) for n_mesh in n_mesh_list])

    def forward(self, x):
        node_list = []
        for layer in self.layers:
            nodes = layer(x)
            node_list.append(nodes)
        return torch.cat(node_list, dim=1)


class ScaledDotProductAttention(nn.Module):
   def __init__(self, opt):
       super(ScaledDotProductAttention, self).__init__()
       self.opt = opt
       self.pos_k = nn.Embedding(opt.n_heads * opt.len_k, opt.d_k)
       self.pos_v = nn.Embedding(opt.n_heads * opt.len_k, opt.d_v)
       self.pos_ids = torch.LongTensor(list(range(opt.n_heads * opt.len_k))).view(1, opt.n_heads, opt.len_k)

   def forward(self, Q, K, V):
       K_pos = self.pos_k(self.pos_ids.cuda())
       V_pos = self.pos_v(self.pos_ids.cuda())
       scores = torch.matmul(Q, (K + K_pos).transpose(-1, -2)) / np.sqrt(self.opt.d_k)
       attn = nn.Softmax(dim=-1)(scores)
       context = torch.matmul(attn, V + V_pos)
       return context, attn


class MultiHeadAttention(nn.Module):
   def __init__(self, opt):
       super(MultiHeadAttention, self).__init__()
       self.opt = opt
       self.W_Q = nn.Linear(opt.d_model, opt.d_k * opt.n_heads)
       self.W_K = nn.Linear(opt.d_model, opt.d_k * opt.n_heads)
       self.W_V = nn.Linear(opt.d_model, opt.d_v * opt.n_heads)
       self.att = ScaledDotProductAttention(opt)
       self.W_O = nn.Linear(opt.n_heads * opt.d_v, opt.d_model)
       self.norm = nn.LayerNorm(opt.d_model)

   def forward(self, Q, K, V):
       residual, batch_size = Q, Q.size(0)
       q_s = self.W_Q(Q).view(batch_size, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)
       k_s = self.W_K(K).view(batch_size, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)
       v_s = self.W_V(V).view(batch_size, -1, self.opt.n_heads, self.opt.d_v).transpose(1,2)
       context, attn = self.att(q_s, k_s, v_s)
       context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.opt.n_heads * self.opt.d_v)
       output = self.W_O(context)
       return self.norm(output + residual), attn


class FgBgAttention(nn.Module):
   def __init__(self, opt):
       super(FgBgAttention, self).__init__()
       self.att = MultiHeadAttention(opt)

   def forward(self, fg_feats, bg_feats):
       output, attn = self.att(fg_feats, bg_feats, bg_feats)
       return output, attn
