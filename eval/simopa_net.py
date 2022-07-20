import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torchsummary import summary
import os

from simopa_cfg import opt
from resnet_4ch import resnet


def roi_align(feature_map, boxes, w, h, outsize=opt.roi_align_size, insize=opt.global_feature_size):
    boxes_ = boxes.clone()
    if boxes_.dim() == 2:
        boxes_ = boxes_.unsqueeze(1)
    B,N,_ = boxes_.shape
    scaled_boxes = torch.zeros_like(boxes_)
    scaled_boxes[:, :, 0::2] = boxes_[:, :, 0::2] * (insize / w[:,None,None]).int()
    scaled_boxes[:, :, 1::2] = boxes_[:, :, 1::2] * (insize / h[:,None,None]).int()
    batch_index = torch.arange(B).view(-1, 1).repeat(1, N).reshape(B, N, 1).to(boxes_.device)
    batch_index = batch_index.float()
    rois = torch.cat((batch_index, scaled_boxes), dim=-1)
    rois = rois.view(B*N, -1)
    pooled_regions = torchvision.ops.roi_align(feature_map, rois,
                                               output_size=(outsize, outsize))
    return pooled_regions


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64):
        super(SelfAttention, self).__init__()
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)

    def forward(self, x):
        x = self.norm(x)
        b, n, _, h = *x.shape, self.heads
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qk)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = rearrange(dots, 'b h n j -> b n h j')
        return attn


class _Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ObjectPlaceNet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(ObjectPlaceNet, self).__init__()
        resnet_layers = int(opt.backbone.split('resnet')[-1])
        if backbone_pretrained:
            backbone = resnet(resnet_layers,
                              opt.without_mask,
                              backbone_pretrained,
                              os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'))
        else:
            backbone = resnet(resnet_layers,
                              opt.without_mask)

        features = list(backbone.children())[:-2]
        backbone = nn.Sequential(*features)
        self.backbone = backbone

        self.global_feature_dim = 512 if opt.backbone in ['resnet18', 'resnet34'] else 2048
        if opt.relation_method is None:
            self.fc_global = nn.Linear(self.global_feature_dim,
                                       opt.class_num, bias=False)

        res = {'inplanes': 2, 'planes': 512, 'expansion': 2, 'blocks': 3}

        if opt.relation_method == 5:
            self.geometric_layers = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Flatten(1),
                nn.Linear((opt.binary_mask_size // 16) ** 2 * 256,
                          opt.geometric_feature_dim, bias=False)
            )

        self.avgpool3x3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1)

        if opt.relation_method is not None:
            self.concatenate_dim = 1024
            if opt.relation_method == 0:
                self.roi_align = roi_align
                self.roi_feature = nn.Linear(self.global_feature_dim * opt.roi_align_size * opt.roi_align_size,
                                             512, bias=False)
                self.region_feature_dim = 1024
                self.fc_region_feature = nn.Linear(self.concatenate_dim, self.region_feature_dim, bias=False)
            elif opt.relation_method in [1,2]:
                self.region_feature_dim = 2048
            else:
                self.roi_feature = nn.Linear(2048, 512, bias=False)
                if opt.relation_method == 3:
                    self.concatenate_dim = 1024
                elif opt.relation_method == 4:
                    self.concatenate_dim += 8
                else:
                    self.concatenate_dim += opt.geometric_feature_dim
                if self.concatenate_dim > 1024:
                    self.region_feature_dim = 1024
                else:
                    self.region_feature_dim = self.concatenate_dim
                self.fc_region_feature = nn.Linear(self.concatenate_dim, self.region_feature_dim, bias=False)

        if opt.attention_method in [0,2]:
            self.refer_attention = SelfAttention(self.region_feature_dim,
                                                 heads=opt.attention_head,
                                                 dim_head=opt.attention_dim_head)
        if opt.attention_method == 0:
            self.fc_weight_learn = nn.Linear(opt.attention_head, 1, bias=False)
        elif opt.attention_method == 1:
            self.fc_weight_learn = nn.Linear(self.region_feature_dim, 1, bias=False)
        elif opt.attention_method == 2:
            self.fc_weight_learn = nn.Linear(opt.attention_head + self.region_feature_dim, 1, bias=False)

        if opt.relation_method is None:
            self.prediction_head = nn.Linear(self.global_feature_dim, opt.class_num, bias=False)
        else:
            fusion_feature_dim = self.region_feature_dim + self.global_feature_dim if not opt.without_global_feature else self.region_feature_dim
            self.prediction_head = nn.Sequential(
                            nn.Linear(fusion_feature_dim, fusion_feature_dim, bias=False),  # 1024*2
                            nn.ReLU(True),
                            nn.Dropout(0.1),
                            nn.Linear(fusion_feature_dim, 512, bias=False),
                            nn.ReLU(True),
                            nn.Linear(512, opt.class_num, bias=False))

    def forward(self, img_cat, target_box, refer_box, target_feature, refer_feature, target_mask, refer_mask, w, h):
        batch_size = img_cat.shape[0]

        global_feature = None
        if opt.without_mask:
            img_cat = img_cat[:,0:3]
        feature_map = self.backbone(img_cat)
        global_feature = self.avgpool1x1(feature_map)
        global_feature = global_feature.flatten(1)

        if opt.relation_method is None:
            prediction = self.prediction_head(global_feature)
            return prediction

        refer_boxes = refer_box[:, :, :4]
        target_boxes = target_box[:,None,:]

        region_feature = None
        if opt.relation_method == 0:
            refer_feature = self.roi_align(feature_map, refer_boxes, w, h)
            refer_feature = self.roi_feature(refer_feature.flatten(1)).view(batch_size, opt.refer_num, -1)
            target_feature = self.roi_align(feature_map, target_boxes, w, h)
            target_feature = self.roi_feature(target_feature.flatten(1)).view(batch_size, -1)
            target_feature = target_feature[:,None,:].repeat(1, opt.refer_num, 1)
            region_feature = torch.cat((refer_feature, target_feature), dim=-1)
            region_feature = self.fc_region_feature(region_feature)
        elif opt.relation_method in [1,2]:
            if opt.relation_method == 1:
                region_feature = target_feature
            else:
                region_feature = torch.mean(torch.cat([refer_feature, target_feature], dim=1), dim=1, keepdim=True)
        elif opt.relation_method in [3,4,5]:
            refer_feature = self.roi_feature(refer_feature)
            target_feature = self.roi_feature(target_feature)
            target_feature = target_feature.repeat(1,opt.refer_num,1)
            region_feature = torch.cat([refer_feature, target_feature], dim=2)
            if opt.relation_method == 3:
                region_feature = self.fc_region_feature(region_feature)
            elif opt.relation_method == 4:
                exp_w, exp_h = w.unsqueeze(1), h.unsqueeze(1)
                refer_x = (refer_boxes[:, :, 0] + refer_boxes[:, :, 2]) / (2 * exp_w)
                refer_y = (refer_boxes[:, :, 1] + refer_boxes[:, :, 3]) / (2 * exp_h)
                refer_w = (refer_boxes[:, :, 2] - refer_boxes[:, :, 0]) / exp_w
                refer_h = (refer_boxes[:, :, 3] - refer_boxes[:, :, 1]) / exp_h
                target_x = (target_boxes[:, :, 0] + target_boxes[:, :, 2]) / (2 * exp_w).repeat(1, opt.refer_num)
                target_y = (target_boxes[:, :, 1] + target_boxes[:, :, 3]) / (2 * exp_h).repeat(1, opt.refer_num)
                target_w = (target_boxes[:, :, 2] - target_boxes[:, :, 0]) / exp_w.repeat(1, opt.refer_num)
                target_h = (target_boxes[:, :, 3] - target_boxes[:, :, 1]) / exp_h.repeat(1, opt.refer_num)
                geometric_feature = torch.stack([refer_x, refer_y, refer_w, refer_h,
                                            target_x, target_y, target_w, target_h],
                                            dim=2)
                fuse_feature = torch.cat([region_feature, geometric_feature], dim=-1)
                region_feature = self.fc_region_feature(fuse_feature)
            else:
                mask_size = opt.binary_mask_size
                target_mask = target_mask.repeat(1,opt.refer_num,1,1).view(
                    batch_size * opt.refer_num, 1, mask_size, mask_size)
                refer_mask = refer_mask.view(batch_size * opt.refer_num, 1, mask_size, mask_size)
                concat_mask = torch.cat([refer_mask, target_mask], dim=1)
                geometric_feature = self.geometric_layers(concat_mask)
                geometric_feature = geometric_feature.view(batch_size, opt.refer_num, -1)
                fused_feature = torch.cat((region_feature, geometric_feature), dim=2)
                region_feature = self.fc_region_feature(fused_feature)

        agg_region_feature = None
        attention_weights = None
        if opt.attention_method is None:
            agg_region_feature = torch.mean(region_feature, dim=1)
        elif opt.attention_method == 1:
            attention_weights = self.fc_weight_learn(region_feature)
            attention_weights = F.softmax(attention_weights, dim=1)
            agg_region_feature = torch.sum(attention_weights * region_feature, dim=1)
        else:
            similarity_vector = self.refer_attention(region_feature)
            similarity_vector = torch.mean(similarity_vector, dim=-1)
            attention_weights = None
            if opt.attention_method == 0:
                attention_weights = self.fc_weight_learn(similarity_vector)
                attention_weights = F.softmax(attention_weights, dim=1)
                agg_region_feature = torch.sum(attention_weights * region_feature, dim=1)
            else:
                combine_feature = torch.cat([region_feature, similarity_vector],dim=2)
                attention_weights = self.fc_weight_learn(combine_feature)
                attention_weights = F.softmax(attention_weights, dim=1)
                agg_region_feature = torch.sum(attention_weights * region_feature, dim=1)
        if opt.without_global_feature:
            prediction = self.prediction_head(agg_region_feature)
        else:
            prediction = self.prediction_head(torch.cat([global_feature, agg_region_feature], dim=-1))
        return prediction, attention_weights


if __name__ == '__main__':
    b = 4
    img_cat = torch.randn(b, 4, 256, 256).cuda()
    target_box = torch.randint(size=(b, 4), low=0, high=256).float().cuda()
    refer_box = torch.randint(size=(b, 5, 6), low=0, high=256).float().cuda()
    target_feat = torch.randn(b, 1, 2048).cuda()
    refer_feat = torch.randn(b, 5, 2048).cuda()
    target_mask = torch.randn(b, 1, 64, 64).cuda()
    refer_mask = torch.randn(b, 5, 64, 64).cuda()
    w = h = (torch.ones(b) * 256).cuda()

    model = ObjectPlaceNet(backbone_pretrained=False).cuda()
    local_pre = model(img_cat, target_box, refer_box, target_feat, refer_feat, target_mask, refer_mask, w, h)
    print(local_pre)
