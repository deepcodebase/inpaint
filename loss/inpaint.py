import logging

import torch
from torch import nn
from torch.cuda.amp import autocast
from torchvision import models

from utils.misc import resize_like


logger = logging.getLogger(__name__)


def binary_mask(mask):
    assert isinstance(mask, torch.Tensor)
    return (mask > 0.5).float()


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, fulls, targets):
        loss = 0.
        for i, (full, target) in enumerate(zip(fulls, targets)):
            loss += self.l1(full, target)
        return loss / len(fulls)


class VGGFeature(nn.Module):

    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
        for para in vgg16.parameters():
            para.requires_grad = False

        # using vgg.named_modules() to check network structure.
        self.vgg16_pool_1 = nn.Sequential(*vgg16.features[0:5])
        self.vgg16_pool_2 = nn.Sequential(*vgg16.features[5:10])
        self.vgg16_pool_3 = nn.Sequential(*vgg16.features[10:17])

    def forward(self, x):

        pool_1 = self.vgg16_pool_1(x)
        pool_2 = self.vgg16_pool_2(pool_1)
        pool_3 = self.vgg16_pool_3(pool_2)

        return [pool_1, pool_2, pool_3]


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, vgg_fulls, vgg_targets):
        loss = 0.
        for i, (vgg_full, vgg_target) in enumerate(
                zip(vgg_fulls, vgg_targets)):
            for feat_full, feat_target in zip(vgg_full, vgg_target):
                loss += self.l1loss(feat_full, feat_target)
        return loss / len(vgg_fulls)


class StyleLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram(self, feature):
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).float()
        with autocast(enabled=False):
            gram_mat = torch.matmul(feature, torch.transpose(feature, 1, 2))
        return gram_mat / (c*h*w)

    def forward(self, vgg_fulls, vgg_targets):
        loss = 0.
        for i, (vgg_full, vgg_target) in enumerate(
                zip(vgg_fulls, vgg_targets)):
            for feat_full, feat_target in zip(vgg_full, vgg_target):
                loss += self.l1loss(
                    self.gram(feat_full), self.gram(feat_target))
        return loss / len(vgg_fulls)


class GradientLoss(nn.Module):

    def __init__(self, img_channels=3):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.img_channels = img_channels

        kernel = torch.FloatTensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]]).view(1, 1, 3, 3)
        kernel = torch.cat([kernel]*img_channels, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=self.img_channels)

    def forward(self, fulls, target, mask):
        loss = 0.
        for i, full in enumerate(fulls):
            mask_resize = resize_like(mask, full)
            grad_full = self.gradient(full) * mask_resize
            grad_target = self.gradient(
                resize_like(target, full)) * mask_resize
            loss += self.l1(grad_full, grad_target)
        return loss / len(fulls)


class TotalVariationLoss(nn.Module):

    def __init__(self, img_channels=3):
        super().__init__()
        self.img_channels = img_channels

        kernel = torch.FloatTensor([
            [0, 1, 0],
            [1, -2, 0],
            [0, 0, 0]]).view(1, 1, 3, 3)
        kernel = torch.cat([kernel]*img_channels, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=self.img_channels)

    def forward(self, fulls, mask):
        loss = 0.
        for i, full in enumerate(fulls):
            grad = self.gradient(full) * resize_like(mask, full)
            loss += torch.mean(torch.abs(grad).float())
        return loss / len(fulls)


class InpaintLoss(nn.Module):

    def __init__(
            self, c_img=3, w_l1=6., w_percep=0.1, w_style=240., w_tv=0.1,
            structure_layers=[0, 1, 2, 3, 4, 5],
            texture_layers=[0, 1, 2]):

        super().__init__()

        self.l_struct = structure_layers
        self.l_text = texture_layers

        self.w_l1 = w_l1
        self.w_percep = w_percep
        self.w_style = w_style
        self.w_tv = w_tv

        self.reconstruction_loss = ReconstructionLoss()

        self.vgg_feature = VGGFeature()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss(c_img)

    def forward(self, fulls, target, mask):

        targets = [resize_like(target, full) for full in fulls]

        loss_total = 0.
        loss_list = {}

        if len(self.l_struct) > 0:

            struct_f = [fulls[i] for i in self.l_struct]
            struct_t = [targets[i] for i in self.l_struct]

            loss_l1 = self.reconstruction_loss(struct_f, struct_t) * self.w_l1

            loss_total += loss_l1

            loss_list['reconstruction_loss'] = loss_l1.item()

        if len(self.l_text) > 0:

            text_f = [targets[i] for i in self.l_text]
            text_t = [fulls[i] for i in self.l_text]

            vgg_f = [self.vgg_feature(f) for f in text_f]
            vgg_t = [self.vgg_feature(t) for t in text_t]

            loss_style = self.style_loss(vgg_f, vgg_t) * self.w_style
            loss_percep = self.perceptual_loss(vgg_f, vgg_t) * self.w_percep
            loss_tv = self.tv_loss(text_f, mask) * self.w_tv

            loss_total = loss_total + loss_style + loss_percep + loss_tv

            loss_list.update({
                'perceptual_loss': loss_percep.item(),
                'style_loss': loss_style.item(),
                'total_variation_loss': loss_tv.item()
            })

        return loss_total, loss_list
