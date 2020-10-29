from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from networks.submodules import *
from layers_package.resample2d_package.resample2d import Resample2d
from layers_package.channelnorm_package.channelnorm import ChannelNorm
from torchvision import ops
from networks.deform_conv import DeformConv
def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
def deformable_conv(in_planes, out_planes, kernel_size=3, strides=1, padding=1, use_bias=False):
    #MFN originally use bias true
    #AANet deform
    return DeformConv(in_planes,out_planes,kernel_size=kernel_size,stride=strides,padding=padding, dilation=1,groups=1,deformable_groups=1,bias=False)
    #pytorch official version
    #return ops.DeformConv2d(in_planes, out_planes, kernel_size, strides, padding, bias=use_bias)
def deconv_MFN(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class ESNet_M(nn.Module):

    def __init__(self, batchNorm=False, lastRelu=True, resBlock=True, maxdisp=-1, input_channel=3, get_features = False):
        super(ESNet_M, self).__init__()
        
        self.batchNorm = batchNorm
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.get_features = get_features
        self.relu = nn.ReLU(inplace=False)
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.resample1 = Resample2d()
        self.channelnorm = ChannelNorm()
        # shrink and extract features
        self.conv1   = conv(self.input_channel, 64, 7, 2)
        if resBlock:
            self.conv2   = ResBlock(64, 128, stride=2)
            self.conv3   = ResBlock(128, 256, stride=2)
            self.conv_redir = ResBlock(256, 32, stride=1)
        else:
            self.conv2   = conv(64, 128, stride=2)
            self.conv3   = conv(128, 256, stride=2)
            self.conv_redir = conv(256, 32, stride=1)

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        if resBlock:
            self.conv3_1 = ResBlock(72, 256)
            self.conv4   = ResBlock(256, 512, stride=2)
            self.conv4_1 = ResBlock(512, 512)
            self.conv5   = ResBlock(512, 512, stride=2)
            self.conv5_1 = ResBlock(512, 512)
            self.conv6   = ResBlock(512, 1024, stride=2)
            self.conv6_1 = ResBlock(1024, 1024)
        else:
            self.conv3_1 = conv(72, 256)
            self.conv4   = conv(256, 512, stride=2)
            self.conv4_1 = conv(512, 512)
            self.conv5   = conv(512, 512, stride=2)
            self.conv5_1 = conv(512, 512)
            self.conv6   = conv(512, 1024, stride=2)
            self.conv6_1 = conv(1024, 1024)

        self.pred_flow6 = predict_flow(1024)

        # iconv with deconv
        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(193+5, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(97+5, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(17+self.input_channel+5, 16, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(1024, 512)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(512)

        self.upconv4 = deconv(512, 256)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(256)

        self.upconv3 = deconv(256, 128)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(128, out_planes=2)

        self.upconv2 = deconv(128, 64)
        self.upfeature2 = deconv_MFN(128,16)
        self.deform2 = deformable_conv(128, 128)
        self.trade_off_conv2 = nn.Conv2d(16, 128, kernel_size=3, stride=1,padding=1, dilation=1, bias=True)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(64, out_planes=2)

        self.upconv1 = deconv(64, 32)
        self.upfeature1 = deconv_MFN(64,16)
        self.bottleneck1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size = 1, padding = 0), nn.ReLU(inplace = True))
        self.deform1 = deformable_conv(64, 64)
        self.trade_off_conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1,padding=1, dilation=1, bias=True)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(32, out_planes=2)

        self.upconv0 = deconv(32, 16)
        self.upfeature0 = deconv_MFN(32,16)
        self.bottleneck0 = nn.Sequential(nn.Conv2d(128, 3, kernel_size = 1, padding = 0), nn.ReLU(inplace = True))
        self.deform0 = deformable_conv(3, 3)
        self.trade_off_conv0 = nn.Conv2d(16, 3, kernel_size=3, stride=1,padding=1, dilation=1, bias=True)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        if self.maxdisp == -1:
            self.pred_flow0 = predict_flow(16)
        else:
            self.disp_expand = ResBlock(16, self.maxdisp)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #self.freeze()
    def _warping(self,input, disp):
        """Warp input 4D tensor using disp (B,1,H,W)"""
        dummy_flow = torch.zeros_like(disp)
        flow_2d = torch.cat((disp, dummy_flow), dim = 1)
        resampled_input = self.resample1(input, -flow_2d)
        return resampled_input
    def _warping_MFN(self, conv_r, disp, mask, feat=None, deform_f=None, trade_off_conv_f=None):
        dummy_flow = torch.zeros_like(disp)
        flow_2d = torch.cat((dummy_flow,disp), dim = 1)
        offset = (flow_2d).unsqueeze(1)
        offset = torch.repeat_interleave(offset, 9, 1)
        S1, S2, S3, S4, S5 = offset.shape
        offset = offset.view(S1, S2*S3, S4, S5)
        x2_warp = deform_f(conv_r, -offset) # 64 ch
        tradeoff = feat # 64 ch
        x2_warp = (x2_warp * nn.functional.sigmoid(mask)) + trade_off_conv_f(tradeoff)
        x2_warp = self.leakyRELU(x2_warp)
        return x2_warp
    def forward(self, input):

        # split left image and right image
        imgs = torch.chunk(input, 2, dim = 1)
        img_left = imgs[0]
        img_right = imgs[1]

        conv1_l = self.conv1(img_left)
        conv2_l = self.conv2(conv1_l)
        conv3a_l = self.conv3(conv2_l)

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)
        
        # Correlate corr3a_l and corr3a_r
        #out_corr = self.corr(conv3a_l, conv3a_r)
        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=40) # for max_disp=40, max disparity considered in the original image is 40*8
        out_corr = self.corr_activation(out_corr)
        out_conv3a_redir = self.conv_redir(conv3a_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

        conv3b = self.conv3_1(in_conv3b)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)
        
        pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        pr3, mask3 = pr3[:,0,:,:].unsqueeze(1), pr3[:,1,:,:].unsqueeze(1)
        upconv2 = self.upconv2(iconv3)
        feat2 = self.leakyRELU(self.upfeature2(iconv3))
        upflow3 = self.upflow3to2(pr3)
        upflow3_interpolated = upsample2d_as(pr3, upflow3, mode="bilinear")
        upmask3_interpolated = upsample2d_as(mask3, upflow3, mode = 'bilinear')
        #conv2_r_warped = self._warping(conv2_r,upflow3_interpolated/4)
        conv2_r_warped = self._warping_MFN(conv2_r, upflow3_interpolated/4, upmask3_interpolated, feat=feat2, deform_f=self.deform2, trade_off_conv_f = self.trade_off_conv2)
        out_corr2 = build_corr_two_side(conv2_l, conv2_r_warped, search_range=range(-2,3))
        out_corr2 = self.corr_activation(out_corr2)
        #diff2 = self.channelnorm(conv2_l-conv2_r_warped)
        concat2 = torch.cat((upconv2, upflow3, conv2_l,out_corr2), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        pr2, mask2 = pr2[:,0,:,:].unsqueeze(1), pr2[:,1,:,:].unsqueeze(1)
        upconv1 = self.upconv1(iconv2)
        feat1 = self.leakyRELU(self.upfeature1(iconv2))
        upflow2 = self.upflow2to1(pr2)
        upflow2_interpolated = upsample2d_as(pr2, upflow2, mode="bilinear")
        upmask2_interpolated = upsample2d_as(mask2, upflow2, mode = 'bilinear')
        #conv1_r_warped = self._warping(conv1_r,upflow2_interpolated/2)
        conv1l_new = self.bottleneck1(upsample2d_as(conv2_l,upflow2_interpolated, mode = 'bilinear')) 
        conv1r_new = self.bottleneck1(upsample2d_as(conv2_r, upflow2_interpolated, mode = 'bilinear'))
        conv1_r_warped = self._warping_MFN(conv1r_new, upflow2_interpolated/2, upmask2_interpolated, feat=feat1, deform_f=self.deform1, trade_off_conv_f = self.trade_off_conv1)
        out_corr1 = build_corr_two_side(conv1l_new, conv1_r_warped, search_range=range(-2,3))
        out_corr1 = self.corr_activation(out_corr1)
        #diff1 = self.channelnorm(conv1_l-conv1_r_warped)
        concat1 = torch.cat((upconv1, upflow2, conv1_l,out_corr1), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        pr1, mask1 = pr1[:,0,:,:].unsqueeze(1), pr1[:,1,:,:].unsqueeze(1)
        upconv0 = self.upconv0(iconv1)
        feat0 = self.leakyRELU(self.upfeature0(iconv1))
        upflow1 = self.upflow1to0(pr1)
        upflow1_interpolated = upsample2d_as(pr1, upflow1, mode="bilinear")
        upmask1_interpolated = upsample2d_as(mask1, upflow1, mode = 'bilinear')
        #conv0_r_warped = self._warping(img_right,upflow1_interpolated)
        img_left_new = self.bottleneck0(upsample2d_as(conv2_l,upflow1_interpolated, mode = 'bilinear'))
        img_right_new = self.bottleneck0(upsample2d_as(conv2_r, upflow1_interpolated,  mode = 'bilinear')) 
        conv0_r_warped = self._warping_MFN(img_right_new, upflow1_interpolated, upmask1_interpolated, feat=feat0, deform_f=self.deform0, trade_off_conv_f = self.trade_off_conv0)
        out_corr0 = build_corr_two_side(img_left_new, conv0_r_warped, search_range=range(-2,3))
        out_corr0 = self.corr_activation(out_corr0)
        #diff0 = self.channelnorm(img_left-conv0_r_warped)
        concat0 = torch.cat((upconv0, upflow1, img_left,out_corr0), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow
        if self.maxdisp == -1:
            pr0 = self.pred_flow0(iconv0)
            pr0 = self.relu(pr0)
        else:
            pr0 = self.disp_expand(iconv0)
            pr0 = F.softmax(pr0, dim=1)
            pr0 = disparity_regression(pr0, self.maxdisp)


        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)


        # can be chosen outside
        if self.get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps
 
    def freeze(self):
        for name, param in self.named_parameters():
            if ('weight' in name) or ('bias' in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

