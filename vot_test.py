# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import time
import vot
from vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
import os
from os.path import realpath, dirname, join
from bilinear import bilinear
from mask_generate import mask_generate
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from Embed import Embed

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

base_path = '/home/zhangzichun/MultiTask/vot2016/'
label_path = '/home/zhangzichun/MultiTask/vot2016seg/'

to_tensor = transforms.ToTensor()

net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()
fc = mask_generate()
fc.load_state_dict(torch.load('./test_save/fc_only.pth'))
fc.eval().cuda()

vots = ['basketball']


#real training
for sequence in vots:
    # start to track
    current_vot = base_path+sequence+'/'

    handle = vot.VOT('polygon', sequence)
    Polygon = handle.region()
    cx, cy, w, h = get_axis_aligned_bbox(Polygon)

    image_file = handle.frame()
    if not image_file:
       sys.exit(0)

    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    im = cv2.imread(current_vot+image_file)  # HxWxC
    state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
    sequenceNo = 0

    while True:

        print('handle '+str(sequenceNo))
        image_file = handle.frame()
        if not image_file:
            break
        im = cv2.imread(current_vot+image_file)  # HxWxC
        im_size = list(im.shape)
        im_size.pop(2)


        state, x_feat = SiamRPN_track(state, im)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        handle.report(Rectangle(res[0], res[1], res[2], res[3]))
        #warped_map = bilinear(x_feat, res)
        mask = fc(x_feat)
        Embed(mask, im_size, res, sequenceNo)

        sequenceNo = sequenceNo+1



