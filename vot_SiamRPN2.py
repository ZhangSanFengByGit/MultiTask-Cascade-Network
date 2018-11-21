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

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

to_tensor = transforms.ToTensor()

base_path = '/home/zhangzichun/MultiTask/vot2016/'
label_path = '/home/zhangzichun/MultiTask/vot2016seg/'
f = open('print2.txt','w')

#vots = os.listdir(base_path)
#vots.remove('list.txt')


# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()
fc = mask_generate().cuda()

# training pack
criterion = torch.nn.MSELoss()
#params = list(net.parameters()) + list(fc.parameters())
#optimizer = torch.optim.Adam(params, lr = 0.001)
optimizer = torch.optim.Adam(fc.parameters(), lr = 0.001)

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())




#training warm up
test_label = Image.open('/home/zhangzichun/MultiTask/vot2016seg/bag/groundtruth/00000000.png').resize([64,64])
test_label = Variable(to_tensor(test_label)[0]).cuda()
test_image = cv2.imread('/home/zhangzichun/MultiTask/vot2016/bag/00000001.jpg')

handle = vot.VOT('polygon', 'bag')
Polygon = handle.region()
cx, cy, w, h = get_axis_aligned_bbox(Polygon)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
state = SiamRPN_init(test_image,target_pos,target_sz,net)
print('training warm up:')

for i in range(5):
    fc.zero_grad()
    net.zero_grad()

    state, x_feat = SiamRPN_track(state, test_image)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    #handle.report(Rectangle(res[0], res[1], res[2], res[3]))

    warped_map = bilinear(x_feat, res)
    mask = fc(warped_map)
    '''res = mask.view(-1) - test_label.type('torch.cuda.FloatTensor').view(-1)
    print('mean distance : {}'.format(torch.mean(res)))
    mask_concat = Variable(torch.ones(64*64,1)).cuda()
    mask_concat = mask_concat - mask
    mask = torch.cat((mask_concat,mask),1)
    print(mask.data)'''
    err = criterion(mask, test_label)
    print('err: {} \n'.format(err.data))
    err.backward()
    optimizer.step()

'''
vots = [bag         blanket    car1      fish2      graduate     handball1       octopus      shaking  soccer2  wiper
ball1       bmx        car2      fish3      gymnastics1  handball2   marching    pedestrian1  sheep    soldier
ball2       bolt1      crossing  fish4      gymnastics2  helicopter  matrix      pedestrian2  singer1  sphere
basketball  bolt2      dinosaur  girl       gymnastics3  iceskater1  motocross1  rabbit       singer2  tiger
birds1      book       fernando  glove      gymnastics4  iceskater2  motocross2  racing       singer3  traffic
birds2      butterfly  fish1     godfather  hand         leaves      nature      road         soccer1  tunnel
]'''
'''
vots = ['bag', 'blanket',    'car1',      'fish2',      'graduate',     'handball1',       'octopus',      'shaking',  'soccer2',  'wiper', \
'ball1',       'bmx',        'car2',      'fish3',      'gymnastics1',  'handball2',   'marching',    'pedestrian1',  'sheep',    'soldier', \
'ball2',       'bolt1',      'crossing',  'fish4',      'gymnastics2',  'helicopter',  'matrix',      'pedestrian2',  'singer1',  'sphere'],\
'basketball',  'bolt2',     'dinosaur',  'girl',      'gymnastics3', 'iceskater1',  'motocross1',  'rabbit',       'singer2',     'tiger', \
'birds1',      'book',       'fernando',  'glove',      'gymnastics4,'  'iceskater2',  'motocross2',  'racing',       'singer3',  'traffic', \
'birds2',     'butterfly',  'fish1',     'godfather', 'hand',        'leaves',     'nature',   'road',       'soccer1',  'tunnel']
'''
vots = ['bag',       'fish2',      'graduate',     'octopus',      'shaking',  'soccer2',   \
'bmx',        'gymnastics1',       'pedestrian1',   'sheep',    'soldier', \
'bolt1',      'crossing',  'fish4',      'gymnastics2',       'pedestrian2',  'singer1',  'sphere',\
'basketball',  'bolt2',     'dinosaur',  'girl',      'gymnastics3',  	'iceskater1',  'motocross1',  'rabbit',       'singer2',     'tiger', \
'birds1',      'book',       'fernando',  'glove',      'gymnastics4',  'iceskater2',  'motocross2',  'racing',    'traffic', \
'birds2',     'butterfly',       'godfather', 'hand',        'leaves',     'nature',   'road']


#real training
for epoch in range(5):
    for sequence in vots:
        print('{} : {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(time.ctime(), sequence))
        f.writelines('{} : {} !!!!!!!!!!!!\n'.format(time.ctime(), sequence))
        # start to track
        current_vot = base_path+sequence+'/'
        current_label = label_path+sequence+'/groundtruth/'

        handle = vot.VOT('polygon', sequence)
        Polygon = handle.region()
        cx, cy, w, h = get_axis_aligned_bbox(Polygon)

        image_file = handle.frame()
        if not image_file:
           sys.exit(0)

        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
        im = cv2.imread(current_vot+image_file)  # HxWxC
        state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
        while True:
            fc.zero_grad()
            net.zero_grad()

            image_file = handle.frame()
            if not image_file:
                break

            label_file = handle.label()
            label_file = current_label + label_file

            label = Image.open(label_file).resize([64,64])
            label = Variable(to_tensor(label)[0]).cuda()
            assert label.size(0) == 64, 'PNG label size wrong'
        

            im = cv2.imread(current_vot+image_file)  # HxWxC
            state, x_feat = SiamRPN_track(state, im)  # track
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            handle.report(Rectangle(res[0], res[1], res[2], res[3]))

            warped_map = bilinear(x_feat, res)
            mask = fc(warped_map)

            '''res = mask.view(-1) - label.type('torch.cuda.FloatTensor').view(-1)
            print('mean distance : {}'.format(torch.mean(res)))
            f.writelines('mean distance : {}'.format(torch.mean(res)))
            mask_concat = Variable(torch.ones(64*64,1)).cuda()
            mask_concat = mask_concat - mask
            mask = torch.cat((mask_concat,mask),1)'''
            err = criterion(mask, label)
            print('err: {} \n'.format(err.data))
            f.writelines('err: {} \n'.format(err.data))
            err.backward()
            optimizer.step()

fc.eval().cpu()
net.eval().cpu()
torch.save(net.state_dict(),'./test_save/net2.pth')
torch.save(fc.state_dict(),'./test_save/fc2.pth')
f.close()

