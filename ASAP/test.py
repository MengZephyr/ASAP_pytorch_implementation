from __future__ import division
from __future__ import print_function

import time
import torch
from random import shuffle
from utils import *
from architecture import architecture
from torchvision.utils import save_image
from vgg import vgg_std, vgg_mean
from PIL import Image

USE_CUDA = torch.cuda.is_available()
print('balin-->', USE_CUDA)
device = torch.device("cuda:1" if USE_CUDA else "cpu")

TestList = get_fileSet_list('../base/')
TestList = get_flagInfo(TestList, '.png')
print(len(TestList))


def checkImgList(pngList):
    fList = []
    for p in pngList:
        img = Image.open(p)
        h, w = img.size
        if float(h)/float(w) > 2 and float(h)/float(w) < 2.5:
            fList = fList + [p]
        if float(h)/float(w) < 0.5 and float(h)/float(w) > 0.4:
            fList = fList + [p]
    return fList


TestList = checkImgList(TestList)
print('# of training = ', len(TestList))

imgS = 256
guiS = 256
K = 8
assert (2**K == imgS)
bsize = 2
TestKK = len(TestList) // bsize

modelArch = architecture(imgH=imgS, imgW=imgS, guiH=guiS, guiW=guiS, K=K, dimG=3, dimOut=3, bsize=bsize, device=device,
                         genCKPName='../ckp/f__Gen.ckp', disCKPName=None)

def getBatchList(pngList):
    gImgList = pngList
    tImgList = []
    for p in gImgList:
        nameInfo = p.split('.')[-2]
        tImgList = tImgList + ['..'+nameInfo + '.jpg']
    return gImgList, tImgList


def draw_iterResult(X, Gt, Mask, nB, ki, prefN):
    simg = []
    for i in range(nB):
        x = X[i, :, :, :]
        g = Gt[i, :, :, :]
        m = Mask[i, :, :, :]
        c = torch.cat([m, x, g], dim=2)
        simg.append(c)
    simg = torch.cat(simg, dim=1)
    save_image(simg, filename=prefN + str(ki) + '.jpg')


kk = 8
for ik in range(kk):
    k = (ik // 1000) % TestKK
    if k == 0:
        shuffle(TestList)
    idList = TestList[k * bsize: (k + 1) * bsize]
    gImgList, tImgList = getBatchList(idList)
    out, gtImg, gtMasks = modelArch.test_Gen(gImgList, tImgList)
    draw_iterResult(out, gtImg, gtMasks, bsize, ik, '../test/f3_')
