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

TrainList = get_fileSet_list('../base/')
TrainList = get_flagInfo(TrainList, '.png')
print(len(TrainList))


def checkImgList(pngList):
    fList = []
    for p in pngList:
        img = Image.open(p)
        h, w = img.size
        if float(h)/float(w) > 2. or float(h)/float(w) < 0.5:
            continue
        fList = fList + [p]
    return fList


TrainList = checkImgList(TrainList)
shuffle(TrainList)
print('# of training = ', len(TrainList))

imgS = 256
guiS = 256
K = 8
assert (2**K == imgS)
bsize = 2
Train_KK = len(TrainList) // bsize
iterations = 100000

modelArch = architecture(imgH=imgS, imgW=imgS, guiH=guiS, guiW=guiS, K=K, dimG=3, dimOut=3, bsize=bsize, device=device,
                         genCKPName='../ckp/g_Asap_5000_Gen.ckp', disCKPName='../ckp/g_Asap_5000_Dis.ckp')


def getBatchList(pngList):
    gImgList = pngList
    tImgList = []
    for p in gImgList:
        nameInfo = p.split('.')[-2]
        tImgList = tImgList + ['..'+nameInfo + '.jpg']
    return gImgList, tImgList


def draw_iterResult(X, Gt, iter, nB, prefN):
    simg = []
    for i in range(nB):
        # x = X[i, :, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
        #     torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        # g = Gt[i, :, :, :] * torch.tensor(vgg_std).view(-1, 1, 1).to(device) + \
        #     torch.tensor(vgg_mean).view(-1, 1, 1).to(device)
        x = X[i, :, :, :]
        g = Gt[i, :, :, :]
        c = torch.cat([g, x], dim=1)
        simg.append(c)
    simg = torch.cat(simg, dim=2)
    save_image(simg, filename=prefN + str(iter) + '.jpg')


IFSumWriter = True
if IFSumWriter:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
betit = -1
for itt in range(betit+1, iterations+1):
    t = time.time()
    k = itt % Train_KK
    if k == 0:
        shuffle(TrainList)
    idList = TrainList[k * bsize: (k + 1) * bsize]
    gImgList, tImgList = getBatchList(idList)
    LossG, pLoss, out, gtImg, gtMasks = modelArch.iter_train_G(gImgList, tImgList)
    LossD = modelArch.iter_train_D(out, gtImg, gtMasks)
    print('Iter_{} --> LossG: {:.4f}, LossP:{:.4f}, LossD: {:.4f}'.format(itt, LossG.item(), pLoss.item(), LossD.item()))
    if IFSumWriter and itt % 500 == 0:
        writer.add_scalar('G_Loss', LossG, itt)
        writer.add_scalar('P_Loss', pLoss, itt)
        writer.add_scalar('D_Loss', LossD, itt)
    if itt % 500 == 0:
        draw_iterResult(out, gtImg, itt, bsize, '../test/t_')
    if itt % 5000 == 0:
        modelArch.save_ckp('../ckp/Asap_'+str(itt), itt)
    if itt % 1000 == 0:
        modelArch.save_ckp('../ckp/t_' + str(itt), itt)




