import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.nn.init import _calculate_correct_fan
import math
from numpy import pi


def siren_uniform(tensor: torch.Tensor, mode: str='fan_in', c:float=6):
    fan = _calculate_correct_fan(tensor, mode)
    std = 1/math.sqrt(fan)
    bound = math.sqrt(c) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def defPixelGrid(H, W, device):
    px = [i for i in range(W)]
    py = [i for i in range(H)]
    gridY, gridX = torch.meshgrid(torch.tensor(py), torch.tensor(px))
    gridY = gridY.to(device)
    gridX = gridX.to(device)
    gridY = gridY.unsqueeze(0).unsqueeze(0)
    gridX = gridX.unsqueeze(0).unsqueeze(0)
    pixelGrid = torch.cat([gridY, gridX], dim=1)
    return pixelGrid


def PoseEncoding(posGrid, K):
    pEn = []
    for k in range(K):
        w = pi / (2.**(k + 1))
        sinp = torch.sin(w * posGrid)
        cosp = torch.cos(w * posGrid)
        pEn.append(torch.cat([sinp, cosp], dim=1))
    pEn = torch.cat(pEn, dim=1)
    return pEn


def defInterGrid(H, W, device):
    px = [2.*i/float(W)-1. for i in range(W)]
    py = [2.*i/float(H)-1 for i in range(H)]
    gridY, gridX = torch.meshgrid(torch.tensor(py), torch.tensor(px))
    gridY = gridY.to(device)
    gridX = gridX.to(device)
    gridY = gridY.unsqueeze(0).unsqueeze(0)
    gridX = gridX.unsqueeze(0).unsqueeze(0)
    pixelGrid = torch.cat([gridY, gridX], dim=1)
    return pixelGrid


class CNN2dLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding, k_relu=-1.):
        super(CNN2dLayer, self).__init__()
        if k_relu < 0:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                      nn.InstanceNorm2d(out_ch),
                                      nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=padding),
                                      nn.InstanceNorm2d(out_ch),
                                      nn.LeakyReLU(k_relu, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class LR_Encoder(nn.Module):
    def __init__(self, in_ch, c_total):
        super(LR_Encoder, self).__init__()
        self.encoder = nn.Sequential(CNN2dLayer(in_ch, 64, 3, 1, 1),  # 256
                                     CNN2dLayer(64, 128, 3, 2, 1),     # 128
                                     CNN2dLayer(128, 256, 3, 2, 1),     # 64
                                     CNN2dLayer(256, 512, 3, 2, 1),     # 32
                                     CNN2dLayer(512, 1024, 3, 2, 1),   # 16
                                     CNN2dLayer(1024, 1024, 3, 1, 1),
                                     CNN2dLayer(1024, 1024, 3, 1, 1),
                                     CNN2dLayer(1024, 1024, 3, 1, 1),
                                     CNN2dLayer(1024, 1024, 3, 1, 1),
                                     CNN2dLayer(1024, 1024, 3, 1, 1),
                                     CNN2dLayer(1024, 1024, 3, 1, 1),
                                     nn.Conv2d(1024, c_total, 1, stride=1, padding=0))

        #self.ini_model_param(self.encoder)

    def ini_model_param(self, net, c=6.):
        for m in net.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(net.state_dict()[m], mode='fan_in', c=c)

    def forward(self, x):
        x = self.encoder(x)
        return x


class pixelWiseFC(nn.Module):
    def __init__(self, inDim, outDim, ifRelu):
        super(pixelWiseFC, self).__init__()
        self.inD = inDim
        self.outD = outDim
        self.ifRelu = ifRelu
        if ifRelu:
            self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, inX, paraT):
        #numE, dimIn, _ = inX.size()
        bsize = paraT.size()[0]
        dimW = self.inD * self.outD
        #print(inX.size(), paraT.size())
        #W = paraT[:, 0:dimW].view(numE, 1, self.outD, self.inD)
        #print("W: ", W.size())
        # [bsize*H*W, outdim, indim]

        #B = paraT[:, dimW:dimW+self.outD].unsqueeze(1).unsqueeze(-1)
        #print("B: ", B.size())
        # [bsize*H*W, outdim, 1]

        #X = inX
        #print("X: ", X.size())
        # [bsize*H*W, indim, 1]

        #Y = torch.matmul(W, inX) + B
        Y = torch.matmul(paraT[:, 0:dimW].view(bsize, 1, self.outD, self.inD), inX) + \
            paraT[:, dimW:dimW+self.outD].unsqueeze(1).unsqueeze(-1)
        if self.ifRelu:
            Y = self.relu(Y)

        # [bsize*H*W, outdim, 1]
        return Y


class pixelWiseAdpNet(nn.Module):
    def __init__(self, in_ch, out_ch, inX, device, FCDims=[64, 64, 64, 64]):
        super(pixelWiseAdpNet, self).__init__()
        #din = in_ch
        # c_total = 0
        # for fd in FCDims:
        #     c_total += din*fd + fd
        #     din = fd
        # c_total += din * out_ch + out_ch

        #self.encoder = LR_Encoder(g_ch, c_total)

        self.pixelWFC = []
        din = in_ch
        for fd in FCDims:
            self.pixelWFC = self.pixelWFC + [pixelWiseFC(din, fd, True)]
            din = fd
        for i, graph in enumerate(self.pixelWFC):
            self.add_module('pixelWFC_{}'.format(i), graph)
        self.outWFC = pixelWiseFC(din, out_ch, False)
        self.actF = nn.Sigmoid()

        self.in_ch = in_ch
        self.FCDims = FCDims
        self.out_ch = out_ch

        self.BSize, _, self.H, self.W = inX.size()
        self.inX = inX  # [bsize, dim, H, W]
        #self.gridXY = gridXY.permute(0, 2, 3, 1)  # [bsize, H, W, dim]
        self.device = device
        #self.ps = ps

    def flattenX(self, x):
        x = x.permute(0, 2, 3, 1)  # [bsize, H, W, dim]
        x = torch.flatten(x, 1, 2)  # [bsize, H*W, dim]
        return x

    def cal_HyperMPL(self, x, p_MPL, h, w):
        x = self.flattenX(x).unsqueeze(-1)
        # x: [bsize, H*W, dim, 1]
        # p_MPL: [bsize, dim]
        #p_MPL = self.flattenX(p_MPL)  #[bsize*H*W, dim]
        #print(p_MPL.size())
        din = self.in_ch
        dim_beg = 0
        for fd, wFC in zip(self.FCDims, self.pixelWFC):
            dim_end = dim_beg + din * fd + fd
            x = wFC(x, p_MPL[:, dim_beg:dim_end])
            dim_beg = dim_end
            din = fd

        dim_end = dim_beg + din * self.out_ch + self.out_ch
        # ppT = adap_params[:, dim_beg:dim_end]
        x = self.outWFC(x, p_MPL[:, dim_beg:dim_end])
        x = self.actF(x).squeeze(-1)
        #x = x.squeeze(-1)

        x = x.view(self.BSize, h, w, self.out_ch)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, adap_params):
        _,_, aH, aW = adap_params.size()
        psH, psW = self.H // aH, self.W//aW
        #print(adap_params.size(), psH, psW)

        out = torch.zeros(self.BSize, self.out_ch, self.H, self.W).to(self.device)
        for ky in range(aH):
            for kx in range(aW):
                y_b = ky*psH
                x_b = kx*psW
                y_e = min(y_b + psH, self.H)
                x_e = min(x_b + psW, self.W)
                #print(y_b, y_e, x_b, x_e)
                x = self.inX[:, :, y_b:y_e, x_b:x_e]
                #s = self.gridXY[:, y_b:y_e, x_b:x_e, :]
                #p_MPL = F.grid_sample(adap_params, s)
                p_MPL = adap_params[:, :, ky, kx]  # [bsize, dim]
                #x = self.cal_HyperMPL(x, p_MPL, y_e-y_b, x_e-x_b)
                out[:, :, y_b:y_e, x_b:x_e] = self.cal_HyperMPL(x, p_MPL, y_e-y_b, x_e-x_b)

        return out


class DiscriminatorPatchCNN(nn.Module):
    def __init__(self, inDim, ifSec=False):
        super(DiscriminatorPatchCNN, self).__init__()
        '''
        Receptive fild: (output_size - 1) * stride_size + kernel_size
        [70 <-- 34 <-- 16 <-- 7 <-- 4 <-- output_size (1)]
        Output image size: 1 + (in_size + 2*padding - (kernel_size-1) - 1) / stride
        [512-->256 --> 128 --> 64 --> 32 --> output_size(31)]
        '''
        if ifSec is True:
            sequence = [spectral_norm(nn.Conv2d(inDim, 64, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)] + \
                       [spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))]
        else:
            sequence = [nn.Conv2d(inDim, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] +\
                       [nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] +\
                       [nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] +\
                       [nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)] + \
                       [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)
        #self.weight_ini(self.model)

    def getLayer(self, id_list, x):
        out = []
        bInd = 0
        for id in id_list:
            if len(out) == 0:
                out.append(self.model[:id](x))
            else:
                tx = out[-1]
                out.append(self.model[bInd:id](tx))
            bInd = id

        return out

    def weight_ini(self, net, c=6.):
        for m in net.state_dict():
            k = m.split('.')
            if k[-1] == 'weight':
                siren_uniform(net.state_dict()[m], mode='fan_in', c=c)

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)


if __name__== '__main__':
    USE_CUDA = True
    device = torch.device("cuda:1" if USE_CUDA else "cpu")
    bSize = 2

    g_F = torch.randn(bSize, 3, 128, 128).to(device)
    #i_F = torch.randn(1, 32, 256, 256).to(device)
    imgS = 256
    K = 8
    gridPixels = defPixelGrid(imgS, imgS, device)
    inX = PoseEncoding(gridPixels, K)
    inX = inX.repeat(bSize, 1, 1, 1)
    _, encoding_ch, _, _ = inX.size()
    print(inX.size())
    gPP = defInterGrid(imgS, imgS, device)
    gPP = gPP.repeat(1, 1, 1, 1)
    FCDims = [64, 64, 64, 64]

    r_model = pixelWiseAdpNet(in_ch=encoding_ch, out_ch=3, inX=inX, device=device, FCDims=FCDims).to(device)

    din = encoding_ch
    c_total = 0
    for fd in FCDims:
        c_total += din * fd + fd
        din = fd
    c_total += din * 3 + 3
    MLP_gen = LR_Encoder(in_ch=3, c_total=c_total).to(device)
    #print(model)

    feat = MLP_gen(g_F)
    out = r_model(feat)
    print('out: ', out.size())
    print('Done')

