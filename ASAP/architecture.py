from models import *
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from vgg import VGG19, vgg_std, vgg_mean, vgg_Normalization


class architecture(object):
    def __init__(self, imgH, imgW, guiH, guiW, K, dimG, dimOut, bsize, device, genCKPName=None, disCKPName=None):
        self.imgH = imgH
        self.imgW = imgW
        self.guiH = guiH
        self.guiW = guiW
        self.device = device
        gridPixels = defPixelGrid(self.imgH, self.imgW, self.device)
        inX = PoseEncoding(gridPixels, K)
        print(inX.size())
        # save_image(inX[0, 30, :, :]*0.5 + 0.5, filename='../test/i_1.jpg')
        # exit(1)
        inX = inX.repeat(bsize, 1, 1, 1)
        _, self.encodeDim, _, _ = inX.size()
        self.dimG = dimG
        self.dimOut = dimOut
        FCDims = [64, 64, 64, 64]

        self.pixel_net = pixelWiseAdpNet(in_ch=self.encodeDim, out_ch=self.dimOut,
                                         inX=inX, device=device, FCDims=FCDims).to(device)

        din = self.encodeDim
        c_total = 0
        for fd in FCDims:
            c_total += din * fd + fd
            din = fd
        c_total += din * self.dimOut + self.dimOut
        self.G_net = LR_Encoder(self.dimG, c_total).to(device)
        print(self.G_net)
        if genCKPName is not None:
            gen_ckp = self.load_ckp(genCKPName)
            self.G_net.load_state_dict(gen_ckp['Generator'])

        self.D_net = DiscriminatorPatchCNN(inDim=self.dimG+self.dimOut, ifSec=False).to(device)
        if disCKPName is not None:
            dis_ckp = self.load_ckp(disCKPName)
            self.D_net.load_state_dict(dis_ckp['Discriminator'])

        self.optimizer_G = torch.optim.Adam(self.G_net.parameters(), lr=1.e-4)
        self.optimizer_D = torch.optim.Adam(self.D_net.parameters(), lr=3.e-4)

        self.Gimg_transform = transforms.Compose([transforms.Resize((self.guiH, self.guiW)), transforms.ToTensor()])
        self.Timg_transform = transforms.Compose([transforms.Resize((self.imgH, self.imgW)), transforms.ToTensor()])
        self.VGG_transform = transforms.Compose([transforms.Resize((self.imgH, self.imgW)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(vgg_mean, vgg_std)])
        self.VGG_Norm = vgg_Normalization(vgg_mean, vgg_std, device)

        self.DataLoss = torch.nn.L1Loss().to(self.device)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.vgg = VGG19(device).to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def ganLoss(self, x, isReal):
        if isReal:
            target_Label = torch.full(x.size(), 1., dtype=torch.float32, device=self.device)
        else:
            target_Label = torch.full(x.size(), 0., dtype=torch.float32, device=self.device)
        return self.criterion(x, target_Label)

    def DNet_FeatLoss(self, gt, x, Layers):
        LFeatures_gt = self.D_net.getLayer(Layers, gt)
        LFeatures_x = self.D_net.getLayer(Layers, x)
        dist = 0.
        for l in range(len(Layers)):
            dist += self.DataLoss(LFeatures_gt[l], LFeatures_x[l])
        return dist

    def setNet_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def ReadImgs(self, filelist, transf):
        ImgTensor = []
        for f in filelist:
            img = Image.open(f).convert('RGB')
            img = transf(img).unsqueeze(0)
            ImgTensor.append(img[:, 0:3, :, :].to(self.device))
        ImgTensor = torch.cat(ImgTensor, dim=0)
        return ImgTensor

    def peceptronLoss(self, gt, x, Layers):
        dist = 0.
        if len(Layers) == 0:
            return self.DataLoss(gt, x)

        gtFeats = self.vgg.get_content_actList(gt, Layers)
        xxFeats = self.vgg.get_content_actList(x, Layers)

        for l in range(len(Layers)):
            dist += self.DataLoss(gtFeats[l], xxFeats[l])
        dist += self.DataLoss(gt, x)
        return dist

    def iter_train_G(self, gImgNList, tImgNList):
        gtMasks = self.ReadImgs(gImgNList, self.Timg_transform)
        gtImg = self.ReadImgs(tImgNList, self.Timg_transform)

        gMasks = self.ReadImgs(gImgNList, self.Gimg_transform)

        self.setNet_requires_grad(self.G_net, True)
        self.setNet_requires_grad(self.D_net, False)
        self.setNet_requires_grad(self.pixel_net, False)
        self.G_net.train()
        self.optimizer_G.zero_grad()

        feat = self.G_net(gMasks)
        out = self.pixel_net(feat)

        exp_Real = self.D_net(torch.cat([out, gtMasks], dim=1))
        discrim_t = self.ganLoss(exp_Real, True)
        dfeat_t = self.DNet_FeatLoss(torch.cat([gtImg, gtMasks], dim=1),
                                     torch.cat([out, gtMasks], dim=1),
                                     [1, 3, 5])
        peceptron_t = self.peceptronLoss(self.VGG_Norm(gtImg), self.VGG_Norm(out), [4, 12, 30])
        #LossG = peceptron_t
        LossG = discrim_t + 10.*dfeat_t + 10.*peceptron_t
        LossG.backward()
        self.optimizer_G.step()
        #save_image(gtMasks, filename='../test/t_3.png')

        return LossG, peceptron_t, out.detach(), gtImg, gtMasks

    def iter_train_D(self, out, gtImg, gtMasks):
        self.setNet_requires_grad(self.D_net, True)
        self.setNet_requires_grad(self.G_net, False)
        self.setNet_requires_grad(self.pixel_net, False)

        self.D_net.train()
        self.optimizer_D.zero_grad()
        D_Fake = self.D_net(torch.cat([out, gtMasks], dim=1))
        fake_Loss = self.ganLoss(D_Fake, False)
        D_real = self.D_net(torch.cat([gtImg, gtMasks], dim=1))
        real_Loss = self.ganLoss(D_real, True)
        LossD = fake_Loss + real_Loss
        LossD.backward()
        self.optimizer_D.step()

        return LossD

    def test_Gen(self, gImgNList, tImgNList):
        gtMasks = self.ReadImgs(gImgNList, self.Timg_transform)
        gtImg = self.ReadImgs(tImgNList, self.Timg_transform)

        gMasks = self.ReadImgs(gImgNList, self.Gimg_transform)

        self.setNet_requires_grad(self.G_net, False)
        self.setNet_requires_grad(self.D_net, False)
        self.setNet_requires_grad(self.pixel_net, False)
        self.G_net.eval()
        with torch.no_grad():
            feat = self.G_net(gMasks)
            out = self.pixel_net(feat)

        return out, gtImg, gtMasks

    def save_ckp(self, savePref, itt):
        torch.save({'itter': itt, 'Generator': self.G_net.state_dict(),
                    'G_optimizer': self.optimizer_G.state_dict()}, savePref+'_Gen.ckp')

        torch.save({'itter': itt, 'Discriminator': self.D_net.state_dict(),
                    'D_optimizer': self.optimizer_D.state_dict()},
                   savePref + '_Dis.ckp')

    def load_ckp(self, fileName):
        ckp = torch.load(fileName, map_location=lambda storage, loc: storage)
        return ckp








