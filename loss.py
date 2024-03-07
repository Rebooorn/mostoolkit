import torch
import numpy as np
import monai
import torch.nn.functional as F
import torchvision

## Some arithmatic helpers
def grad_x(vol):
    # vol.shape = [c, w, h, d]
    # return the gradient of vol in x direction
    c, w, h, d = vol.shape
    grad = (vol[:, 1:, :, :] - vol[:, :-1, :, :]) / 2
    grad = torch.cat([torch.zeros([c, 1, h, d]).cuda(), grad], dim=1)
    return grad

def grad_y(vol):
    c, w, h, d = vol.shape
    grad = (vol[:, :, 1:, :] - vol[:, :, :-1, :]) / 2
    grad = torch.cat([torch.zeros([c, w, 1, d]).cuda(), grad], dim=2)
    return grad

def grad_z(vol):
    c, w, h, d = vol.shape
    grad = (vol[:, :, :, 1:] - vol[:, :, :, :-1]) / 2
    grad = torch.cat([torch.zeros([c, w, h, 1]).cuda(), grad], dim=3)
    return grad

## For debug
import matplotlib.pyplot as plt

class MultipleOutputLoss(torch.nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class dice_loss_(torch.nn.Module):
    def __init__(self, smooth=0.1):
        super(dice_loss_, self).__init__()
        self.smooth = smooth

    def __call__(self, y_hat, y):

        bs, out_channel, w, h = y_hat.shape

        y_hat = torch.sigmoid(y_hat)
        y = y.unsqueeze(1)
        y_onehot = torch.zeros_like(y_hat)
        y_onehot.scatter_(1, y.type(torch.int64), 1)

        y_flat = y_onehot.view(-1)
        y_hat_flat = y_hat.view(-1)

        intersection = (y_flat * y_hat_flat).sum()

        return 1 - (2 * intersection + self.smooth) / (y_flat.sum() + y_hat_flat.sum() + self.smooth)


class DiceLoss(torch.nn.Module):
    'Modification from monai.losses.DiceLoss'
    def __init__(self, include_background=True):
        super(DiceLoss, self).__init__()
        self.dice_loss = monai.losses.DiceLoss(to_onehot_y=True, include_background=include_background)
        # print('CAUTION: loaded diceloss contains sigmoid already')

    def __call__(self, y_hat, y):
        # calculate dice loss using torch methods
        # y_hat.shape = [bs, out_channel, w, h]
        # y.shape = [bs, w, h]
        return self.dice_loss(torch.sigmoid(y_hat), y)

    def forward(self, y_hat, y):
        return self.dice_loss(torch.sigmoid(y_hat), y)
        # return self.dice_loss(y_hat, y.unsqueeze(1))


class DiceCELoss(torch.nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, weight, include_background=True):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, include_background=include_background)
        self.cross_entropy = CELoss(weight, include_background)
        # print('CAUTION: loaded diceloss contains sigmoid already')

    def forward(self, y_pred, y_true):
        # y_pred = torch.sigmoid(y_pred)
        dice = self.dice(torch.sigmoid(y_pred), y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy


class CELoss(torch.nn.Module):
    """Xentropy loss"""

    def __init__(self, weight, include_background=True):
        super().__init__()
        if not include_background:
            weight[0] = 0.
        weight = torch.tensor(np.array(weight)).float().cuda()
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, y_pred, y_true):
        # y_pred.shape = [B, C, DWH...]
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return cross_entropy


# loss for registration, used in https://arxiv.org/ftp/arxiv/papers/1711/1711.01666.pdf
def get_bending_energy(disp):
    c, w, h, c = disp.shape
    dxx = grad_x(grad_x(disp))
    dyy = grad_y(grad_y(disp))
    dzz = grad_z(grad_z(disp))
    dxy = grad_x(grad_y(disp))
    dxz = grad_x(grad_z(disp))
    dyz = grad_y(grad_z(disp))
    e = torch.sum(dxx ** 2) + torch.sum(dyy ** 2) + torch.sum(dzz ** 2) + 2 * torch.sum(dxy ** 2) \
        + 2 * torch.sum(dxz ** 2) + 2 * torch.sum(dyz ** 2)
    return e / (c * w * h * c)

class CELoss_bending_energy(torch.nn.Module):
    def __init__(self, one_hot=False, regularization=True):
        """ params: one_hot: set False when the input will not be one-hot encoded. regularization: whether to
        include bending energy regulariser.

        GOOD LUCK CONVERGING!
        """
        super().__init__()
        self.one_hot = one_hot
        self.regularization = regularization

    def forward(self, y_pred, y_true, disp):
        if self.one_hot is False:
            # y_pred has one extra dim
            y_pred = F.one_hot(y_pred.long().squeeze(-1)).float()
            y_true = F.one_hot(y_true.long()).float()
        celoss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        if self.regularization:
            bending_energy = get_bending_energy(torch.squeeze(disp))
        else:
            bending_energy = 0
        return celoss + bending_energy


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())

        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class VGGPerceptualLoss3D(VGGPerceptualLoss):
    def __init__(self, resize=True, direction='xy', sigmoid=False):
        super().__init__(resize)
        self.direction = direction
        self.sigmoid = sigmoid

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1, 1)
            target = target.repeat(1, 3, 1, 1, 1)
        input = (input-self.mean.view(1,3,1,1,1)) / self.std.view(1,3,1,1,1)
        target = (target-self.mean.view(1,3,1,1,1)) / self.std.view(1,3,1,1,1)
        if self.resize:
            input = self.transform(input, mode='trilinear', size=(224, 224, input.shape[-1]), align_corners=False)
            target = self.transform(target, mode='trilinear', size=(224, 224, target.shape[-1]), align_corners=False)
        loss = 0.0
        if self.direction == 'yz':
            x = torch.cat([input[i] for i in range(input.shape[0])], dim=-1).permute(3, 0, 1, 2)
            y = torch.cat([target[i] for i in range(target.shape[0])], dim=-1).permute(3, 0, 1, 2)
        elif self.direction == 'xz':
            x = torch.cat([input[i] for i in range(input.shape[0])], dim=2).permute(2, 0, 1, 3)
            y = torch.cat([target[i] for i in range(target.shape[0])], dim=2).permute(2, 0, 1, 3)
        else:
            # xy
            x = torch.cat([input[i] for i in range(input.shape[0])], dim=1).permute(1,0,2,3)
            y = torch.cat([target[i] for i in range(target.shape[0])], dim=1).permute(1,0,2,3)
        # print(x.shape)

        # xx = x.cpu().detach().numpy()[64,0]
        # yy = y.cpu().detach().numpy()[64,0]
        # fig, axes = plt.subplots(1,2)
        # axes[0].imshow(xx)
        # axes[1].imshow(yy)
        # plt.show()

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if self.sigmoid:
                x = torch.sigmoid(x)
                y = torch.sigmoid(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

if __name__ == '__main__':
    # d_disp = np.arange(50) ** 3

    # d_disp = d_disp.reshape(1, 1, 1, 50)
    # d_disp = np.repeat(d_disp, 50, axis=2)
    # d_disp = np.repeat(d_disp, 50, axis=1)
    # d_disp = np.repeat(d_disp, 3, axis=0)

    # d_disp = torch.Tensor(d_disp).long().cuda()
    # # e = get_bending_energy(d_disp)
    # # print(e)

    # d_pred = torch.randint(0, 3, size=[1, 50, 50, 50]).cuda()
    # d_true = d_pred.clone()

    # loss = CELoss_bending_energy(regularization=True)
    # l = loss.forward(y_pred=d_pred, y_true=d_true, disp=d_disp)
    # print(l)
    y = torch.rand(2, 3, 100, 100, 100)
    y_hat = torch.rand(2, 3, 100, 100, 100)
    loss = VGGPerceptualLoss3D(resize=False)
    l = loss(y, y_hat)
    print(l)