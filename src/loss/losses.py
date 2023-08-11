# yapf: disable
import torch.nn as nn
from .ssim import MS_SSIM

class GeneratorLoss(nn.Module):

    def __init__(self, device, alpha=0.3, loss='mse'):
        super(GeneratorLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = MS_SSIM()
        self.mse = nn.MSELoss()

        if loss == "mse":
            self.loss = lambda watermark, img : self.mse(self.alpha * watermark + img, img)
        elif loss == "ssim":
            self.loss = lambda watermark, img : 0.16 * self.l1(self.alpha * watermark + img, img) + 0.84 * self.ssim(self.alpha * watermark + img, img)
        else:
            assert(False)

    def forward(self, watermark, img):
        return self.loss(watermark, img)

class BobLoss(nn.Module):

    def __init__(self, device):
        super(BobLoss, self).__init__()
        self.device = device
        self.mse = nn.BCELoss()

    def forward(self, pred_masks, ground_truth_masks):
        return self.mse(pred_masks, ground_truth_masks)