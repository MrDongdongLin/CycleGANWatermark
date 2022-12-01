import random
import time
import datetime
import sys
import cv2
import os

from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
from math import log10, sqrt, exp
from PIL import Image, ImageFilter

from torchvision.utils import save_image


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False, weights=None):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3):
        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3, weights=None):
        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)


def generate_random_watermarks(watermark_size, batch_size=4):
    z = torch.zeros((batch_size, watermark_size), dtype=torch.float).random_(0, 2)
    return z


try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from torchvision import transforms
from PIL import Image
import torch.nn as nn

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


class Processor(nn.Module):
    """
    Processor
    """
    def __init__(self):
        super(Processor, self).__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        return 'EmptyDefense (Identity)'


preprocess = transforms.Compose([
    transforms.ToTensor()])


def jpeg_compression(image, path, setting, device, i):
    image = tensor_to_image2(image[0])
    image = Image.fromarray(image, mode='RGB')
    # image.save(os.path.join(root_path, 'jpeg', str(setting), f'{i}.jpeg'), quality=setting)
    image.save(os.path.join('D:/Deepfake/Datasets/test/', f'{i}.jpeg'), quality=setting)

    image = Image.open(os.path.join('D:/Deepfake/Datasets/test/', f'{i}.jpeg')).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)
    return image


def tensor_to_image(input):
    # mean = ([0.5] * 3)
    # std = ([0.5] * 3)
    # unorm = transforms.Normalize(
    #     mean=[-m / s for m, s in zip(mean, std)],
    #     std=[1 / s for s in std]
    # )
    output = input.permute(1, 2, 0)
    # for adv, id in zip(adv_image, frame_ids):
    output = output.cpu().detach().numpy() * 255.0
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR).astype(np.uint8)

    return output

def tensor_to_image2(tensor):
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def unnorm(mean, std, input):
    # mean = ([0.5] * 3)
    # std = ([0.5] * 3)
    unorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    output = unorm(input)

    return output


def get_transform(opt, method=Image.BICUBIC, convert=True):
    transform_list = []
    # if grayscale:
    #     transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))

    if 'flip' in opt.preprocess:
        transform_list.append(transforms.RandomHorizontalFlip())

    if 'norm' in opt.preprocess:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if convert:
        transform_list += [transforms.ToTensor()]
        # if grayscale:
        #     transform_list += [transforms.Normalize((0.5,), (0.5,))]
        # else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transform_list


def gaussian_noise(input, stddev):
    noise = Variable(input.data.new(input.size()).normal_(0, stddev))
    output = input + noise
    return output


class JPEGFilter(Processor):
    def __init__(self, quality=75):
        super(JPEGFilter, self).__init__()
        self.quality = quality

    def forward(self, x):
        return JPEGEncodingDecoding.apply(x, self.quality).to(x.device)


class JPEGEncodingDecoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quality):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=quality)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", JPEGEncodingDecoding)


class GaussianBlurFilter(Processor):
    def __init__(self, kernelsize=2):
        super(GaussianBlurFilter, self).__init__()
        self.kernelsize = kernelsize

    def forward(self, x):
        return GaussianBlurEncodeDecode.apply(x, self.kernelsize).to(x.device)


class GaussianBlurEncodeDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernelsize):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            img.filter(ImageFilter.GaussianBlur(radius=kernelsize))
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=100)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", GaussianBlurEncodeDecode)


def gaussian_blur(input, radius):
    input = tensor_to_image(input[0])
    input = Image.fromarray(input, mode='RGB')
    input = input.filter(ImageFilter.GaussianBlur(radius=radius))
    virtualpath = BytesIO()
    input.save(virtualpath, 'JPEG', quality=100)
    image = Image.open(virtualpath).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    image = image.cuda()
    return image


class ColorFilter(Processor):
    def __init__(self, factor=0.2):
        super(ColorFilter, self).__init__()
        self.factor = factor

    def forward(self, x, method):
        return ColorEncodeDecode.apply(x, method, self.factor).to(x.device)


class ColorEncodeDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, method, factor):
        if method == 'brightness':
            color_jitter = transforms.ColorJitter(brightness=(0.99 + factor, 1.01 + factor))
        elif method == 'contrast':
            color_jitter = transforms.ColorJitter(contrast=(0.99 + factor, 1.01 + factor))
        elif method == 'saturation':
            color_jitter = transforms.ColorJitter(saturation=(0.99 + factor, 1.01 + factor))
        else:
            raise NameError('HiThere')
        lst_img = []
        for img in x:
            img = color_jitter(img)
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=100)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", ColorEncodeDecode)


def color(input, method, setting):
    if method == 'brightness':
        color_jitter = transforms.ColorJitter(brightness=(0.99 + setting, 1.01 + setting))
    elif method == 'contrast':
        color_jitter = transforms.ColorJitter(contrast=(0.99 + setting, 1.01 + setting))
    elif method == 'saturation':
        color_jitter = transforms.ColorJitter(saturation=(0.99 + setting, 1.01 + setting))
    else:
        raise NameError('HiThere')
    image = color_jitter(input)
    image_ = tensor_to_image(image[0])
    image_ = Image.fromarray(image_, mode='RGB')
    virtualpath = BytesIO()
    image_.save(virtualpath, 'JPEG', quality=100)
    image = Image.open(virtualpath).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    image = image.cuda()
    return image


def random_JPEG_compression(x):
    # def __call__(self, img):
    qf = random.randrange(10, 90)
    lst_img = []
    for img in x:
        img = _to_pil_image(img.detach().clone().cpu())
        virtualpath = BytesIO()
        img.save(virtualpath, 'JPEG', quality=qf)
        lst_img.append(_to_tensor(Image.open(virtualpath)))
    return x.new_tensor(torch.stack(lst_img)).cuda()
    # outputIoStream = BytesIO()
    # img.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    # outputIoStream.seek(0)
    # return _to_tensor(Image.open(outputIoStream))


def random_gaussian_noise(input):
    stddev = random.uniform(0.05, 0.5)
    noise = Variable(input.data.new(input.size()).normal_(0, stddev))
    output = input + noise
    return output.cuda()


def random_gaussian_blur(x):
    # def __call__(self, img):
    kernelsize = random.randrange(1, 10)
    lst_img = []
    for img in x:
        img = _to_pil_image(img.detach().clone().cpu())
        img.filter(ImageFilter.GaussianBlur(radius=kernelsize))
        virtualpath = BytesIO()
        img.save(virtualpath, 'JPEG', quality=100)
        lst_img.append(_to_tensor(Image.open(virtualpath)))
    return x.new_tensor(torch.stack(lst_img)).cuda()


def random_color_enhance(x, method):
    setting = random.uniform(0.2, 2.0)
    if method == 'brightness':
        color_jitter = transforms.ColorJitter(brightness=(0.99 + setting, 1.01 + setting))
    elif method == 'contrast':
        color_jitter = transforms.ColorJitter(contrast=(0.99 + setting, 1.01 + setting))
    elif method == 'saturation':
        color_jitter = transforms.ColorJitter(saturation=(0.99 + setting, 1.01 + setting))
    else:
        raise NameError('HiThere')
    image = color_jitter(x)
    image_ = tensor_to_image(image[0])
    image_ = Image.fromarray(image_, mode='RGB')
    virtualpath = BytesIO()
    image_.save(virtualpath, 'JPEG', quality=100)
    image = Image.open(virtualpath).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    image = image.cuda()
    return image