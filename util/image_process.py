import random
import torch
import cv2
import numpy as np
import math
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.autograd import Variable
from torchvision import transforms
import torchvision.transforms.functional as F
# import albumentations as A

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()
preprocess = transforms.Compose([
    transforms.ToTensor()])


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


def random_JPEG_compression(x):
    # def __call__(self, img):
    qf = random.randrange(10, 100, 10)
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
    stddev = random.uniform(0.05, 0.2)
    # noise = Variable(input.data.new(input.size()).normal_(0, stddev))
    # output = input + noise
    output = torch.clamp(input.clone() + (torch.randn(
        [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]) * stddev).to(input.device), 0, 1)
    return output.cuda()


def random_gaussian_blur(x):
    # def __call__(self, img):
    kernelsize = random.randrange(1, 19+1, 2)
    output = F.gaussian_blur(x, kernel_size=kernelsize)
    return output
    # lst_img = []
    # for img in x:
    #     img = _to_pil_image(img.detach().clone().cpu())
    #     img.filter(ImageFilter.GaussianBlur(radius=kernelsize))
    #     virtualpath = BytesIO()
    #     img.save(virtualpath, 'JPEG', quality=100)
    #     lst_img.append(_to_tensor(Image.open(virtualpath)))
    # return x.new_tensor(torch.stack(lst_img)).cuda()


def random_color_enhance(x, method):
    setting = random.uniform(0.2, 1.0)
    if method == 'brightness':
        color_jitter = transforms.ColorJitter(brightness=(0.99 + setting, 1.01 + setting))
    elif method == 'contrast':
        color_jitter = transforms.ColorJitter(contrast=(0.99 + setting, 1.01 + setting))
    elif method == 'saturation':
        color_jitter = transforms.ColorJitter(saturation=(0.99 + setting, 1.01 + setting))
    else:
        raise NameError('HiThere')
    image = color_jitter(x)
    # image_ = tensor_to_image(image[0])
    # image_ = Image.fromarray(image_, mode='RGB')
    # virtualpath = BytesIO()
    # image_.save(virtualpath, 'JPEG', quality=100)
    # image = Image.open(virtualpath).convert('RGB')
    # image = preprocess(image).unsqueeze(0)
    # image = image.cuda()
    image = image.to(x.device)
    return image


def data_augmantation(images):
    x = random.random()
    if 0.5 <= x < 0.6:
        outputs = random_JPEG_compression(images)
    elif 0.6 <= x < 0.7:
        outputs = random_gaussian_noise(images)
    elif 0.7 <= x < 0.8:
        outputs = random_gaussian_blur(images)
    elif 0.8 <= x < 0.86:
        outputs = random_color_enhance(images, 'brightness')
    elif 0.86 <= x < 0.93:
        outputs = random_color_enhance(images, 'contrast')
    elif 0.93 <= x < 1.0:
        outputs = random_color_enhance(images, 'saturation')
    else:
        return images
    # print(x)
    # print(outputs.size())
    return outputs


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


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


def gaussian_noise(input, stddev):
    # noise = Variable(input.data.new(input.size()).normal_(0, stddev))
    # output = torch.clamp(input + noise, -1, 1)
    # output = A.GaussNoise(var_limit=stddev, p=1)
    output = torch.clamp(input.clone() + (torch.randn(
        [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]) * (stddev**0.5)).to(input.device), 0, 1)
    return output


class GaussianBlurFilter(Processor):
    def __init__(self, kernelsize=2):
        super(GaussianBlurFilter, self).__init__()
        self.kernelsize = kernelsize

    def forward(self, x):
        return GaussianBlurEncodeDecode.apply(x, self.kernelsize).to(x.device)


class GaussianBlurEncodeDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernelsize):
        images = F.gaussian_blur(x, kernel_size=kernelsize)
        images = images.to(x.device)
        return images
        # lst_img = []
        # for img in x:
        #     img = _to_pil_image(img.detach().clone().cpu())
        #     img.filter(ImageFilter.GaussianBlur(radius=kernelsize))
        #     virtualpath = BytesIO()
        #     img.save(virtualpath, 'JPEG', quality=100)
        #     lst_img.append(_to_tensor(Image.open(virtualpath)))
        # return x.new_tensor(torch.stack(lst_img))

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", GaussianBlurEncodeDecode)


def _gaussian_blur(img, ks):
    output = F.gaussian_blur(img, kernel_size=ks)
    # output = _gb(img)
    return output


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
        images = color_jitter(x)
        images = images.to(x.device)
        return images
        # lst_img = []
        # for img in x:
        #     img = color_jitter(img)
        #     img = _to_pil_image(img.detach().clone().cpu())
        #     virtualpath = BytesIO()
        #     img.save(virtualpath, 'JPEG', quality=100)
        #     lst_img.append(_to_tensor(Image.open(virtualpath)))
        # return x.new_tensor(torch.stack(lst_img))

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", ColorEncodeDecode)


