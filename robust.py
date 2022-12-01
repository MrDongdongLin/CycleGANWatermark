"""
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python robust.py --dataroot /pubdata/ldd/winter2summer --name winter2summer_cyclegan --model cyclegan_wb --num_test 100 --eval --encoder_path /home/lindd/projects/CycleGANWatermark/wmextractor/stegastamp_256_200_encoder.pth --decoder_path /home/lindd/projects/CycleGANWatermark/wmextractor/stegastamp_256_200_decoder.pth
"""
import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import get_acc, unnorm
import torchvision.utils as vutils
from models.networks import StegaStampEncoder, StegaStampDecoder
from util.image_process import JPEGFilter, gaussian_noise, ColorFilter, _gaussian_blur
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers


    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    Encoder = StegaStampEncoder(
        opt.crop_size,
        opt.input_nc,
        opt.watermark_size,
        # WATERMARK_UPSIZE=50,
        return_residual=False,
    )
    Encoder = Encoder.cuda()
    Encoder.load_state_dict(torch.load(opt.encoder_path, map_location=model.device))
    Encoder.eval()
    Decoder = StegaStampDecoder(opt.crop_size, opt.input_nc, opt.watermark_size)
    Decoder = Decoder.cuda()
    Decoder.load_state_dict(torch.load(opt.decoder_path, map_location=model.device))
    Decoder.eval()

    psnr = PSNR().to(model.device)
    ssim = SSIM().to(model.device)

    if not os.path.exists(os.path.join(opt.results_dir, opt.name, 'robust/bitacc/')):
        os.makedirs(os.path.join(opt.results_dir, opt.name, 'robust/bitacc/'))
    if not os.path.exists(os.path.join(opt.results_dir, opt.name, 'robust/images/')):
        os.makedirs(os.path.join(opt.results_dir, opt.name, 'robust/images/'))

    args_qf = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    args_stddev = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # args_blur = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    args_blur = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    args_color = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    sum_bitacc_fake_B = 0
    sum_bitacc_fake_A = 0
    sum_bitacc_fake_EA = 0
    sum_bitacc_fake_EB = 0
    sum_list_jpeg = np.zeros((4, len(args_qf)))
    sum_list_noise = np.zeros((4, len(args_stddev)))
    sum_list_blur = np.zeros((4, len(args_blur)))
    sum_list_color_bright = np.zeros((4, len(args_color)))
    sum_list_color_contrast = np.zeros((4, len(args_color)))
    sum_list_color_saturat = np.zeros((4, len(args_color)))
    sum_list_jpeg_psnr = np.zeros((4, len(args_qf)))
    sum_list_noise_psnr = np.zeros((4, len(args_stddev)))
    sum_list_blur_psnr = np.zeros((4, len(args_blur)))
    sum_list_color_bright_psnr = np.zeros((4, len(args_color)))
    sum_list_color_contrast_psnr = np.zeros((4, len(args_color)))
    sum_list_color_saturat_psnr = np.zeros((4, len(args_color)))

    log_robust_jpeg = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/robustness_jpeg_256.txt"), "w", encoding="utf-8")
    log_robust_noise = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/robustness_noise_256.txt"), "w", encoding="utf-8")
    log_robust_blur = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/robustness_blur_256.txt"), "w", encoding="utf-8")
    log_robust_color_bright = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/robustness_color_bright_256.txt"), "w", encoding="utf-8")
    log_robust_color_contrast = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/robustness_color_contrast_256.txt"), "w", encoding="utf-8")
    log_robust_color_saturat = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/robustness_color_saturat_256.txt"), "w", encoding="utf-8")

    log_psnr_jpeg = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/psnr_jpeg_256.txt"), "w", encoding="utf-8")
    log_psnr_noise = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/psnr_noise_256.txt"), "w", encoding="utf-8")
    log_psnr_blur = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/psnr_blur_256.txt"), "w", encoding="utf-8")
    log_psnr_color_bright = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/psnr_color_bright_256.txt"), "w", encoding="utf-8")
    log_psnr_color_contrast = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/psnr_color_contrast_256.txt"), "w", encoding="utf-8")
    log_psnr_color_saturat = open(os.path.join(opt.results_dir, opt.name, "robust/bitacc/psnr_color_saturat_256.txt"), "w", encoding="utf-8")

    img_save_path = os.path.join(opt.results_dir, opt.name, 'robust')

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        real_A = visuals['real_A'].to(model.device)
        real_B = visuals['real_B'].to(model.device)
        fake_A = visuals['fake_A'].to(model.device)
        fake_B = visuals['fake_B'].to(model.device)
        real_wA = model.real_wA.to(model.device)
        real_wB = model.real_wB.to(model.device)
        fake_EA = Encoder(real_wA[: real_A.size(0)], real_A)
        fake_EB = Encoder(real_wB[: real_B.size(0)], real_B)

        bitacc_fake_B = get_acc(opt, fake_B, Decoder, real_wA)
        bitacc_fake_A = get_acc(opt, fake_A, Decoder, real_wB)
        bitacc_fake_EA = get_acc(opt, fake_EA, Decoder, real_wA)
        bitacc_fake_EB = get_acc(opt, fake_EB, Decoder, real_wB)

        sum_bitacc_fake_B += bitacc_fake_B
        sum_bitacc_fake_A += bitacc_fake_A
        sum_bitacc_fake_EA += bitacc_fake_EA
        sum_bitacc_fake_EB += bitacc_fake_EB

        mean = ([0.5] * 3)
        std = ([0.5] * 3)
        fake_B_std = fake_B  # unnorm(mean, std, ganw_A)
        fake_A_std = fake_A  #unnorm(mean, std, ganw_B)
        fake_EA_std = fake_EA  #unnorm(mean, std, fake_Aw)
        fake_EB_std = fake_EB  #unnorm(mean, std, fake_Bw)

        for i, setting in enumerate(args_qf):
            fake_B_jpeg = JPEGFilter(quality=setting)(fake_B_std)
            fake_A_jpeg = JPEGFilter(quality=setting)(fake_A_std)
            fake_EA_jpeg = JPEGFilter(quality=setting)(fake_EA_std)
            fake_EB_jpeg = JPEGFilter(quality=setting)(fake_EB_std)

            bitacc_fake_B_jpeg = get_acc(opt, fake_B_jpeg, Decoder, real_wA)
            bitacc_fake_A_jpeg = get_acc(opt, fake_A_jpeg, Decoder, real_wB)
            bitacc_fake_EA_jpeg = get_acc(opt, fake_EA_jpeg, Decoder, real_wA)
            bitacc_fake_EB_jpeg = get_acc(opt, fake_EB_jpeg, Decoder, real_wB)

            sum_list_jpeg[0][i] += bitacc_fake_B_jpeg
            sum_list_jpeg[1][i] += bitacc_fake_A_jpeg
            sum_list_jpeg[2][i] += bitacc_fake_EA_jpeg
            sum_list_jpeg[3][i] += bitacc_fake_EB_jpeg
            sum_list_jpeg_psnr[0][i] += psnr(fake_B_std, fake_B_jpeg)
            sum_list_jpeg_psnr[1][i] += psnr(fake_A_std, fake_A_jpeg)
            sum_list_jpeg_psnr[2][i] += psnr(fake_EA_std, fake_EA_jpeg)
            sum_list_jpeg_psnr[3][i] += psnr(fake_EB_std, fake_EB_jpeg)

            vutils.save_image(fake_B_jpeg, os.path.join(opt.results_dir, opt.name, 'robust/images/jpegCA_{}.png'.format(setting)), normalize=False, padding=0)
            vutils.save_image(fake_A_jpeg, os.path.join(opt.results_dir, opt.name, 'robust/images/jpegCB_{}.png'.format(setting)), normalize=False,
                              padding=0)
            vutils.save_image(fake_EA_jpeg, os.path.join(opt.results_dir, opt.name, 'robust/images/jpegEA_{}.png'.format(setting)), normalize=False,
                              padding=0)
            vutils.save_image(fake_EB_jpeg, os.path.join(opt.results_dir, opt.name, 'robust/images/jpegEB_{}.png'.format(setting)), normalize=False,
                              padding=0)

        for i, setting in enumerate(args_stddev):
            fake_B_noise = gaussian_noise(fake_B, setting)
            fake_A_noise = gaussian_noise(fake_A, setting)
            fake_EA_noise = gaussian_noise(fake_EA, setting)
            fake_EB_noise = gaussian_noise(fake_EB, setting)

            bitacc_fake_B_noise = get_acc(opt, fake_B_noise, Decoder, real_wA)
            bitacc_fake_A_noise = get_acc(opt, fake_A_noise, Decoder, real_wB)
            bitacc_fake_EA_noise = get_acc(opt, fake_EA_noise, Decoder, real_wA)
            bitacc_fake_EB_noise = get_acc(opt, fake_EB_noise, Decoder, real_wB)

            sum_list_noise[0][i] += bitacc_fake_B_noise
            sum_list_noise[1][i] += bitacc_fake_A_noise
            sum_list_noise[2][i] += bitacc_fake_EA_noise
            sum_list_noise[3][i] += bitacc_fake_EB_noise
            sum_list_noise_psnr[0][i] += psnr(fake_B, fake_B_noise)
            sum_list_noise_psnr[1][i] += psnr(fake_A, fake_A_noise)
            sum_list_noise_psnr[2][i] += psnr(fake_EA, fake_EA_noise)
            sum_list_noise_psnr[3][i] += psnr(fake_EB, fake_EB_noise)

            vutils.save_image(fake_B_noise, os.path.join(opt.results_dir, opt.name, "robust/images/noiseCA_{}.png".format(setting)), normalize=False,
                              padding=0)
            vutils.save_image(fake_A_noise, os.path.join(opt.results_dir, opt.name, "robust/images/noiseCB_{}.png".format(setting)), normalize=False,
                              padding=0)
            vutils.save_image(fake_EA_noise, os.path.join(opt.results_dir, opt.name, "robust/images/noiseEA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EA_noise, os.path.join(opt.results_dir, opt.name, "robust/images/noiseEB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)

        for i, setting in enumerate(args_blur):
            # fake_A_blur = GaussianBlurFilter(kernelsize=setting)(fake_A_std)
            # fake_B_blur = GaussianBlurFilter(kernelsize=setting)(fake_B_std)
            # fake_Aw_blur = GaussianBlurFilter(kernelsize=setting)(fake_Aw_std)
            # fake_Bw_blur = GaussianBlurFilter(kernelsize=setting)(fake_Bw_std)
            # print(setting)
            fake_B_blur = _gaussian_blur(fake_B_std, setting)
            fake_A_blur = _gaussian_blur(fake_A_std, setting)
            fake_EA_blur = _gaussian_blur(fake_EA_std, setting)
            fake_EB_blur = _gaussian_blur(fake_EB_std, setting)

            bitacc_fake_B_blur = get_acc(opt, fake_B_blur, Decoder, real_wA)
            bitacc_fake_A_blur = get_acc(opt, fake_A_blur, Decoder, real_wB)
            bitacc_fake_EA_blur = get_acc(opt, fake_EA_blur, Decoder, real_wA)
            bitacc_fake_EB_blur = get_acc(opt, fake_EB_blur, Decoder, real_wB)

            sum_list_blur[0][i] += bitacc_fake_B_blur
            sum_list_blur[1][i] += bitacc_fake_A_blur
            sum_list_blur[2][i] += bitacc_fake_EA_blur
            sum_list_blur[3][i] += bitacc_fake_EB_blur
            sum_list_blur_psnr[0][i] += psnr(fake_B_std, fake_B_blur)
            sum_list_blur_psnr[1][i] += psnr(fake_A_std, fake_A_blur)
            sum_list_blur_psnr[2][i] += psnr(fake_EA_std, fake_EA_blur)
            sum_list_blur_psnr[3][i] += psnr(fake_EA_std, fake_EB_blur)

            vutils.save_image(fake_B_blur, os.path.join(opt.results_dir, opt.name, "robust/images/blurCA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_A_blur, os.path.join(opt.results_dir, opt.name, "robust/images/blurCB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EA_blur, os.path.join(opt.results_dir, opt.name, "robust/images/blurEA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EB_blur, os.path.join(opt.results_dir, opt.name, "robust/images/blurEB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)

        for i, setting in enumerate(args_color):
            fake_B_color_bright = ColorFilter(factor=setting)(fake_B_std, 'brightness')
            fake_A_color_bright = ColorFilter(factor=setting)(fake_A_std, 'brightness')
            fake_EA_color_bright = ColorFilter(factor=setting)(fake_EA_std, 'brightness')
            fake_EB_color_bright = ColorFilter(factor=setting)(fake_EB_std, 'brightness')

            bitacc_fake_B_color_bright = get_acc(opt, fake_B_color_bright, Decoder, real_wA)
            bitacc_fake_A_color_bright = get_acc(opt, fake_A_color_bright, Decoder, real_wB)
            bitacc_fake_EA_color_bright = get_acc(opt, fake_EA_color_bright, Decoder, real_wA)
            bitacc_fake_EB_color_bright = get_acc(opt, fake_EB_color_bright, Decoder, real_wB)

            sum_list_color_bright[0][i] += bitacc_fake_B_color_bright
            sum_list_color_bright[1][i] += bitacc_fake_A_color_bright
            sum_list_color_bright[2][i] += bitacc_fake_EA_color_bright
            sum_list_color_bright[3][i] += bitacc_fake_EB_color_bright
            sum_list_color_bright_psnr[0][i] += psnr(fake_B_std, fake_B_color_bright)
            sum_list_color_bright_psnr[1][i] += psnr(fake_A_std, fake_A_color_bright)
            sum_list_color_bright_psnr[2][i] += psnr(fake_EA_std, fake_EA_color_bright)
            sum_list_color_bright_psnr[3][i] += psnr(fake_EA_std, fake_EB_color_bright)

            fake_B_color_contrast = ColorFilter(factor=setting)(fake_B_std, 'contrast')
            fake_A_color_contrast = ColorFilter(factor=setting)(fake_A_std, 'contrast')
            fake_EA_color_contrast = ColorFilter(factor=setting)(fake_EA_std, 'contrast')
            fake_EB_color_contrast = ColorFilter(factor=setting)(fake_EB_std, 'contrast')

            bitacc_fake_B_color_contrast = get_acc(opt, fake_B_color_contrast, Decoder, real_wA)
            bitacc_fake_A_color_contrast = get_acc(opt, fake_A_color_contrast, Decoder, real_wB)
            bitacc_fake_EA_color_contrast = get_acc(opt, fake_EA_color_contrast, Decoder, real_wA)
            bitacc_fake_EB_color_contrast = get_acc(opt, fake_EB_color_contrast, Decoder, real_wB)

            sum_list_color_contrast[0][i] += bitacc_fake_B_color_contrast
            sum_list_color_contrast[1][i] += bitacc_fake_A_color_contrast
            sum_list_color_contrast[2][i] += bitacc_fake_EA_color_contrast
            sum_list_color_contrast[3][i] += bitacc_fake_EB_color_contrast
            sum_list_color_contrast_psnr[0][i] += psnr(fake_B_std, fake_B_color_contrast)
            sum_list_color_contrast_psnr[1][i] += psnr(fake_A_std, fake_A_color_contrast)
            sum_list_color_contrast_psnr[2][i] += psnr(fake_EA_std, fake_EA_color_contrast)
            sum_list_color_contrast_psnr[3][i] += psnr(fake_EA_std, fake_EB_color_contrast)

            fake_B_color_saturate = ColorFilter(factor=setting)(fake_B_std, 'saturation')
            fake_A_color_saturate = ColorFilter(factor=setting)(fake_A_std, 'saturation')
            fake_EA_color_saturate = ColorFilter(factor=setting)(fake_EA_std, 'saturation')
            fake_EB_color_saturate = ColorFilter(factor=setting)(fake_EB_std, 'saturation')

            bitacc_fake_B_color_saturate = get_acc(opt, fake_B_color_saturate, Decoder, real_wA)
            bitacc_fake_A_color_saturate = get_acc(opt, fake_A_color_saturate, Decoder, real_wB)
            bitacc_fake_EA_color_saturate = get_acc(opt, fake_EA_color_saturate, Decoder, real_wA)
            bitacc_fake_EB_color_saturate= get_acc(opt, fake_EB_color_saturate, Decoder, real_wB)

            sum_list_color_saturat[0][i] += bitacc_fake_B_color_saturate
            sum_list_color_saturat[1][i] += bitacc_fake_A_color_saturate
            sum_list_color_saturat[2][i] += bitacc_fake_EA_color_saturate
            sum_list_color_saturat[3][i] += bitacc_fake_EB_color_saturate
            sum_list_color_saturat_psnr[0][i] += psnr(fake_B_std, fake_B_color_saturate)
            sum_list_color_saturat_psnr[1][i] += psnr(fake_A_std, fake_A_color_saturate)
            sum_list_color_saturat_psnr[2][i] += psnr(fake_EA_std, fake_EA_color_saturate)
            sum_list_color_saturat_psnr[3][i] += psnr(fake_EA_std, fake_EB_color_saturate)

            vutils.save_image(fake_B_color_bright, os.path.join(opt.results_dir, opt.name, "robust/images/brightCA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_A_color_bright, os.path.join(opt.results_dir, opt.name, "robust/images/brightCB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EA_color_bright, os.path.join(opt.results_dir, opt.name, "robust/images/brightEA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EB_color_bright, os.path.join(opt.results_dir, opt.name, "robust/images/brightEB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_B_color_contrast, os.path.join(opt.results_dir, opt.name, "robust/images/contrastCA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_A_color_contrast, os.path.join(opt.results_dir, opt.name, "robust/images/contrastCB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EA_color_contrast, os.path.join(opt.results_dir, opt.name, "robust/images/contrastEA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EB_color_contrast, os.path.join(opt.results_dir, opt.name, "robust/images/contrastEB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_B_color_saturate, os.path.join(opt.results_dir, opt.name, "robust/images/saturationCA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_A_color_saturate, os.path.join(opt.results_dir, opt.name, "robust/images/saturationCB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EA_color_saturate, os.path.join(opt.results_dir, opt.name, "robust/images/saturationEA_{}.png".format(setting)),
                              normalize=False,
                              padding=0)
            vutils.save_image(fake_EB_color_saturate, os.path.join(opt.results_dir, opt.name, "robust/images/saturationEB_{}.png".format(setting)),
                              normalize=False,
                              padding=0)

    # pure
    log_robust_jpeg.write(
        str(sum_bitacc_fake_EA / opt.num_test) + ' ' + str(sum_bitacc_fake_EB / opt.num_test) + ' ' + str(
            sum_bitacc_fake_B / opt.num_test) + ' ' + str(
            sum_bitacc_fake_A / opt.num_test) + '\n')
    log_robust_noise.write(
        str(sum_bitacc_fake_EA / opt.num_test) + ' ' + str(sum_bitacc_fake_EB / opt.num_test) + ' ' + str(
            sum_bitacc_fake_B / opt.num_test) + ' ' + str(
            sum_bitacc_fake_A / opt.num_test) + '\n')
    log_robust_blur.write(
        str(sum_bitacc_fake_EA / opt.num_test) + ' ' + str(sum_bitacc_fake_EB / opt.num_test) + ' ' + str(
            sum_bitacc_fake_B / opt.num_test) + ' ' + str(
            sum_bitacc_fake_A / opt.num_test) + '\n')
    log_robust_color_bright.write(
        str(sum_bitacc_fake_EA / opt.num_test) + ' ' + str(sum_bitacc_fake_EB / opt.num_test) + ' ' + str(
            sum_bitacc_fake_B / opt.num_test) + ' ' + str(
            sum_bitacc_fake_A / opt.num_test) + '\n')
    log_robust_color_contrast.write(
        str(sum_bitacc_fake_EA / opt.num_test) + ' ' + str(sum_bitacc_fake_EB / opt.num_test) + ' ' + str(
            sum_bitacc_fake_B / opt.num_test) + ' ' + str(
            sum_bitacc_fake_A / opt.num_test) + '\n')
    log_robust_color_saturat.write(
        str(sum_bitacc_fake_EA / opt.num_test) + ' ' + str(sum_bitacc_fake_EB / opt.num_test) + ' ' + str(
            sum_bitacc_fake_B / opt.num_test) + ' ' + str(
            sum_bitacc_fake_A / opt.num_test) + '\n')

    for i in range(np.shape(sum_list_jpeg)[1]):
        log_robust_jpeg.write(str(sum_list_jpeg[2][i]/opt.num_test) + ' ' + str(sum_list_jpeg[3][i]/opt.num_test)
                              + ' ' + str(sum_list_jpeg[0][i]/opt.num_test) + ' ' + str(sum_list_jpeg[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_noise)[1]):
        log_robust_noise.write(str(sum_list_noise[2][i]/opt.num_test) + ' ' + str(sum_list_noise[3][i]/opt.num_test)
                              + ' ' + str(sum_list_noise[0][i]/opt.num_test) + ' ' + str(sum_list_noise[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_blur)[1]):
        log_robust_blur.write(str(sum_list_blur[2][i]/opt.num_test) + ' ' + str(sum_list_blur[3][i]/opt.num_test)
                              + ' ' + str(sum_list_blur[0][i]/opt.num_test) + ' ' + str(sum_list_blur[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_color_bright)[1]):
        log_robust_color_bright.write(str(sum_list_color_bright[2][i]/opt.num_test) + ' ' + str(sum_list_color_bright[3][i]/opt.num_test)
                              + ' ' + str(sum_list_color_bright[0][i]/opt.num_test) + ' ' + str(sum_list_color_bright[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_color_contrast)[1]):
        log_robust_color_contrast.write(str(sum_list_color_contrast[2][i]/opt.num_test) + ' ' + str(sum_list_color_contrast[3][i]/opt.num_test)
                              + ' ' + str(sum_list_color_contrast[0][i]/opt.num_test) + ' ' + str(sum_list_color_contrast[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_color_saturat)[1]):
        log_robust_color_saturat.write(str(sum_list_color_saturat[2][i]/opt.num_test) + ' ' + str(sum_list_color_saturat[3][i]/opt.num_test)
                              + ' ' + str(sum_list_color_saturat[0][i]/opt.num_test) + ' ' + str(sum_list_color_saturat[1][i]/opt.num_test) + '\n')
    # psnr
    for i in range(np.shape(sum_list_jpeg_psnr)[1]):
        log_psnr_jpeg.write(str(sum_list_jpeg_psnr[2][i]/opt.num_test) + ' ' + str(sum_list_jpeg_psnr[3][i]/opt.num_test)
                              + ' ' + str(sum_list_jpeg_psnr[0][i]/opt.num_test) + ' ' + str(sum_list_jpeg_psnr[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_noise_psnr)[1]):
        log_psnr_noise.write(str(sum_list_noise_psnr[2][i]/opt.num_test) + ' ' + str(sum_list_noise_psnr[3][i]/opt.num_test)
                              + ' ' + str(sum_list_noise_psnr[0][i]/opt.num_test) + ' ' + str(sum_list_noise_psnr[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_blur_psnr)[1]):
        log_psnr_blur.write(str(sum_list_blur_psnr[2][i]/opt.num_test) + ' ' + str(sum_list_blur_psnr[3][i]/opt.num_test)
                              + ' ' + str(sum_list_blur_psnr[0][i]/opt.num_test) + ' ' + str(sum_list_blur_psnr[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_color_bright_psnr)[1]):
        log_psnr_color_bright.write(str(sum_list_color_bright_psnr[2][i]/opt.num_test) + ' ' + str(sum_list_color_bright_psnr[3][i]/opt.num_test)
                              + ' ' + str(sum_list_color_bright_psnr[0][i]/opt.num_test) + ' ' + str(sum_list_color_bright_psnr[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_color_contrast_psnr)[1]):
        log_psnr_color_contrast.write(str(sum_list_color_contrast_psnr[2][i]/opt.num_test) + ' ' + str(sum_list_color_contrast_psnr[3][i]/opt.num_test)
                              + ' ' + str(sum_list_color_contrast_psnr[0][i]/opt.num_test) + ' ' + str(sum_list_color_contrast_psnr[1][i]/opt.num_test) + '\n')
    for i in range(np.shape(sum_list_color_saturat_psnr)[1]):
        log_psnr_color_saturat.write(str(sum_list_color_saturat_psnr[2][i]/opt.num_test) + ' ' + str(sum_list_color_saturat_psnr[3][i]/opt.num_test)
                              + ' ' + str(sum_list_color_saturat_psnr[0][i]/opt.num_test) + ' ' + str(sum_list_color_saturat_psnr[1][i]/opt.num_test) + '\n')

    log_robust_jpeg.close()
    log_robust_noise.close()
    log_robust_blur.close()
    log_robust_color_bright.close()
    log_robust_color_contrast.close()
    log_robust_color_saturat.close()
    log_psnr_jpeg.close()
    log_psnr_noise.close()
    log_psnr_color_bright.close()
    log_psnr_color_contrast.close()
    log_psnr_color_saturat.close()
    log_psnr_blur.close()
    webpage.save()  # save the HTML
