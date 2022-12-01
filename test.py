"""
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name winter2summer_cyclegan --model cyclegan_watermark

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import gen_watermark_bits, get_acc
from torchvision.utils import save_image
from util.visualizer import Visualizer
from models.networks import StegaStampDecoder


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.display_port = 8097
    opt.display_ncols = 4
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

    Decoder = StegaStampDecoder(opt.crop_size, opt.input_nc, opt.watermark_size)
    Decoder = Decoder.cuda()
    Decoder.load_state_dict(torch.load(opt.decoder_path, map_location=model.device))
    Decoder.eval()

    acc_gan_A, acc_gan_B, acc_idt_A, acc_idt_B, acc_rec_A, acc_rec_B = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    epoch_iter = 0
    for i, data in enumerate(dataset):
        epoch_iter += opt.batch_size
        counter = i
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        if opt.model == 'cyclegan_wb':
            real_A = visuals['real_A'].to(model.device)
            real_B = visuals['real_B'].to(model.device)
            fake_A = visuals['fake_A'].to(model.device)
            fake_B = visuals['fake_B'].to(model.device)
            real_wA = model.real_wA.to(model.device)
            real_wB = model.real_wB.to(model.device)

            trainAB_path = os.path.join(opt.test_outputs, 'trainAB')
            trainBA_path = os.path.join(opt.test_outputs, 'trainBA')
            testAB_path = os.path.join(opt.test_outputs, 'testAB')
            testBA_path = os.path.join(opt.test_outputs, 'testBA')
            if not os.path.exists(trainAB_path):
                os.makedirs(trainAB_path)
            if not os.path.exists(trainBA_path):
                os.makedirs(trainBA_path)
            if not os.path.exists(testAB_path):
                os.makedirs(testAB_path)
            if not os.path.exists(testBA_path):
                os.makedirs(testBA_path)
            if i < 4000:
                direction = 'AtoB'
                visualizer.save_paired_img(trainAB_path, real_A, fake_B, i, direction)
                direction = 'BtoA'
                visualizer.save_paired_img(trainBA_path, real_B, fake_A, i, direction)
            else:
                direction = 'AtoB'
                visualizer.save_paired_img(testAB_path, real_A, fake_B, i, direction)
                direction = 'BtoA'
                visualizer.save_paired_img(testBA_path, real_B, fake_A, i, direction)

            # bitacc_gan_A, bitacc_gan_B, bitacc_idt_A, bitacc_idt_B, bitacc_rec_A, bitacc_rec_B = visualizer.print_current_bitacc(
            #     model, 0, epoch_iter, opt.batch_size, opt.phase)
            bitacc_gan_A = get_acc(opt, fake_B, Decoder, real_wA)
            bitacc_gan_B = get_acc(opt, fake_A, Decoder, real_wB)


            acc_gan_A += bitacc_gan_A
            acc_gan_B += bitacc_gan_B

        if opt.model == 'pix2pix':
            real_A = visuals['real_A'].to(model.device)
            fake_B = visuals['fake_B'].to(model.device)
            this_batch_size = min(opt.batch_size, real_A.size(0))
            real_wA, real_wB = gen_watermark_bits(opt, this_batch_size, model.device)

            testABpp_path = os.path.join(opt.test_outputs, 'testABpp')
            testBApp_path = os.path.join(opt.test_outputs, 'testBApp')

            if not os.path.exists(testABpp_path):
                os.makedirs(testABpp_path)
            if not os.path.exists(testBApp_path):
                os.makedirs(testBApp_path)
            if opt.test_direction == 'AtoB':
                visualizer.save_paired_img(testABpp_path, real_A, fake_B, i, opt.test_direction)
                bitacc_gan_A = get_acc(opt, fake_B, Decoder, real_wA)
            else:
                visualizer.save_paired_img(testBApp_path, real_A, fake_B, i, opt.test_direction)
                bitacc_gan_B = get_acc(opt, fake_B, Decoder, real_wB)

    avg_gan_A = acc_gan_A / counter
    avg_gan_B = acc_gan_B / counter

    message = '(Test epoch: %d, iters: %d, bitacc_gan_A: %.3f, bitacc_gan_B: %.3f) ' % (
        0, epoch_iter, avg_gan_A, avg_gan_B)

    print(message)  # print the message
    with open(os.path.join(opt.test_outputs, opt.phase + '_bitacc_log.txt'),
              "a") as log_file:
        log_file.write('%s\n' % message)  # save the message

    webpage.save()  # save the HTML
