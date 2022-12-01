"""
Example:
    Train a CycleGAN watermark model:
        python train.py --dataroot /pubdata/ldd/winter2summer --name winter2summer_cyclegan --model cyclegan_wb
    Train a CycleGAN surrogate model:
        python train.py --dataroot /pubdata/ldd/winter2summer --name winter2summer_cyclegan_surrogate --model cycle_gan
"""
import time
import os
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import gen_watermark_bits, get_acc
from models.networks import StegaStampDecoder

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    # for testing
    if opt.model == 'cyclegan_wb':
        opt.phase = 'test'  # chosen from test set
        val_dataset = create_dataset(opt)
        opt.phase = 'train'
    if opt.model == 'pix2pix':
        if opt.data_direction == 'AtoB':
            opt.phase = 'testAB'  # chosen from test set
            val_dataset = create_dataset(opt)
            opt.phase = 'trainAB'
        else:
            opt.phase = 'testBA'  # chosen from test set
            val_dataset = create_dataset(opt)
            opt.phase = 'trainBA'
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # print bitwise accuracy for training and testing
            if total_iters % opt.display_freq == 0:
                if opt.model == 'cyclegan_wb':
                    visualizer.print_current_bitacc(model, epoch, epoch_iter, opt.batch_size, opt.phase)
                    opt.phase = 'test'
                    acc_gan_A, acc_gan_B, acc_idt_A, acc_idt_B, acc_rec_A, acc_rec_B = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    for j, val_data in enumerate(val_dataset):
                        counter = j
                        if j > opt.num_val:  # only apply our model to opt.num_test images.
                            break
                        model.set_input(val_data)  # unpack data from data loader
                        model.test()  # run inference
                        bitacc_gan_A, bitacc_gan_B, bitacc_idt_A, bitacc_idt_B, bitacc_rec_A, bitacc_rec_B = visualizer.print_current_bitacc(
                            model, epoch, epoch_iter, opt.batch_size, opt.phase)

                        acc_gan_A += bitacc_gan_A
                        acc_gan_B += bitacc_gan_B
                        acc_idt_A += bitacc_idt_A
                        acc_idt_B += bitacc_idt_B
                        acc_rec_A += bitacc_rec_A
                        acc_rec_B += bitacc_rec_B
                    avg_gan_A = acc_gan_A / counter
                    avg_gan_B = acc_gan_B / counter
                    avg_idt_A = acc_idt_A / counter
                    avg_idt_B = acc_idt_B / counter
                    avg_rec_A = acc_rec_A / counter
                    avg_rec_B = acc_rec_B / counter

                    message = '(Test epoch: %d, iters: %d, bitacc_gan_A: %.3f, bitacc_gan_B: %.3f, bitacc_idt_A: %.3f, bitacc_idt_B: %.3f, bitacc_rec_A: %.3f, bitacc_rec_B: %.3f) ' % (
                        epoch, epoch_iter, avg_gan_A, avg_gan_B, avg_idt_A, avg_idt_B, avg_rec_A, avg_rec_B)

                    print(message)  # print the message
                    with open(os.path.join(opt.checkpoints_dir, opt.name, opt.phase + '_bitacc_log.txt'),
                              "a") as log_file:
                        log_file.write('%s\n' % message)  # save the message
                    opt.phase = 'train'
                if opt.model == 'pix2pix':
                    # python train.py --dataroot /pubdata/ldd/landscape/surrogate --gpu_ids 0 --dataset_mode aligned --phase trainAB --direction AtoB
                    results = model.get_current_visuals()
                    real_A = results['real_A']
                    fake_B = results['fake_B']
                    real_B = results['real_B']
                    this_batch_size = min(opt.batch_size, real_A.size(0))
                    real_wA, real_wB = gen_watermark_bits(opt, this_batch_size, model.device)

                    Decoder = StegaStampDecoder(opt.crop_size, opt.input_nc, opt.watermark_size)
                    Decoder = Decoder.cuda()
                    Decoder.load_state_dict(torch.load(opt.decoder_path, map_location=model.device))
                    Decoder.eval()

                    bitacc_gan_wA = get_acc(opt, fake_B, Decoder, real_wA)
                    bitacc_gan_wB = get_acc(opt, fake_B, Decoder, real_wB)

                    message = '(Validate epoch: %d, iters: %d, bitacc_gan_wA: %.3f, bitacc_gan_wB: %.3f) ' % (
                        epoch, epoch_iter, bitacc_gan_wA, bitacc_gan_wB)
                    print(message)
                    with open(os.path.join(opt.checkpoints_dir, opt.name, opt.phase + '_bitacc_log.txt'),
                              "a") as log_file:
                        log_file.write('%s\n' % message)  # save the message

                    if opt.data_direction == 'AtoB':
                        opt.phase = 'testAB'
                    else:
                        opt.phase = 'testBA'
                    acc_gan_wA, acc_gan_wB = 0.0, 0.0
                    for j, val_data in enumerate(val_dataset):
                        counter = j
                        if j > opt.num_val:  # only apply our model to opt.num_test images.
                            break
                        model.set_input(val_data)  # unpack data from data loader
                        model.test()  # run inference

                        results = model.get_current_visuals()
                        real_A = results['real_A']
                        fake_B = results['fake_B']
                        real_B = results['real_B']
                        this_batch_size = min(opt.batch_size, real_A.size(0))

                        bitacc_gan_wA = get_acc(opt, fake_B, Decoder, real_wA)
                        bitacc_gan_wB = get_acc(opt, fake_B, Decoder, real_wB)

                        acc_gan_wA += bitacc_gan_wA
                        acc_gan_wB += bitacc_gan_wB

                    avg_gan_wA = acc_gan_wA / counter
                    avg_gan_wB = acc_gan_wB / counter

                    message = '(Test epoch: %d, iters: %d, bitacc_gan_wA: %.3f, bitacc_gan_wB: %.3f) ' % (
                        epoch, epoch_iter, avg_gan_wA, avg_gan_wB)
                    print(message)
                    with open(os.path.join(opt.checkpoints_dir, opt.name, opt.phase + '_bitacc_log.txt'),
                              "a") as log_file:
                        log_file.write('%s\n' % message)  # save the message

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
