import numpy as np
import os
import sys
import ntpath
import time
import torch
import torchvision.utils as vutils
from . import util, html
from subprocess import Popen, PIPE
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.fid import FrechetInceptionDistance as FID


try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, use_wandb=False):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
        if use_wandb:
            ims_dict[label] = wandb.Image(im)
    webpage.add_images(ims, txts, links, width=width)
    if use_wandb:
        wandb.log(ims_dict)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0
        self.ncols = opt.display_ncols
        
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_wandb:
            self.wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        # self.log_bitacc_name = os.path.join(opt.checkpoints_dir, opt.name, 'bitacc_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    if self.opt.last_activation == 'tanh':
                        image_numpy = util.tensor2im(image)
                    else:
                        image_numpy = util.tensor2im_unorm(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        if self.opt.last_activation == 'tanh':
                            image_numpy = util.tensor2im(image)
                        else:
                            image_numpy = util.tensor2im_unorm(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0,'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                if self.opt.last_activation == 'tanh':
                    image_numpy = util.tensor2im(image)
                else:
                    image_numpy = util.tensor2im_unorm(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})


        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                if self.opt.last_activation == 'tanh':
                    image_numpy = util.tensor2im(image)
                else:
                    image_numpy = util.tensor2im_unorm(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    if self.opt.last_activation == 'tanh':
                        image_numpy = util.tensor2im(image)
                    else:
                        image_numpy = util.tensor2im_unorm(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        if self.use_wandb:
            self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_bitacc(self, model, epoch, iters, this_batchsize, phase):
        # plot the bitwise accuracy
        real_wA = model.real_wA
        real_wB = model.real_wB
        results = model.get_current_visuals()
        ganw_A = model.DEC(results['fake_B'])
        ganw_B = model.DEC(results['fake_A'])
        idtw_A = model.DEC(results['idt_A'])
        idtw_B = model.DEC(results['idt_B'])
        recw_A = model.DEC(results['rec_B'])
        recw_B = model.DEC(results['rec_A'])
        wm_gan_A = (ganw_A > 0).long()
        wm_gan_B = (ganw_B > 0).long()
        wm_idt_A = (idtw_A > 0).long()
        wm_idt_B = (idtw_B > 0).long()
        wm_rec_A = (recw_A > 0).long()
        wm_rec_B = (recw_B > 0).long()
        bitacc_gan_A = (wm_gan_A == real_wA).float().mean(dim=1).sum().item() / this_batchsize
        bitacc_gan_B = (wm_gan_B == real_wB).float().mean(dim=1).sum().item() / this_batchsize
        bitacc_idt_A = (wm_idt_A == real_wA).float().mean(dim=1).sum().item() / this_batchsize
        bitacc_idt_B = (wm_idt_B == real_wB).float().mean(dim=1).sum().item() / this_batchsize
        bitacc_rec_A = (wm_rec_A == real_wA).float().mean(dim=1).sum().item() / this_batchsize
        bitacc_rec_B = (wm_rec_B == real_wB).float().mean(dim=1).sum().item() / this_batchsize

        if phase == 'train':
            message = '(epoch: %d, iters: %d, bitacc_gan_A: %.3f, bitacc_gan_B: %.3f, bitacc_idt_A: %.3f, bitacc_idt_B: %.3f, bitacc_rec_A: %.3f, bitacc_rec_B: %.3f) ' % (
            epoch, iters, bitacc_gan_A, bitacc_gan_B, bitacc_idt_A, bitacc_idt_B, bitacc_rec_A, bitacc_rec_B)

            print(message)  # print the message
            with open(os.path.join(self.opt.checkpoints_dir, self.opt.name, phase + '_bitacc_log.txt'), "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
        else:
            return bitacc_gan_A, bitacc_gan_B, bitacc_idt_A, bitacc_idt_B, bitacc_rec_A, bitacc_rec_B

    def print_current_vis(self, model, epoch, iters, phase):
        # plot the bitwise accuracy
        real_WA = model.real_WA
        real_WB = model.real_WB
        results = model.get_current_visuals()
        ganw_A = model.RNet(results['fake_B'])
        ganw_B = model.RNet(results['fake_A'])

        psnr = PSNR()
        ssim = SSIM()
        fid = FID(feature=64)
        psnr_ganA = psnr(real_WA, ganw_A)
        psnr_ganB = psnr(real_WB, ganw_B)

        ssim_ganA = ssim(real_WA, ganw_A)
        ssim_ganB = ssim(real_WB, ganw_B)
        fid.update(real_WA, real=True)
        fid.update(ganw_A, real=False)
        fid_ganA = fid.compute().numpy()
        fid.update(real_WB, real=True)
        fid.update(ganw_B, real=False)
        fid_ganB = fid.compute().numpy()

        if phase == 'train':
            message = '(epoch: %d, iters: %d, psnr_A: %.3f, psnr_B: %.3f, ssim_A: %.3f, ssim_B: %.3f, nc_A: %.3f, nc_B: %.3f) ' % (
            epoch, iters, psnr_ganA, psnr_ganB, ssim_ganA, ssim_ganB, fid_ganA, fid_ganB)

            print(message)  # print the message
            with open(os.path.join(self.opt.checkpoints_dir, self.opt.name, phase + '_visual_log.txt'), "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
        else:
            return psnr_ganA, psnr_ganB, ssim_ganA, ssim_ganB, fid_ganA, fid_ganB

    def save_paired_img(self, saved_path, cover_img_A, container_img, i, direction):
        for j, img in enumerate(zip(cover_img_A, container_img)):
            result_img = torch.cat([img[0].unsqueeze(0), img[1].unsqueeze(0)], 3)
            # print(result_img.size())
            if direction == 'AtoB':
                resultImgName = '%s/AB_batch%04d_id%04d.png' % (saved_path, i, j)
            else:
                resultImgName = '%s/BA_batch%04d_id%04d.png' % (saved_path, i, j)
            vutils.save_image(result_img, resultImgName, normalize=False, padding=0)