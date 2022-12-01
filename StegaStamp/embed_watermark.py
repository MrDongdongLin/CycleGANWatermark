import argparse
import os
import glob
import PIL

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR

parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument(
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder.",
    default="D:\\Deepfake\\Methods\\CycleGAN_Watermark\\StegaStamp\\encoder\\stegastamp_100_encoder.pth"
)
parser.add_argument(
    "--data_dir", type=str, help="Directory with images.",
    default="D:\\Deepfake\\Datasets\\AgingFace\\train\\B"
)
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to.",
    default="D:\\Deepfake\\Datasets\\AgingFace\\train\\Bw100"
)
parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images.",
    default=128
)
parser.add_argument(
    "--identical_watermarks", type=bool, default=True, help="If this option is provided use identical watermarks. Otherwise sample arbitrary watermarks."
)
parser.add_argument(
    "--check", type=bool, default=True, help="Validate watermark detection accuracy."
)
parser.add_argument(
    "--decoder_path", type=str, help="Provide trained StegaStamp decoder to verify watermark detection accuracy.",
    default="D:\\Deepfake\\Methods\\CycleGAN_Watermark\\StegaStamp\\decoder\\stegastamp_100_decoder.pth"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed to sample watermarks.")
parser.add_argument("--nrow", type=int, default=3)
parser.add_argument("--watermark_size", type=int, default=100)
# parser.add_argument("--cuda", type=int, default=0)


args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


def to_rgb(image):
    rgb_image = PIL.Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def generate_random_watermarks(watermark_size, batch_size=4):
    z = torch.zeros((batch_size, watermark_size), dtype=torch.float).random_(0, 2)
    return z


uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

# if int(args.cuda) == -1:
#     device = torch.device("cpu")
# else:
#     device = torch.device("cuda:0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('evaluating on', device)

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if image.mode != "RGB":
            image = to_rgb(image)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

def load_data():
    global dataset, dataloader

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

def load_models():
    global HideNet, RevealNet
    global watermark_SIZE
    
    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path, map_location=device)
    watermark_SIZE = args.watermark_size  # state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        watermark_SIZE,
        # WATERMARK_UPSIZE=50,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, watermark_SIZE
    )

    # kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path, map_location=device))
    HideNet.load_state_dict(torch.load(args.encoder_path, map_location=device))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)


def embed_watermarks():
    all_watermarked_images = []
    all_watermarks = []

    print("watermarking the images...")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # generate identical watermarks
    watermarks = generate_random_watermarks(watermark_SIZE, 1)
    watermarks = watermarks.view(1, watermark_SIZE).expand(BATCH_SIZE, watermark_SIZE)
    watermarks = watermarks.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    bitwise_accuracy = 0
    psnr_all = 0.0
    ssim_all = 0.0
    psnr = PSNR().to(device)
    ssim = SSIM().to(device)

    for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):

        # generate arbitrary watermarks
        if not args.identical_watermarks:
            watermarks = generate_random_watermarks(watermark_SIZE, BATCH_SIZE)
            watermarks = watermarks.view(BATCH_SIZE, watermark_SIZE)
            watermarks = watermarks.to(device)

        images = images.to(device)

        watermarked_images = HideNet(watermarks[: images.size(0)], images)
        all_watermarked_images.append(watermarked_images.detach().cpu())
        all_watermarks.append(watermarks[: images.size(0)].detach().cpu())
        # print("hello", watermarks[0], watermarks[1])

        if args.check:
            detected_watermarks = RevealNet(watermarked_images)
            detected_watermarks = (detected_watermarks > 0).long()
            bitwise_accuracy += (detected_watermarks[: images.size(0)].detach() == watermarks[: images.size(0)]).float().mean(dim=1).sum().item()

            psnr_bs = psnr(images, watermarked_images)
            ssim_bs = ssim(images, watermarked_images)
            # ssim_bs = ssim(images, watermarked_images)
            # ssim_bs = ssim(images, watermarked_images, data_range=1, size_average=True, )
            # print("Batch Avg PSNR: {:.5f} - Batch Avg SSIM: {:.5f}".format(psnr_bs, ssim_bs))
            psnr_all += psnr_bs
            ssim_all += ssim_bs

            if i == 2:
                save_image(images[:args.nrow*args.nrow], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=args.nrow)
                save_image(watermarked_images[:args.nrow*args.nrow], os.path.join(args.output_dir, "test_samples_watermarked.png"),
                           nrow=args.nrow)
                save_image(torch.abs(images - watermarked_images)[:args.nrow*args.nrow],
                           os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=args.nrow)


    dirname = args.output_dir
    if not os.path.exists(os.path.join(dirname, "watermarked_images")):
        os.makedirs(os.path.join(dirname, "watermarked_images"))

    all_watermarked_images = torch.cat(all_watermarked_images, dim=0).cpu()
    all_watermarks = torch.cat(all_watermarks, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "embedded_watermarks.txt"), "w")
    for idx in range(len(all_watermarked_images)):
        image = all_watermarked_images[idx]
        watermark = all_watermarks[idx]
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        save_image(image, os.path.join(args.output_dir, "watermarked_images", f"{filename}"), padding=0)
        watermark_str = "".join(map(str, watermark.cpu().long().numpy().tolist()))
        f.write(f"{filename} {watermark_str}\n")
    f.close()

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_watermarks)
        print(f"Bitwise accuracy on watermarked images: {bitwise_accuracy}")
        print("Avg PSNR: {:.5f} - Avg SSIM: {:.5f}".format(psnr_all/len(dataloader), ssim_all/len(dataloader)))

        # save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        # save_image(watermarked_images[:49], os.path.join(args.output_dir, "test_samples_watermarked.png"), nrow=7)
        # save_image(torch.abs(images - watermarked_images)[:49], os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)


def main():

    load_data()
    load_models()

    embed_watermarks()


if __name__ == "__main__":
    main()
