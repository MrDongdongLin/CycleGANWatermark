## Train StegaStamp

Load images from CelebA and train the StegaStamp. Trained models can be found in `checkpoints`.

```shell
cd StegaStamp
python train.py \
--data_dir /media/hdddati1/donny/Datasets/CelebA/ \
--output_dir /media/ssddati1/donny/Methods/CycleGANWatermark/StegaStamp \
--fingerprint_length 200 \
--image_resolution 256 \
--num_epochs 50 \
--batch_size 64
```

## Embed watermark

```shell
python embed_watermark.py \
--encoder_path /media/ssddati1/donny/Methods/CycleGAN_Watermark/encoder/stegastamp_100_encoder.pth \
--decoder_path /media/ssddati1/donny/Methods/CycleGAN_Watermark/decoder/stegastamp_100_decoder.pth \
--data_dir /media/hdddati1/donny/Datasets/FFHQ/train/A_w100_r128/watermarked_images \
--output_dir /media/hdddati1/donny/Datasets/FFHQ/train/A_ww100_r128 \
--image_resolution 128 \
--watermark_size 100 \
--batch_size 64 \
--identical_watermarks True \
--check True
```

```shell
python train.py --dataroot /pubdata/ldd/FFHQ/512/ --name man2woman_cyclegan_ganw0_idw0_cycw0_batch --model cyclegan_wb --decoder_path /home/lindd/projects/CycleGANWatermark/wmextractor/stegastamp_256_200_decoder.pth --watermark_size 200 --gpu_ids 0,1 --max_dataset_size 3000 --lambda_ganw 0 --lambda_idw 0 --lambda_cycw 0 --norm batch --batch_size 16
```

