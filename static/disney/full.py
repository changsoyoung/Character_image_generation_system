import sys
import os

import numpy as np
import argparse
import scipy.ndimage
import PIL.Image
import face_alignment
from torch.autograd.grad_mode import enable_grad

import math

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator

import matplotlib.pyplot as plt
from util import *

# 누끼 import 추가
from rembg import remove

plt.rcParams['figure.dpi'] = 150

os.chdir("./static/disney")

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def image_align(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17, :2]  # left-right
        lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
        lm_eyebrow_right = lm[22 : 27, :2]  # left-right
        lm_nose          = lm[27 : 31, :2]  # top-down
        lm_nostrils      = lm[31 : 36, :2]  # top-down
        lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
        lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        img.save(dst_file, 'PNG')


#ffhq 지혜 / 있는걸 전부 다 align
RAW_IMAGES_DIR = '../img'
ALIGNED_IMAGES_DIR = './aligned_images'

if not os.path.exists(ALIGNED_IMAGES_DIR):
    os.makedirs(ALIGNED_IMAGES_DIR)

landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

for img_name in os.listdir(RAW_IMAGES_DIR):
    raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)

    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
 
        aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, f'align-{img_name}')

        image_align(raw_img_path, aligned_face_path, face_landmarks)


#projector
def gaussian_loss(v):
    # [B, 9088]
    loss = (v-gt_mean) @ gt_cov_inv @ (v-gt_mean).transpose(1,0)
    return loss.mean()

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=3000)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--n_mean_latent', type=int, default=10000)
    #parser.add_argument('files', metavar='FILES', nargs='+')

    args = parser.parse_args(args=[])
    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


    imgfile = 'aligned_images/align-1.jpeg'
    imgs = []

    img = transform(Image.open(imgfile).convert('RGB'))
    imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    imgs.shape

    g_ema = Generator(resize, 512, 8)
    ensure_checkpoint_exists('face.pt')
    g_ema.load_state_dict(torch.load('face.pt')['g_ema'], strict=False)
    g_ema = g_ema.to(device).eval()


    with torch.no_grad():
        latent_mean = g_ema.mean_latent(50000)
        latent_in = list2style(latent_mean)

    # get gaussian stats
    if not os.path.isfile('inversion_stats.npz'):
        with torch.no_grad():
            source = list2style(g_ema.get_latent(torch.randn([10000, 512]).cuda())).cpu().numpy()
            gt_mean = source.mean(0)
            gt_cov = np.cov(source, rowvar=False)

        # We show that style space follows gaussian distribution
        # An extension from this work https://arxiv.org/abs/2009.06529
        np.savez('inversion_stats.npz', mean=gt_mean, cov=gt_cov)

    data = np.load('inversion_stats.npz')
    gt_mean = torch.tensor(data['mean']).cuda().view(1,-1).float()
    gt_cov_inv = torch.tensor(data['cov']).cuda()

    # Only take diagonals
    mask = torch.eye(*gt_cov_inv.size()).cuda()
    gt_cov_inv = torch.inverse(gt_cov_inv*mask).float()

    step = 3000
    lr = 1.0

    percept = lpips.LPIPS(net='vgg', spatial=True).to(device)
    latent_in.requires_grad = True

    optimizer = optim.Adam([latent_in], lr=lr, betas=(0.9, 0.999))

    min_latent = None
    min_loss = 100
    pbar = tqdm(range(step))
    latent_path = []

    for i in pbar:
        t = i / step
    #     lr = get_lr(t, lr)
        if i > 0 and i % 500 == 0:
            lr *= 0.2
        latent_n = latent_in

        img_gen, _ = g_ema(style2list(latent_n))

        batch, channel, height, width = img_gen.shape

        if height > 256:
            img_gen = F.interpolate(img_gen, size=(256,256), mode='area')

        p_loss = 20*percept(img_gen, imgs).mean()
        mse_loss = 1*F.mse_loss(img_gen, imgs)
        g_loss = 1e-3*gaussian_loss(latent_n)

        loss = p_loss + mse_loss + g_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())
            
        if loss.item() < min_loss:
            min_loss = loss.item()
            min_latent = latent_in.detach().clone()

        pbar.set_description(
            (
                f'loss: {loss.item():.4f}; '
                f'perceptual: {p_loss.item():.4f}; '
                f'mse: {mse_loss.item():.4f}; gaussian: {g_loss.item():.4f} lr: {lr:.4f}'
            )
        )
        
    latent_path.append(latent_in.detach().clone()) # last latent vector

    print(min_loss)

    img_gen, _ = g_ema(style2list(latent_path[-1]))
    img_gen, _ = g_ema(style2list(min_latent))


    torch.save({'latent': min_latent}, 'inversion_codes/04.pt')


#toonify
device = 'cuda' #@param ['cuda', 'cpu']

generator1 = Generator(256, 512, 8, channel_multiplier=2).eval().to(device)
generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device).eval()

mean_latent1 = load_model(generator1, 'face.pt')
mean_latent2 = load_model(generator2, 'disney.pt')

truncation = .5
face_seed =  44654 #@param {type:"number"} 
disney_seed =  686868 #@param {type:"number"} 
    
with torch.no_grad():
    torch.manual_seed(face_seed)
    source_code = torch.randn([1, 512]).to(device)
    latent1 = generator1.get_latent(source_code, truncation=truncation, mean_latent=mean_latent1)
    source_im, _ = generator1(latent1)

    torch.manual_seed(disney_seed)
    reference_code = torch.randn([1, 512]).to(device)
    latent2 = generator2.get_latent(reference_code, truncation=truncation, mean_latent=mean_latent2)
    reference_im, _ = generator2(latent2)

num_swap =  6
alpha =  0.5

early_alpha = 0

def toonify(latent1, latent2):
    with torch.no_grad():
        noise1 = [getattr(generator1.noises, f'noise_{i}') for i in range(generator1.num_layers)]
        noise2 = [getattr(generator2.noises, f'noise_{i}') for i in range(generator2.num_layers)]

        out1 = generator1.input(latent1[0])
        out2 = generator2.input(latent2[0])
        out = (1-early_alpha)*out1 + early_alpha*out2

        out1, _ = generator1.conv1(out, latent1[0], noise=noise1[0])
        out2, _ = generator2.conv1(out, latent2[0], noise=noise2[0])
        out = (1-early_alpha)*out1 + early_alpha*out2

        skip1 = generator1.to_rgb1(out, latent1[1])
        skip2 = generator2.to_rgb1(out, latent2[1])
        skip = (1-early_alpha)*skip1 + early_alpha*skip2

        i = 2
        for conv1_1, conv1_2, noise1_1, noise1_2, to_rgb1, conv2_1, conv2_2, noise2_1, noise2_2, to_rgb2 in zip(
            generator1.convs[::2], generator1.convs[1::2], noise1[1::2], noise1[2::2], generator1.to_rgbs,
            generator2.convs[::2], generator2.convs[1::2], noise2[1::2], noise2[2::2], generator2.to_rgbs
        ):


            conv_alpha = early_alpha if i < num_swap else alpha
            out1, _ = conv1_1(out, latent1[i], noise=noise1_1)
            out2, _ = conv2_1(out, latent2[i], noise=noise2_1)
            out = (1-conv_alpha)*out1 + conv_alpha*out2
            i += 1

            conv_alpha = early_alpha if i < num_swap else alpha
            out1, _ = conv1_2(out, latent1[i], noise=noise1_2)
            out2, _ = conv2_2(out, latent2[i], noise=noise2_2)
            out = (1-conv_alpha)*out1 + conv_alpha*out2
            i += 1

            conv_alpha = early_alpha if i < num_swap else alpha
            skip1 = to_rgb1(out, latent1[i], skip)
            skip2 = to_rgb2(out, latent2[i], skip)
            skip = (1-conv_alpha)*skip1 + conv_alpha*skip2

            i += 1

    image = skip.clamp(-1,1)
    
    return image

result = toonify(latent1, latent2)

latent_real = torch.load('inversion_codes/04.pt')['latent']
latent_real = style2list(latent_real)

source_im, _ = generator1(latent_real)

result2 = toonify(latent_real, latent2)

display_image(result2)

#remove_background code 첨부
f_img = Image.open('result.png').convert('RGB')

x = 215
y = 86
w = 555
h = 556

cropped_image = f_img.crop((x, y, x+w, y+h))

# 누끼
output = remove(cropped_image) # remove background

background_img = Image.open('background_img.png')
foreground_img = output

foreground_img = foreground_img.resize(background_img.size)

combined = Image.alpha_composite(background_img.convert('RGBA'), foreground_img.convert('RGBA'))

combined.save('../photo/result_disney.png', 'png')