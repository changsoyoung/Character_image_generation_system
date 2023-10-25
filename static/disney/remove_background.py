# util.py에서 정의한 함수가 작동이 안돼서 일단 다 import 해 봄..
import sys
import os

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
from tqdm import tqdm

import lpips
from model import Generator

from util import *


# 누끼 import 추가
from rembg import remove
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#import torchvision.transforms.functional as TF

# transform도 안써도됨
transform = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

# 이미지 크롭
# 이미지 경로 -> 디즈니화된 이미지 result로 저장된거 가져옴
# img 위 코드에 이미 다른걸로 정의한 것 같아서 이름 f_img로 설정
f_img = Image.open('result.png').convert('RGB')
#f_img  = Image.open('C:\SOAT\result.png').convert('RGB')

x = 215
y = 86
w = 555
h = 556

cropped_image = f_img.crop((x, y, x+w, y+h))
cropped_image.save('cropped_image.png', 'png')


# 누끼
output = remove(cropped_image) # remove background
# output.save('/content/drive/MyDrive/img/1.png', 'png') # save image
output.save('output.png', 'png')


# back_ground image 따로 만들어서 파일에 저장해놓음
# foreground_img는 cropped되어 저장된 사진
#background_img = Image.open('C:\SOAT\background_img.png')

background_img = Image.open('background_img.png')
foreground_img = output

foreground_img = foreground_img.resize(background_img.size)

combined = Image.alpha_composite(background_img.convert('RGBA'), foreground_img.convert('RGBA'))

# 이미지 display 써서 해보려했지만 실패... 걍 pil로 함
#combined_rgb = combined.convert("RGB")
#tensor_combined = transform(combined_rgb)
#pil_image = TF.to_pil_image(tensor_combined)

#cropped_image2 = pil_image.crop((x, y, x+w, y+h))

#cropped_tensor = TF.to_tensor(cropped_image2)

#display_final_image(cropped_tensor)

#display안 쓰고 할 거면 이것만 하면 되긴 됨
# combined.save('result3.png', 'png')
combined.save('fff_result.png', 'png')

#combined_rgb = combined.convert("RGB")


#tensor_combined = transform(combined_rgb)

# 텐서를 pil 이미지로 변환
#pil_image = transforms.ToPILImage()(tensor_combined)
#pil_image_rgb = pil_image.convert("RGB")
#pil_image_rgb.save('result2.png', 'png')


#plt.imshow(combined)
#plt.show()

# 흰 배경에 누끼 딴 이미지 f_result.png로 저장
#display_final_image(tensor_combined)
#combined.save('/content/drive/MyDrive/img/3.png', 'png')