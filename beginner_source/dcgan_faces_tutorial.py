# -*- coding: utf-8 -*-
"""
DCGAN Tutorial
==============

**저자**: `Nathan Inkawhich <https://github.com/inkawhich>`__
**번역**: `JaeJoong Lee <https://github.com/JaeLee18>`__

"""


######################################################################
# 개요
# ------------
# 
# 이 튜토리얼은 예제를 해보면서 DCGAN에 대한 소개를 제공합니다.
# 우리는 생성적 적대 신경망 (GAN)으로 실제 연예인들의 사진을 많이 이용해 훈련시켜서 새로운 연예인 사진들을 생성하게 할 것 입니다.
# 대부분의 코드들은 `pytorch/examples <https://github.com/pytorch/examples>`__, 에서 온 DCGAN 구현을 사용했습니다.
# 또한, 이 튜토리얼은 구현에 대해 자세히 설명하고, 어떻게 그리고 왜 이 신경망이 작동하는지에 대해 알려줍니다.
# GAN에 대한 사전지식은 필요하지 않지만 처음 보시는 분들은 어떻게 작동되는지에 대해 이해하려고 시간이 좀 필요할 수도 있습니다.
# 시간을 아끼기 위해서는 GPU가 한 개나 두 개 정도 필요로 합니다.
# 자 그러면, 처음부터 시작해봅시다. 
# 
# 생성적 적대 신경망
# -------------------------------
# 
# GAN이란 무엇인가?
# ~~~~~~~~~~~~~~
# 
# GAN은 딥러닝 모델이 데이터의 분포를 배우게 해서 새로운 데이터를 그 분포로부터 생성하게끔 가르치는 프레임워크입니다.
# 
# GAN은 Ian Goodfellow에 의해 2014년에 만들어졌으며 처음으로 
#`Generative Adversarial Nets <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__.
# 이라는 논문에 소개되었습니다.
# GAN은 *generator* 와 *discriminator* 라는 두개의 모델로 이루어져 있습니다.
# generator의 역할은 훈련용 이미지처럼 보이는 'fake' 이미지들을 만들어내는것입니다.
# 또한, discriminator의 역할은 이미지를 보고 과연 그게 진짜 훈련용 이미지 인지 아니면 generator에서 생성된 가짜 이미지인지 확인하는것입니다.
# 훈련 과정 중에는 generator는 지속적으로 더 좋은 fake 이미지들을 생성하여 discriminator를 속이려 하고 그러는 와중에
# discriminator는 더 철저하게 구별하여 진짜 훈련용 이미지와 fake 이미지들을 구별해냅니다.
# 이러한 경쟁은 generator가 정말 훈련용 데이터 이미지들중 하나처럼 보이는 이미지를 생성해내고 
# discriminator는 항상 50%의 신뢰도를 가지고 generator의 결과물이 fake인지 진짜인지 판별할 때 끝이 나게됩니다.
# 
# 이제 튜토리얼에서 사용될 표기법들을 정의할텐데 discriminator에 사용될것들부터 시작하겠습니다.
# :math:`x` 를 이미지를 나타내는 데이터라 하겠습니다.
# :math:`D(x)` 는 discriminator 신경망으로 :math:`x` 가  generator에서 온게 아닌 실제 훈련데이터일 확률을 반환합니다.
# 우리는 이미지를 다루기 때문에 :math:`D(x)` 의 입력값으로 C(채널)H(세로)W(가로)가 3x64x64인 이미지를 사용합니다.
# 직관적으로, :math:`D(x)` 는 :math:`x` 가 실제 훈련데이터에서 올 경우에는 높은값을 반환하고 generator에서 온 경우라면 낮은값을 반환합니다.
# :math:`D(x)` 는 전통적인 이진 분류기로 생각될 수도 있습니다.
# 
# generator의 표기법에 대해 알아보겠습니다.
# :math:`z` 는 잠재 공간 벡터로 일반적인 정규분포로부터 샘플되어집니다.
# :math:`G(z)` 는 generator 함수를 나타내며 :math:`z` 의 잠재 벡터를 데이터 공간으로 맵핑합니다.
# :math:`G` 의 목적은 (:math:`p_{data}`) 로 부터 온 훈련 데이터의 분포를 예측하여 generator가 fake 샘플을 
# 예측분포치인  (:math:`p_g`) 로 부터 생성하는것입니다.
#
# 그래서 :math:`D(G(z))` 는 generator :math:`G` 의 결과물이 진짜 이미지인지 나타내는 확률입니다.
# `Goodfellow’s paper <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__,
# 에 묘사된것 처럼 :math:`D` 와 :math:`G` 는 최소최대 게임(minimax game)을 하는데 :math:`D` 는 올바르게 진짜와 fake 
# 를 구분하는 확률 (:math:`logD(x)`) 을 극대화시키면서 :math:`G` 는 :math:`D` 가  결과값을 fake 로 예측하는 확률 (:math:`log(1-D(G(x)))`) 을 최소화 하는 게임입니다.
# 논문에서,  GAN의 손실함수는 
# .. math:: \underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim p_{z}(z)}\big[log(1-D(G(z)))\big]
# 로 정의하였습니다.

# 이론적으로, 이 최소최대 게임(minimax game)의 답은
# :math:`p_g = p_{data}` 이며 discriminator가 무작위로 입력값이 진짜인지 fake 인지 예측하는 경우 입니다.
# 그러나, GAN의 수렴 이론은 현재에도 활발히 연구되고 있으며 현실적인 신경만들은 항상 이러한 수준까지 훈련되지는 않습니다.
#
# DCGAN이란?
# ~~~~~~~~~~~~~~~~
# 
# DCGAN은 위에 설명된 GAN의 확장개념이지만 다른 점이라면 discriminator에는 convolutional 계층을 
# generator에는 convolutional-transpose 계층이 각각 사용됩니다. 
# DCGAN은 처음으로 Radford et. al. 이 작성한 `Unsupervised Representation Learning With
# Deep Convolutional Generative Adversarial
# Networks <https://arxiv.org/pdf/1511.06434.pdf>`__  이라는 논문에서 소개가 되었습니다.
# discriminator는  strided
# `convolution <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`__ 층과, 
# `batch
# norm <https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d>`__ 층, 그리고
# `LeakyReLU <https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU>`__ 를
# 활성함수로 구성되어 있습니다.
# 입력값은 3x64x64의 이미지이며 결과값은 입력값이 실제 데이터 분포에서 왔을 확률입니다.
# generator는  
# `convolutional-transpose <https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d>`__ 층,
# batch norm 층, 그리고
# `ReLU <https://pytorch.org/docs/stable/nn.html#relu>`__ 를 활성함수로 구성되어 있습니다.
# 입력값은 잠재 벡터인 :math:`z` 이며 정규 분포에서 뽑아내진 값입니다.
# 결과값은 3x64x64 의 RGB 이미지 입니다.
# strided conv-transpose 층은 잠재 벡터가 같은 모양의 이미지로 변환되는 것을 가능하게 합니다.
# 이 논문에서 저자들은 옵티마이저를 설정하는 방법, 손실함수 계산 방법, 모델 가중치 초기화 방법들을 설명해주었는데
# 이 모든 방법들은 아래에 전부 설명이 될 예정입니다.
# 

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# 재현하기 위한 random seed 설정
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 새로운 결과를 보기 원한다면 사용하세요.
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


######################################################################
# 입력값들
# ------
# 
# 실행하기 위한 입력값들을 정의해봅시다:
# 
# -  **dataroot** - 데이터셋 폴더의 루트에 대한 경로입니다. 데이터셋에 대해서는 다음 섹션에서
#    더 이야기 할 것입니다.
# -  **workers** - DataLoader로 데이터를 불러올때 사용될 worker 쓰레드의 갯수입니다.
# -  **batch_size** - 훈련때 사용될 배치 크기 입니다. DCGAN 논문에서는 128로 사용했습니다.
# -  **image_size** - 훈련때 사용될 이미지의 크기입니다. 초기값은 64x64로 설정 되어있고,
#    다른 크기를 원한다면 D와 G의 구조도 다시 설정해야합니다. 
#    자세하게 보려면 `여기 <https://github.com/pytorch/examples/issues/70>`__ 를 봐주세요.
# -  **nc** - 입력값으로 사용되는 이미지의 색상 채널 갯수입니다. 컬러 이미지들은 3입니다.
# -  **nz** - 잠재 벡터의 길이
# -  **ngf** - generator를 통해 이용되는 feature map의 깊이와 관련되어있습니다.
# -  **ndf** - discriminator를 통해 전파되는 feature map의 깊이를 설정합니다.
# -  **num_epochs** - 훈련 때 실행되는 에포크의 수입니다. 더 좋은 결과를 위해선 많은 에포크가 필요하지만
#    그만큼 더 많은 시간이 필요합니다.
# -  **lr** - 훈련 때 사용되는 러닝레이트 입니다. DCGAN에서 나와있는 대로 
#    0.0002로 설정합니다.
# -  **beta1** - beta1 는 Adam 옵티마이저에 사용됩니다. 논문에 소개된 대로
#    0.5로 설정합니다.
# -  **ngpu** - 사용가능한 GPU의 갯수입니다. 0으로 설정되면 코드는 CPU를 사용하며, 
#    0보다 큰 수일 때는 해당하는 GPU개수만큼 사용하게됩니다.
# 

# 데이터셋의 루트 경로
dataroot = "data/celeba"

# dataloader에 사용되는 worker 개수
workers = 2

# 훈련때 사용되는 배치 사이즈
batch_size = 128

# 훈련 이미지의 크기입니다. 모든 이미지들은 transformer를 사용되어 크기 변환됩니다.
image_size = 64

# 훈련때 사용될 이미지 갯수입니다. 컬러 이미지에는 3으로 사용됩니다.
nc = 3

# 잠재 벡터 z의 크기입니다. (예) generator 입력값의 크기)
nz = 100

# generator 안의 feature map 크기
ngf = 64

# discriminator 안의 feature map 크기
ndf = 64

# 훈련 에포크 갯수
num_epochs = 5

# 옵티마이저의 러닝레이트
lr = 0.0002

# Adam 옵티마이저의 Beta1 하이퍼 파라미터
beta1 = 0.5

# 사용가능한 GPU 갯수. CPU모드 로 설정하려면 0을 사용하세요.
ngpu = 1


######################################################################
# 데이터
# ----
# 
# 이번 튜토리얼에서 우리는 `Celeb-A Faces
# dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ 링크된 사이트에서 다운받을 수 있고 또는 
# `Google
# Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__. 에서도 다운 받을 수 있는
# 데이터를 이용합니다.
# 데이터셋은 *img_align_celeba.zip* 이라는 파일로 다운받아집니다. 
# 다운이 완료된후, *celeba* 라는 폴더를 만들고 zip 파일을 그 폴더에 풀어주세요.
# 그 다음에  *dataroot* 을 방금 만드신 *celeba* 로 설정 해주시면 됩니다.
# 폴더 구조는 다음과 같아야 합니다.
#
# ::
# 
#    /path/to/celeba
#        -> img_align_celeba  
#            -> 188242.jpg
#            -> 173822.jpg
#            -> 284702.jpg
#            -> 537394.jpg
#               ...
# 
# 폴더 구조를 이렇게 만드는 것은 중요한 과정이고 ImageFolder 데이터셋 class를 사용할 예정이라 
# 데이터셋의 하위폴더에 주어진대로 위치해야합니다.
# 이제 우리는 데이터셋과  dataloder 를 설정하고
# 실행할 하드웨어 장치 설정과 훈련때 사용되는 데이터들을 시각화 할 수 있습니다.
# 

# ImageFolder를 우리가 설정해놓은 대로 사용할 수 있습니다.
# 데이터셋 만들기
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# dataloader 만들기
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 어떤 장치로 실행할지 설정합니다.
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 몇몇개의 훈련 이미지를 시각화합니다.
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))



######################################################################
# 구현
# --------------
# 
# 입력값 파라미터들과 데이터셋으로 이제 우리는 코드 구현을 할 수 있습니다.
# 우선 우리는 가중치 초기화 전력부터 시작으로 그다음에 generator, discriminator, 손실함수
# 그리고 훈련 과정에 대해 자세히 알아보겠습니다.
# 
# 가중치 초기화
# ~~~~~~~~~~~~~~~~~~~~~
# 
# DCGAN 논문에서 저자들은 구체적으로 가중치는 평균 0과 표준편차 0.02 를 갖는 정규분포를 이용해서 모든 모델들이
# 초기화되어야 한다고 밝히고 있습니다.
# ``weights_init`` 함수는 초기화된 모델을 입력값으로 받고 모든 convolutional, convolutional-transpose
# 그리고 batch normalization 층들을 기준에 맞도록 초기화 합니다.
# 이 함수는 초기화 이후에 즉시 모델들에게 적용됩니다.
# 

#  netG와 netD를 호출한뒤에 사용자 정의된 가중치 초기화를 실시합니다.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


######################################################################
# Generator
# ~~~~~~~~~
# 
# generator인 :math:`G` 는 잠재 벡터 공간(:math:`z`)을  데이터 공간에 대응시키도록 고안되었습니다.
# 우리의 데이터가 이미지이므로 :math:`z` 를 데이터 공간으로 변환한다는 뜻은 사실상 훈련 이미지와 같은 크기(예) 3x64x64) 의
# 이미지를 만든다는것과 같습니다. 
# 구현에서는  2차원의 strided convolutional transpose 층과 2차원 batch norm 층, 그리고 relu 활성화 함수의 조합으로 
# 결과값을 도출해낼수 있습니다.
# generator의 결과값은 tanh 함수로 들어가서 :math:`[-1,1]` 의 데이터 범위로 반환됩니다.
# conv-transpose 층 다음에 batch norm 함수들이 존재한다는것을 알아둘 필요가 있습니다. 
# 왜냐하면 이 배치에 대한 구성이 DCGAN 논문의 핵심적인 기여이기 때문입니다.
# 이러한 층들은 훈련도중 변화도의 흐름에 대해 도움을 줍니다.
# DCGAN 논문에서 generator 의 이미지는 아래와 같이 보입니다.
#
# .. figure:: /_static/img/dcgan_generator.png
#    :alt: dcgan_generator
#
# 입력 섹션에서 우리가 설정한 입력값에 대한 설정들(*nz*, *ngf*, and *nc*) 이 어떻게 코드 안에 있는 generator 구조에 영향을 미치는지 알아둬야합니다.
# *nz* 는 z 입력 벡터의 길이고,  *ngf* 는 generator를 통해 전파되는 feature maps 의 크기와 연관이 있습니다.
# 또한, *nc* 는 결과값 이미지에 대한 채널의 갯수입니다 (RGB이미지는 3으로 설정해야 합니다.)
# 아래는 generator 코드입니다.
# 

# Generator 코드

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력값은 Z이며 convolution으로 들어갑니다.
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 상태 크기. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 상태 크기. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 상태 크기. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 상태 크기. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 상태 크기. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


######################################################################
# 이제 우리는 generator 를 초기화하고  ``weights_init`` 
# 함수를 적용 할 수 있습니다.
# 출력되는 모델을 확인해서 어떻게 generator 객체가
# 구성되어있는지 확인해보세요.
# 

# generator 만들기.
netG = Generator(ngpu).to(device)

# 멀티 gpu를 설정 할 수 있습니다. 
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# weights_init 함수를 적용해서 모든 가중치를 무작위로
# 평균이 0이고 표준편차가 0.2인 것으로 초기화 합니다.
netG.apply(weights_init)

# 모델을 출력해 봅니다.
print(netG)


######################################################################
# Discriminator
# ~~~~~~~~~~~~~
# 
# 앞서 말한대로 discriminator :math:`D` 는 이진 분류 신경망에 해당하며
# 이미지를 입력값으로 입력값으로 사용된 이미지가 가짜 이미지에 대해서 어떤 확률로 진짜 이미지인지 
# 결과값을 출력해줍니다.
# :math:`D` 는 3x64x64 크기의 이미지를 입력값으로 받으며 Conv2d, 
# BatchNorm2d, LeakyReLU 층의 조합으로 처리됩니다. 결과값은은 Sigmoid 활성화 함수를 이용해서
# 마지막으로 확률이 나오게 됩니다. 이 신경망 구조는 문제를 해결하는데 필요에 따라 더 많은 층을 추가하여
# 확장 될수도 있지만, strided convolution, BatchNorm, 그리고  LeakyReLUs층이 주로 
# 사용되어야 합니다.
# DCGAN 논문은 다운 샘플을 하기위해 pooling을 사용하는것 보다 strided convolution을
# 사용하는것이 더 좋다고 언급하고 있는데 그 이유로는 신경망이 신경망 자체의 pooling
# 함수를 배우도록 하기 떄문이라고 하고 있습니다.
# 또한 batch norm 과 leaky relu 함수들은 :math:`G` 와 :math:`D` 의
# 훈련 과정에서 gradient 흐름이 잘 가도록 중요한 역할을 합니다.
# 

#########################################################################
# Discriminator 코드

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력값은 (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


######################################################################
# 이제 generator와 같이 discriminator 또한 생성 할 수 있습니다.
# ``weights_init`` 함수를 적용하고 모델의 구조를 출력해봅시다.
# 

# Discriminator 를 생성합니다.
netD = Discriminator(ngpu).to(device)

# 멀티 gpu 를 설정할 수 있습니다.
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# weights_init 함수를 적용해서 모든 가중치를 무작위로
# 평균이 0이고 표준편차가 0.2인 것으로 초기화 합니다.
netD.apply(weights_init)

# 모델을 출력합니다.
print(netD)


######################################################################
# 손실 함수와 옵티마이저
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# :math:`D` 와  :math:`G` 을 구현해놨기 때문에 어떻게 손실 함수와 
# 옵티마이저를 통해 훈련될수 있을지 정할 수 있습니다.
# 우리는 Binary Cross Entropy 손실함수 
# (`BCELoss <https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss>`__) 
# 를 사용할 예정이며 PyTorch 에는 
# 
# .. math:: \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
# 
# 로 정의되어 있습니다.
# 
# 이 함수가 어떻게 손실 함수에서 log 부분을 계산하는지 관심있게 봐야 합니다. 
# (예)  :math:`log(D(x))` 와 :math:`log(1-D(G(z)))`).
# 우리는 어떤 부분의  BCE 식이 :math:`y` 의 입력값으로 이용될지 지정할 수 도 있습니다.
# 이 부분은 곧 등장할 훈련 반복문에서 사용될것이지만 :math:`y` (예) GT 라벨)를 바꿈으로서
# 어떤 부분을 계산하기를 원하는지 정할 수 있다는것을 이해하는게 굉장히 중요합니다.
# 
# 다음으로 우리는 진짜 데이터 라벨을 1로 가짜 데이터 라벨을 0으로 정의합니다.
# 이 라벨들은 :math:`D` 와 :math:`G` 의 손실을 계산할때 사용되며 
# 이러한 라벨들은 원래의 GAN 논문의 관례가 사용되었습니다.
# 마침내 우리는 두개의 다른 옵티마이저를 설정했습니다.
# 하나는 :math:`D` 이며 다른 하나는 :math:`G` 에 해당 합니다.
# DCGAN 논문에서 밝혔듯이 둘 다 Adam 옵티마이저를 사용하고 러닝 레이트는 0.0002이며
# Beta1 은 0.5로 설정합니다. 
# generator의 학습 진행을 계속해서 추적하기 위해서 우리는 가우시안 분포 (예) fixed_noise)에서 
# 얻은 고정된 잠재 백터의 배치를 생성할것 입니다.
# 훈련 반복문에서 우리는 주기적으로 이  fixed_noise 를 :math:`G` 의 입력값으로 집어넣고,
# 반복을 거치면서 노이즈에서 이미지의 형태가 나오는것을 보게 될 것입니다.
# 

# BCELoss 함수 초기화
criterion = nn.BCELoss()

# 잠재 벡터의 배치를 만들고 우리는 generator의 진행을 시각화 할때
# 이용 할 것입니다.
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 학습과정때 사용될 진짜와 가짜 데이터 라벨들을 설정 합니다.
fake_label = 0.

# G 와 D 에 Adam 옵티마이저를 설정 합니다.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


######################################################################
# 학습
# ~~~~~~~~
# 
# 마침내 우리는 GAN 프레임워크를 정의 하는 부분을 다 마치고 훈련을 시킬 수 있습니다.
# GAN을 훈련시킬때에는 알 수 없는 현상들이 나타나는데 잘못된 하이퍼파라미터 셋팅은
# 모델 붕괴로 가게 할 수 있고 뭐가 잘못되었는지 설명은 거의 할 수 없습니다.
# 여기서 우리는 최고의 사례라고 불리는 `ganhacks <https://github.com/soumith/ganhacks>`__ 
# 을 보면서 Goodfellow의 논문에서 Algorithm 1 을 적용할 예정입니다.
# 즉, 우리는  “진짜 이미지와 가짜 이미지의 다른 미니 배치 작성” 을 할 예정이고 또한 G의 목적 함수 :math:`logD(G(z))` 를 
# 최대화 하도록 조정합니다. 
# 학습 과정은 두개의 메인 파트로 나누어 집니다. 파트1은 Discriminator를 업데이트 하는것이고
# 파트2는 Generator를 업데이트 하는것입니다.
# 
# **파트 1 - Discriminator 학습시키기**
# 
# 다시 한번 말씀 드리자면 discriminator를 학습시키는 목적은 주어진 입력값이
#  진짜인지 가짜인지 정확히 분류하는 확률을 최대화 시키는것 입니다.
# Goodfellow의 말에 의하면 우리는 “ discriminator를 자신의 상승시키는 stochastic gradient
# 를 이용하여 업데이트” 해야하길 원해야 한다고 하고 있습니다.
# 우리는 :math:`log(D(x)) + log(1-D(G(z)))` 가 최대화 되도록 원합니다.
# ganhacks 이 제안한 분리된 미니 배치에 의하면, 우리는 이것을 두 단계로 나누어 계산해야합니다.
# 첫째로 우리는 훈련 세트의 실제 샘플들의 배치를 만들고 :math:`D` 로 전파하여 
# 손실 (:math:`log(D(x))`) 을 계산합니다. 그 뒤에 역전파로 gradient를 계산합니다.
# 둘째로 우리는 현재의 generator 로 부터 가짜 샘플의 배치를 생성하고 이것을 :math:`D`
# 로 전파시킨다음 손실 (:math:`log(1-D(G(z)))`) 을 계산합니다. 그리고 * 축적된 * gradient
# 들을 역전파 합니다.
# 이제, 전부 진짜 와 전부 가짜 배치들로 부터 축적된 gradient을 가지고 우리는 Discriminator의 옵티마이저
# 를 최적화 시킬것입니다.
# 
# **파트2 - Generator 학습시키기**
# 
# 본래의 논문이 서술했듯, 우리는 :math:`log(1-D(G(z)))` 을 최소화 하도록
# generator를 학습시켜서 더 나은 가짜를 만들기를 원합니다.
# 언급한 대로 이것은 Goodfellow가 초기 학습 단계에는 충분한 gradient를 제공하지 못한다는 것을
# 보여주었습니다.
# 이것을 고치기 위해 우리는 대신에 :math:`log(D(G(z)))` 을 최대화 하려고 합니다.
# 코드에서 우리는 다음과 같은 것을 달성하고자 합니다.
# 파트1에서의 Generator의 결과값을 Discriminator을 이용해서 분류하고 
# G의 손실을 * 진짜 라벨을 GT로 * 이용해서 계산하고, 
# G의 gradient를 역전파로 계산하고 그리고
# 마침내 G의 파라미터들을 옵티마이저 최적화를 통해 업데이트 합니다.
# 진짜 라벨을 GT 라벨로 이용하여 손실 함수를 계산하는것은 직관적이지 않아 보일 수도 있지만 
# 이것은 BCELoss의 :math:`log(x)` 부분을 (:math:`log(1-x)` 이 아니라)
# 정확히 우리가 원하는것을 계산하기 위해 사용합니다.
#
# 마침내, 우리는 통계적인 보고와 각 에포크의 마지막에 우리는
# 우리의 fixed_noise 배치를 generator를 통해 시각적으로 G의 학습 경과를 보기위해
# 입력시킬 수 있습니다.
# 학습 통계치 보고는 다음과 같습니다.
# 
# -  **Loss_D** - 모든 실제와 가짜 배치들 (:math:`log(D(x)) + log(D(G(z)))`) 의
#    손실 합을 discriminator 의 손실로 계산함.
# -  **Loss_G** - :math:`log(D(G(z)))` 로 계산된 generator의 손실
# -  **D(x)** - 모든 실제 배치에 대한 discriminator 의 평균 결과값.
#    이것은 1에 가까운 수에서 시작하여 이론적으로는 G 가 점점 더 나아지면서 0.5에 수렴합니다.
#    왜 그런지 생각해 보세요.
# -  **D(G(z))** - 모든 가짜 배치에 대한 discriminator 결과값의 평균.
#    D 전의 첫 값은 업데이트 되고 두번째 값은 D 이후에 업데이트 됩니다. 이러한 값을은 
#    0 근처에서 시작하여 G 가 나아지면서 0.5에 수렴합니다.
#    왜 그런지 생각해보세요. #
# 
# ** 알림: ** 이 단계는 얼마큼의 에포크로 설정 했냐에 따라 매우 오래걸릴 수도 있습니다.
# 원한다면 몇몇 데이터들을 데이터셋에서 지우셔도 됩니다.
# 

# 학습 반복문

# 진행을 추적하기 위한 배열들
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# 각 에포크
for epoch in range(num_epochs):
    # dataloader의 각 배치
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) D 신경망 업데이트: log(D(x)) + log(1 - D(G(z))) 최대화
        ###########################
        ## 전부 진짜 배치를 이용한 학습
        netD.zero_grad()
        # 배치 준비
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # D를 통해 진짜 배치 전파
        output = netD(real_cpu).view(-1)
        # 전부 진짜인 배치에 대한 손실 계산
        errD_real = criterion(output, label)
        # D를 통한 역전파로 gradient 계산
        errD_real.backward()
        D_x = output.mean().item()

        ## 전부 가짜인 배치 학습
        # 잠재 벡터의 배치 생성
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # G를 이용하여 가짜 이미지 배치 생성
        fake = netG(noise)
        label.fill_(fake_label)
        # D를 이용해서 모두 가짜인 배치 분류
        output = netD(fake.detach()).view(-1)
        # 전부 가짜인 배치를 이용하여 D의 손실 계산
        errD_fake = criterion(output, label)
        # 이 배치를 이용하여 gradient 계산
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 전부 가짜인 배치와 전부 진짜인 배치의 gradient 합치기
        errD = errD_real + errD_fake
        # D 업데이트 하기
        optimizerD.step()

        ############################
        # (2) G 신경망 업데이트: log(D(G(z))) 최대화 하기
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  #  가짜 라벨들은 generator 비용은 진짜 라벨로 처리
        #  방금 D 를 업데이트 했기 때문에, D를 통한 전부 가짜인 배치 전파를 실행
        output = netD(fake).view(-1)
        # 이 결과값을 기반으로 G의 손실 계산
        errG = criterion(output, label)
        # G의 gradient 계산
        errG.backward()
        D_G_z2 = output.mean().item()
        # G 업데이트
        optimizerG.step()
        
        # 학습 통계치 출력
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # 나중에 그래프 그리기위해 손실값들 저장
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        #  fixed_nois에 대한 G의 결과값과을 저장해서 어떻게 generator가 행동하고 있는지 확인
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1


######################################################################
# 결과
# -------
# 
# 마침내, 우리가 무엇을 했는지 확인해 볼 수 있습니다. 이곳에서 우리는 세가지 다른
# 결과를 확인할것 입니다. 첫째, 우리는 어떻게 D와 G의 손실이 학습 도중에 변화하는지
# 확인합니다. 둘째, 우리는 fixed_noise배치에서 매 에포크에 대한 G의 결과값을 시각화합니다.
# 셋째, 우리는 G로부터의 가자 데이터 배치 다음으로 진짜 데이터 배치를 확인합니다.
# 
# ** 손실 V.S. 학습 반복수 **
# 
# 아래는 D와 G의 손실과 학습 반복수에 대한 그래프 입니다.
# 

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


######################################################################
# ** G의 진행에 대한 시각화**
# 
# 어떻게 우리가 fixed_noise 배치에 대한 generator의 결과값을 훈련의 매 에포크 후에
# 저장했는지 기억해야 합니다. 이제, 우리는 G의 학습 진행을 애니메이션과 함께 
# 시각화 할수 있습니다. 재생 버튼을 눌러 애니메이션을 시작해 보세요.
# 

#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


######################################################################
# ** 실제 이미지 vs. 가짜 이미지**
# 
# 마침내 실제 이미지와 가짜 이미지를 직접 옆에 두고 비교해봅시다.
# 

# dataloader에서 진짜 이미지 배치를 가져옵니다.
real_batch = next(iter(dataloader))

# 진짜 이미지 그리기
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# 마지막 에포크에서의 가짜 이미지 그리기
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()


######################################################################
# 다음으로 해야할 것들
# ----------------
# 
# 마침내 우리는 우리의 여정 마지막에 도착했습니다. 하지만 여기서부터
# 해야할 것들이 남아 있습니다.
# 아래의 것들을 시도해 보세요.
# 
# -  더 오래 학습하여 얼마나 좋은 결과를 얻는지 확인하기
# -  다른 데이터셋을 갖기위해 모델을 변경해보고 가능하다면 이미지사이즈와 모델 구조 변경해보기
# -  다른 멋진 GAN 프로젝트 확인하기
#    ` 여기 <https://github.com/nashory/gans-awesome-applications>`__
# -  음악을 생성해내는 GAN 확인해보기
#    ` 음악 <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>`__
# 
