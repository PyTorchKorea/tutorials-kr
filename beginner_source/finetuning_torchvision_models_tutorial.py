"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__
**번역:** `길정경 <https://github.com/kil-jung-keong>`__

"""


######################################################################
# 이 튜토리얼에서는 어떻게 미세조정을 하는지와 각각의 모델들이 1000-클래스의 이미지넷 데이터셋에서 사전학습된 `토치비전 모델
# <https://pytorch.org/docs/stable/torchvision/models.html>`__ 에서 특징 추출을 하는지에 대하여 깊게 살펴볼 것입니다. 
# 이 튜토리얼은 몇 개의 현대적인 CNN 구조들에 대하여 깊은 해석을 할 수 있게끔 해주며, 어떠한 파이토치 모델에도 미세조정을 진행할 수 있도록 직관력을 길러줄 것입니다. 
# 각각의 모델 구조들이 모두 달랐기 때문에, 어떠한 시나리오에서도 적용가능한 표준화된 미세조정 코드가 없었습니다.
# 따라서 연구자들은 각각의 모델에 현존하는 구조들을 보고 직접 적절하게 커스터마이징을 해야했습니다. 
# 
# 이 문서에서는 두 가지 종류의 transfer learning(전이 학습)에 대하여 보일 것입니다 : 미세조정과 특징 추출
# 
# **미세조정** 에서는, 새로운 문제에 적용하기 위하여 사전학습된 모델에서 시작하여 *모든* 모델의 parameters(파라미터들)을 업데이트합니다. 즉, 전체 모델을 재훈련 시킵니다. 
# **특징 추출** 에서는, 사전학습된 모델에서 시작하여 가장 마지막 층의 가중치만 업데이트 하는 방식으로 예측을 이끌어냅니다. 
# 이것을 특징 추출이라 부르는데, 그 이유는 사전학습된 CNN을 고정된 특징 추출기로서 사용하며, 출력층만 변형시키기 때문입니다. 
# 전이 학습에 대한 기술적인 정보가 더 필요하다면 `여기 <https://cs231n.github.io/transfer-learning/>`__ 나 `여기 <https://ruder.io/transfer-learning/>`__ 를 참고하기 바랍니다. 
# 일반적으로 전이 학습 방법들은 모두 다음의 단계들을 따릅니다:
# 
# -  사전 학습된 모델을 초기화(initialize)합니다.
# -  마지막 층(들)의 모양을 새로운 데이터셋의 클래스들의 개수와 같게 만듭니다.
# -  학습하는 동안 어떤 파라미터들을 학습시킬지 정하기 위하여 최적화 알고리즘을 정의합니다. 
# -  학습 단계를 진행시킵니다.
# 

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


######################################################################
# 입력
# ------
# 
# 다음 내용들이 동작을 위하여 필요한 모든 파라미터들의 변경사항입니다. 
# `여기 <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`__ 에서 다운받을 수 있는 *hymenoptera_data* 라는 데이터셋을 사용할 것입니다. 
# 이 데이터셋은 **벌** 과 **개미** 의 두 개의 클래스들을 포함하며, 이 데이터셋은 커스텀 데이터셋을 새로 만드는 대신, `ImageFolder <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder>`__ 
# 을 사용할 수 있도록 구조적으로 구성되어있습니다. 
# ``model_name`` 은 당신이 사용하고 싶은 모델을 입력하는 것이며, 다음의 리스트에서 반드시 선택되어져야만 합니다:
# 
# ::
# 
#    [resnet, alexnet, vgg, squeezenet, densenet, inception]
# 
# 다른 입력값들은 다음과 같습니다: ``num_classes`` 는 데이터셋에 있는 클래스들의 개수를 의미하고, 
# ``batch_size`` 는 학습을 위한 배치의 크기를 의미하며,
# ``num_epochs`` 는 작동시키고 싶은 학습 에포크들의 개수를 의미하고, 마지막으로
# ``feature_extract`` 는 사전 학습을 시키고 있거나 특징 추출을 진행하고 있을 때를 의미하는 불 방식의 입력값입니다. 
# 만약 ``feature_extract = False`` 이라면, 모델은 사전 학습이 완료된 것이며, 모든 모델의 파라미터들이 업데이트 된 것입니다.
# 만약 ``feature_extract = True`` 이라면, 다른 파라미터들은 고정된 채로 가장 마지막 층의 파라미터들만이 업데이트 된것입니다. 
# 

# 다음은 가장 윗 단계의 데이터 디렉토리입니다. 여기서는 이미지폴더(ImageFolder) 데이터 형식을 따릅니다. 
data_dir = "./data/hymenoptera_data"

# [resnet, alexnet, vgg, squeezenet, densenet, inception]에서 모델을 선택합니다. 
model_name = "squeezenet"

# 데이터 셋 안에 있는 클래스들의 개수
num_classes = 2

# 학습을 위한 배치사이즈 (당신이 얼만큼의 메모리를 가지고 있느냐에 따라 달라질 수 있습니다.)
batch_size = 8

# 학습을 위한 에포크의 개수
num_epochs = 15

# 특징 추출을 위한 표기. 
# 만약 거짓으로 되어있다면, 전체 모델을 미세조정하는 것이고, 참으로 되어있다면 재구성된 층의 파라미터들만 업데이트하는 것입니다. 
feature_extract = True


######################################################################
# 도움 함수들
# ----------------
# 
# 적절한 모델들을 위한 코드를 작성하기 전에, 몇 개의 도움 함수들에 대하여 정의해보겠습니다. 
# 
# 모델 학습과 검증 코드
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# ``train_model`` 함수는 주어진 모델에 대한 학습과 검증을 담당합니다. 
# 모델이 인셉션(Inception)일 때 입력값으로서 파이토치 모델, 데이터로더 딕셔너리(dictionary), 손실 함수, 옵티마이저, 학습과 검증을 위한 특정한 수의 에포크, 그리고 불방식의 플래그가 필요합니다. 
# *is_inception* 플래그는 *Inception v3* 모델을 수용하기 위한 플래그이며, `여기 <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>`__ 
# 에 묘사된 것과 같이 이 구조는 보조적인 출력을 사용하므로  전체적인 모델의 손실은 보조적인 출력과 최종적인 출력을 모두 고려해야합니다. 
# 함수는 특정한 개수의 에포크만큼 훈련이 되며, 각각의 에포크는 전체 검증 단계만큼 실행이 됩니다. 
# 이 함수는 또한 모델의 최고 성능(검증 정확도)을 추적하며, 마지막 훈련에서는 최고의 성능을 보이는 모델을 반환합니다.  
# 각각의 에포크를 마친 후에는 학습과 검증 정확도가 출력됩니다. 
# 

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에포크는 학습과 검증의 단계가 존재합니다. 
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정합니다
            else:
                model.eval()   # 모델을 평가 모드로 설정합니다

            running_loss = 0.0
            running_corrects = 0

            # 전체 데이터에 대하여 반복합니다. 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 파라미터의 변화도를 0으로 설정합니다. 
                optimizer.zero_grad()

                # 순전파
                # 학습을 진행하고 있을 때만 과거를 추적한다. 
                with torch.set_grad_enabled(phase == 'train'):
                    # 모델의 출력을 얻고, 손실값을 계산한다. 
                    # 보조적인 출력값이 있는 inception 모델일 경우는 특별한 경우입니다.
                    # 따라서 이 모델이 학습 모드일 경우 최종적인 결과값과 보조 출력값을 더함으로써 손실을 계산하지만 테스트 모드에서는 최종적인 출력값만 고려합니다. 
                    if is_inception and phase == 'train':
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958 를 참고하였습니다.
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 훈련 단계일 때만 진행하는 역전파 + 최적화 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 모델의 가중치를 깊은 복사를 이용하여 얻습니다. 
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 최고의 모델 가중치를 불러옵니다. 
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


######################################################################
# 모델 파라미터들에 대한 .requires_grad 기여를 설정하는 방법
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 이 도움 함수는 특징을 추출할 때 거짓으로 설정함으로써 ``.requires_grad`` 기능이 모델의 파라미터에 기여할 수 있도록 합니다. 
# 기본적으로 사전 학습된 모델을 불러올 때, 모든 파라미터들은 ``.requires_grad=True`` 으로 설정되어있으나, 이는 처음부터 학습을 시키거나 미세조정을 진행할 때만 필요합니다. 
# 그러나, 만약 특징 추출을 하고 있고 새롭게 초기화된 층의 변화도만 계산하고 싶다면 그외 모든 파라미터들은 변화도가 필요하지 않습니다. 
# 이 부분에 대해서는 나중에 더 기술하도록 하겠습니다. 
# 

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


######################################################################
# 네트워크를 초기화 하고 재구성(Reshape) 하는 법 
# -----------------------------------
# 
# 이제 가장 흥미로운 부분에 대하여 기술하겠습니다.
# 여기서는 각 네트워크의 모양을 어떻게 바꾸는지에 대하여 설명하도록 하겠습니다.
# 참고로, 이것은 자동적인 과정이 아니며 각 모델에 대하여 고유한 특징을 가집니다. 
# 보통 FC 층이 위치하는 CNN 모델의 마지막 층의 노드들의 개수는 데이터셋에서 클래스의 개수와 동일합니다.  
#  
# 모든 모델들이 이미지넷 데이터셋에서 사전학습이 되었기 때문에, 그들은 모두 사이즈가 1000인 출력층을 가지며 각 클래스에 대하여 하나의 노드를 가집니다. 
# 
# 이 문서는 마지막 계층의 모양을 이전의 입력값의 수와 같게 만드는 것이며 데이터셋의 클래스들의 개수와 같은 수의 출력값들을 가지게 하는 것이 목적입니다.
# 이어지는 다음 장부터는 각각의 모델의 구조를 어떻게 바꾸는지에 대하여 이야기해보도록 하겠습니다. 
# 하지만 먼저, 미세조정과 특징 추출의 차이에 대해서 중요한 세부사항이 있습니다. 
# 특징 추출 시에, 가장 마지막 층의 변수들의 파라미터들만 업데이트 하고 싶어합니다, 즉, 재구성하는 층의 파라미터들만 업데이트 하고 싶어하는 것입니다. 
# 따라서, 바꾸지 않는 파라미터들일 경우에는 기울기를 계산할 필요가 없기 때문에 효율성을 위해서 .requires_grad 모듈을 False로 놓습니다. 
# 기본적으로 이 모듈이 True로 설정되어있기 때문에 이는 중요한 부분입니다. 
# 그리고 나면, 새로운 층을 초기화하고 새로운 파라미터들이 ``.requires_grad=True`` 이 되게 하기때문에 오직 새로운 계층의 파라미터들만이 업데이트됩니다. 
# 미세조정을 할 때에는 .required_grad를 기본값인 True로 설정해 놓는다.
# 
# 최종적으로, inception_v3 는 입력 사이즈로 (299,299) 를 요하며, 다른 모델들은 모두 (224,224) 사이즈입니다. 
# 
# Resnet
# ~~~~~~
# 
# Resnet은 `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__ 에서 소개되었습니다. 
# Resnet18, Resnet34, Resnet50, Resnet101 과 Resnet152의 다양한 사이즈의 모델이 있으며, 각각은 모두 토치비전 모델에서 사용이 가능합니다. 
# 여기서는 데이터셋이 작고 2개의 클래스만을 가지고 있기 때문에 Resnet18을 사용하였습니다.
# 모델을 출력해보면, 마지막 층은 아래에 나와있듯이 완전연결계층입니다. 
# 
# ::
# 
#    (fc): Linear(in_features=512, out_features=1000, bias=True) 
# 
# 따라서, 512개의 입력 특징들과 2개의 출력 특징들을 가지는 선형 층이 되도록 하기 위하여 ``model.fc`` 층을 재초기화해야합니다:
# 
# ::
# 
#    model.fc = nn.Linear(512, num_classes)
# 
# Alexnet
# ~~~~~~~
# 
# Alexnet은 `ImageNet Classification with Deep
# Convolutional Neural
# Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`__
# 에서 소개되었으며 이미지넷 데이터셋을 사용한 CNN중 가장 처음으로 성공적인 모델이였습니다. 
# 이 모델의 구조를 출력해보면, 분류기의 6번째 층으로부터 모델의 출력값이 나오는 것을 볼 수 있습니다. 
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     ) 
# 
# 우리가 가진 데이터셋에 이 모델을 적용하고 싶다면 다음과 같이 재초기화 해야합니다. 
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# VGG
# ~~~
# 
# VGG는 `Very Deep Convolutional Networks for
# Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`__ 에서 소개되었습니다. 
# 토치비전은 다양한 길이와 배치 정규화 층들을 가진 8개 버전의 VGG 모델을 제공합니다. 
# 여기서는 배치 정규화 층을 가진 VGG-11 모델을 사용합니다. 
# 출력층은 Alexnet과 비슷합니다.
# 
# ::
# 
#    (classifier): Sequential(
#        ...
#        (6): Linear(in_features=4096, out_features=1000, bias=True)
#     )
# 
# 따라서, 출력층을 수정하기 위하여 같은 기술을 사용합니다. 
# 
# ::
# 
#    model.classifier[6] = nn.Linear(4096,num_classes)
# 
# Squeezenet
# ~~~~~~~~~~
# 
# Squeeznet은 `SqueezeNet:
# AlexNet-level accuracy with 50x fewer parameters and <0.5MB model
# size <https://arxiv.org/abs/1602.07360>`__ 에서 소개되었으며 여기에 소개된 어떠한 모델들보다 다른 출력 구조를 사용합니다. 
# 토치비전은 2가지 버전의 Squeezenet을 제공하며, 여기서는 1.0 버전을 사용합니다.
# 출력은 분류기의 가장 첫 번째 층인 1x1 합성곱 층으로부터 나옵니다:
# 
# ::
# 
#    (classifier): Sequential(
#        (0): Dropout(p=0.5)
#        (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#        (2): ReLU(inplace)
#        (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
#     ) 
# 
# 이 네트워크를 수정하기 위하여 2 깊이의 출력 특징맵을 구성하기위하여 Conv2d 층을 재초기화 합니다. 
# 
# ::
# 
#    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
# 
# Densenet
# ~~~~~~~~
# 
# Densenet은 `Densely Connected Convolutional
# Networks <https://arxiv.org/abs/1608.06993>`__ 에서 소개되었습니다. 
# 토치비전은 4가지 버전의 Densenet을 제공하지만 여기서는 Densenet-121을 사용합니다. 
# 출력층은 1024개의 입력 특징들을 가진 선형층입니다:
# 
# ::
# 
#    (classifier): Linear(in_features=1024, out_features=1000, bias=True) 
# 
# 네트워크를 재구성하기위하여, 분류기의 선형층을 다음과 같이 재초기화합니다. 
# 
# ::
# 
#    model.classifier = nn.Linear(1024, num_classes)
# 
# Inception v3
# ~~~~~~~~~~~~
# 
# 마지막으로, Inception v3 모델은 `Rethinking the Inception
# Architecture for Computer
# Vision <https://arxiv.org/pdf/1512.00567v1.pdf>`__ 에서 처음 소개되었습니다. 
# 이 네트워크는 학습할 때 2개의 출력층을 가지고 있기 때문에 특별합니다. 
# 두 번재 출력은 보조 출력이라고 알려져있으며, 모델의 AuxLogits 부분을 포함하고 있습니다. 
# 주요한 출력은 모델의 가장 마지막에 위치한 선형 층입니다.
# 참고로, 추론할때 주요한 출력을 고려합니다. 
# 보조 출력과 주요 출력은 다음과 같이 보여질 수 있습니다:
# 
# ::
# 
#    (AuxLogits): InceptionAux(
#        ...
#        (fc): Linear(in_features=768, out_features=1000, bias=True)
#     )
#     ...
#    (fc): Linear(in_features=2048, out_features=1000, bias=True)
# 
# 이 모델을 미세조정하기 위해서는 이 두 층들을 모두 재구성해야합니다. 
# 이는 다음과 같이 시행될 수 있습니다. 
# 
# ::
# 
#    model.AuxLogits.fc = nn.Linear(768, num_classes)
#    model.fc = nn.Linear(2048, num_classes)
# 
# 많은 모델들이 비슷한 출력 구조를 가지고 있지만 각각은 반드시 다르게 다루어져야합니다. 
# 또한, 출력된 모델의 구조를 확인하고 데이터셋의 클래스들의 개수와 출력 특징들의 개수가 같은 것을 확인해야합니다. 
# 

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 이 if 문에 설정될 변수를 초기화합니다.
    # 
    # 각각의 변수는 모델에 따라 달라집니다. 
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# 학습을 위하여 모델을 초기화합니다. 
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# 방금 인스턴스화 한 모델을 출력합니다. 
print(model_ft) 


######################################################################
# Load Data
# ---------
# 
# 입력 사이즈가 어떤식으로 구성되어야할지 알기 때문에, data
# transforms, image datasets, 그리고 dataloaders를 초기화할 수 있습니다. 
# 참고로, `여기 <https://pytorch.org/docs/master/torchvision/models.html>`__ 에 묘사된 것 처럼 모델들은 정규화된 값으로 사전학습 되었습니다. 
# 

# 학습을 위한 데이터 증강과 정규화
# 확인을 위한 정규화 
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# 학습과 확인을 위한 데이터셋 생성
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# 학습과 학인을 위한 데이터로더 생성
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# GPU가 사용 가능하다면 찾습니다. 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# 옵티마이저 생성
# --------------------
# 
# 모델 구조가 맞다면, 사전학습과 특징 추출을 위한 마지막 단계는 원하는 파라미터들만을 업데이트 하는 옵티마이저를 생성하는 것입니다. 
# 재구성하기 이전에 사전학습된 모델을 불러온 이후에는 만약 ``feature_extract=True`` 이라면 모든 파라미터들을 알려진대로 놓아야하고 
# ``.requires_grad`` 이 False로 되어있어야 합니다. 
# 그리고 재초기화 된 층의 파라미터들이 기본적으로 ``.requires_grad=True`` 으로 되어있어야 합니다. 
# 그리고 지금부터 *.requires_grad=True 인 모든 파라미터들은 최적화되어야합니다.*
# 다음으로, 파라미터들과 입력값의 리스트를 만들고 이 리스트를 SGD 알고리즘에 넣습니다. 
# 
# 이를 확인하기 위하여, 학습을 위해 출력된 파라미터들을 확인합니다. 
# 미세조정을 할 때에 이 리스트는 길어야하며 모든 모델의 파라미터들을 포함하고 있어야합니다. 
# 그러나, 특징 추출을 할 때에 이 리스트는 짧아야하고 재구성된 층들의 가중치와 편향들만을 포함해야 합니다. 
# 

# GPU에 모델을 보냅니다. 
model_ft = model_ft.to(device)

# 이 학습에서 파라미터들이 최적화되고 업데이트 되도록 모읍니다. 
# 만약 미세조정을 한다면 모든 파라미터들을 업데이트해야할 것입니다. 
# 그러나, 만약 특징 추출을 하고 있다면, 방금 초기화 한것, 즉 requires_grad가 True인 파라미터들만을 업데이트 합니다. 
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# 모든 파라미터들이 최적화 된것을 관찰합니다. 
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


######################################################################
# 학습과 확인 스텝을 구동합니다. 
# --------------------------------
# 
# 최종적으로, 마지막 단계는 모델의 손실을 설정하고 난 후 설정된 개수의 에포크 만큼 학습과 확인 함수를 구동하는 것입니다. 
# 에포크의 수에 따라 cpu에서 시간이 걸릴 수 있습니다. 
# 또한, 기본적으로 설정된 학습률은 모든 모델들에서 최적이지 않으며, 따라서 최고의 성능을 얻기 위해서는 각각의 모델에 따라서 조정하는 것이 필수적입니다. 
# 

# 손실 함수를 설정합니다. 
criterion = nn.CrossEntropyLoss()

# 학습과 평가
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


######################################################################
# 처음부터 학습된 모델 간의 비교 
# ------------------------------------------
# 
# 그저 재미로, 전이학습을 사용하지 않았을 때 어떻게 모델이 학습하는지 관찰해보도록 하겠습니다.
# 미세 조정 및 특징 추출의 성능은 다음과 같습니다.
# 주로 데이터 세트에 그 성능이 결정되지만 일반적으로 두 전이 학습 방법은 처음부터 훈련된 모델에 비해 훈련 시간과 전체 정확도 측면에서 유리한 결과를 산출합니다.
# 
# 

# 사전 학습 되지 않은 버전의 모델을 초기화합니다. 
scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
_,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# 처음부터 훈련된 모델과 전이학습을 시행한 모델의 학습 곡선을 검증 정확도 대 학습 에포크의 개수에 대하여 그려보았습니다. 
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()


######################################################################
# 최종적인 결론과 향후 방향
# -----------------------------------
# 
# 다른 모델들도 학습을 시켜보고 얼마나 좋은 성능을 보이는지 관찰해보세요. 
# 또한, 역전파에서 대부분의 기울기를 계산할 필요가 없기 때문에 특징 추출에서 적은 시간이 걸린다라는 것을 명심하세요. 
# 당신은 여기서부터 많은 것들을 할 수 있습니다:
# 
# -  더 어려운 데이터 셋으로 이 코드를 구동할 수 있으며 전이학습의 장점을 관찰할 수 있습니다. 
# -  여기서 묘사된 방법들을 사용하고, 새로운 분야에서 (i.e. NLP, 오디오, 등등) 다른 모델을 업데이트 하기 위하여 전이학습을 사용할 수 있습니다. 
# -  모델에 만족하면 ONNX 모델로 내보내거나 하이브리드 프런트 엔드를 사용하여 모델을 추적하여 더 많은 속도와 최적화 기회를 얻을 수 있습니다.
# 

