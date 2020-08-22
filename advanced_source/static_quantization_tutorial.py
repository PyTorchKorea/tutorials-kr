"""
(실험용) PyTorch에서 eager mode로 실행하는 정적 양자화 
=========================================================

**Author**: `Raghuraman Krishnamoorthi <https://github.com/raghuramank100>`_

**Edited by**: `Seth Weidman <https://github.com/SethHWeidman/>`_

**번역**: `Choi Yoonjeong <https://github.com/potatochips178/>`_


이 튜토리얼은 모델의 정확성을 더욱 높이기 위한 두가지 발전된 기술 - 채널별 양자화(per-channel quantization)와 양자화 자각 훈련(quantization-aware training)- 을 설명할 뿐 아니라 어떻게 훈련 후 정적 양자화(post-training static quantization)가 동작하는지 보여줍니다. 
최근 양자화는 CPU를 통해서만 사용 가능하기 때문에 GPU/CUDA를 사용하지 않을 것을 알립니다. 

이 튜토리얼의 마지막에서는 PyTorch에서 양자화가 어떻게 속도를 오르게 하는 반면 모델 크기를 줄이는 결과가 나오는지 볼 수 있습니다. 
게다가 `여기서 <https://arxiv.org/abs/1806.08342>`_ 보인 발전된 양자화 기술들을 쉽게 적용시키는 법을 볼 수 있을 것이며 양자화된 모델들은 그렇지 않은 모델보다 훨씬 더 적은 정확성 일치를 가집니다. 

주의: ``MobileNetV2`` 모델 구조 정의하기, 데이터 로더 정의하기 등과 같은 다른 PyTorch 레포지토리에서 공통으로 사용되는 많은 코드를 사용할 예정입니다. 
물론 사용자들이 이것을 읽기 바랍니다. 그러나 만약 양자화 특징을 얻길 원한다면 자유롭게 건너뛰고 "4. 훈련 후 정적 양자화" 부분으로 넘어가면 됩니다. 

필요한 것을들 불러오면서 시작하겠습니다:
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization

# # warnings 만들기
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# 반복 가능한 결과를 위한 무작위 시드 지정하기
torch.manual_seed(191009)

######################################################################
# 1. 모델 구조
# ---------------------
#
# 먼저 양자화를 할 수 있도록 몇몇 중요한 수정을 적용한 MobileNetV2 모델 구조를 정의하겠습니다: 
#
# - 덧셈을 ``nn.quantized.FloatFunctional`` 로 대체하기
# - 신경망 처음과 끝에 ``QuantStub`` 와 ``DeQuantStub`` 삽입하기
# - ReLU6를 ReLU로 교체하기
#
# 노트: 이 코드는 `이곳에서 <https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py>`_ 가져왔습니다. 

from torch.quantization import QuantStub, DeQuantStub

def _make_divisible(v, divisor, min_value=None):
    """
    이 함수는 원본 ts repo에서 가져왔습니다. 
    모든 계층들이 8로 나뉠수 있는 채널 숫자를 가지는 것을 보장합니다. 
    다음에서 확인할 수 있습니다:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 내림은 10% 이상으로 내려가지 않는 것을 보장합니다. 
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # ReLU로 교체하기
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # torch.add을 floatfunctional로 교체하기
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 메인 클래스
        인자들:
            num_classes (int): 클래스 개수
            width_mult (float): 넓이 승수 - 이 양을 통해 각 계층에 있는 채널 개수를 조정 
            inverted_residual_setting: 네트워크 구조
            round_nearest (int): 각 계층의 채널 수를 이 숫자의 배수로 반올림. 1부터 반올림하지 않음으로 설정 가능.
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # 사용자가 t, c, n, s가 요구된다는 것을 안다는 전제하에 첫 번째 요소만 확인하기
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # 첫 번째 계층 만들기
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        #  반전된 나머지 블락 만들기
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # 나머지 여러 계층들 만들기
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # nn.Sequential 로 만들기
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # classifier 만들기
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.quant(x)

        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # 양자화에 앞서 Conv+BN 와 Conv+BN+Relu 모듈을 융합시키기
    # 이 연산자는 숫자를 변경하지 않음
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

######################################################################
# 2. 헬퍼(Helper) 함수
# ---------------------
#
# 다음으로 모델 평가를 위한 여러 헬퍼 함수를 만들도록 하겠습니다. 
# 이것들은 대부분 `이곳으로부터 <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_  왔습니다. 
# 

class AverageMeter(object):
    """계산하고 평균과 현재 값 저장하기"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """k의 구체적인 값을 위해서 k 상위 예측에 대한 정확성을 계산하기"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

######################################################################
# 3. 데이터셋과 데이터 로더 정의하기
# ----------------------------------
#
# 마지막 주요 설정 단계로서, 훈련과 테스트셋을 위해 데이터로더를 정의해야 합니다. 
#
# ImageNet Data
# ^^^^^^^^^^^^^
#
# 이 튜토리얼에서 만드는 특정한 데이터셋은 ImageNet 데이터에서 1000개의 이미지들을 포함하고 이미지는 각 클래스로부터 옵니다. 
# (250MB가 넘는 이 데이터셋은 비교적 쉽게 다운로드할 수 있을 정도로 크기가 작습니다)
# 이 커스텀 데이터셋을 다운받을 수 있는 URL은 다음과 같습니다: 
#
# .. code::
#
#     https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip
#
# Python을 사용해 데이터를 로컬로 다운받기 위해선 다음과 같이 하면 됩니다:
#
# .. code:: python
#
#     import requests
#
#     url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip`
#     filename = '~/Downloads/imagenet_1k_data.zip'
#
#     r = requests.get(url)
#
#     with open(filename, 'wb') as f:
#         f.write(r.content)
#
# 이 튜토리얼을 실행하기 위해서, `메이크파일(makefile) <https://github.com/pytorch/tutorials/blob/master/Makefile>`_ 로부터 만들어진 `코드 <https://github.com/pytorch/tutorials/blob/master/Makefile#L97-L98>`_ 를 사용해 데이터를 다운로드받고 알맞은 장소로 이동시켜야 합니다. 
# 
# 
#
# 반면에, 전체 ImageNet 데이터셋을 사용해 이 튜토리얼에 있는 코드를 실행하기 위해서는 `여기서 <https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet>`_ ``torchvision`` 을 사용한 데이터를 다운받아야 합니다. 
# 예를 들어, 트레이닝 셋을 다운받고 거기에 표준 변환을 적용시키기 위해서는 다음과 같이 해야 합니다:
#
# .. code:: python
#
#     import torchvision
#     import torchvision.transforms as transforms
#
#     imagenet_dataset = torchvision.datasets.ImageNet(
#         '~/.data/imagenet',
#         split='train',
#         download=True,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])
#
# 데이터를 다운로드한 후, 데이터 읽기에 사용할 데이터로더를 정의할 함수를 아래와 같이 보여야 합니다. 
# 이 함수들은 대부분은 `여기에서 <https://github.com/pytorch/vision/blob/master/references/detection/train.py>`_ 가져왔습니다. 

def prepare_data_loaders(data_path):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

######################################################################
# 다음으로, 미리 학습된 MobileNetV2 모델을 가져와야 합니다. 
# ``torchvision`` `이곳에서 <https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py#L9>`_ 에서 데이터를 다운받을 수 있는 URL을 제공합니다.

data_path = 'data/imagenet_1k'
saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 30

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

######################################################################
# 다음으로, "모델 혼합"을 하겠습니다; 이것은 정확도를 향상시키는 동시에 메모리 접근을 줄여 모델을 빠르게 만듭니다. 
# 어떤 모델에도 적용 가능하지만 특히 양자화된 모델에서 흔합니다. 

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# 모델 혼합하기
float_model.fuse_model()

# Conv+BN+Relu 와 Conv+Relu의 혼합 기록
print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)

######################################################################
# 드디어 정확도의 "기준"에 도달하기위해, 양자화되지 않은 모델과 혼합된 모델의 정확도를 확인해 봅시다. 

num_eval_batches = 10

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

######################################################################
# 모델이 14.0 MB밖에 되지 않는다는 것에 비해 ImageNet의 확실한 기준을 가진 300개의 이미지에서 78%의 정확도를 볼 수 있습니다. 
#
# 이것이 비교할 기준이 됩니다. 다음으로, 다른 양자화 방법을 사용해봅시다. 
#
# 4. 훈련 후 정적 양자화(Post-training static quantization)
# ---------------------------------------------------------
#
# 훈련 후 정적 양자화는 동적 양자화처럼 가중치를 실수에서 정수로 변환하는 것을 포함할 뿐 아니라 추가적인 첫 번째 피딩 네트워크를 지나가는 데이터의 배치 단계의 수행과 다른 활성값의 분산 결과 계산이 필요합니다
# (특히, 이것은 `옵저버(observer)` 모듈을 이 데이터를 기록하는 다른 지점에 삽입함으로서 실행됩니다).
# 이 분산은 어떻게 특별히 다른 활성값이 추론 시간에 양자화되야 하는지를 결정하는데 사용됩니다(단순한 기술은 전체 범위의 활성값을 256단계로 나누는 것이지만 더 정교한 방법을 사용하겠습니다).
# 중요하게, 이 추가적인 단계는 양자화된 값이 모든 연산 사이에서 실수로 변환하는 것 대신 - 그리고 다시 정수로 돌아감 - 이 값들이 연산들을 지나갈 수 있도록 만들어 상당한 속도 향상을 이룹니다. 

num_calibration_batches = 10

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Conv, bn 와 relu 혼합
myModel.fuse_model()

# 양자화 형태 구체화하기
# 간단한 min/max 범위의 평가와 텐서별 가중치의 양자화로 시작하기
myModel.qconfig = torch.quantization.default_qconfig
print(myModel.qconfig)
torch.quantization.prepare(myModel, inplace=True)

# 첫번째 측정
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

# 트레이닝 셋과 대응시키기
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# 양자화된 모델로 변환하기
torch.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

print("Size of model after quantization")
print_size_of_model(myModel)

top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

######################################################################
# 양자화된 모델을 위해, 같은 300개 이미지에 대해서 62% 이하인 상당히 낮은 정확성을 보겠습니다.
# 그럼에도 불구하고, 모델의 사이즈를 거의 4x 정도가 줄은 3.6 MB 이하로 감소시켰습니다.
#
# 추가적으로, 단순히 다른 양자화 형태를 사용하면 정확성을 꽤 향살시킬 수 있습니다. 
# x86의 양자화를 위해 추천하는 형태를 통해 같은 행위를 반복하면 됩니다. 
# 이 형태는 다음과 같습니다. 
#
# - 각 채널 기본에 있는 가중치를 양자화합니다.
# - 활성값의 히스토그램을 모으는 히스그램 옵저버(observer)를 사용해 최선의 방법인 양자화 파라미터를 고릅니다.
#

per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)

######################################################################
# 단지 이 양자화 형태 메소드를 바꾸는 것만으로 76%의 정확성 증가가 나옵니다!
# 아직, 위에서 구한 78%의 기준보다는 1-2% 정도 떨어집니다.
# 그래서 이제 양자화 자각 훈련(quantization aware training)을 해봅시다. 
#
# 5. 양자화-자각 훈련(Quantization-aware training)
# ------------------------------------------------
#
# 양자화 자각 훈련(QAT)는 보편적으로 가장 높은 정확성을 내는 양자화 방법입니다.
# QAT를 사용하면, 모든 가중치와 활성값들은 정방향과 역방향 모두 지나가면서 훈련을 하는 동안 가짜로 양자화됩니다:
# 즉,  실수 값은 int8 값과 유사하게 반올림되지만 모든 계산들은 부동소수점 수로 진행됩니다.
# 따라서 훈련하는 동안 모든 가중치 보정은 모델이 결과적으로 양자화 될 것이라는 사실을 "자각"한 체로 만들어집니다; 
# 그러므로 양자화한 이후, 이 방법은 동적 양자화나 훈련 후 정적 양자화보다는 더 높은 정확성을 산출해 낼 것입니다. 
#   
# QAT를 실행하는 데 있어 전체적인 실행 흐름은 이전과 매우 유사합니다:
#
# - 이전과 같은 모델을 사용할 수 있습니다: 양자화-자각 훈련을 위해 추가적인 준비가 필요하지 않습니다. 
# - 옵저버(observer)를 지정하는 대신에 가중치와 활성값 다음에 어떤 가짜-양자화를 사용할 지 지정하는 ``qconfig`` 사용이 필요합니다. 
#
# 첫 번째로 훈련 함수를 정의하겠습니다:

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

######################################################################
# 이전처럼 모델을 융합하겠습니다.

qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

######################################################################
# 마침내, 양자화-자각 훈련을 위한 모델을 준비하기 위해 ``prepare_qat`` 는 "가짜 양자화"를 실행합니다. 

torch.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)

######################################################################
# 높은 정확성을 가진 양자화된 모델을 훈련하기는 추론을 통한 정확한 숫자 모델링을 요구합니다. 
# 그러므로 양자화 자각 훈련을 위해, 다음을 통해 훈련을 수정합니다:
#
# - 추론 수치와 더 많은 일치를 위해 배치 기준을 실행하는 동안 훈련의 마지막을 향하는 평균과 분산을 사용하는 것으로 변경니다. 
# - 또한 양자화 파라미터들(크기와 제로 포인트)을 고정하고 가중치를 정밀하게 맞춥니다.

num_train_batches = 20

# 훈련시키고 각 에폭(epoch)후에 정확성을 확인하기
for nepoch in range(8):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # 양자화 인자 고정하기
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # 배치 기준을 평균과 분산 측정으로 고정하기
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # 각 에폭 후에 정확성 확인하기
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

#####################################################################
# 여기서 작은 숫자 에폭으로 양자화-자각 훈련을 실행했습니다. 
# 그럼에도 불구하고, 양자화-자각 훈련은 전체 이미지넷 데이터셋에서 부동소수점의 71.9%와 가까운 71%를 넘어선 정확성을 산출했습니다. 
#
# 양자화-자각 훈련에 대해서 더 많은 것들:
#
# - QAT는 더 많은 디버깅을 허용하는 훈련 후 양자화 기술의 상위 기술입니다. 
#   예를 들어, 만약 모델의 정확성이 가중치나 활성값 양자화에 의해 제한되었다면 분석 가능합니다. 
# - 실제 양자화 계산의 수치를 만드는데 가짜 양자화를 사용했기 때문에 부동소수점에서 양자화된 모델의 정확성을 시뮬레이트할 수 있습니다.
# - 훈련 후 양자화를 쉽게 모방할 수 있습니다. 
#
# 양자화를 통해 속도 높이기
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# 마침내, 위에서 언급한 무언가를 확정해 봅시다: 양자화된 모델이 실제로 추론을 더 빠르게 실행할까요? 테스트해 봅시다.

def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # 적은 이미지 배치(batch)로 미리 짜인 모델 실행하기
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)

######################################################################
# 맥북프로에서 로컬로 이것을 실행하면 보통 모델에 대해서 61ms 가, 양자화된 모델에서는 20ms가 걸리며 이는 평균적으로 2-4x 정도 속도가 오른 것을 설명합니다. 
# 부동소수점 모델과 양자화된 모델을 비교 했습니다.
#
# 결론
# ----------
#
# 이 튜토리얼에서, 실제로 무엇을 실행하는지 그리고 PyTorch에서 어떻게 사용하는지 설명하기 위해 두 개의 양자화 모델 - 훈련 후 정적 양자화와 양자화 자각 훈련 - 을 보였습니다. 
#
# 읽어주셔서 감사합니다! 언제나 그렇듯, 피드백을 환영하며 만약 이슈가 있다면 `여기에 <https://github.com/pytorch/pytorch/issues>`_ 남겨주면 고맙겠습니다.
#  

