"""
**파이토치(PyTorch) 소개** ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
`TensorBoard Support <tensorboardyt_tutorial.html>`_ ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

파이토치(PyTorch) 소개
=======================

아래 영상이나 `유튜브 <https://www.youtube.com/watch?v=IC0_FRiX-sw>`__ 를 보고 따라해보세요.
(역자 주: 영어 영상으로 영상 오른쪽 아래의 CC 버튼과 설정 버튼을 눌러 자동 번역 자막을 켤 수 있습니다.)

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/IC0_FRiX-sw" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

파이토치 텐서(Tensor)
-----------------------

영상의 `03:50 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=230s>`__ 부터 따라해보겠습니다.

먼저 파이토치(PyTorch)를 불러옵니다.

"""

import torch

######################################################################
# 몇 가지 기본적인 텐서(tensor) 조작법을 살펴보겠습니다.
# 먼저 텐서를 만드는 몇 가지 방법입니다:
#

z = torch.zeros(5, 3)
print(z)
print(z.dtype)


#########################################################################
# 위에서 0으로 채워진 5x3 행렬(matrix)을 만들고, 데이터 유형(datatype)을
# 조회해보았습니다. 0은 32비트 부동 소수점(floating point) 숫자로, 이는
# PyTorch의 기본값입니다.
#
# 정수(integer)를 쓰고 싶으면 어떻게 할까요? 언제든 기본 값을 재정의할 수
# 있습니다:
#

i = torch.ones((5, 3), dtype=torch.int16)
print(i)


######################################################################
# 기본값을 바꾸면 텐서를 출력할 때 이를 보여주는 것을 확인할 수 있습니다.
#
# 학습 가중치(weight)를 무작위로 초기화하는 것이 일반적이며, 종종
# 결과를 재현하기 위해 의사 난수 생성기(PRNG; Pseudo Random Number Generator)에
# 특정 시드(seed) 값을 사용합니다:
#

torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # 새로운 값

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # 시드 값이 동일하므로 r1 값 재현


#######################################################################
# PyTorch 텐서는 산술 연산을 직관적으로 수행합니다. 비슷한 모양(shape)의
# 텐서는 더하거나 곱하기 등을 할 수 있습니다. 스칼라 값의 연산은 텐서 전체에
# 적용됩니다:
#

ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # 모든 원소들에 2를 곱합니다
print(twos)

threes = ones + twos       # 모양(shape)이 같기 때문에 덧셈이 가능합니다
print(threes)              # 각 원소(element)끼리 더해집니다
print(threes.shape)        # 입력 텐서와 같은 모양(shape)을 갖습니다

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# 아래 주석을 제거하면 런타임 에러(runtime error)가 발생합니다
# r3 = r1 + r2


######################################################################
# 가능한 수리 연산의 예를 몇 개 살펴보겠습니다:
#

r = (torch.rand(2, 2) - 0.5) * 2 # -1부터 1 사이의 값
print('A random matrix, r:')
print(r)

# 일반적인 수리 연산을 지원하며:
print('\nAbsolute value of r:')
print(torch.abs(r))

# 삼각함수 또한 가능합니다:
print('\nInverse sine of r:')
print(torch.asin(r))

# 그리고 행렬식(determinant)나 특이값 분해(signular value decomposition)과 같은
# 선형대수 연산들도 지원합니다:
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# 또한 통계(statistical)나 집계(aggregate) 연산들도 가능합니다:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))


##########################################################################
# GPU에서 병렬 연산을 하는 방법 등, PyTorch 텐서의 성능(power)에 대해서
# 알아야 할 다른 내용들도 있습니다 - 이는 다른 영상에서 더 깊게 다루도록 하겠습니다.
#
# PyTorch 모델(Model)
# --------------------
#
# 영상의 `10:00 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=600s>`__ 부터 따라해보겠습니다.
#
# PyTorch에서 모델(model)을 표현하는 방법을 알아보겠습니다.
#

import torch                     # PyTorch의 모든 것을 위해 불러오기
import torch.nn as nn            # PyTorch 모델의 부모 객체인 torch.nn.Module을 위해 불러오기
import torch.nn.functional as F  # 활성 함수(activation function)를 위해 불러오기


#########################################################################
# .. figure:: /_static/img/mnist.png
#    :alt: le-net-5 diagram
#
# *Figure: LeNet-5*
#
# 위의 그림은 딥러닝(Deep Learning)의 폭발적 성장을 이끈 가장 초기의 합성
# 신경망(convolutional neural net) 중 하나인 LeNet-5을 나타낸 것입니다.
# 이는 손으로 쓴 작은 숫자 이미지(MNIST 데이터셋)를 읽어서, 이미지에 표현된
# 숫자(digit)를 정확하게 분류하도록 만들어졌습니다.
#
# 어떻게 동작하는지 요약하면 다음과 같습니다:
#
# -  C1 계층(layer)은 합성곱 계층(convolutional layer)로, 학습(training) 과정에서
#    입력 이미지를 훑어보고 특징(feature)을 학습합니다. 이미지에서 학습한 특징을
#    어디에서 보았는지를 맵(map)으로 출력합니다. 이 "활성화 맵(activation map)"은
#    S2 계층에서 다운샘플링(downsample)됩니다.
# -  C3 계층은 또다른 합성곱 계층으로, C1 계층의 활성화 맵을 훑어보면서
#    특징들의 *조합(combination)* 을 찾습니다. 이러한 특징 조합의 공간적
#    위치(spatial location)를 표현하는 활성화 맵을 출력하며, 이는 S4 계층에서
#    다운샘플링됩니다.
# -  끝으로, 마지막 부분의 완전히 연결된 F5, F6 및 OUTPUT 계층은 최종 활성화 맵을
#    가져와 10개 숫자를 나타내는 상자 중 하나로 분류하는 *분류기(classifier)* 입니다.
#
# 이 간단한 신경망을 코드로는 어떻게 표현할 수 있을까요?
#

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1개의 입력 이미지 채널(흑백), 6개의 출력 채널, 3x3의 정사각형의 합성곱
        # 커널(convolution kernel)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 아핀(affine) 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6은 이미지의 차원
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 윈도우(window)로 맥스 풀링(max pooling)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 정사각형이면 숫자 하나만 지정할 수 있음
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 배치(batch) 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


############################################################################
# 이 코드를 살펴보면, 위의 그림(diagram)과 구조적으로 유사함을 발견할 수
# 있을 것입니다.
#
# 이것은 일반적인 PyTorch 모델의 구조를 보여줍니다:
#
# -  ``torch.nn.Module`` 을 상속받습니다 - 모듈은 중첩(nested)될 수 있습니다.
#    사실, ``Conv2d`` 와 ``Linear`` 계층 클래스들 또한 ``torch.nn.Module`` 을
#    상속받았습니다.
# -  모델에는 모델의 계층을 실체화(instantiate)하고 필요한 데이터
#    아티팩트(artifact)를 실체화하는 ``__init__()`` 함수가 있습니다.
#    (예. NLP 모델은 어휘집(vocabulary)을 불러옵니다.)
# -  모델에는 실제로 연산이 이뤄지는 ``forward()`` 함수가 있습니다:
#    입력이 신경망 계층들과 다양한 함수를 통과하며 출력을 계산(generate)합니다.
# -  그 외에도, 모델 클래스에는 일반적인 Python 클래스와 마찬가지로 모델의 연산을
#    도와줄 속성과 메소드 등을 추가할 수 있습니다.
#
# 이제 이 객체를 실체화(instantize)하고 샘플 입력을 실행해보겠습니다.
#

net = LeNet()
print(net)                         # 이 객체는 자신에 대해서 무엇을 알려줄까요?

input = torch.rand(1, 1, 32, 32)   # 32x32 흑백 이미지를 위한 대용물(stand-in)
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # forward()를 직접 호출하지 않습니다
print('\nRaw output:')
print(output)
print(output.shape)


##########################################################################
# 위에서 몇 가지 중요한 일들이 발생했습니다:
#
# 먼저, ``LeNet`` 클래스를 실체화하고 ``net`` 객체를 출력했습니다.
# ``torch.nn.Module`` 의 하위 클래스(subclass)는 생성된 계층과 각 계층의 모양과
# 매개변수를 출력합니다. 이는 모델이 어떻게 동작하는지 간단하게 파악할 수 있는
# 편리한 개요를 제공합니다.
#
# 그 아래에서는 색상 채널 1개의 32x32짜리 이미지를 나타내는 가짜(dummy) 입력을
# 만들었습니다. 일반적으로는 이미지 타일을 불러온 뒤 그걸 이러한 모양의 텐서로
# 변환합니다.
#
# 텐서에 추가 차원이 있다는 것을 눈치채셨을 수도 있습니다 - 이는 *배치(batch) 차원*
# 입니다. PyTorch 모델은 모든 데이터가 *묶음(batch)* 으로 다뤄진다고 가정합니다.
# 예를 들어, 위와 같은 이미지 타일 16개짜리 묶음은 ``(16, 1, 32, 32)`` 의 모양을
# 갖습니다. 여기에서는 하나의 이미지만 사용하기 때문에 1개 짜리 묶음인
# ``(1, 1, 32, 32)`` 모양(shape)으로 만들었습니다.
#
# 이후 모델을 함수처럼 호출하여 추론(inference)을 하도록 합니다: ``net(input)``.
# 이 호출의 출력값은 입력이 어떠한 숫자를 표현하는지에 대한 모델의 신뢰도를 나타냅니다.
# (하지만 이 모델은 아직 아무것도 학습하지 ㅇ낳았으므로 출력으로부터 어떠한 신호도
# 기대하지 않는 것이 좋습니다.) ``output`` 의 모양을 살펴보면 배치 차원도 있으며,
# 입력의 배치 차원과 일치함을 알 수 있습니다. 만약 입력 배치로 16개를 전달했다면,
# ``output`` 의 모양은 ``(16, 10)`` 이 될 것입니다.
#
# Dataset과 Dataloader
# ------------------------
#
# 영상의 `14:00 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=840s>`__ 부터 따라해보겠습니다.
#
# 아래에서는 TorchVision에서 바로 다운로드 할 수 있는 오픈 액세스(open-access) 데이터셋들 중
# 하나를 사용하여, 모델에서 사용할 수 있도록 이미지를 변환하는 방법과 DataLoader를 사용하여
# 데이터 묶음을 모델에 공급하는 방법을 보여주고 있습니다.
#
# 가장 먼저 해야 할 일은 입력 이미지를 PyTorch 텐서로 변환하는 것입니다.
#

#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


##########################################################################
# 여기에서는 입력에 대해 2가지 변환 작업을 지정하였습니다:
#
# -  ``transforms.ToTensor()`` 는 Pillow로 이미지를 불러와 PyTorch 텐서로
#    변환합니다.
# -  ``transforms.Normalize()`` 는 평균이 0이고 표준 편차가 0.5가 되도록
#    텐서의 값들을 조절합니다. 대부분의 활성 함수는 x = 0 주변에서 가장 큰
#    변화도(gradient)를 갖고 때문에, 데이터를 가운데로 정렬(centering)하여
#    학습 속도를 빠르게 할 수 있습니다.
#
# 자르기(crop), 가운데로 정렬하기(centering), 회전하기(rotate) 및 반사하기
# (reflect) 등을 포함하여 더 많은 변환 작업들을 사용할 수 있습니다.
#
# 다음으로, CIFAR10 데이터셋의 인스턴스를 생성합니다. 이것은 6종류의 동물들
# (새, 고양이, 사슴, 개, 개구리, 말)과 4종류의 운송수단(비행기, 자동차, 배, 트럭)을
# 합친, 총 10개의 객체 분류(class)를 나타내는 32x32 컬러 이미지 타일 세트입니다:
#

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


##########################################################################
# .. note::
#      위 셀(cell)을 실행하면 데이터셋 다운로드에 시간이 다소 걸릴 수 있습니다.
#
# PyTorch에서 데이터셋을 생성하는 예제입니다. (위의 CIFAR-10과 같은) 다운로드
# 가능한 데이터셋은 ``torch.utils.data.Dataset``의 하위 클래스입니다. PyTorch의
# ``Dataset`` 클래스에는 TorchVision, Torchtext 및 TorchAudio의 다운로드 가능한
# 데이터셋들이 있으며, ``torchvision.datasets.ImageFolder`` 와 같이 폴더에서
# 라벨링된 이미지를 읽어오는 유틸리티 데이터셋 클래스들도 있습니다. 또한
# ``Dataset`` 의 하위 클래스를 직접 만들수도 있습니다.
#
# 데이터셋을 실체화(instantiate)할 때, 몇 가지 지정해야 할 내용이 있습니다:
#
# -  데이터를 저장하려는 파일시스템 경로(path)
# -  학습 시 이 데이터셋을 사용하는지 여부; 대부분의 데이터셋은 학습용과
#    테스트용으로 분할합니다.
# -  이전에 데이터셋을 다운로드한 적 없는 경우 다운로드 할지 여부
# -  데이터에 적용할 변환 작업(transformation)
#
# 데이터셋이 준비되면 이를 ``DataLoader`` 에 제공합니다:
#

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


##########################################################################
# ``Dataset`` 하위 클래스는 데이터에 대한 접근을 감싸고(wrap) 있으며,
# 제공하는 데이터의 종류에 특화되어 있습니다. ``DataLoader`` 는 데이터에 대해
# *아무것도* 모르며 ``Dataset`` 이 제공하는 입력 텐서를 지정한 매개변수를
# 사용하여 묶음(batch)으로 만듭니다.
#
# 위 예제에서는 ``trainset`` 으로부터 4개씩 묶음으로 만들어 무작위의 순서로
# (``shuffle=True``), 2개의 작업자(worker)를 사용하도록 ``DataLoader`` 에
# 요청하였습니다.
#
# ``DataLoader`` 가 제공하는 데이터 배치(batch)를 시각화해보겠습니다:
#

import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # 비정규화(unnormalize)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 임의의 학습 이미지를 가져옵니다
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지를 표시합니다
imshow(torchvision.utils.make_grid(images))
# 정답(label)을 출력합니다
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 위의 셀을 실행하면 4개의 이미지와 각 이미지에 해당하는 정답(label)이
# 표시됩니다.
#
# PyTorch 모델 학습하기
# ---------------------------
#
# 영상의 `17:10 <https://www.youtube.com/watch?v=IC0_FRiX-sw&t=1030s>`__ 부터 따라해보겠습니다.
#
# 지금까지 했던 것들을 모아보고, 모델을 훈련시켜보겠습니다:
#

#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


#########################################################################
# 먼저, 학습과 테스트를 위한 데이터셋이 필요합니다. 이전에 다운로드한 적 없다면,
# 아래 셀을 실행하면 데이터셋이 다운로드될 것입니다. (1분 정도 소요될 수 있습니다.)
#

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


######################################################################
# ``DataLoader`` 의 출력을 점검해보겠습니다:
#

import matplotlib.pyplot as plt
import numpy as np

# 이미지를 표시하는 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 학습용 이미지를 무작위로 몇 개 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 표시
imshow(torchvision.utils.make_grid(images))
# 정답 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


##########################################################################
# 이제 아래 모델을 학습해보곘습니다. 영상 앞쪽에서 얘기했던 LeNet을 3컬러
# 이미지용으로 변형한 것이므로 익숙해보일 수 있습니다.
#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


######################################################################
# 마지막으로 손실 함수(loss function)와 옵티마이저(optimizer)를 준비합니다:
#

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


##########################################################################
# 영상의 앞 부분에서 논의했던 것처럼, 손실 함수는 이상적인 결과와 모델의 예측 값이
# 얼마나 멀리 떨어져있는지를 측정합니다. 교차 엔트로피 손실(cross-entropy loss)은
# 위와 같은 분류 모델에서 사용하는 일반적인 손실함수입니다.
#
# **옵티마이저(optimizer)** 는 학습을 주도합니다. 여기에서는 직관적(straightforward)
# 최적화 알고리즘 중 하나인 *확률적 경사 하강법(stochastic gradient descent)* 를
# 구현하는 옵티마이저를 만들었습니다. 이 알고리즘에서 사용하는 학습률(learning rate, ``lr``)
# 과 모멘텀(momentum)과 같은 매개변수 외에도, 모델의 모든 학습 가능한 가중치의 모음을
# ``net.parameters()`` 로 전달합니다. 이러한 가중치들의 모음을 옵티마이저가 조절합니다.
#
# 마지막으로, 이 모든 것을 학습 루프(loop) 안에서 조합합니다. 아래 셀을 실행하면
# 몇 분 정도 걸릴 수 있습니다.
#

for epoch in range(2):  # 데이터셋을 수 차례 반복

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 입력 값 가져오기
        inputs, labels = data

        # 매개변수 경사도를 0으로 설정
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        if i % 2000 == 1999:    # 매 2000 미니-배치마다 출력
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


########################################################################
# 여기에서는 **훈련 에포크(epoch)를 2번만** 반복합니다.(1번 행) 즉,
# 전체 학습 데이터셋을 2번 반복합니다. 매번 반복할 때마다 **학습 데이터를
# 반복하며** (4번 행) 변형된 입력 이미지와 정답을 묶음으로 제공하는 안쪽
# 반복문이 있습니다.
#
# **변화도를 0으로 만드는 것** (9번 행)도 중요한 단계입니다. 변화도는
# 배치 단위로 누적됩니다; 각 배치 때마다 재설정하지 않으면, 계속 누적되어
# 잘못된 변화도 값을 제공하므로 학습이 불가능해집니다.
#
# 12번 행에서는 배치를 **모델에 제공하여 예측값을 요청** 하였습니다. 이어지는
# 행(13)에서는 (모델 예측값인) ``output`` 과 (정답인) ``labels`` 간의
# 손실을 계산하였습니다.
#
# 14번 행에서는 ``backward()`` 단계를 수행하고, 얼마나 학습해야 하는지를 알려주기
# 위해 변화도를 계산하였습니다.
#
# 15번 행에서는 옵티마이저가 한 학습 단계를 수행합니다. ``backward()`` 호출을
# 통해 얻은 변화도를 사용하여 손실을 줄이는 방향으로 모델의 가중치를 조금씩
# 이동합니다.
#
# 반복문의 나머지 부분은 에포크 횟수나 얼마나 학습이 진행되었는지, 학습 과정에서
# 수집된 손실에 대해 가볍게 보고합니다.
#
# **위 셀을 실행하면** 아래와 같은 내용들이 표시됩니다:
#
# ::
#
#    [1,  2000] loss: 2.235
#    [1,  4000] loss: 1.940
#    [1,  6000] loss: 1.713
#    [1,  8000] loss: 1.573
#    [1, 10000] loss: 1.507
#    [1, 12000] loss: 1.442
#    [2,  2000] loss: 1.378
#    [2,  4000] loss: 1.364
#    [2,  6000] loss: 1.349
#    [2,  8000] loss: 1.319
#    [2, 10000] loss: 1.284
#    [2, 12000] loss: 1.267
#    Finished Training
#
# 손실이 단조 감소하고 있는 것에 유의해주세요. 이는 모델이 학습 데이터셋을 통해
# 성능을 계속 개선해나가고 있음을 나타냅니다.
#
# 마지막 단계로 모델이 데이터셋을 단순히 "외우고" 있는 것이 아니라, *일반적인*
# 학습을 진행 중인지를 확인해봐야 합니다. 이것은 **과적합(overfitting)** 이라고
# 하는데, 일반적으로 데이터셋이 너무 작거나(일반적인 학습을 위한 예시가 부족함),
# 모델이 데이터셋을 잘 따라하게 만드는데 필요한 것보다 많은 학습 매개변수를
# 가지고 있음을 나타냅니다.
#
# 이것이 데이터셋을 학습용과 테스트용으로 나누는 이유입니다. 모델의
# 일반성(generality)을 점검하기 위해, 학습하지 않았던 데이터들로 예측을
# 해보곘습니다:
#

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#########################################################################
# 지금까지 잘 따라오셨다면, 이 시점에서 모델이 대략 50% 가량의 정확도를 보임을
# 알 수 있습니다. 이는 최고 수준(state-of-the-art)은 아니지만,
# 무작위로 찍었을 때 보이는 10%의 정확도보다는 훨씬 낫습니다.
# 이는 모델이 일반적인 학습을 헀음을 보여줍니다.
#
