Pytorch Cheat Sheet

---

# Imports(선언)

## General

.. code-block:: python

```
import torch                                        # 루트 패키지
from torch.utils.data import Dataset, Dataloader    # 데이터셋 표현과 로딩

```

## Neural Network API

.. code-block:: python

```
import torch.autograd as autograd         # 계산 그래프
from torch import Tensor                  # 계산 그래프 안에서 텐서 노드
import torch.nn as nn                     # 신경망(neural networks)
import torch.nn.functional as F           # 계층, 활성화 함수 그 외의 것
import torch.optim as optim               # 옵티마이저 (경사 하강법, ADAM, etc.)
from torch.jit import script, trace       # 하이브리드 프론트엔드 데코레이터 tracing jit

```

참고

`autograd <https://pytorch.org/docs/stable/autograd.html>`**,
`nn <https://pytorch.org/docs/stable/nn.html>`**,
`functional <https://pytorch.org/docs/stable/nn.html#torch-nn-functional>`__,
 `optim <https://pytorch.org/docs/stable/optim.html>`__

## Torchscript and JIT

.. code-block:: python

```
torch.jit.trace()         # 모듈이나 함수 그리고 예제 데이터 인풋을 취하고 
													# 모델이 진행되는 동안 데이터가 마주하는 계산 단계를 추적

@script                   # 추적 중인 코드 내에서 데이터의 흐름을 나타내기 위한 데코레이터

```

참고 :  `Torchscript <https://pytorch.org/docs/stable/jit.html>`__

## ONNX

.. code-block:: python

```
torch.onnx.export(model, dummy data, xxxx.proto)       # 훈련된 모델, 더미 데이터 및 원하는 파일 이름을 
																											 # 사용하여 ONNX 형식 모델을 내보내기

model = onnx.load("alexnet.proto")                     # ONNX 모델 불러오기
onnx.checker.check_model(model)                        # 모델 IR이 잘 형성되었는지 확인
                                                       

onnx.helper.printable_graph(model.graph)               # 읽을 수 있는 그래프 표현을 출력

```

참고 :  `onnx <https://pytorch.org/docs/stable/onnx.html>`__

## Vision

.. code-block:: python

```
from torchvision import datasets, models, transforms     # vision 데이터셋
                                                         # 모델들
                                                         # 변환들

import torchvision.transforms as transforms              # 합성 가능한 변환들

```

참고 : 
`torchvision <https://pytorch.org/docs/stable/torchvision/index.html>`__

## Distributed Training

.. code-block:: python

```
import torch.distributed as dist          # 분산 통신
from multiprocessing import Process       # 메모리 공유 프로세스

```

참고 : `distributed <https://pytorch.org/docs/stable/distributed.html>`__
,
`multiprocessing <https://pytorch.org/docs/stable/multiprocessing.html>`__

# Tensors

## Creation

.. code-block:: python

```
torch.randn(*size)              # N(0,1)에서 서로 독립인 값을 가지는 텐서
torch.[ones|zeros](*size)       # 모두 1이나 0의 값을 가지는 텐서
torch.Tensor(L)                 # 중첩된 리스트 혹은 numpy 배열을 통한 텐서 생성
x.clone()                       # x를 복제
with torch.no_grad():           # Autograd가 더이상 텐서를 추적하지 못하도록 하는 코드
requires_grad=True              # 인수를 참(True)로 설정시, 나중의 미분 계산을 위해 계산 과정을 기록

```

참고 :  `tensor <https://pytorch.org/docs/stable/tensors.html>`__

## Dimensionality

.. code-block:: python

```
x.size()                              # 객체의 차원을 tuple형태로 반환
torch.cat(tensor_seq, dim=0)          # 치수를 따라 텐서를 연결
x.view(a,b,...)                       # x의 크기를 (a,b,...)로 재조정
x.view(-1,a)                          # x를 크기(b,a)로 재조정
x.transpose(a,b)                      # a차원과 b차원 간의 변경
x.permute(*dims)                      # 차원들간의 재배열
x.unsqueeze(dim)                      # 텐서에 축을 추가
x.unsqueeze(dim=2)                    # (a,b,c) 텐서-> (a,b,1,c) 텐서

```

참고 :  `tensor <https://pytorch.org/docs/stable/tensors.html>`__

## Algebra

.. code-block:: python

```
A.mm(B)       # 행렬 간 곱
A.mv(x)       # 행렬-벡터 간 곱
x.t()         # x의 전치행렬 

```

참고 :  `math operations <https://pytorch.org/docs/stable/torch.html?highlight=mm#math-operations>`__

## GPU Usage

.. code-block:: python

```
torch.cuda.is_available                                 # cuda를 체크합니다.
x.cuda()                                                # x의 데이터를 CPU에서 GPU로 옮기고
                                                        # 새로운 객체를 반환

x.cpu()                                                 # x의 데이터를 GPU에서 CPU로 옮기고
                                                        # 새로운 객체를 반환합니다.

if not args.disable_cuda and torch.cuda.is_available(): # 장치 확인 코드
    args.device = torch.device('cuda')                  # 그리고 모듈화
else:                                                   #
    args.device = torch.device('cpu')                   #

net.to(device)                                          # 매개 변수와 버퍼를 장치의 텐서로
                                                        # 재귀적으로 변환
                                                       

mytensor.to(device)                                     # 장치(gpu, cpu)로 텐서들을 복

```

참고 :  `cuda <https://pytorch.org/docs/stable/cuda.html>`__

# Deep Learning

.. code-block:: python

```
nn.Linear(m,n)                                # m개의 뉴런에서 n개의 뉴런으로 연결되는
																							# 완전연결(fully connected) 레이어 

nn.ConvXd(m,n,s)                              # m개의 채널에서 n개의 채널로 연결되는 
                                              # X 차원 컨벌루션(conv)레이어  where X⍷{1,2,3}
                                              # X는 {1,2,3} 중 하나고 커널 사이즈는 s

nn.MaxPoolXd(s)                               # X 차원 pooling 레이어
                                              # 위와 같은 표기법

nn.BatchNorm                                  # batch norm 레이어
nn.RNN/LSTM/GRU                               # 순환(recurrent) 레이어들
nn.Dropout(p=0.5, inplace=False)              # 아무 차원 입력에 대한 드랍아웃(dropout) 레이어 
nn.Dropout2d(p=0.5, inplace=False)            # 2차원 채널별 드롭아웃
nn.Embedding(num_embeddings, embedding_dim)   # 인덱스에서 내장 벡터로 매핑

```

참고 : `nn <https://pytorch.org/docs/stable/nn.html>`__

## Loss Functions

.. code-block:: python

```
nn.X                                  # X에는 BCELoss, CrossEntropyLoss,
                                      # L1Loss, MSELoss, NLLLoss, SoftMarginLoss,
                                      # MultiLabelSoftMarginLoss, CosineEmbeddingLoss,
                                      # KLDivLoss, MarginRankingLoss, HingeEmbeddingLoss
                                      # ,CosineEmbeddingLoss가 있다.

```

참고 :  `loss functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`__

## Activation Functions

.. code-block:: python

```
nn.X                                  # X에는 ReLU, ReLU6, ELU, SELU, PReLU, LeakyReLU,
                                      # Threshold, HardTanh, Sigmoid, Tanh,
                                      # LogSigmoid, Softplus, SoftShrink,
                                      # Softsign, TanhShrink, Softmin, Softmax,
                                      # Softmax2d 또는 LogSoftmax가 있음

```

참고 : `activation functions <https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity>`__

## Optimizers

.. code-block:: python

```
opt = optim.x(model.parameters(), ...)      # 옵티마이저 생성
opt.step()                                  # 가중치(weights) 업데이트
optim.X                                     # X에는 SGD, Adadelta, Adagrad, Adam,
                                            # SparseAdam, Adamax, ASGD,
                                            # LBFGS, RMSProp 또는 Rprop가 있음

```

See `optimizers <https://pytorch.org/docs/stable/optim.html>`__

## Learning rate scheduling

.. code-block:: python

```
scheduler = optim.X(optimizer,...)      # 학습률 스케줄러 생성
scheduler.step()                        # epoch의 시작할 때 학습률 업데이트
optim.lr_scheduler.X                    # X에는 LambdaLR, StepLR, MultiStepLR,
													              # ExponentialLR 또는 ReduceLROnPLateau가 있음

```

참고 : `learning rate scheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`__

# Data Utilities

## Datasets

.. code-block:: python

```
Dataset                    # 데이터셋을 나타내는 추상클래스
TensorDataset              # 텐서의 형태로 라벨링 되어 있는 데이터셋
Concat Dataset             # 데이터셋을 연결시켜주는 클래스

```

참고 : 
`datasets <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`__

## Dataloaders and DataSamplers

.. code-block:: python

```
DataLoader(dataset, batch_size=1, ...)      # 개별 데이터 지점들의 구조에 관계없이 데이터 배치를 불러옴

sampler.Sampler(dataset,...)                # 데이터셋에서 샘플링하는 방법을 다루는 추상 클래스

sampler.XSampler where ...                  # X에는 Sequential, Random, Subset,
                                            # WeightedRandom 또는 Distributed가 있다

```

참고 : 
`dataloader <https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader>`__

## 여기도 참고하세요

- `Deep Learning with PyTorch: A 60 Minute Blitz <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>`__
*([pytorch.org](http://pytorch.org/))*
- `PyTorch Forums <https://discuss.pytorch.org/>`__
*([discuss.pytorch.org](http://discuss.pytorch.org/))*
- `PyTorch for Numpy users <https://github.com/wkentaro/pytorch-for-numpy-users>`__
*([github.com/wkentaro/pytorch-for-numpy-users](http://github.com/wkentaro/pytorch-for-numpy-users))*
