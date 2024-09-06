"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
**TensorBoard Support** ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

PyTorch TensorBoard 지원
===========================

**번역**: `박정은 <https://github.com/Angela-Park-JE/>`_

아래 영상이나 `youtube <https://www.youtube.com/watch?v=6CEld3hZgqc>`_\를 참고하세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/6CEld3hZgqc" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

시작하기에 앞서
----------------

이 튜토리얼을 실행하기 위해서 PyTorch, TorchVision, 
Matplotlib 그리고 TensorBoard를 설치해야 합니다.

``conda`` 사용 시:

.. code-block:: sh

    conda install pytorch torchvision -c pytorch
    conda install matplotlib tensorboard

``pip`` 사용 시:

.. code-block:: sh

    pip install torch torchvision matplotlib tensorboard

한번 종속된 모듈을 설치하고 나서, 
설치한 환경에서 이 노트북을 다시 시작합니다.


개요
------------

이 notebook에서는 Fashion-MNIST 데이터셋에 대해 
변형된 LeNet-5를 학습시킬 것입니다.
Fashion-MNIST는 의류의 종류를 나타내는 10개의 클래스 레이블을 포함하는 
다양한 의류의 타일 이미지 세트입니다.

"""

# PyTorch 모델과 훈련 필수 요소
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 이미지 데이터셋과 이미지 조작
import torchvision
import torchvision.transforms as transforms

# 이미지 시각화
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard 지원
from torch.utils.tensorboard import SummaryWriter

# 만약 Google Colab처럼 TensorFlow가 설치된 환경을 사용 중이라면 
# 아래의 코드를 주석 해제하여 
# TensorBoard 디렉터리에 임베딩을 저장할 때의 버그를 방지하세요.

# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

######################################################################
# TensorBoard에서 이미지 나타내기
# -----------------------------
#
# 먼저, 데이터셋에서 TensorBoard로 샘플 이미지를 추가합니다:
#

# 데이터셋을 모아서 사용 가능하도록 준비하기
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# 훈련과 검증으로 분할하여 각각 ./data에 저장하기
training_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
validation_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)


validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=2)

# 클래스 레이블
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# 인라인 이미지 시각화를 위한 함수
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # 비정규화(unnormalize)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 4개의 이미지로부터 배치 하나를 추출하기
dataiter = iter(training_loader)
images, labels = next(dataiter)

# 이미지를 나타내기 위한 격자 생성
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)


########################################################################
# 위에서 TorchVision과 Matplotlib을 사용하여 
# 입력 데이터의 미니 배치를 시각적으로 배열한 격자를 만들었습니다. 아래에서는 TensorBoard에서 사용될 
# 이미지를 기록하기 위해 ``SummaryWriter`` 의 ``add_image()`` 를 호출하고, 
# 또한 ``flush()`` 를 호출하여 이미지가 즉시 디스크에 기록되도록 합니다.
#

# log_dir 인수 기본값은 "runs"입니다 - 하지만 구체적으로 정하는 것이 좋습니다.
# 위에서 torch.utils.tensorboard.SummaryWriter를 가져왔습니다.
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# TensorBoard 로그 디렉터리에 이미지 데이터 쓰기(write)
writer.add_image('Four Fashion-MNIST Images', img_grid)
writer.flush()

# 눈으로 보기 위해서는 커맨드 라인에서 TensorBoard를 시작하세요:
#   tensorboard --logdir=runs
# ...그런 다음 브라우저에서 http://localhost:6006/ 를 열어보세요.


##########################################################################
# 만약 TensorBoard를 커맨드 라인에서 구동시켜 
# 그것을 새 브라우저 탭(보통 `localhost:6006 <localhost:6006>`__)에서 열었다면, 
# IMAGES 탭에서 이미지 격자를 확인할 수 있을 것입니다.
#
# 훈련 시각화를 위한 스칼라 그래프 그리기
# --------------------------------------
#
# TensorBoard는 훈련 진행 과정과 효과를 추적하기에 
# 유용합니다. 아래에서 훈련 루프를 실행하고 몇몇 지표를 추적하며 
# TensorBoard에서 사용할 데이터를 저장할 것입니다.
#
# 이미지 타일을 분류할 모델과 옵티마이저 
# 그리고 훈련의 손실 함수를 정의해 봅시다:
#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


##########################################################################
# 이제 단일 에폭을 훈련하고, 매 1000 배치마다 훈련 셋과 검증 셋의
# 손실을 평가해 봅니다:
#

print(len(validation_loader))
for epoch in range(1):  # 데이터셋을 여러 번 반복
    running_loss = 0.0

    for i, data in enumerate(training_loader, 0):
        # 기본 훈련 루프
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # 매 1000 미니 배치마다...
            print('Batch {}'.format(i + 1))
            # 검증 셋과 비교
            running_vloss = 0.0

            # 평가 모드에서는 일부 모델의 특정 작업을 생략할 수 있습니다 예시: 드롭아웃 레이어
            net.train(False) # 평가 모드로 전환, 예시: 정규화(regularisation) 끄기
            for j, vdata in enumerate(validation_loader, 0):
                vinputs, vlabels = vdata
                voutputs = net(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
            net.train(True) # 훈련 모드로 돌아가기, 예시: 정규화 켜기

            avg_loss = running_loss / 1000
            avg_vloss = running_vloss / len(validation_loader)

            # 배치별 평균 실행 손실을 기록
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch * len(training_loader) + i)

            running_loss = 0.0
print('Finished Training')

writer.flush()


#########################################################################
# 열린 TensorBoard로 전환하여 SCALARS탭을 살펴보세요.
#
# 모델 시각화하기
# ----------------------
#
# TensorBoard는 모델 내 데이터 흐름을 검사하는 데에도 유용합니다.
# 이를 위해, 모델과 샘플 입력을 이용해 ``add_graph()`` 메소드를
# 호출합니다:
#

# 다시, 이미지의 미니 배치 하나를 가져옵니다.
dataiter = iter(training_loader)
images, labels = next(dataiter)

# add_graph()는 샘플 입력이 모델을 통과하는 과정을 추적하고,
# 이를 그래프로 시각화합니다.
writer.add_graph(net, images)
writer.flush()


#########################################################################
# TensorBoard로 전환하면, GRAPHS 탭이 보일 것입니다.
# “NET” 노드를 더블 클릭하여 모델 내 계층과 데이터 흐름을
# 확인하세요.
#
# Visualizing Your Dataset with Embeddings
# ----------------------------------------
#
# The 28-by-28 image tiles we’re using can be modeled as 784-dimensional
# vectors (28 \* 28 = 784). It can be instructive to project this to a
# lower-dimensional representation. The ``add_embedding()`` method will
# project a set of data onto the three dimensions with highest variance,
# and display them as an interactive 3D chart. The ``add_embedding()``
# method does this automatically by projecting to the three dimensions
# with highest variance.
#
# Below, we’ll take a sample of our data, and generate such an embedding:
#

# Select a random subset of data and corresponding labels
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# Extract a random subset of data
images, labels = select_n_random(training_set.data, training_set.targets)

# get the class labels for each image
class_labels = [classes[label] for label in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.flush()
writer.close()


#######################################################################
# Now if you switch to TensorBoard and select the PROJECTOR tab, you
# should see a 3D representation of the projection. You can rotate and
# zoom the model. Examine it at large and small scales, and see whether
# you can spot patterns in the projected data and the clustering of
# labels.
#
# For better visibility, it’s recommended to:
#
# - Select “label” from the “Color by” drop-down on the left.
# - Toggle the Night Mode icon along the top to place the
#   light-colored images on a dark background.
#
# Other Resources
# ---------------
#
# For more information, have a look at:
#
# - PyTorch documentation on `torch.utils.tensorboard.SummaryWriter <https://pytorch.org/docs/stable/tensorboard.html?highlight=summarywriter>`__
# - Tensorboard tutorial content in the `PyTorch.org Tutorials <https://tutorials.pytorch.kr/>`__
# - For more information about TensorBoard, see the `TensorBoard
#   documentation <https://www.tensorflow.org/tensorboard>`__
