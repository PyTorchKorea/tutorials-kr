"""
`Introduction <introyt1_tutorial.html>`_ ||
`Tensors <tensors_deeper_tutorial.html>`_ ||
`Autograd <autogradyt_tutorial.html>`_ ||
`Building Models <modelsyt_tutorial.html>`_ ||
**TensorBoard Support** ||
`Training Models <trainingyt.html>`_ ||
`Model Understanding <captumyt.html>`_

파이토치 텐서보드 지원
===========================

아래 `유튜브 <https://www.youtube.com/watch?v=6CEld3hZgqc>`__ 를 따라 하세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/6CEld3hZgqc" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

시작하기 전에
----------------

이 튜토리얼을 진행하기 위해 PyTorch와 TorchVision, Matplotlib, TensorBoard 설치가 필요합니다. 

 ``conda`` 로 설치:

 ``conda install pytorch torchvision -c pytorch`` 
 ``conda install matplotlib tensorboard`` 

 ``pip`` 으로 설치:

 ``pip install torch torchvision matplotlib tensorboard`` 

필요한 라이브러리들이 모두 설치되면, 설치된 파이썬 환경에서 노트북(주피터 노트북)을 재실행하세요.


소개
------------

이번 튜토리얼에서는, Fashion-MNIST 데이터셋을 활용하여 LeNet-5의 변형모델을 학습시킬 것입니다.
 Fashion-MNIST는 10개의 클래스의 다양한 옷을 나타내는 이미지셋입니다.

"""

# 파이토치와 학습에 필요한 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 이미지 데이터셋과 이미지 조작
import torchvision
import torchvision.transforms as transforms

# 이미지 표현
import matplotlib.pyplot as plt
import numpy as np

# 파이토치가 제공하는 텐서보드
from torch.utils.tensorboard import SummaryWriter


######################################################################
# 텐서보드에서 이미지 보기
# -----------------------------
#
# 데이터셋에서 텐서보드로 샘플이미지를 추가하는 것으로 시작해 봅시다:
#

# 데이터셋 수집 및 데이터셋을 사용할 수 있도록 준비
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# ./data폴더에 학습셋과 검증셋 분할하여 저장
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

# 클래스 라벨
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# 이미지를 내부에서 표현할 수 있도록 도와주는 함수
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 4개의 이미치 배치 추출
dataiter = iter(training_loader)
images, labels = dataiter.next()

# 이미지로부터 4개의 그리드를 생성하고 보여주기
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)


########################################################################
# 위에서 TorchVision과 Matplotlib를 사용해 입력 데이터의 미니배치를 보여주기 위한 그리드를 생성하였습니다.
# 아래에서는 ``add_image()``를 사용해 텐서보드에 사용된 이미지를 기록하기 위한``SummaryWriter``를 불러옵니다.
# 또한 디스크에 이미지를 정확하게 쓰기 위한 ``flush()``를 불러옵니다.
#

# 기본적으로 기록할 디렉터리는 runs이지만 구체적일 수록 좋음
# torch.utils.tensorboard.SummaryWriter는 위에서 선언됨
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# 탠서보드 로그 디렉토리에 이미지 데이터 출력하기
writer.add_image('Four Fashion-MNIST Images', img_grid)
writer.flush()

# 텐서보드를 시작하고 보기 위해서 아래의 명령어를 입력:
#   tensorboard --logdir=runs
# ...텐서보드를 열기 위해 http://localhost:6006/ 로 접속


##########################################################################
# 만약 명령어로 텐서보드를 시작하고 새로운 브라우저 탭에서 열 경우(보통은 `localhost:6006 <localhost:6006>`__) Images탭에서 이미지 그리드를 볼 수 있습니다. 

# 학습을 시각화하기 위한 스칼라 그래프화
# --------------------------------------
#
# TensorBoard is useful for tracking the progress and efficacy of your
# training. Below, we’ll run a training loop, track some metrics, and save
# the data for TensorBoard’s consumption.
# 텐서보드는 학습의 진행상황과 잘 되고 있는지 추적하기에 유용합니다.
# 아래에서는 학습을 진행하고 몇가지 매트릭을 추적하고, 텐서보드에 데이터를 저장할 것입니다. 


# Let’s define a model to categorize our image tiles, and an optimizer and
# loss function for training:
# 학습을 위한 loss함수와 옵티마이저, 이미지를 분류하기 위한 모델을 정의해봅시다 :

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
# 이제 1에폭을 학습해 보며 1000배치를 학습할 때마다 검증셋 loss를 평가해봅시다. :
# 

print(len(validation_loader))
for epoch in range(1):  # 데이터셋을 여러번 반복
    running_loss = 0.0

    for i, data in enumerate(training_loader, 0):
        # 기본 학습 과정
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # 100 미니배치마다 실행
            print('Batch {}'.format(i + 1))
            # 검증셋에 대하여 loss확인
            running_vloss = 0.0

            net.train(False) # 검증에는 변화도가 필요없음
            for j, vdata in enumerate(validation_loader, 0):
                vinputs, vlabels = vdata
                voutputs = net(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
            net.train(True) # 다음 학습을 위해 변화도를 다시 True로 변경

            avg_loss = running_loss / 1000
            avg_vloss = running_vloss / len(validation_loader)

            # 배치마다 평균 loss 기록
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch * len(training_loader) + i)

            running_loss = 0.0
print('Finished Training')

writer.flush()


#########################################################################
# 텐서보드로 전환해 SCALARS 탭을 보세요.
#
# 모델의 시각화
# ----------------------
#
# 텐서보드는 모델 안에서 데이터 흐름을 시험하는데 사용할 수도 있습니다.
# 그러기 위해서는, 텐서 보드를 열 때 모델과 샘플 인풋을 인자로 받는 ``add_graph()`` 메소드를 호출하세요.
#

# 다시 이미지의 한번의 미니배치를 가져오기
dataiter = iter(training_loader)
images, labels = dataiter.next()

# add_graph() 는 모델을 통해 샘플 입력을 추적하고, 그래프를 만듬
writer.add_graph(net, images)
writer.flush()


#########################################################################
# 텐서보드로 전환하면 GRAPHS 탭을 볼수 있을 것입니다.
# 모델의 레이어와 데이터 흐름도를 보기 위해 "NET"항목을 더블클릭하세요.
#
# 임베딩을 통한 데이터셋 시각화
# ----------------------------------------
#
# 우리가 사용할 28*28 이미지들은 784차원의 벡터로 모델링 될 수 있습니다.(28 \* 28 = 784). 
# 이것을 낮은 차원에 투영할 수 있습니다.
#  ``add_embedding()`` 메소드는 데이터를 삼차원에 큰 분산으로 투영하고 3D 차트로 보여줄것입니다.
#  ``add_embedding()`` 는 데이터를 삼차원에 큰 분산으로 투영함으로써 자동으로 수행해줍니다.
#
# 아래에서 우리는 데이터의 샘플을 가지고 이러한 임베딩을 생성할 것입니다.
#

# 데이터와 해당 라벨들을 랜덤으로 선택 
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# 데이터의 랜덤셋 추출
images, labels = select_n_random(training_set.data, training_set.targets)

# 각 이미지의 클래스 라벨 추출
class_labels = [classes[label] for label in labels]

# 임베딩 기록
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.flush()
writer.close()


#######################################################################
# 이제 텐서보드로 전환하고 PROJECTOR 탭을 선택한다면 투영된 백터의 3D 표현을 볼 수 있습니다. 모델을 확대하고 회전할 수 있습니다.
# 크고 작은 스케일로 시험해보고 투영화 된 데이터와 군집화된 라벨들에서 패턴을 착을 수 있는지 보세요.
#
# 더 나은 시각화를 위해 아래를 추천합니다. :
#
# - 왼쪽의 “Color by”로 부터 "label" 선택
# - 밝은 색의 이미지를 어두운 배경에서 보고싶으면 상단의 야간 모드 아이콘을 전환 
#
# 다른 자료
# ---------------
#
# 더 많은 정보를 위해 아래를 확인하세요.:
#
# - PyTorch documentation on `torch.utils.tensorboard.SummaryWriter <https://pytorch.org/docs/stable/tensorboard.html?highlight=summarywriter>`__
# - Tensorboard tutorial content in the `PyTorch.org Tutorials <https://tutorials.pytorch.kr/>`__
# - For more information about TensorBoard, see the `TensorBoard
#   documentation <https://www.tensorflow.org/tensorboard>`__
