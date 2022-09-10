# -*- coding: utf-8 -*-
"""
Ray Tune을 이용한 하이퍼파라미터 튜닝
===================================
**번역**: `심형준 <http://github.com/95hj>`_
하이퍼파라미터 튜닝은 보통의 모델과 매우 정확한 모델간의 차이를 만들어 낼 수 있습니다. 
종종 다른 학습률(Learnig rate)을 선택하거나 layer size를 변경하는 것과 같은 간단한 작업만으로도 모델 성능에 큰 영향을 미치기도 합니다.
다행히, 최적의 매개변수 조합을 찾는데 도움이 되는 도구가 있습니다.
`Ray Tune <https://docs.ray.io/en/latest/tune.html>`_ 은 분산 하이퍼파라미터 튜닝을 위한 업계 표준 도구입니다. 
Ray Tune은 최신 하이퍼파라미터 검색 알고리즘을 포함하고 TensorBoard 및 기타 분석 라이브러리와 통합되며 기본적으로
`Ray' 의 분산 기계 학습 엔진
<https://ray.io/>`_ 을 통해 교육을 지원합니다.
이 튜토리얼은 Ray Tune을 파이토치 학습 workflow에 통합하는 방법을 알려줍니다.
CIFAR10 이미지 분류기를 훈련하기 위해 `파이토치 문서에서 이 튜토리얼을 <https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html>`_ 확장할 것입니다.
아래와 같이 약간의 수정만 추가하면 됩니다.
1. 함수에서 데이터 로딩 및 학습 부분을 감싸두고,
2. 일부 네트워크 파라미터를 구성 가능하게 하고,
3. 체크포인트를 추가하고 (선택 사항),
4. 모델 튜닝을 위한 검색 공간을 정의합니다.
|
이 튜토리얼을 실행하기 위해 아래의 패키지가 설치되어 있는지 확인하십시오.
-  ``ray[tune]``: 배포된 하이퍼파라미터 튜닝 라이브러리
-  ``torchvision``: 데이터 트랜스포머의 경우
설정 / Imports
---------------
import들로 시작합니다.
"""
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

######################################################################
# 대부분의 import들은 파이토치 모델을 빌드하는데 필요합니다. 
# 마지막 세 개의 import들만 Ray Tune을 사용하기 위한 것입니다.
#
# Data loaders
# ------------
# data loader를 자체 함수로 감싸두고 전역 데이터 디렉토리로 전달합니다. 
# 이런 식으로 서로 다른 실험들 간에 데이터 디렉토리를 공유할 수 있습니다.


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

######################################################################
# 구성 가능한 신경망
# ---------------------------
# 구성 가능한 파라미터만 튜닝이 가능합니다. 
# 이 예시를 통해 fully connected layer 크기를 지정할 수 있습니다.


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

######################################################################
# 학습 함수
# ------------------
# 흥미롭게 하기 위해 `파이토치 문서에서 <https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html>`_ 
# 예제에 일부를 변경하여 소개합니다.
#
# 훈련 스크립트를 ``train_cifar(config, checkpoint_dir=None, data_dir=None)`` 함수로 감싸둡니다. 
# 짐작할 수 있듯이, ``config`` 매개변수는 훈련할 하이퍼파라미터를 받습니다. ``checkpoint_dir`` 매개변수는 체크포인트를
# 복원하는 데 사용됩니다. ``data_dir`` 은 데이터를 읽고 저장하는 디렉토리를 지정하므로, 
# 여러 실행들이 동일한 데이터 소스를 공유할 수 있습니다.
#
# .. code-block:: python
#
#     net = Net(config["l1"], config["l2"])
#
#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(
#             os.path.join(checkpoint_dir, "checkpoint"))
#         net.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)
#
# 또한, 옵티마이저의 학습률(learning rate)을 구성할 수 있습니다.
#
# .. code-block:: python
#
#     optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
#
# 또한 학습 데이터를 학습 및 검증 세트로 나눕니다. 따라서 데이터의 80%는 모델 학습에 사용하고, 
# 나머지 20%에 대해 유효성 검사 및 손실을 계산합니다. 학습 및 테스트 세트를 반복하는 배치 크기도 구성할 수 있습니다.
#
# DataParallel을 이용한 GPU(다중)지원 추가
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 이미지 분류는 GPU를 사용할 때 이점이 많습니다. 운좋게도 Ray Tune에서 파이토치의 추상화를 계속 사용할 수 있습니다. 
# 따라서 여러 GPU에서 데이터 병렬 훈련을 지원하기 위해 모델을 ``nn.DataParallel`` 으로 감쌀 수 있습니다.
#
# .. code-block:: python
#
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#     net.to(device)
#
# ``device`` 변수를 사용하여 사용 가능한 GPU가 없을 때도 학습이 가능한지 확인합니다. 
# 파이토치는 다음과 같이 데이터를 GPU메모리에 명시적으로 보내도록 요구합니다.
#
# .. code-block:: python
#
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#
# 이 코드는 이제 CPU들, 단일 GPU 및 다중 GPU에 대한 학습을 지원합니다. 
# 특히 Ray는 `부분GPU <https://docs.ray.io/en/master/using-ray-with-gpus.html#fractional-gpus>`_ 도 지원하므로 
# 모델이 GPU 메모리에 적합한 상황에서는 테스트 간에 GPU를 공유할 수 있습니다. 이는 나중에 다룰 것입니다.
#
# Ray Tune과 소통하기
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 가장 흥미로운 부분은 Ray Tune과의 소통입니다.
#
# .. code-block:: python
#
#     with tune.checkpoint_dir(epoch) as checkpoint_dir:
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#     tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
#
# 여기서 먼저 체크포인트를 저장한 다음 일부 메트릭을 Ray Tune에 다시 보냅니다. 특히, validation loss와 accuracy를 
# Ray Tune으로 다시 보냅니다. 그 후 Ray Tune은 이러한 메트릭을 사용하여 최상의 결과를 유도하는 하이퍼파라미터 구성을 
# 결정할 수 있습니다. 이러한 메트릭들은 또한 리소스 낭비를 방지하기 위해 성능이 좋지 않은 실험을 조기에 중지하는 데 사용할 수 있습니다.
#
# 체크포인트 저장은 선택사항이지만 `Population Based Training <https://docs.ray.io/en/master/tune/tutorials/tune-advanced-tutorial.html>`_ 
# 과 같은 고급 스케줄러를 사용하려면 필요합니다. 또한 체크포인트를 저장하면 나중에 학습된 모델을 로드하고 평가 세트(test set)에서 검증할 수 있습니다.
#
# Full training function
# ~~~~~~~~~~~~~~~~~~~~~~
#
# 전체 코드 예제는 다음과 같습니다.


def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

######################################################################
# 보다시피, 대부분의 코드는 원본 예제에서 직접 적용되었습니다.
#
# Test set 정확도(accuracy)
# -----------------
# 일반적으로 머신러닝 모델의 성능은 모델 학습에 사용되지 않은 데이터를 사용해 테스트합니다. 
# Test set 또한 함수로 감싸둘 수 있습니다.


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

######################################################################
# 이 함수는 또한 ``device`` 파라미터를 요구하므로, test set 평가를 GPU에서 수행할 수 있습니다.
#
# 검색 공간 구성
# ----------------------------
# 마지막으로 Ray Tune의 검색 공간을 정의해야 합니다. 예시는 아래와 같습니다.
#
# .. code-block:: python
#
#     config = {
#         "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#
# ``tune.sample_from()`` 함수를 사용하면 고유한 샘플 방법을 정의하여 하이퍼파라미터를 얻을 수 있습니다. 
# 이 예제에서 ``l1`` 과 ``l2`` 파라미터는 4와 256 사이의 2의 거듭제곱이어야 하므로 4, 8, 16, 32, 64, 128, 256입니다. 
# ``lr`` (학습률)은 0.0001과 0.1 사이에서 균일하게 샘플링 되어아 합니다. 마지막으로, 배치 크기는 2, 4, 8, 16중에서 선택할 수 있습니다.
#
# 각 실험에서, Ray Tune은 이제 이러한 검색 공간에서 매개변수 조합을 무작위로 샘플링합니다. 
# 그런 다음 여러 모델을 병렬로 훈련하고 이 중에서 가장 성능이 좋은 모델을 찾습니다. 또한 성능이 좋지 않은 실험을 조기에 종료하는 ``ASHAScheduler`` 를 사용합니다.
#
# 상수 ``data_dir`` 파라미터를 설정하기 위해 ``functools.partial`` 로 ``train_cifar`` 함수를 감싸둡니다. 또한 각 실험에 사용할 수 있는 자원들(resources)을 Ray Tune에 알릴 수 있습니다.
#
# .. code-block:: python
#
#     gpus_per_trial = 2
#     # ...
#     result = tune.run(
#         partial(train_cifar, data_dir=data_dir),
#         resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         checkpoint_at_end=True)
#
# 파이토치 ``DataLoader`` 인스턴스의 ``num_workers`` 을 늘리기 위해 CPU 수를 지정하고 사용할 수 있습니다. 
# 각 실험에서 선택한 수의 GPU들은 파이토치에 표시됩니다. 실험들은 요청되지 않은 GPU에 액세스할 수 없으므로 같은 자원들을 사용하는 중복된 실험에 대해 신경쓰지 않아도 됩니다.
#
# 부분 GPUs를 지정할 수도 있으므로, ``gpus_per_trial=0.5`` 와 같은 것 또한 가능합니다. 이후 각 실험은 GPU를 공유합니다. 사용자는 모델이 여전히 GPU메모리에 적합한지만 확인하면 됩니다.
#
# 모델을 훈련시킨 후, 가장 성능이 좋은 모델을 찾고 체크포인트 파일에서 학습된 모델을 로드합니다. 이후 test set 정확도(accuracy)를 얻고 모든 것들을 출력하여 확인할 수 있습니다.
#
# 전체 주요 기능은 다음과 같습니다.


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)


######################################################################
# 코드를 실행하면 결과는 다음과 같습니다.
#
# ::
#
#     Number of trials: 10 (10 TERMINATED)
#     +-----+------+------+-------------+--------------+---------+------------+--------------------+
#     | ... |   l1 |   l2 |          lr |   batch_size |    loss |   accuracy | training_iteration |
#     |-----+------+------+-------------+--------------+---------+------------+--------------------|
#     | ... |   64 |    4 | 0.00011629  |            2 | 1.87273 |     0.244  |                  2 |
#     | ... |   32 |   64 | 0.000339763 |            8 | 1.23603 |     0.567  |                  8 |
#     | ... |    8 |   16 | 0.00276249  |           16 | 1.1815  |     0.5836 |                 10 |
#     | ... |    4 |   64 | 0.000648721 |            4 | 1.31131 |     0.5224 |                  8 |
#     | ... |   32 |   16 | 0.000340753 |            8 | 1.26454 |     0.5444 |                  8 |
#     | ... |    8 |    4 | 0.000699775 |            8 | 1.99594 |     0.1983 |                  2 |
#     | ... |  256 |    8 | 0.0839654   |           16 | 2.3119  |     0.0993 |                  1 |
#     | ... |   16 |  128 | 0.0758154   |           16 | 2.33575 |     0.1327 |                  1 |
#     | ... |   16 |    8 | 0.0763312   |           16 | 2.31129 |     0.1042 |                  4 |
#     | ... |  128 |   16 | 0.000124903 |            4 | 2.26917 |     0.1945 |                  1 |
#     +-----+------+------+-------------+--------------+---------+------------+--------------------+
#
#
#     Best trial config: {'l1': 8, 'l2': 16, 'lr': 0.00276249, 'batch_size': 16, 'data_dir': '...'}
#     Best trial final validation loss: 1.181501
#     Best trial final validation accuracy: 0.5836
#     Best trial test set accuracy: 0.5806
#
# 대부분의 실험은 자원 낭비를 막기 위해 일찍 중단되었습니다. 가장 좋은 결과를 얻은 실험은 58%의 정확도를 달성했으며, 이는 테스트 세트에서 확인할 수 있습니다.
#
# 이것이 전부입니다! 이제 파이토치 모델의 매개변수를 조정할 수 있습니다.
#
