(베타) 컴퓨터 비전 튜토리얼을 위한 양자화된 전이학습(Quantized Transfer Learning)
============================================================================================

.. tip::
   이 튜토리얼을 최대한 활용하시려면, 다음의 링크를 이용하시길 추천합니다.
   `Colab 버전 <https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/quantized_transfer_learning_tutorial.ipynb>`_.
   이를 통해 아래에 제시된 정보로 실험을 해 볼 수 있습니다.

**Author**: `Zafar Takhirov <https://github.com/z-a-f>`_
**Reviewed by**: `Raghuraman Krishnamoorthi <https://github.com/raghuramank100>`_
**Edited by**: `Jessica Lin <https://github.com/jlin27>`_
**번역**: `정재민 <https://github.com/jjeamin>`_

이 튜토리얼은 `Sasank Chilamkurthy <https://chsasank.github.io/>`_ 가 작성한
:doc:`/beginner/transfer_learning_tutorial` 을 기반으로 합니다.

전이학습(Transfer learning)은 다른 데이터셋에 적용하기 위해서 미리 학습된 모델을 사용하는 기술을 말합니다.
전이학습을 사용하는 2가지 주요 방법이 있습니다.


1. **고정 된 특징 추출기로써 ConvNet**: 여기서는 마지막 몇개의 계층(일명 “헤드(the head)”, 일반적으로 완전히 연결된 계층)
   을 제외하고 네트워크의 모든 매개 변수 가중치를 `“고정(freeze)” <https://arxiv.org/abs/1706.04983>`_ 합니다.
   마지막 계층은 임의의 가중치로 초기화된 새로운 계층으로 대체되며 오직 이 계층만 학습됩니다.


2. **ConvNet 미세조정(Finetuning)**: 랜덤 초기화 대신, 미리 학습된 네트워크를 이용하여 모델을 초기화합니다.
   이후 평소처럼 학습을 진행하지만 다른 데이터셋을 사용합니다.
   평소처럼 학습이 진행되지만 다른 데이터셋을 사용합니다.

   출력의 수가 다를 수 있기 때문에, 일반적으로 신경망에서 헤드(또는 그 일부)는 교체됩니다.
   이 방법에서는 학습률을 더 작은 수로 설정하는 것이 일반적입니다.
   이는 네트워크가 이미 학습되었기 때문이며 새로운 데이터셋으로 "미세조정(finetuning)"하려면 약간의 변경만이 필요합니다.


또한 위의 두 방법을 결합할 수도 있습니다.
먼저 특징 추출기를 고정(freeze)하고 헤드(the head)를 학습시킵니다.
그런 다음 특징 추출기(또는 그 일부)를 고정해제(unfreeze)하고 학습률을
더 작은 수로 설정한 다음 학습을 계속할 수 있습니다.


이번 파트에서는 첫번째 방법을 사용해 양자화된 모델로 특징을 추출해봅시다.


파트 0. 요구사항
---------------------

전이 학습(transfer learning)을 시작하기 전에,
데이터 불러오기 / 시각화와 같은 "요구사항(prerequisites)"을 검토하겠습니다.

.. code:: python

    # Imports
    import copy
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import time

    plt.ion()

Nightly Build 설치하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


PyTorch의 베타(beta)를 사용할 것이므로 최신 버전의 ``torch`` 와 ``torchvision`` 을 설치하는 것을 권장합니다.
로컬(local) 설치에 대한 최신 지침은 `여기 <https://pytorch.org/get-started/locally/>`_ 에서 찾을 수 있습니다.
예를 들어 GPU 지원 없이 설치하려면 :


.. code:: shell

   pip install numpy
   pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
   # CUDA 지원은 https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html를 사용하세요.


데이터 불러오기
~~~~~~~~~~~~~~~~~~

.. note :: 이번 섹션은 원본 전이학습(Transfer Learning) 튜토리얼과 동일합니다.

``torchvision`` 과 ``torch.utils.data`` 패키지를 사용하여 데이터를 불러옵니다.

여기서 풀고자 하는 문제는 이미지로부터 **개미** 와 **벌** 을 분류하는 것입니다.
이 데이터셋은 개미와 벌에 대해 각각 120장의 학습용 이미지, 75개의 검증용 이미지를 포함합니다.
이는 일반화하기에는 아주 작은 데이터셋입니다.
하지만 우리는 전이학습(Transfer Learning)을 사용하기 때문에, 일반화를 꽤 잘 할 수 있을 것입니다.

이 데이터셋은 imagenet의 아주 작은 일부입니다.

.. note :: `여기 <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_ 에서 데이터를 다운로드 받아 ``data`` 디렉토리에 압축을 푸세요.


.. code:: python

    import torch
    from torchvision import transforms, datasets

    # 학습을 위한 데이터 보강(Data augmentation)과 정규화
    # 검증을 위한 정규화
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                  shuffle=True, num_workers=8)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


일부 이미지 시각화하기
~~~~~~~~~~~~~~~~~~~~~~

데이터 보강을 이해하기 위해 일부 학습용 이미지를 시각화 해보겠습니다.

.. code:: python

    import torchvision

    def imshow(inp, title=None, ax=None, figsize=(5, 5)):
      """Imshow for Tensor."""
      inp = inp.numpy().transpose((1, 2, 0))
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      inp = std * inp + mean
      inp = np.clip(inp, 0, 1)
      if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
      ax.imshow(inp)
      ax.set_xticks([])
      ax.set_yticks([])
      if title is not None:
        ax.set_title(title)

    # 학습 데이터의 배치를 얻습니다.
    inputs, classes = next(iter(dataloaders['train']))

    # 배치로부터 격자 형태의 이미지를 만듭니다.
    out = torchvision.utils.make_grid(inputs, nrow=4)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    imshow(out, title=[class_names[x] for x in classes], ax=ax)


모델 학습을 위한 지원 함수
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

다음은 모델을 학습하기 위한 일반 함수 입니다.

- 학습률(learning rate)을 관리합니다(schedules).
- 최적의 모델을 저장합니다.

.. code:: python

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
      """
      Support function for model training.
      모델 학습을 위한 지원 함수
      매개변수:
        model: 학습할 모델
        criterion: 최적화 기준(손실)
        optimizer: 학습에 사용할 옵티마이저
        scheduler: ``torch.optim.lr_scheduler`` 의 인스턴스
        num_epochs: 에폭의 수
        device: 학습을 동작시킬 장치. 'cpu' 또는 'cuda'여야 합니다.
      """
      since = time.time()

      best_model_wts = copy.deepcopy(model.state_dict())
      best_acc = 0.0

      for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭에는 학습 및 검증 단계가 있습니다.
        for phase in ['train', 'val']:
          if phase == 'train':
            model.train()  # 모델을 학습 모드로 설정하기
          else:
            model.eval()   # 모델을 평가 모드로 설정하기

          running_loss = 0.0
          running_corrects = 0

          # 데이터 반복하기
          for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 매개 변수 기울기를 0으로 설정하기
            optimizer.zero_grad()

            # 순전파
            # 학습 동안만 연산 기록을 추적하기
            with torch.set_grad_enabled(phase == 'train'):
              outputs = model(inputs)
              _, preds = torch.max(outputs, 1)
              loss = criterion(outputs, labels)

              # 역전파 + 학습 단계에서만 최적화
              if phase == 'train':
                loss.backward()
                optimizer.step()

            # 통계 보기
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
          if phase == 'train':
            scheduler.step()

          epoch_loss = running_loss / dataset_sizes[phase]
          epoch_acc = running_corrects.double() / dataset_sizes[phase]

          print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

          # 모델 복사하기
          if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

      time_elapsed = time.time() - since
      print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
      print('Best val Acc: {:4f}'.format(best_acc))

      # 최적의 모델 가중치 불러오기
      model.load_state_dict(best_model_wts)
      return model


모델 예측을 시각화하기 위한 지원 함수
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

일부 이미지에 대한 예측을 출력하는 일반 함수

.. code:: python

    def visualize_model(model, rows=3, cols=3):
      was_training = model.training
      model.eval()
      current_row = current_col = 0
      fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

      with torch.no_grad():
        for idx, (imgs, lbls) in enumerate(dataloaders['val']):
          imgs = imgs.cpu()
          lbls = lbls.cpu()

          outputs = model(imgs)
          _, preds = torch.max(outputs, 1)

          for jdx in range(imgs.size()[0]):
            imshow(imgs.data[jdx], ax=ax[current_row, current_col])
            ax[current_row, current_col].axis('off')
            ax[current_row, current_col].set_title('predicted: {}'.format(class_names[preds[jdx]]))

            current_col += 1
            if current_col >= cols:
              current_row += 1
              current_col = 0
            if current_row >= rows:
              model.train(mode=was_training)
              return
        model.train(mode=was_training)


파트 1. 양자화된 특징 추출기(Quantized Feature Extractor)를 기반으로 사용자 지정 분류기 훈련하기
---------------------------------------------------------------------------------------------------

이번 섹션에서는 “고정된(frozen)” 양자화 특징 추출기를 사용하고 그 위에 사용자 지정 분류기 헤드를
학습합니다. 부동 소수점 모델과 다르게 양자화된 모델에는 학습 가능한 매개 변수가 없으므로
requires_grad = False를 설정할 필요가 없습니다. 자세한 내용은 `설명서 <https://pytorch.org/docs/stable/quantization.html>`_ 를 참조하세요.

미리 학습된 모델을 불러옵니다: 이번 예제에서는 `ResNet-18 <https://pytorch.org/hub/pytorch_vision_resnet/>`_ 을 사용할 것입니다.

.. code:: python

    import torchvision.models.quantization as models

    # 나중에 사용할 수 있게 `fc` 에 필터의 수가 필요합니다.
    # 여기서 각 출력 샘플의 크기는 2로 설정합니다.
    # 또한, nn.Linear(num_ftrs, len(class_names))로 일반화 할 수 있습니다.
    model_fe = models.resnet18(pretrained=True, progress=True, quantize=True)
    num_ftrs = model_fe.fc.in_features


이 시점에서 미리 학습된 모델을 수정해야 합니다. 모델의 시작과 끝에는 양자화/역양자화 블록이 있습니다.
그러나 특징 추출기만 사용하기 때문에 역양자화(dequantization) 계층은 선형 계층(헤드) 바로 전으로 이동시켜야 합니다.
가장 쉬운 방법은 모델을 ``nn.Sequential`` 모듈로 감싸는 것입니다.

첫번째 단계는 ResNet 모델에서 특징 추출기를 분리하는 것입니다.
이 예제에서는 ``fc`` 를 제외한 모든 계층을 특징 추출기로 사용해야 하지만, 실제로는 필요한 만큼 많은 부분을 사용할 수 있습니다.
이것은 합성곱 계층 중 일부를 교체하려는 경우에도 유용합니다.

.. note:: 양자화 모델에서 특징 추출기를 분리할 때 양자화를 유지하려는 부분의 시작과 끝에 수동으로 양자화/역양자화를 배치해야 합니다.

아래 함수는 사용자 지정 헤드로 모델을 생성하는 함수입니다.

.. code:: python

    from torch import nn

    def create_combined_model(model_fe):
      # 1 단계. 특징 추출기를 분리합니다.
      model_fe_features = nn.Sequential(
        model_fe.quant,  # Quantize the input
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant,  # 출력을 역양자화하기
      )

      # 2 단계. 새로운 "헤드(head)"를 만듭니다.
      new_head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 2),
      )

      # 3 단계. 결합하고 양자 스텁(stubs)을 잊으면 안됩니다.
      new_model = nn.Sequential(
        model_fe_features,
        nn.Flatten(1),
        new_head,
      )
      return new_model

.. warning:: 현재 양자화된 모델은 CPU에서만 실행할 수 있습니다.
  그러나 모델의 양자화 되지 않은 부분은 GPU로 보낼 수 있습니다.

.. code:: python

    import torch.optim as optim
    new_model = create_combined_model(model_fe)
    new_model = new_model.to('cpu')

    criterion = nn.CrossEntropyLoss()

    # 헤드(the head)만 훈련 한다는 점을 유의하세요
    optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

    # 7 에폭마다 0.1배씩 학습률이 감소
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


학습과 평가
~~~~~~~~~~~~~~~~~~

이 단계는 CPU에서 약 15 ~ 25분 걸립니다. 양자화된 모델은 CPU에서만 실행되기 때문에
GPU에서는 훈련을 실행할 수 없습니다.

.. code:: python

    new_model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=25, device='cpu')

    visualize_model(new_model)
    plt.tight_layout()


파트 2. 양자화 가능한 모델 미세조정(Finetuning)
--------------------------------------------------------

이번 파트에서는 전이학습(Transfer Learning)을 사용하여 특징 추출기(Feature Extractor)를
미세조정(Finetuning) 합니다. 파트 1과 2 모두에서 특징 추출기는 양자화됩니다. 차이점은 파트 1에서
미리 학습 된 양자화 모델을 사용합니다. 이번 파트에서, 우리는 관심있는 데이터셋으로 미세조정(Finetuning)한 후
양자화된 특징 추출기를 생성하므로, 양자화의 장점을 가지면서 전이 학습(Transfer Learning)으로 더 나은 정확도를
얻을 수 있습니다. 특정한 예제에서는 학습용 셋은 매우 작기 때문에(120개의 이미지) 전체 모델을
미세조정(Finetuning)하는 장점이 불분명 합니다. 그러나 여기에 표시된 절차는 더욱 더 큰 데이터셋을 사용한 전이 학습(Transfer Learning)의
정확도를 향상시킵니다.

미리 학습된 특징 추출기는 양자화가 가능해야 합니다.
양자화가 가능한지 확인하기 위해서 다음 단계를 수행하세요:

   1. ``torch.quantization.fuse_modules`` 를 사용하여 ``(Conv, BN, ReLU)`` ,
      ``(Conv, BN)``, 그리고 ``(Conv, ReLU)`` 를 융합합니다.
   2. 특징 추출기를 사용자 지정 헤드와 연결합니다. 이를 위해서 특징 추출기의 출력을 역으로 양자화 해야합니다.
   3. 특징 추출기의 적합한 위치에 가짜 양자화 모듈을 삽입하여 학습하는 동안에 양자화를 모방합니다.

(1) 단계의 경우 멤버 메서드(member method) ``fuse_model`` 이 있는
``torchvision/models/quantization`` 의 모델을 사용합니다.
이 함수는 모든 ``conv`` , ``bn`` , 그리고 ``relu`` 모듈을 통합합니다.
사용자 지정 모델의 경우, 수동으로 통합할 모듈의 목록과 함께 ``torch.quantization.fuse_modules`` API를 호출해야합니다.


(2) 단계는 이전 섹션에서 사용한 ``create_combined_model`` 함수에 의해서 수행됩니다.


(3) 단계는 가짜 양자화 모듈을 삽입하는 ``torch.quantization.prepare_qat`` 를 사용하여 수행됩니다.


(4) 단계로 모델을 "미세조정(Finetuning)"한 후, 완전하게 양자화된 버전으로 변환(5단계) 할 수 있습니다.


미세조정(Finetuning) 모델을 양자화된 모델로 변환하려면 ``torch.quantization.convert`` 함수를
호출 할 수 있습니다. (이 경우 특징 추출기만 양자화 됩니다.)


.. note:: 랜덤 초기화 때문에 여러분의 결과가 튜토리얼에 표시된 결과와 다를 수 있습니다.

.. code:: python

    # `quantize=False` 를 주목하세요
    model = models.resnet18(pretrained=True, progress=True, quantize=False)
    num_ftrs = model.fc.in_features

    # 1 단계
    model.train()
    model.fuse_model()
    # 2 단계
    model_ft = create_combined_model(model)
    model_ft[0].qconfig = torch.quantization.default_qat_qconfig  # Use default QAT configuration
    # 3 단계
    model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)


모델 미세조정
~~~~~~~~~~~~~~~~~~~~

현재 튜토리얼에서는 전체 모델이 미세조정 되었습니다.
일반적으로 이것은 더 높은 정확도로 이어질 것입니다.
그러나 여기서는 크기가 작은 학습용 데이터셋을 사용했기 때문에 결국 과적합하게 됩니다.


4 단계. 모델 미세조정하기

.. code:: python

    for param in model_ft.parameters():
      param.requires_grad = True

    model_ft.to(device)  # GPU에서 미세조정(Finetuning) 할 수 있습니다.

    criterion = nn.CrossEntropyLoss()

    # 이미 모든 것이 학습된 상태이므로 학습률이 낮습니다.
    # 더 작은 Learning rate에 주목하세요
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)

    # 학습률을 몇 에폭마다 0.3배 감소시키기
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.3)

    model_ft_tuned = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                 num_epochs=25, device=device)

5 단계. 양자화된 모델로 변환하기

.. code:: python

    from torch.quantization import convert
    model_ft_tuned.cpu()

    model_quantized_and_trained = convert(model_ft_tuned, inplace=False)


양자화된 모델이 일부 이미지에서 어떻게 동작하는지 살펴보겠습니다.

.. code:: python

    visualize_model(model_quantized_and_trained)

    plt.ioff()
    plt.tight_layout()
    plt.show()
