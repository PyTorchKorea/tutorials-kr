Intel® Extension for PyTorch* 백엔드
=====================================

`torch.compile`을 통해 더 원할한 작동을 위해, Intel® Extension for PyTorch는 ``ipex``라는 백엔드를 구현했습니다. 이 백엔드는 Intel 플랫폼에서 하드웨어 자원 사용 효율성을 개선하여 성능을 향상시키는 것을 목표로 합니다. `ipex` 백엔드는 모델 컴파일을 위한 Intel® Extension for PyTorch에 설계된 추가 커스터마이징을 통해 구현되었습니다.

사용 예시
~~~~~~~~~~~~~

FP32 학습
----------

아래 예제를 통해, 여러분은 FP32 데이터 유형으로 모델을 학습할 때 `torch.compile`과 함께 `ipex` 백엔드를 사용하는 방법을 배울 수 있습니다.
.. code:: python

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = 'datasets/cifar10/'

   transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   train_dataset = torchvision.datasets.CIFAR10(
     root=DATA,
     train=True,
     transform=transform,
     download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(
     dataset=train_dataset,
     batch_size=128
   )

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
   model.train()

   #################### 코드 변경 부분 ####################
   import intel_extension_for_pytorch as ipex

   # 선택적으로 다음 API를 호출하여 프런트엔드 최적화를 적용합니다.
   model, optimizer = ipex.optimize(model, optimizer=optimizer)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   for batch_idx, (data, target) in enumerate(train_loader):
       optimizer.zero_grad()
       output = compile_model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()


BF16 학습
----------

아래 예시를 통해 BFloat16 데이터 유형으로 모델 학습을 위해 `torch.compile` 와 함께 `ipex` 백엔드를 활용하는 방법을 알아보세요.

.. code:: python

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = 'datasets/cifar10/'

   transform = torchvision.transforms.Compose([
     torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   train_dataset = torchvision.datasets.CIFAR10(
     root=DATA,
     train=True,
     transform=transform,
     download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(
     dataset=train_dataset,
     batch_size=128
   )

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum=0.9)
   model.train()

   #################### 코드 변경 부분 ####################
   import intel_extension_for_pytorch as ipex

   # Invoke the following API optionally, to apply frontend optimizations
   model, optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.cpu.amp.autocast():
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()
           output = compile_model(data)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()


FP32 추론
--------------

아래 예시를 통해 `ipex` 백엔드를 `torch.compile`와 함께 활용하여 FP32 데이터 유형으로 모델을 추론하는 방법을 알아보세요.

.. code:: python

   import torch
   import torchvision.models as models

   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex
   
   # 선택적으로 다음 API를 호출하여 프런트엔드 최적화를 적용합니다.
   model = ipex.optimize(model, weights_prepack=False)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.no_grad():
       compile_model(data)


BF16 추론
--------------

아래 예시를 통해 `ipex` 백엔드를 `torch.compile`와 함께 활용하여 BFloat16 데이터 유형으로 모델을 추론하는 방법을 알아보세요.

.. code:: python

   import torch
   import torchvision.models as models

   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### code changes ####################
   import intel_extension_for_pytorch as ipex

   # Invoke the following API optionally, to apply frontend optimizations
   model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)

   compile_model = torch.compile(model, backend="ipex")
   ######################################################

   with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
       compile_model(data)
