분산 RPC 프레임워크를 사용하여 매개변수 서버 구현하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Author** : `Rohan Varma <https://github.com/rohan-varma>`_
  **번역**\ : `김은솔 <https://github.com/hongsam123>`_

선수과목:

-  `PyTorch Distributed Overview <../beginner/dist_overview.html>`__
-  `RPC API documents <https://pytorch.org/docs/master/rpc.html>`__

이 자습서는 PyTorch의 `분산 RPC 프레임워크(Distributed RPC framework) <https://pytorch.org/docs/stable/rpc.html>`_ 를 사용하여 매개변수 서버를 구현하는 간단한 예제를 소개합니다. 매개변수 서버 프레임워크란 서버의 집합이 대형 임베딩 테이블(embedding tables)과 같은 매개변수를 저장하고, 몇몇 트레이너들이 최신의 매개변수를 검색하기 위해 매개변수 서버를 질의하는 패러다임입니다. 이러한 트레이너는 로컬에서 훈련 루프를 실행할 수 있고 때때로 최신의 매개변수를 가져오기 위해 매개변수 서버와 동기화할 수 있습니다. 매개변수 서버 접근방식에 관한 더 자세한 정보는  `이 문서 <https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf>`_ 를 통해 확인하시면 됩니다.

우리는 분산 RPC 프레임워크를 사용하여 여러 트레이너들이 동일한 매개변수 서버와 통신하기 위해 RPC를 사용하고 원격 매개변수 서버 인스턴스 상태에 접근하기 위해 `RRef <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef>`_ 를 사용하는 예제를 빌드합니다. 각 트레이너는 분산형 autograd를 사용하여 여러 노드에서 autograd 그래프 스티칭(stitching)을 통해 분산 방식으로 전용 역방향 패스(dedicated backward pass)를 시작합니다.

**Note**\ : 이 자습서에서는 모델을 여러 머신으로 분할하거나 네트워크 트레이너가 다른 머신에서 호스트된 매개변수를 가져오는 매개변수 서버 학습 전략을 구현하는데 유용한 분산 RPC 프레임워크의 사용을 다룹니다. 만약 여러 GPU에서 모델을 복제하려는 경우에는 `Distributed Data Parallel tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_ 을 참고해주세요. 강화 학습과 RNN 사용 사례는 `RPC tutorial <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`_ 에서 다루고 있습니다.

익숙한 것부터 시작해보겠습니다. 필요한 모듈을 가져오고 MNIST 데이터셋에서 학습할 간단한 ConvNet을 정의합니다. 아래 네트워크는 주로 `pytorch/examples repo <https://github.com/pytorch/examples/tree/master/mnist>`_ 에서 정의된 네트워크로부터 채택되었습니다.

.. code-block:: python

   import argparse
   import os
   import time
   from threading import Lock

   import torch
   import torch.distributed.autograd as dist_autograd
   import torch.distributed.rpc as rpc
   import torch.multiprocessing as mp
   import torch.nn as nn
   import torch.nn.functional as F
   from torch import optim
   from torch.distributed.optim import DistributedOptimizer
   from torchvision import datasets, transforms

   # --------- pytorch/examples에서 가져온 학습할 NMIST 네트워크 -----

   class Net(nn.Module):
       def __init__(self, num_gpus=0):
           super(Net, self).__init__()
           print(f"Using {num_gpus} GPUs to train")
           self.num_gpus = num_gpus
           device = torch.device(
               "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
           print(f"Putting first 2 convs on {str(device)}")
           # 첫 번째 cuda 장치에, cuda 장치가 없는 경우 CPU에 conv layer를 배치합니다.
           self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
           self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
           # 두 번째 cuda 장치(있는 경우)에 나머지 네트워크를 배치합니다.
           if "cuda" in str(device) and num_gpus > 1:
               device = torch.device("cuda:1")

           print(f"Putting rest of layers on {str(device)}")
           self.dropout1 = nn.Dropout2d(0.25).to(device)
           self.dropout2 = nn.Dropout2d(0.5).to(device)
           self.fc1 = nn.Linear(9216, 128).to(device)
           self.fc2 = nn.Linear(128, 10).to(device)

       def forward(self, x):
           x = self.conv1(x)
           x = F.relu(x)
           x = self.conv2(x)
           x = F.max_pool2d(x, 2)

           x = self.dropout1(x)
           x = torch.flatten(x, 1)
           # 필요한 경우 tensor를 다음 장치로 이동합니다.
           next_device = next(self.fc1.parameters()).device
           x = x.to(next_device)

           x = self.fc1(x)
           x = F.relu(x)
           x = self.dropout2(x)
           x = self.fc2(x)
           output = F.log_softmax(x, dim=1)
           return output
다음으로 나머지 스크립트에 유용한 몇 가지 도우미 함수(helper functions)를 정의합니다. 원격 노드에 있는 객체의 주어진 메서드를 호출하는 함수를 정의하기 위해서 `rpc_sync <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.rpc_sync>`_ 와 `RRef <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef>`_ 를 사용합니다. 아래에서 원격 객체에 대한 핸들이 ``rref`` 인자에 의해 주어지고, 우리는 이것을 소유 노드(owning node)인 ``rref.owner()``\에서 실행합니다. 호출자 노드(caller node)에서는 이 명령을 ``rpc_sync``\의 사용을 통해 즉, 응답이 수신될 때까지 차단함으로써 동기화하여 실행합니다.


.. code-block:: python

   # --------- Helper Methods --------------------
   
   # 지역노드(local node)에서, 첫번째 arg를 RRef에 의해 얻어진 값으로 하여 메서드를 호출합니다.
   # 다른 arg들은 호출된 함수에 인자로서 전달됩니다.
   # 인스턴스 메서드를 호출하는데에 유용합니다.
   # 메서드는 클래스 메서드를 포함하여 일치하는 어떤 함수든 될 수 있습니다.
   def call_method(method, rref, *args, **kwargs):
       return method(rref.local_value(), *args, **kwargs)

   # RRef가 주어지면 RRef가 가지는 값에 대해 호출하여 메서드에서 전달된 결과를 반환합니다. 
   # 이 호출은 RRef를 가지고 주어진 인수를 전달하는 원격 노드에서 수행됩니다.
   # 예 : 만약 RRef에서 얻은 값이 Foo type인 경우, remote_method(Foo.bar, rref, arg1, arg2)는
   # 원격 노드에서 <foo_instance>.bar(arg1, arg2)를 호출하고 결과를 다시 얻은 것과 동일합니다.

   def remote_method(method, rref, *args, **kwargs):
       args = [method, rref] + list(args)
       return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)
이제 우리는 매개변수 서버를 정의 할 준비가 되었습니다. ``nn.Module``\를 서브클래스로 하고 위에서 정의한 네트워크에 핸들을 저장합니다. 모델을 호출하기 전 입력이 전송되는 장치가 될 입력 장치 또한 저장합니다.

.. code-block:: python

   # --------- Parameter Server --------------------
   class ParameterServer(nn.Module):
       def __init__(self, num_gpus=0):
           super().__init__()
           model = Net(num_gpus=num_gpus)
           self.model = model
           self.input_device = torch.device(
               "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
다음으로 forward pass를 정의합니다. 현재 분산 RPC 프레임워크는 RPC를 통한 CPU tensor 전송만을 지원하기 때문에 모델 출력 장치에 관계없이 출력을 CPU로 옮깁니다. 우리는 호출자/피호출자에서의 다른 장치(CPU/GPU)의 가능성으로 인해 RPC를 통한 CUDA tensor 전송을 의도적으로 비활성화 하였지만, 추후 릴리즈에서는 이를 지원할 수 있습니다.

.. code-block:: python

   class ParameterServer(nn.Module):
   ...
       def forward(self, inp):
           inp = inp.to(self.input_device)
           out = self.model(inp)
           # 이 출력은 1.5.0 부터 오직 CPU tensor만 허용하는 RPC를 통해 전달됩니다.
           # 이러한 이유로 tensor는 GPU 메모리의 안팎으로 이동되어야 합니다. 
           out = out.to("cpu")
           return out
다음으로 학습 및 검증 목적에 유용한 몇 가지 기타 함수를 정의합니다. 첫 번째로 ``get_dist_gradients``\는 분산 Autograd 컨텍스트 ID(Distributed Autograd context ID)를 받고, 분산 autograd에 의해 계산된 변화도를 검색하기 위해 ``dist_autograd.get_gradients`` API를 호출합니다. 더 자세한 정보는 `distributed autograd documentation <https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework>`_ 에서 찾아볼 수 있습니다. 프레임워크가 현재 RPC에 의한 tensor 전송만을 지원하기 때문에, 우리는 결과 사전(resulting dictionary)을 통해 반복하고 각 tensor를 CPU tensor로 변환합니다. 다음으로는 ``get_param_rrefs``\가 모델 매개변수를 통해 반복하고 그것을 (local) `RRef <https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RRef>`_ 로서 래핑(wrap)합니다. 이 메서드는 트레이너 노드에 의해 RPC를 통하여 호출되고 최적화 될 매개변수 목록을 반환합니다. 이는 `Distributed Optimizer <https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim>`_ 에 대한 입력으로서 요구되어지며, ``RRef``\ s의 목록으로 그것이 최적화 할 모든 매개변수를 필요로합니다.


.. code-block:: python

   # 이 모델에 누적된 변화도를 검색하기 위해 dist autograd를 사용합니다.
   # 주로 검증을 위해 사용됩니다.
   def get_dist_gradients(self, cid):
       grads = dist_autograd.get_gradients(cid)
       # 이 출력은 1.5.0부터 CPU tensor만 허용하는 RPC를 통해 전달됩니다.
       # 이러한 이유로 tensor는 GPU 메모리의 안팎으로 이동되어야 합니다. 
       cpu_grads = {}
       for k, v in grads.items():
           k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
           cpu_grads[k_cpu] = v_cpu
       return cpu_grads
   
   # RRef에서 지역 매개변수를 래핑합니다.
   # 매개변수를 원격으로 최적화하는 DistributedOptimizer를 구축하는데 필요합니다.
   def get_param_rrefs(self):
       param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
       return param_rrefs
마지막으로 매개변수 서버를 초기화하는 메서드를 생성할것입니다. 모든 프로세스에서 매개변수 서버 인스턴스는 오직 하나만 있으며, 모든 트레이너는 동일한 매개변수 서버와 통신하고 저장된 동일한 모델을 업데이트합니다. ``run_parameter_server``\에서 볼 수 있듯이, 서버 자체는 독립적인 작업을 수행하지 않습니다. 서버는 트레이너로부터 요청을 기다리고 (아직 정의되지 않았음) 요청된 함수를 실행함으로써 응답합니다.

.. code-block:: python

   # global 매개변수 서버 인스턴스
   param_server = None
   # 하나의 매개변수 서버만을 확보하기 위한 Lock
   global_lock = Lock()


   def get_parameter_server(num_gpus=0):
       """
       모든 트레이너 프로세스에 singleton 매개변수 서버 반환
       """
       global param_server
       # ParameterServer에 대한 핸들이 하나만 있는지 확인
       with global_lock:
           if not param_server:
               # 한 번의 구성
               param_server = ParameterServer(num_gpus=num_gpus)
           return param_server

   def run_parameter_server(rank, world_size):
       # 매개변수 서버는 모델의 호스트 역할만 하고 트레이너의 요청에 응답합니다.
       # rpc.shutdown()은 기본적으로 모든 수행자가 완료할 때까지 기다립니다.
       # 이는 매개변여수 서버가 모든 트레이너가 완료하는것을 기다린 후 종료하는것을 의미합니다.
       print("PS master initializing RPC")
       rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
       print("RPC initialized! Running parameter server...")
       rpc.shutdown()
       print("RPC shutdown on parameter server.")
위의 ``rpc.shutdown()``은 즉시 매개변수 서버를 종료하지는 않습니다. 대신 모든 작업자(이 경우에서는 트레이너)들이 ``rpc.shutdown()``를 호출할 때까지 기다립니다. 이를 통해 모든 트레이너(아직 정의하지 않았음)가 그들의 프로세스를 완료하기 전에는 매개변수 서버가 오프라인 상태가 되지 않음을 보장합니다.

다음으로는 ``TrainerNet`` 클래스를 정의합니다. 이것은 또한 ``nn.Module``\의 서브클래스가 되며, ``__init__`` 메서드는 매개변수 서버에 RRef, 또는 원격 참조(Remote Reference)를 얻기 위해 ``rpc.remote`` API를 사용합니다. 여기서 우리는 매개변수 서버를 우리의 로컬 프로세스에 복사하는 것이 아니라 ``self.param_server_rref``\를 별도의 프로세스에 있는 매개변수 서버에 대한 분산 공유 포인터(distributed shared pointer)로 생각할 수 있습니다.

.. code-block:: python

   # --------- Trainers --------------------

   # 트레이너에 의해 학습된 네트워크에 해당하는 nn.Module
   # forward() 메서드는 단순히 주어진 매개변수 서버에서 네트워크를 호출합니다.
   class TrainerNet(nn.Module):
       def __init__(self, num_gpus=0):
           super().__init__()
           self.num_gpus = num_gpus
           self.param_server_rref = rpc.remote(
               "parameter_server", get_parameter_server, args=(num_gpus,))
다음으로 ``get_global_param_rrefs``\라고 불리는 메서드를 정의합니다. 이 메서드의 필요성에 동기를 주기 위해서  `DistributedOptimizer <https://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim>`_ 의, 특히 API 서명에 관한 문서를 읽어보는것이 좋습니다. 옵티마이저는 최적화될 원격 매개변수에 해당하는 ``RRef``\s의 리스트를 전달 받아야 합니다. 그래서 우리는 여기서 필요한 ``RRef``\ s를 얻습니다. 주어진 ``TrainerNet``\가 유일하게 상호작용하는 원격 작업자가 ``ParameterServer``\이므로, 간단히 ``ParameterServer``\에서 ``remote_method``\를 호출합니다. 우리가 ``ParameterServer`` 클래스에서 정의했던 ``get_param_rrefs`` 메서드를 사용합니다. 이 메서드는 최적화가 필요한 매개변수에 ``RRef``\ s 목록을 반환합니다. 이 경우 ``TrainerNet``\는 그것의 자체 매개변수를 정의하지 않습니다. 만약 정의하였다면 각 매개변수를 ``RRef``\에 래핑하고, 그것을 ``DistributedOptimizer``\에 대한 입력으로 포함시켜야 합니다.

.. code-block:: python

   class TrainerNet(nn.Module):
   ...
       def get_global_param_rrefs(self):
           remote_params = remote_method(
               ParameterServer.get_param_rrefs,
               self.param_server_rref)
           return remote_params
이제 ``ParameterServer``\에 정의된 네트워크의 forward pass를 실행하기 위해 (동기화된)RPC를 호출하는 ``forward`` 메서드를 정의할 준비가 되었습니다. ``ParameterServer``\에 대한 원격 핸들인 ``self.param_server_rref``\를 RPC 호출에 전달합니다. 이 호출은 ``ParameterServer``\가 실행중인 노드에 RPC를 보내고, ``forward`` pass를 호출하며, 모델의 출력에 해당하는 ``Tensor``\를 반환합니다.

.. code-block:: python

   class TrainerNet(nn.Module):
   ...
       def forward(self, x):
           model_output = remote_method(
               ParameterServer.forward, self.param_server_rref, x)
           return model_output
트레이너를 완전히 정의하였으므로, 이제 네트워크와 옵티마이저를 생성하고 네트워크를 통해 일부의 입력을 실행하며 손실을 계산하는 신경망 훈련 루프를 작성할 차례입니다. 훈련 루프는 로컬 훈련 프로그램과 비슷하지만 머신에 분산되어 있는 네트워크의 특성상 약간의 수정이 존재합니다. 

아래에서는 ``TrainerNet``\을 초기화하고 ``DistributedOptimizer``\를 구축합니다. 위에서 언급했던것 처럼 최적화하기를 원하는 모든 전역(분산 트레이닝에 참여하는 모든 노드에 걸쳐) 매개변수를 전달해야 합니다. 추가적으로, 사용할 로컬 옵티마이저(이 경우에는 SGD)를 전달합니다. 로컬 옵티마이저를 생성했던 것과 같은 방식으로 기본 옵티마이저 알고리즘을 구성할 수 있습니다. ``optimizer.SGD``\에 대한 모든 인자가 적절하게 전달됩니다. 예를 들자면, 우리는 모든 로컬 옵티마이저의 학습률로 사용될 사용자 지정 학습률을 전달합니다. 

.. code-block:: python

   def run_training_loop(rank, num_gpus, train_loader, test_loader):
       # 일반적인 신경망을 forward/backward/optimizer 단계로,
       # 하지만 분산 방식으로 실행합니다.
       net = TrainerNet(num_gpus=num_gpus)
       # DistributedOptimizer 구축
       param_rrefs = net.get_global_param_rrefs()
       opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
다음으로 메인 훈련 루프를 정의합니다. PyTorch의 `DataLoader <https://pytorch.org/docs/stable/data.html>`_ 에 의해 제공되는 반복 가능 항목(iterables)을 통해 반복을 수행합니다. 일반적인 forward/backward/optimizer 루프를 작성하기 전에, 먼저 `Distributed Autograd context <https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.context>`_ 안에서 로직을 래핑합니다. 이것은 모델의 순방향 패스(forward pass)에 호출되는 RPCs를 기록하기 위해 필요하며, 역방향 패스(backward pass)에 참여하는 모든 분산된 작업자를 포함하는 적절한 그래프를 구성할 수 있습니다. 분산된 autograd 컨텍스트는 특정한 반복에 해당하는 변화도를 누적하고 최적화하기 위한 식별자의 역할을 하는 ``context_id``\를 반환합니다.

이 로컬 작업자에 대해 역방향 패스를 시작하는 일반적인 ``loss.backward()``\를 호출하는 것과는 다르게, ``dist_autograd.backward()``\를 호출하고 우리가 역방향 패스를 시작하기 원하는 루트인 ``loss`` 뿐만 아니라 context_id도 또한 넘겨줍니다. 또 ``context_id``\를 옵티마이저 호출에 넘겨주는데, 이것은 모든 노드를 통과하는 특정 역방향 패스에 의해 계산되어지는 변화도를 조회할 수 있어야 합니다.

.. code-block:: python

   def run_training_loop(rank, num_gpus, train_loader, test_loader):
   ...
       for i, (data, target) in enumerate(train_loader):
           with dist_autograd.context() as cid:
               model_output = net(data)
               target = target.to(model_output.device)
               loss = F.nll_loss(model_output, target)
               if i % 5 == 0:
                   print(f"Rank {rank} training batch {i} loss {loss.item()}")
               dist_autograd.backward(cid, [loss])
               # dist autograd가 성공적으로 실행되었고 변화도가 반환되었는지를 확인합니다.
               assert remote_method(
                   ParameterServer.get_dist_gradients,
                   net.param_server_rref,
                   cid) != {}
               opt.step(cid)

        print("Training complete!")
        print("Getting accuracy....")
        get_accuracy(test_loader, net)
다음으로는 일반적인 로컬 모델과 마찬가지로 학습을 마친 후의 모델의 정확도를 간단히 계산합니다. 그러나 위에서 이 함수에 전달하는 ``net``\이 ``TrainerNet``\의 인스턴스이므로 순방향 패스는 투명한 방식으로 RPC를 호출합니다.

.. code-block:: python

   def get_accuracy(test_loader, model):
       model.eval()
       correct_sum = 0
       # 가능한 경우 GPU를 사용하여 평가합니다.
       device = torch.device("cuda:0" if model.num_gpus > 0
           and torch.cuda.is_available() else "cpu")
       with torch.no_grad():
           for i, (data, target) in enumerate(test_loader):
               out = model(data, -1)
               pred = out.argmax(dim=1, keepdim=True)
               pred, target = pred.to(device), target.to(device)
               correct = pred.eq(target.view_as(pred)).sum().item()
               correct_sum += correct

       print(f"Accuracy {correct_sum / len(test_loader.dataset)}")
다음으로 RPC의 초기화를 담당하는 ``ParameterServer``\의 메인 루프로 ``run_parameter_server``\를 정의했던 방식과 비슷하게 트레이너를 위한 비슷한 루프를 정의합니다. 다른점은 트레이너가 위에서 정의했던 훈련 루프를 실행해야한다는 것입니다.

.. code-block:: python
   
   # 트레이너를 위한 메인 루프
   def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
       print(f"Worker rank {rank} initializing RPC")
       rpc.init_rpc(
           name=f"trainer_{rank}",
           rank=rank,
           world_size=world_size)

       print(f"Worker {rank} done initializing RPC")

       run_training_loop(rank, num_gpus, train_loader, test_loader)
       rpc.shutdown()
``run_parameter_server``\와 비슷하게 ``rpc.shutdown()``\은 기본적으로 노드가 종료되기 전 모든 작업자(트레이너와 매개변수 서버 모두)가 ``rpc.shutdown()``\을 호출할 때까지 기다립니다. 이것은 노드가 정상적으로 종료되고 다른 노드가 온라인이 될 것으로 예상하는 동안 오프라인 상태가 되지 않습니다.

이제 트레이너와 매개변수 서버 각 코드를 완성하였으며, 트레이너와 매개변수 서버를 시작하는 코드를 추가하는 것만 남았습니다. 먼저 매개변수 서버와 트레이너들에 적용되는 다양한 인자를 받아들어야 합니다. ``world_size``\는 학습에 참여할 노드들의 총 수에 해당하며 모든 트레이너와 매개변수 서버의 합계입니다. 또한 0 (단일 매개변수 서버를 실행할)에서 ``world_size - 1``\까지 각 프로세스에 고유한 ``rank``\를 전달해야 합니다. ``master_addr``\와 ``master_port``\는 0순위 프로세스가 실행중인 위치를 확인하는데 사용될 인자이고, 서로를 검색하기 위해 개별노드에 의해 사용됩니다. 로컬에서 이 예제를 테스트하기 위해서는 생성된 모든 인스턴스에 ``localhost``\와 동일한 ``master_port``\을 전달하기만 하면됩니다. 이 예제는 데모 목적으로 0~2개의 GPU만을 지원하지만, 추가 GPU를 사용하도록 패턴을 확장할 수 있습니다. 

.. code-block:: python

   if __name__ == '__main__':
       parser = argparse.ArgumentParser(
           description="Parameter-Server RPC based training")
       parser.add_argument(
           "--world_size",
           type=int,
           default=4,
           help="""Total number of participating processes. Should be the sum of
           master node and all training nodes.""")
       parser.add_argument(
           "rank",
           type=int,
           default=None,
           help="Global rank of this process. Pass in 0 for master.")
       parser.add_argument(
           "num_gpus",
           type=int,
           default=0,
           help="""Number of GPUs to use for training, Currently supports between 0
            and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
       parser.add_argument(
           "--master_addr",
           type=str,
           default="localhost",
           help="""Address of master, will default to localhost if not provided.
           Master must be able to accept network traffic on the address + port.""")
       parser.add_argument(
           "--master_port",
           type=str,
           default="29500",
           help="""Port that master is listening on, will default to 29500 if not
           provided. Master must be able to accept network traffic on the host and port.""")

       args = parser.parse_args()
       assert args.rank is not None, "must provide rank argument."
       assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
       os.environ['MASTER_ADDR'] = args.master_addr
       os.environ["MASTER_PORT"] = args.master_port
이제 명령줄 인자에 따라 매개변수 서버 또는 트레이너에 해당하는 프로세스를 생성합니다. 전달된 순위가 0이라면 ``ParameterServer``\를 만들고 아니라면 ``TrainerNet``\를 만듭니다. 실행하려는 함수에 해당하는 하위 프로세스를 시작하기 위해 ``torch.multiprocessing``\를 사용하고, ``p.join()``\으로 메인 스레드에서 이 프로세스가 완료될 때까지 기다립니다. 트레이너를 초기화 하는 경우, MNIST 데이터셋에서 데이터 로더를 학습 및 테스트하기 위해 PyTorch의 `dataloaders <https://pytorch.org/docs/stable/data.html>`_ 를 사용합니다.

.. code-block:: python

   processes = []
   world_size = args.world_size
   if args.rank == 0:
       p = mp.Process(target=run_parameter_server, args=(0, world_size))
       p.start()
       processes.append(p)
   else:
       # 사용할 데이터 가져오기
       train_loader = torch.utils.data.DataLoader(
           datasets.MNIST('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])),
           batch_size=32, shuffle=True,)
       test_loader = torch.utils.data.DataLoader(
           datasets.MNIST(
               '../data',
               train=False,
               transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                           ])),
           batch_size=32,
           shuffle=True,
       )
       # 이 노드에서 작업자 훈련 시작
       p = mp.Process(
           target=run_worker,
           args=(
               args.rank,
               world_size, args.num_gpus,
               train_loader,
               test_loader))
       p.start()
       processes.append(p)

   for p in processes:
       p.join()
예제를 로컬에서 실행하려면 별도의 터미널 창에서 생성하려는 서버 및 작업자에 대해 다음 명령어 작업을 수행하시면 됩니다. ``python rpc_parameter_server.py --world_size=WORLD_SIZE --rank=RANK`` 예를 들면 world size가 2인 마스터 노드의 경우 명령어는 ``python rpc_parameter_server.py --world_size=2 --rank=0``\와 같습니다. 그런 다음 트레이너는 별도의 창에서 ``python rpc_parameter_server.py --world_size=2 --rank=1`` 명령어로 시작할 수 있으며, 하나의 서버와 하나의 트레이너로 학습을 시작합니다. 이 자습서에서는 0~2개의 GPU를 사용하여 학습이 발생한다고 가정하고, 이때 인자는 학습 스크립트에 ``--num_gpus=N``\를 전달하며 구성할 수 있습니다.

명령어 라인 인자에 ``--master_addr=ADDRESS``\와 ``--master_port=PORT``\를 전달하여 마스터 작업자가 수신하는 주소와 포트번호를 표시할 수 있습니다. (트레이너와 마스터 노드가 다른 머신에서 실행하는 기능성을 테스트하기 위해)
