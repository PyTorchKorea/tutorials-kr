Fully Sharded Data Parallel(FSDP) 시작하기
======================================================

**저자**: `Hamid Shojanazeri <https://github.com/HamidShojanazeri>`__, `Yanli Zhao <https://github.com/zhaojuanmao>`__, `Shen Li <https://mrshenli.github.io/>`__
**번역:**: `이진혁 <https://github.com/uddk6215>__`

.. note::
   |edit| 이 튜토리얼을 `github <https://github.com/pytorch/tutorials/blob/main/intermediate_source/FSDP_tutorial.rst>`__에서 보고 수정할 수 있습니다.

대규모 AI 모델을 학습하는 것은 많은 컴퓨팅 파워와 리소스를 필요로 하는 어려운 작업입니다.
또한 이러한 대규모 모델의 학습을 위해서는 엔지니어링 측면에서 상당한 복잡성이 따릅니다.
PyTorch 1.11에서 출시된 `PyTorch FSDP <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/>`__는 이러한 작업을 더 쉽게 만들어줍니다.

이 튜토리얼에서는 간단한 MNIST 모델에 `FSDP APIs <https://pytorch.org/docs/stable/fsdp.html>`__를 사용하는 방법을 보여줍니다. 
이 방법은 `HuggingFace BERT models <https://huggingface.co/blog/zero-deepspeed-fairscale>`__이나 최대 1조 개의 매개변수를 가진
`GPT 3 models up to 1T parameters <https://pytorch.medium.com/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff>`__
같은 더 큰 모델로 확장될 수 있습니다. 
예제로 사용된 DDP MNIST 코드는 `here <https://github.com/yqhu/mnist_examples>`__에서 가져왔습니다.


FSDP의 작동 방식
--------------
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__ (DDP) 학습에서는,
각 process/ worker가 모델의 복제본을 소유하고 데이터 배치를 처리한 후, 최종적으로 all-reduce를 사용하여 서로 다른 worker들의 변화도를 합산합니다. 
DDP에서는 모델 가중치와 옵티마이저 상태가 모든 worker들에 걸쳐 복제됩니다. 
FSDP는 모델 파라미터, 옵티마이저 상태, 변화도를 DDP rank들에 걸쳐 샤딩하는 데이터 병렬 처리 방식입니다.

FSDP로 학습할 때, GPU 메모리 사용량은 모든 work들에 걸쳐 DDP로 학습할 때보다 작습니다. 
이로 인해 더 큰 모델이나 배치 크기를 디바이스에 맞출 수 있어 매우 큰 모델의 학습이 가능해집니다. 
다만 이는 통신량 증가라는 비용을 수반합니다. 이 때 발생하는 오버헤드는 통신과 계산을 중첩하는 등의 내부 최적화를 통해 줄어듭니다.

.. figure:: /_static/img/distributed/fsdp_workflow.png
   :width: 100%
   :align: center
   :alt: FSDP workflow

   FSDP Workflow

FSDP는 고수준에서 다음과 같이 작동합니다.

*생성자에서*

* 모델 파라미터들을 샤딩하고 각 랭크는 자신의 샤드만 유지합니다.

*순전파 경로에서*

* all_gather를 실행하여 모든 랭크로부터 모든 샤드를 수집해 이 FSDP 유닛의 전체 파라미터를 복원합니다.
* 순전파 연산을 실행합니다.
* 방금 수집한 파라미터 샤드를 폐기합니다.

*역전파 경로에서*

* all_gather를 실행하여 모든 랭크로부터 모든 샤드를 수집해 이 FSDP 유닛의 전체 파라미터를 복원합니다.
* 역전파 연산을 실행합니다.
* reduce_scatter를 실행하여 변화도를 동기화합니다.
* 파라미터를 폐기합니다.

FSDP의 샤딩을 쉽게 이해하는 한 가지 방법은 DDP에서 수행되는 변화도에 대한 all-reduce연산을 reduce-scatter와 all-gather 2개로 분해하는 것입니다. 
구체적으로는, 역전파 과정에서 FSDP는 변화도를 축소하고 분산시켜 각 랭크가 변화도의 샤드를 소유하도록 합니다. 
그런 다음 옵티마이저 단계에서 매개변수의 해당 샤드를 업데이트합니다. 
마지막으로, 후속 순전파에서 all-gather 연산을 수행하여 갱신된 매개변수가 담긴 샤드를 수집하고 결합합니다.

.. figure:: /_static/img/distributed/fsdp_sharding.png
   :width: 100%
   :align: center
   :alt: FSDP allreduce

   FSDP Allreduce

FSDP 사용 방법
---------------
여기서는 시연 목적으로 MNIST 데이터셋으로 훈련을 수행할 간단한 예제 모델을 사용해보겠습니다. 이 API들과 로직은 더 큰 모델의 학습에도 적용될 수 있습니다.

*설정*

1.1 PyTorch와 Torchvision 설치

설치에 대한 정보는 `Get Started guide <https://pytorch.org/get-started/locally/>`__ 를 참조바랍니다.
다음 코드 snippet들을 "FSDP_mnist.py"라는 Python 스크립트에 추가합니다.

1.2  필요한 패키지 임포트

.. note::

    이 튜토리얼은 PyTorch 버전 1.12 이상을 대상으로 합니다.
    이전 버전을 사용하고 있다면, `size_based_auto_wrap_policy`의 모든 인스턴스를 `default_auto_wrap_policy`로 교체하시기 바랍니다.

.. code-block:: python

    # 출처: https://github.com/pytorch/examples/blob/master/mnist/main.py
    import os
    import argparse
    import functools
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms


    from torch.optim.lr_scheduler import StepLR

    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        CPUOffload,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap,
    )

1.3 분산 학습 설정
앞서 언급했듯이 FSDP는 분산 학습 환경이 필요한 데이터 병렬화의 한 유형입니다. 
따라서 여기서는 분산 학습을 위한 process를 초기화하고 정리하는 두 가지 함수를 사용합니다.

.. code-block:: python

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

2.1  손글씨 숫자 분류를 위한 예제 모델을 정의

.. code-block:: python

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
        
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

2.2 학습 함수 정의

.. code-block:: python

    def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
        model.train()
        ddp_loss = torch.zeros(2).to(rank)
        if sampler:
            sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            loss.backward()
            optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

2.3 검증 함수 정의

.. code-block:: python

    def test(model, rank, world_size, test_loader):
        model.eval()
        correct = 0
        ddp_loss = torch.zeros(3).to(rank)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(rank), target.to(rank)
                output = model(data)
                ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
                ddp_loss[2] += len(data)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        if rank == 0:
            test_loss = ddp_loss[0] / ddp_loss[2]
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
                1.   * ddp_loss[1] / ddp_loss[2]))

2.4 모델을 FSDP로 래핑하는 분산 학습 함수 정의
**주의: FSDP 모델을 저장하기 위해서는 각 랭크에서 state_dict를 호출한 다음, 랭크 0에서 전체 상태를 저장해야 합니다.**

.. code-block:: python

    def fsdp_main(rank, world_size, args):
        setup(rank, world_size)

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset1 = datasets.MNIST('../data', train=True, download=True,
                            transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                            transform=transform)

        sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
        sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

        train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
        test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
        cuda_kwargs = {'num_workers': 2,
                        'pin_memory': True,
                        'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100
        )
        torch.cuda.set_device(rank)
        
        
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)

        model = Net().to(rank)

        model = FSDP(model)

        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        init_start_event.record()
        for epoch in range(1, args.epochs + 1):
            train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
            test(model, rank, world_size, test_loader)
            scheduler.step()

        init_end_event.record()

        if rank == 0:
            print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
            print(f"{model}")

        if args.save_model:
            # 모든 랭크에서 학습이 완료되었는지 확인하기 위해 barrier를 사용합니다.
            dist.barrier()
            states = model.state_dict()
            if rank == 0:
                torch.save(states, "mnist_cnn.pt")
        
        cleanup()



2.5 마지막으로, 인자를 파싱하고 메인 함수를 설정

.. code-block:: python

    if __name__ == '__main__':
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
        args = parser.parse_args()

        torch.manual_seed(args.seed)

        WORLD_SIZE = torch.cuda.device_count()
        mp.spawn(fsdp_main,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True)


 FSDP 모델의 특정 시간(학습 루프의 실행 시간)을 측정하기 위해 CUDA 이벤트를 기록했습니다. 전체 CUDA 이벤트 시간은 110.85 초였습니다.

.. code-block:: bash

    python FSDP_mnist.py

    CUDA event elapsed time on training loop 40.67462890625sec

FSDP로 모델을 래핑하면, 모델은 다음과 같이 보일 것입니다. 모델이 하나의 FSDP 유닛으로 래핑된 것을 볼 수 있습니다.
다음으로, fsdp_auto_wrap_policy를 추가하는 것을 살펴보고 차이점에 대해 논의할 것입니다.

.. code-block:: bash

    FullyShardedDataParallel(
    (_fsdp_wrapped_module): FlattenParamsWrapper(
        (_fpw_module): Net(
        (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
        (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        (dropout1): Dropout(p=0.25, inplace=False)
        (dropout2): Dropout(p=0.5, inplace=False)
        (fc1): Linear(in_features=9216, out_features=128, bias=True)
        (fc2): Linear(in_features=128, out_features=10, bias=True)
        )
    )
 )

다음은 PyTorch Profiler로 캡처한 g4dn.12.xlarge AWS EC2 인스턴스의 4개 GPU에서 FSDP MNIST 학습 시 최대 메모리 사용량입니다.

.. figure:: /_static/img/distributed/FSDP_memory.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   FSDP Peak Memory Usage

FSDP에 *fsdp_auto_wrap_policy* 를 적용하지 않으면, FSDP는 전체 모델을 하나의 FSDP 유닛에 넣게 되어 계산 효율성과 메모리 효율성이 감소합니다.
작동 방식은 다음과 같습니다. 예를 들어, 모델에 100개의 Linear 층이 있다고 가정해 봅시다. FSDP(model)을 실행하면, 전체 모델을 감싸는 하나의 FSDP 유닛만 생성됩니다.
이 경우, allgather 연산이 100개 모든 선형 층의 전체 매개변수를 수집하게 되어, 매개변수 값 샤딩을 통한 CUDA 메모리 절약 효과가 없어집니다.
또한, 100개의 선형 층 전체에 대해 하나의 대규모 allgather 연산만 수행되므로, 층 간 통신과 계산을 동시에 처리할 수 없습니다.

이러한 문제를 피하기 위해, fsdp_auto_wrap_policy를 사용할 수 있습니다. 해당 방식은 지정된 조건(예: 크기 제한)이 충족되면 
현재 FSDP 유닛 단위를 마무리하고 새로운 단위를 자동으로 시작합니다.

이렇게 하면 여러 개의 FSDP 유닛 단위가 생기고, 한 번에 하나의 FSDP 유닛 단위만 전체 매개변수를 수집하면 됩니다. 
예를 들어, 5개의 FSDP 유닛 단위가 있다 가정하고 각 단위가 20개의 선형 층을 포함한다고 가정해 봅시다. 
그러면 순전파 과정에서 첫 번째 FSDP 단위는 처음 20개 선형 층의 매개변수들만 모으고, 계산을 수행한 후 이 매개변수들을 버리고 다음 20개 층으로 넘어갑니다.
이런 방식으로, 어느 시점에서도 각 rank(GPU)는 100개가 아닌 20개의 선형 층의 매개변수와 변화도 값만 실제로 메모리에 유지하게 됩니다


2.4에서 이를 구현하기 위해 auto_wrap_policy를 정의하고 FSDP 래퍼에 전달하고, 다음 예시에서 my_auto_wrap_policy는 층의 매개변수 수가 100개보다 크면 
해당 층을 FSDP로 래핑하거나 샤딩할 수 있다고 정의합니다. 층의 매개변수 수가 100개 미만이면 FSDP에 의해 다른 작은 층들과 함께 래핑됩니다.
최적의 auto wrap policy를 찾는 것은 어려운 과제입니다. PyTorch는 향후 이 설정을 위한 자동 튜닝 기능을 추가할 예정입니다. 
자동 튜닝 도구 없이는 다양한 auto wrap policy들을 실험적으로 사용하여 워크플로우를 프로파일링하고 최적의 것을 찾는 것이 좋습니다.

.. code-block:: python

    my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=20000
        )
    torch.cuda.set_device(rank)
    model = Net().to(rank)

    model = FSDP(model,
        fsdp_auto_wrap_policy=my_auto_wrap_policy)

fsdp_auto_wrap_policy를 적용하면, 모델은 다음과 같은 구조를 가지게 됩니다.

.. code-block:: bash

    FullyShardedDataParallel(
  (_fsdp_wrapped_module): FlattenParamsWrapper(
    (_fpw_module): Net(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (dropout1): Dropout(p=0.25, inplace=False)
      (dropout2): Dropout(p=0.5, inplace=False)
      (fc1): FullyShardedDataParallel(
        (_fsdp_wrapped_module): FlattenParamsWrapper(
          (_fpw_module): Linear(in_features=9216, out_features=128, bias=True)
        )
      )
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )
  )


.. code-block:: bash

    python FSDP_mnist.py

    CUDA event elapsed time on training loop 41.89130859375sec

다음은 auto_wrap policy를 적용하여 FSDP를 사용한 MNIST 학습의 최대 메모리 사용량입니다. 이는 PyTorch Profiler로 캡처한 4개의 GPU가 있는 g4dn.12.xlarge AWS EC2 인스턴스에서 측정되었습니다.
auto_wrap policy를 적용하지 않은 FSDP와 비교했을 때, 각 디바이스의 최대 메모리 사용량이 약 75MB에서 66MB로 감소한 것을 관찰할 수 있었습니다.

.. figure:: /_static/img/distributed/FSDP_autowrap.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   Auto_wrap policy를 사용한 FSDP의 최대 메모리 사용량

*CPU 오프로딩*: FSDP를 사용해도 모델이 너무 커서 GPU에 맞지 않는 경우, CPU 오프로딩이 도움이 될 수 있습니다.

현재는 매개변수와 변화도 값의 CPU 오프로딩만 지원됩니다. cpu_offload=CPUOffload(offload_params=True)를 전달하여 활성화할 수 있습니다.

매개변수와 변화도 값이 옵티마이저와 함께 작동하기 위해 같은 디바이스에 있어야 하므로, 현재는 암묵적으로 변화도 값의 CPU 오프로딩도 활성화됩니다.

이 API는 변경될 수 있습니다. 기본값은 None이며, 이 경우 오프로딩이 수행되지 않습니다.

이 기능을 사용하면 호스트와 디바이스 간 텐서의 빈번한 복사로 인해 학습 속도가 상당히 느려질 수 있지만, 

메모리 효율성을 개선하고 더 큰 규모의 모델을 학습하는 데 도움이 될 수 있습니다.

2.4에서는 이를 FSDP 래퍼에 추가하기만 하면 됩니다.


.. code-block:: python

    model = FSDP(model,
        fsdp_auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True))


DDP와 비교해보겠습니다. 2.4에서 모델을 평범하게 DDP로 래핑하고, 변경 사항을 'DDP_mnist.py'에 저장한다면 다음과 같습니다.

.. code-block:: python

    model = Net().to(rank)
    model = DDP(model)


.. code-block:: bash

    python DDP_mnist.py

    CUDA event elapsed time on training loop 39.77766015625sec

다음은 PyTorch 프로파일러로 캡처한, 4개의 GPU로 g4dn.12.xlarge AWS EC2 인스턴스에서 DDP를 사용하여 MNIST를 학습시킨 모델의 최대 메모리 사용량입니다.

.. figure:: /_static/img/distributed/DDP_memory.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   Auto_wrap policy를 사용한 DDP의 최대 메모리 사용량

여기서 정의한 간단한 예제와 작은 MNIST 모델을 고려할 때, DDP와 FSDP의 최대 메모리 사용량 차이를 관찰할 수 있습니다.
DDP에서는 각 process가 모델의 복제본을 가지고 있어, 모델 매개변수, 옵티마이저 상태, 그리고 변화도를 DDP 랭크에 걸쳐 샤딩하는 FSDP에 비해 메모리 사용량이 더 높습니다.
auto_wrap policy를 사용한 FSDP의 최대 메모리 사용량이 가장 낮고, 그 다음으로 FSDP, 마지막으로 DDP 순입니다.

또한, 전체 학습 과정의 실행 시간을 보면, 작은 모델과 단일 머신에서 학습시키는 것을 고려할 때는, auto_wrap policy를 사용 여부에 관계없이 FSDP는 DDP와 거의 비슷한 속도로 수행되었습니다.
이 예제는 대부분의 실제 애플리케이션을 대표하지 않습니다. DDP와 FSDP 사이의 자세한 분석과 비교는 이 `blog post  <https://pytorch.medium.com/6c8da2be180d>`__ 를 참조바랍니다.
