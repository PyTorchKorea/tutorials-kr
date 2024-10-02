Fully Sharded Data Parallel(FSDP) 시작하기
======================================================

**저자**: `Hamid Shojanazeri <https://github.com/HamidShojanazeri>`__, `Yanli Zhao <https://github.com/zhaojuanmao>`__, `Shen Li <https://mrshenli.github.io/>`__
**번역:** `이진혁 <https://github.com/uddk6215>__`

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
각 process/ worker가 모델의 복제본을 소유하고 데이터 배치를 처리한 후, 최종적으로 all-reduce를 사용하여 서로 다른  worker들의 변화도를 합산합니다. 
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
    This tutorial is intended for PyTorch versions 1.12 and later. 
    If you are using an earlier version, replace all instances of `size_based_auto_wrap_policy` with `default_auto_wrap_policy`.
    이 튜토리얼은 PyTorch 버전 1.12 이상을 대상으로 합니다.
    이전 버전을 사용하고 있다면, 모든 `size_based_auto_wrap_policy`의 인스턴스를 `default_auto_wrap_policy`로 교체하시기 바랍니다.

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

1.3 Distributed training setup. As we mentioned FSDP is a type of data parallelism which requires a distributed training environment, so here we use two helper functions to initialize the processes for distributed training and clean up.

.. code-block:: python

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

2.1  Define our toy model for handwritten digit classification. 

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

2.2 Define a train function 

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

2.3 Define a validation function 

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

2.4 Define a distributed train function that wraps the model in FSDP

**Note: to save the FSDP model, we need to call the state_dict on each rank then on Rank 0 save the overall states.**

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
            # use a barrier to make sure training is done on all ranks
            dist.barrier()
            states = model.state_dict()
            if rank == 0:
                torch.save(states, "mnist_cnn.pt")
        
        cleanup()



2.5 Finally, parse the arguments and set the main function

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


We have recorded cuda events to measure the time of FSDP model specifics. The CUDA event time was 110.85 seconds.

.. code-block:: bash

    python FSDP_mnist.py

    CUDA event elapsed time on training loop 40.67462890625sec

Wrapping the model with FSDP, the model will look as follows, we can see the model has been wrapped in one FSDP unit.
Alternatively, we will look at adding the fsdp_auto_wrap_policy next and will discuss the differences. 

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

The following is the peak memory usage from FSDP MNIST training on g4dn.12.xlarge AWS EC2 instance with 4 GPUs captured from PyTorch Profiler. 


.. figure:: /_static/img/distributed/FSDP_memory.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   FSDP Peak Memory Usage

Applying *fsdp_auto_wrap_policy* in FSDP otherwise, FSDP will put the entire model in one FSDP unit, which will reduce computation efficiency and memory efficiency. 
The way it works is that, suppose your model contains 100 Linear layers. If you do FSDP(model), there will only be one FSDP unit which wraps the entire model. 
In that case, the allgather would collect the full parameters for all 100 linear layers, and hence won't save CUDA memory for parameter sharding.
Also, there is only one blocking allgather call for the all 100 linear layers, there will not be communication and computation overlapping between layers. 

To avoid that, you can pass in an fsdp_auto_wrap_policy, which will seal the current FSDP unit and start a new one automatically when the specified condition is met (e.g., size limit).
In that way you will have multiple FSDP units, and only one FSDP unit needs to collect full parameters at a time. E.g., suppose you have 5 FSDP units, and each wraps 20 linear layers.
Then, in the forward, the 1st FSDP unit will allgather parameters for the first 20 linear layers, do computation, discard the parameters and then move on to the next 20 linear layers. So, at any point in time, each rank only materializes parameters/grads for 20 linear layers instead of 100.


To do so in 2.4 we define the auto_wrap_policy and pass it to FSDP wrapper, in the following example, my_auto_wrap_policy defines that a layer could be wrapped or sharded by FSDP if the number of parameters in this layer is larger than 100.
If the number of parameters in this layer is smaller than 100, it will be wrapped with other small layers together by FSDP. 
Finding an optimal auto wrap policy is challenging, PyTorch will add auto tuning for this config in the future. Without an auto tuning tool, it is good to profile your workflow using different auto wrap policies experimentally and find the optimal one.

.. code-block:: python

    my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=20000
        )
    torch.cuda.set_device(rank)
    model = Net().to(rank)

    model = FSDP(model,
        fsdp_auto_wrap_policy=my_auto_wrap_policy)

Applying the fsdp_auto_wrap_policy, the model would be as follows:

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

The following is the peak memory usage from FSDP with auto_wrap policy of MNIST training on a g4dn.12.xlarge AWS EC2 instance with 4 GPUs captured from PyTorch Profiler. 
It can be observed that the peak memory usage on each device is smaller compared to FSDP without auto wrap policy applied, from ~75 MB to 66 MB.

.. figure:: /_static/img/distributed/FSDP_autowrap.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   FSDP Peak Memory Usage using Auto_wrap policy

*CPU Off-loading*: In case the model is very large that even with FSDP wouldn't fit into GPUs, then CPU offload can be helpful here. 

Currently, only parameter and gradient CPU offload is supported. It can be enabled via passing in cpu_offload=CPUOffload(offload_params=True).

Note that this currently implicitly enables gradient offloading to CPU in order for params and grads to be on the same device to work with the optimizer. This API is subject to change. The default is None in which case there will be no offloading.

Using this feature may slow down the training considerably, due to frequent copying of tensors from host to device, but it could help improve memory efficiency and train larger scale models. 

In 2.4 we just add it to the FSDP wrapper


.. code-block:: python

    model = FSDP(model,
        fsdp_auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True))


Compare it with DDP, if in 2.4 we just normally wrap the model in DPP, saving the changes in “DDP_mnist.py”.

.. code-block:: python

    model = Net().to(rank)
    model = DDP(model)


.. code-block:: bash

    python DDP_mnist.py

    CUDA event elapsed time on training loop 39.77766015625sec

The following is the peak memory usage from DDP MNIST training on g4dn.12.xlarge AWS EC2 instance with 4 GPUs captured from PyTorch profiler. 

.. figure:: /_static/img/distributed/DDP_memory.gif
   :width: 100%
   :align: center
   :alt: FSDP peak memory

   DDP Peak Memory Usage using Auto_wrap policy


Considering the toy example and tiny MNIST model we defined here, we can observe the difference between peak memory usage of DDP and FSDP. 
In DDP each process holds a replica of the model, so the memory footprint is higher compared to FSDP which shards the model parameters, optimizer states and gradients over DDP ranks.
The peak memory usage using FSDP with auto_wrap policy is the lowest followed by FSDP and DDP. 

Also, looking at timings, considering the small model and running the training on a single machine, FSDP with and without auto_wrap policy performed almost as fast as DDP.
This example does not represent most of the real applications, for detailed analysis and comparison between DDP and FSDP please refer to this `blog post  <https://pytorch.medium.com/6c8da2be180d>`__ .
