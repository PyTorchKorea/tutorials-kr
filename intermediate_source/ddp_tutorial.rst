분산 데이터 병렬 처리 시작하기
===================================
**저자**: `Shen Li <https://mrshenli.github.io/>`_

**감수**: `Joe Zhu <https://github.com/gunandrose4u>`_

**번역**: `조병근 <https://github.com/Jo-byung-geun>`_

.. note::
   |edit| 이 튜토리얼의 소스 코드는 `GitHub <https://github.com/pytorch/tutorials/blob/master/intermediate_source/ddp_tutorial.rst>`__ 에서 확인하고 변경해 볼 수 있습니다.

선수과목(Prerequisites):

-  `PyTorch 분산 처리 개요 <../beginner/dist_overview.html>`__
-  `분산 데이터 병렬 처리 API 문서 <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
-  `분산 데이터 병렬 처리 문서 <https://pytorch.org/docs/master/notes/ddp.html>`__


`분산 데이터 병렬 처리 <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__\(DDP)는
여러 기기에서 실행할 수 있는 데이터 병렬 처리를 모듈 수준에서 구현합니다.
DDP를 사용하는 어플리케이션은 여러 작업(process)을 생성하고 작업 당 단일 DDP 인스턴스를 생성해야 합니다.
DDP는 `torch.distributed <https://tutorials.pytorch.kr/intermediate/dist_tuto.html>`__
패키지의 집합 통신(collective communication)을 사용하여 변화도(gradient)와 버퍼를 동기화합니다.
좀 더 구체적으로, DDP는 ``model.parameters()``\에 의해 주어진 각 파라미터에 대해 Autograd hook을 등록하고,
hook은 역방향 전달에서 해당 변화도가 계산될 때 작동합니다.
다음으로 DDP는 이 신호를 사용하여 작업 간에 변화도 동기화를 발생시킵니다. 자세한 내용은
`DDP design note <https://pytorch.org/docs/master/notes/ddp.html>`__\를 참조하십시오.


DDP의 권장 사용법은, 여러 장치에 있을 수 있는 각 모델 복제본당 하나의 작업을 생성하는 것입니다.
DDP 작업은 동일한 기기 또는 여러 기기에 배치할 수 있지만 GPU 장치는 작업 간에 공유할 수 없습니다.
이 튜토리얼에서는 기본 DDP 사용 사례에서 시작하여,
checkpointing 모델 및 DDP와 모델 병렬 처리의 결합을 포함한 추가적인 사용 사례를 보여줍니다.


.. note::
    이 튜토리얼의 코드는 8-GPU 서버에서 실행되지만 다른 환경에서도 쉽게 적용할 수 있습니다.

``DataParallel``\과 ``DistributedDataParallel`` 간의 비교
----------------------------------------------------------

내용에 들어가기에 앞서 복잡성이 증가했음에도 불구하고
``DataParallel``\에 ``DistributedDataParallel`` 사용을 고려하는 이유를 생각해봅시다.

- 첫째, ``DataParallel``\은 단일 작업, 멀티쓰레드이며 단일 기기에서만 작동하는 반면,
  ``DistributedDataParallel``\은 다중 작업이며 단일 및 다중 기기 학습을 전부 지원합니다.
  ``DataParallel``\은 쓰레드간 GIL 경합, 복제 모델의 반복 당 생성, 산란 입력 및 수집 출력으로 인한
  추가적인 오버헤드로 인해 일반적으로 단일 시스템에서조차 ``DistributedDataParallel``\보다 느립니다.
- 모델이 너무 커서 단일 GPU에 맞지 않을 경우 **model parallel**\을 사용하여 여러 GPU로 분할해야 한다는
  `prior tutorial <https://tutorials.pytorch.kr/intermediate/model_parallel_tutorial.html>`__\을 떠올려 보세요.
  ``DistributedDataParallel``\은 **model parallel**\에서 실행되지만 ``DataParallel``\은 이때 실행되지 않습니다.
  DDP를 모델 병렬 처리와 결합하면 각 DDP 작업은 모델 병렬 처리를 사용하며
  모든 작업은 데이터 병렬 처리를 사용합니다.
- 모델이 여러 대의 기기에 존재해야 하거나 사용 사례가 데이터 병렬화 패러다임에 맞지 않는 경우,
  일반적인 분산 학습 지원을 보려면 `the RPC API <https://pytorch.org/docs/stable/rpc.html>`__\를 참조하십시오.



기본적인 사용법
---------------

DDP 모듈을 생성하기 전에 반드시 우선 작업 그룹을 올바르게 설정해야 합니다. 자세한 내용은
`PYTORCH로 분산 어플리케이션 개발하기 <https://tutorials.pytorch.kr/intermediate/dist_tuto.html>`__\에서 확인할 수 있습니다.

.. code:: python

    import os
    import sys
    import tempfile
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    import torch.multiprocessing as mp

    from torch.nn.parallel import DistributedDataParallel as DDP

    # 윈도우 플랫폼에서 torch.distributed 패키지는
    # Gloo backend, FileStore 및 TcpStore 만을 지원합니다.
    # FileStore의 경우, init_process_group 에서
    # init_method 매개변수를 로컬 파일로 설정합니다.
    # 다음 예시:
    # init_method="file:///f:/libtmp/some_file"
    # dist.init_process_group(
    #    "gloo",
    #    rank=rank,
    #    init_method=init_method,
    #    world_size=world_size)
    # TcpStore의 경우 리눅스와 동일한 방식입니다.

    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # 작업 그룹 초기화
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

이제 DDP로 감싸여진 Toy 모듈을 생성하고 더미 입력 데이터를 입력해 보겠습니다.
우선 DDP는 0순위 작업에서부터 DDP 생성자의 다른 모든 작업들에게 모델의 상태를 전달하므로,
다른 모델의 매개 변수 초기값들에서 시작하는 다른 DDP 작업들에 대하여 걱정할 필요가 없습니다.

.. code:: python

    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    def demo_basic(rank, world_size):
        print(f"Running basic DDP example on rank {rank}.")
        setup(rank, world_size)

        # 모델을 생성하고 순위 아이디가 있는 GPU로 전달
        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        cleanup()


    def run_demo(demo_fn, world_size):
        mp.spawn(demo_fn,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)

보여지는 바와 같이 DDP는 하위 수준의 분산 커뮤니케이션 세부 사항을 포함하고
로컬 모델처럼 깔끔한 API를 제공합니다. 변화도 동기화 통신(gradient synchronization communications)은
역전파 전달(backward pass)간 수행되며 역전파 계산(backward computation)과 겹치게 됩니다.
``backword()``\가 반환되면 ``param.grad``\에는 동기화된 변화도 텐서(synchronized gradient tensor)가 포함되어 있습니다.
기본적으로 DDP는 작업 그룹을 설정하는데 몇 개의 LoCs만이 필요하지만 보다 다양하게 사용하는 경우 주의가 필요합니다.

비대칭 작업 속도
--------------------

DDP에서는 생성자, 순전파(forward pass) 및 역전파 전달 호출 지점이 분산 동기화 지점(distribute synchronization point)입니다.
서로 다른 작업이 동일한 수의 동기화를 시작하고 동일한 순서로 이러한 동기화 지점에 도달하여
각 동기화 지점을 거의 동시에 진입을 요구합니다.
그렇지 않으면 빠른 작업이 일찍 도착하고 다른 작업의 대기 시간이 초과될 수 있습니다.
따라서 사용자는 작업 간의 작업량을 균형 있게 분배할 필요가 있습니다.
때때로 비대칭 작업(skewed processing) 속도는 다음과 같은 이유로 인하여 불가피하게 발생합니다.
예를 들어, 네트워크 지연, 리소스 경쟁(resource contentions), 예측하지 못한 작업량 급증 등입니다.
이러한 상황에서 시간 초과를 방지하려면, `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__\를
호출할 때 충분한 ``timeout``\값을 전달해야 합니다.

체크포인트를 저장하고 읽어오기
------------------------------

학습 중에 ``torch.save``\와 ``torch.load`` 로 모듈의 체크포인트를 만들고 그 체크포인트로부터 복구하는 것이 일반적입니다.
더 자세한 내용은 `SAVING AND LOADING MODELS <https://tutorials.pytorch.kr/beginner/saving_loading_models.html>`__\를 참고하세요.
DDP를 사용할 때, 최적의 방법은 모델을 한 작업에만 저장하고
그 모델을 모든 작업에 쓰기 과부하(write overhead)를 줄이며 읽어오는 것입니다.
이는 모든 작업이 같은 매개변수로부터 시작되고 변화도는
역전파 전달로 동기화되므로 옵티마이저(optimizer)는
매개변수를 동일한 값으로 계속 설정해야 하기 때문에 정확합니다. 이러한 최적화를 사용하는 경우,
저장이 완료되기 전에 불러오는 어떠한 작업도 시작하지 않도록 해야 합니다. 더불어, 모듈을 읽어올 때
작업이 다른 기기에 접근하지 않도록 적절한 ``map_location`` 인자를 제공해야합니다.
``map_location``\값이 없을 경우, ``torch.load``\는 먼저 모듈을 CPU에 읽어온 다음 각 매개변수가
저장된 위치로 복사하여 동일한 장치를 사용하는 동일한 기기에서 모든 작업을 발생시킵니다.
더 추가적인 실패 복구와 엘라스틱(elasticity support)은 `TorchElastic <https://pytorch.org/elastic>`__\을 참고하세요.

.. code:: python

    def demo_checkpoint(rank, world_size):
        print(f"Running DDP checkpoint example on rank {rank}.")
        setup(rank, world_size)

        model = ToyModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        if rank == 0:
            # 모든 작업은 같은 매개변수로부터 시작된다고 생각해야 합니다.
            # 무작위의 매개변수와 변화도는 역전파 전달로 동기화됩니다.
            # 그럼으로, 하나의 작업은 모델을 저장하기에 충분합니다.
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        # 작업 0이 저장한 후 작업 1이 모델을 읽어오도록 barrier()를 사용합니다.
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)

        loss_fn(outputs, labels).backward()
        optimizer.step()

        # 파일삭제를 보호하기 위해 아래에 dist.barrier()를 사용할 필요는 없습니다.
        # DDP의 역전파 전달 과정에 있는 AllReduce 옵스(ops)가 동기화 기능을 수행했기 때문에

        if rank == 0:
            os.remove(CHECKPOINT_PATH)

        cleanup()

모델 병렬 처리를 활용한 DDP
------------------------------

DDP는 다중 GPU 모델에서도 작동합니다.
다중 GPU 모델을 활용한 DDP는 대용량의 데이터를 가진 대용량 모델을 학습시킬 때 특히 유용합니다.

.. code:: python

    class ToyMpModel(nn.Module):
        def __init__(self, dev0, dev1):
            super(ToyMpModel, self).__init__()
            self.dev0 = dev0
            self.dev1 = dev1
            self.net1 = torch.nn.Linear(10, 10).to(dev0)
            self.relu = torch.nn.ReLU()
            self.net2 = torch.nn.Linear(10, 5).to(dev1)

        def forward(self, x):
            x = x.to(self.dev0)
            x = self.relu(self.net1(x))
            x = x.to(self.dev1)
            return self.net2(x)

다중 GPU 모델을 DDP로 전달할 때, ``device_ids``\와 ``output_device``\를 설정하지 않아야 합니다.
입력 및 출력 데이터는 어플리케이션 또는 모델 ``forward()``\에 의해 적절한 장치에 배치됩니다.

.. code:: python

    def demo_model_parallel(rank, world_size):
        print(f"Running DDP with model parallel example on rank {rank}.")
        setup(rank, world_size)

        # 작업을 위한 mp_model 및 장치 설정
        dev0 = (rank * 2) % world_size
        dev1 = (rank * 2 + 1) % world_size
        mp_model = ToyMpModel(dev0, dev1)
        ddp_mp_model = DDP(mp_model)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        # 출력값은 dev1에 저장
        outputs = ddp_mp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(dev1)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        cleanup()


    if __name__ == "__main__":
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        run_demo(demo_basic, world_size)
        run_demo(demo_checkpoint, world_size)
        run_demo(demo_model_parallel, world_size)

Initialize DDP with torch.distributed.run/torchrun
--------------------------------------------------------------------

We can leverage PyTorch Elastic to simplify the DDP code and initialize the job more easily.
Let's still use the Toymodel example and create a file named ``elastic_ddp.py``.

.. code:: python

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim

    from torch.nn.parallel import DistributedDataParallel as DDP

    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))

    def demo_basic():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")

        # create model and move it to GPU with id rank
        device_id = rank % torch.cuda.device_count()
        model = ToyModel().to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(device_id)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    if __name__ == "__main__":
        demo_basic()

One can then run a `torch elastic/torchrun<https://pytorch.org/docs/stable/elastic/quickstart.html>`__ command
on all nodes to initialize the DDP job created above:

.. code:: bash
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py

We are running the DDP script on two hosts, and each host we run with 8 processes, aka, we
are running it on 16 GPUs. Note that ``$MASTER_ADDR`` must be the same across all nodes.

Here torchrun will launch 8 process and invoke ``elastic_ddp.py``
on each process on the node it is launched on, but user also needs to apply cluster
management tools like slurm to actually run this command on 2 nodes.

For example, on a SLURM enabled cluster, we can write a script to run the command above
and set ``MASTER_ADDR`` as:

.. code:: bash
    export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

Then we can just run this script using the SLURM command: ``srun --nodes=2 ./torchrun_script.sh``.
Of course, this is just an example; you can choose your own cluster scheduling tools
to initiate the torchrun job.

For more information about Elastic run, one can check this
`quick start document <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ to learn more.
