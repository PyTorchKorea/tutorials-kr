DeviceMesh 시작하기
=====================================================

**저자**: `Iris Zhang <https://github.com/wz337>`__, `Wanchao Liang <https://github.com/wanchaol>`__
**역자:** `강동석 <https://github.com/ehdtjr>`_

.. note::
   |edit| 이 튜토리얼은 `github <https://github.com/PyTorchKorea/tutorials-kr/blob/master/recipes_source/distributed_device_mesh.rst>`__ 에서 보거나 편집할 수 있습니다.

사전 준비(Prerequisites):

- `분산 통신 패키지 - torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__
- Python 3.8 - 3.11
- PyTorch 2.2


분산 학습을 위해 분산 통신기(communicator), 즉 NVIDIA Collective Communication Library(NCCL) 통신기를 설정하는 일은 상당한 어려움이 될 수 있습니다. 서로 다른 병렬화 방식을 조합해야 하는 작업이라면,
각 병렬화 방식마다 NCCL 통신기(예: :class:`ProcessGroup`)를 직접 설정하고 관리해야 합니다. 이 과정은 복잡하고 오류가 발생하기 쉽습니다.
:class:`DeviceMesh` 는 이 과정을 단순화할 수 있고, 더 다루기 쉽게 만들며 오류 발생 가능성도 줄여줍니다.

DeviceMesh란 무엇인가
---------------------
:class:`DeviceMesh` 는 :class:`ProcessGroup` 을 관리하는 상위 수준의 추상화입니다.
서로 다른 하위 프로세스 그룹에 대해 랭크(rank)를 어떻게 올바르게 설정할지 고민하지 않고도, 노드 간(inter-node) 및 노드 내(intra-node) 프로세스 그룹을 손쉽게 만들 수 있습니다.
또한 :class:`DeviceMesh` 를 통해 다차원 병렬화에 사용되는 내부의 프로세스 그룹과 디바이스를 쉽게 관리할 수 있습니다.

.. figure:: /_static/img/distributed/device_mesh.png
   :width: 100%
   :align: center
   :alt: PyTorch DeviceMesh

DeviceMesh가 유용한 이유
------------------------
DeviceMesh는 여러 병렬화 방식을 조합(composability)해야 하는 다차원 병렬화(예: 3차원 병렬)를 다룰 때 유용합니다. 예를 들어, 병렬화 방식이 호스트 간 통신과 각 호스트 내부의 통신을 모두 요구하는 경우가 그렇습니다.
위 이미지는 동일한 구성의 환경에서 각 호스트 내부의 디바이스를 연결하고, 각 디바이스를 다른 호스트의 대응 디바이스와 연결하는 2D 메시를 만들 수 있음을 보여줍니다.

DeviceMesh가 없다면, 어떤 병렬화를 적용하기 전에 각 프로세스마다 NCCL 통신기와 CUDA 디바이스를 직접 설정해야 하며, 이는 꽤 복잡한 작업입니다.
다음 코드는 :class:`DeviceMesh` 없이 하이브리드 샤딩(hybrid sharding) 2차원 병렬 패턴을 설정하는 예시입니다.
먼저 샤드(shard) 그룹과 복제 그룹을 직접 계산하고, 각 랭크에 알맞은 그룹을 할당해야 합니다.

.. code-block:: python

    import os

    import torch
    import torch.distributed as dist

    # 월드 토폴로지 이해
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Running example on {rank=} in a world with {world_size=}")

    # 2차원 형태의 병렬 패턴을 관리하기 위한 프로세스 그룹 생성
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    # 샤드 그룹 생성 (예: (0, 1, 2, 3), (4, 5, 6, 7))
    # 각 랭크에 올바른 샤드 그룹 할당
    num_node_devices = torch.cuda.device_count()
    shard_rank_lists = list(range(0, num_node_devices // 2)), list(range(num_node_devices // 2, num_node_devices))
    shard_groups = (
        dist.new_group(shard_rank_lists[0]),
        dist.new_group(shard_rank_lists[1]),
    )
    current_shard_group = (
        shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
    )

    # 복제 그룹 생성 (예: (0, 4), (1, 5), (2, 6), (3, 7))
    # 각 랭크에 올바른 복제 그룹 할당
    current_replicate_group = None
    shard_factor = len(shard_rank_lists[0])
    for i in range(num_node_devices // 2):
        replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
        replicate_group = dist.new_group(replicate_group_ranks)
        if rank in replicate_group_ranks:
            current_replicate_group = replicate_group

위 코드를 실행하려면 PyTorch Elastic을 활용할 수 있습니다. ``2d_setup.py`` 라는 파일을 만든 뒤,
`torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ 명령을 실행하세요.

.. code-block:: python

    torchrun --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400 2d_setup.py

.. note::
    예시를 간단히 보여주기 위해 단일 노드만 사용해 2D 병렬을 시뮬레이션하고 있습니다. 이 코드는 멀티 호스트 환경에서도 그대로 사용할 수 있습니다.

:func:`init_device_mesh` 를 활용하면 위의 2D 설정을 단 두 줄로 끝낼 수 있고, 필요할 때는
내부의 :class:`ProcessGroup` 에도 접근할 수 있습니다.


.. code-block:: python

    from torch.distributed.device_mesh import init_device_mesh
    mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))

    # `get_group` API를 통해 내부 프로세스 그룹에 접근할 수 있습니다.
    replicate_group = mesh_2d.get_group(mesh_dim="replicate")
    shard_group = mesh_2d.get_group(mesh_dim="shard")

``2d_setup_with_device_mesh.py`` 라는 파일을 만든 뒤,
`torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ 명령을 실행하세요.

.. code-block:: python

    torchrun --nproc_per_node=8 2d_setup_with_device_mesh.py


HSDP에서 DeviceMesh를 사용하는 방법
-------------------------------

Hybrid Sharding Data Parallel(HSDP)은 호스트 내부에서는 FSDP를, 호스트 간에는 DDP를 수행하는 2D 전략입니다.

DeviceMesh가 간단한 설정으로 모델에 HSDP를 적용하는 데 어떻게 도움이 되는지 예시로 살펴보겠습니다. DeviceMesh를 사용하면
샤드 그룹과 복제 그룹을 직접 만들고 관리하지 않아도 됩니다.

.. code-block:: python

    import torch
    import torch.nn as nn

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard as FSDP


    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.net1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(self.relu(self.net1(x)))


    # HSDP: MeshShape(2, 4)
    mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_replicate", "dp_shard"))
    model = FSDP(
        ToyModel(), device_mesh=mesh_2d
    )

``hsdp.py`` 라는 파일을 만든 뒤,
`torch elastic/torchrun <https://pytorch.org/docs/stable/elastic/quickstart.html>`__ 명령을 실행하세요.

.. code-block:: python

    torchrun --nproc_per_node=8 hsdp.py

사용자 정의 병렬 방식에서 DeviceMesh를 사용하는 방법
--------------------------------------------------------
대규모 학습 환경에서는 더 복잡한 사용자 정의 병렬 학습 구성을 다뤄야 할 수도 있습니다. 예를 들어, 서로 다른 병렬화 방식에 맞춰 하위 메시(sub-mesh)를 나누어 사용해야 할 수 있습니다.
DeviceMesh를 사용하면 상위 메시에서 하위 메시를 잘라내고, 상위 메시를 초기화할 때 이미 만들어진 NCCL 통신기를 그대로 재사용할 수 있습니다.

.. code-block:: python

    from torch.distributed.device_mesh import init_device_mesh
    mesh_3d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp"))

    # 상위 메시에서 하위 메시를 잘라낼 수 있습니다.
    hsdp_mesh = mesh_3d["replicate", "shard"]
    tp_mesh = mesh_3d["tp"]

    # `get_group` API를 통해 내부 프로세스 그룹에 접근할 수 있습니다.
    replicate_group = hsdp_mesh["replicate"].get_group()
    shard_group = hsdp_mesh["shard"].get_group()
    tp_group = tp_mesh.get_group()


결론
----------
지금까지 :class:`DeviceMesh` 와 :func:`init_device_mesh` 를 살펴보고,
이를 활용해 클러스터에 분산된 디바이스의 배치를 표현하는 방법도 알아봤습니다.

더 자세한 내용은 다음 자료를 참고하세요.

- `2D parallel combining Tensor/Sequence Parallel with FSDP <https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py>`__
- `Composable PyTorch Distributed with PT2 <https://static.sched.com/hosted_files/pytorch2023/d1/%5BPTC%2023%5D%20Composable%20PyTorch%20Distributed%20with%20PT2.pdf>`__
