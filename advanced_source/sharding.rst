TorchRec 샤딩 방식 살펴보기
===========================

이 튜토리얼에서는 ``EmbeddingPlanner`` 와 ``DistributedModelParallel`` API를 통해 
임베딩 테이블의 샤딩(Sharding) 방식을 다루며, 각기 다른 샤딩 구성을 명시적으로 설정해 봄으로써 
샤딩 방식에 따른 성능상의 이점을 탐구합니다.

설치
------------

필수 요구사항: - python >= 3.7

TorchRec을 사용할 때는 CUDA 환경을 사용하는 것을 강력히 권장합니다.   
CUDA를 사용할 경우: - cuda >= 11.0

.. code:: python

    # conda를 설치하면 condatoolkit 11.3과 함께 pytorch를 쉽게 설치할 수 있습니다.
    !sudo rm Miniconda3-py37_4.9.2-Linux-x86_64.sh Miniconda3-py37_4.9.2-Linux-x86_64.sh.*
    !sudo wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
    !sudo chmod +x Miniconda3-py37_4.9.2-Linux-x86_64.sh
    !sudo bash ./Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -f -p /usr/local

.. code:: python

    # PyTorch 설치 (cudatoolkit 11.3 포함)
    !sudo conda install pytorch cudatoolkit=11.3 -c pytorch-nightly -y

torchRec을 설치하면 자동으로 `FBGEMM <https://github.com/pytorch/fbgemm>`__,도 함께 설치됩니다.
FBGEMM은 CUDA 커널과 GPU 연산이 포함된 연산 라이브러리 모음입니다.

.. code:: python

    # torchrec 설치
    !pip3 install torchrec-nightly

Colab 환경에서 다중 프로세싱을 사용하기 위해 multiprocess 패키지를 설치해야 합니다.
이 패키지는 IPython 환경에서 멀티프로세싱 프로그래밍이 가능하게 합니다.

.. code:: python

    !pip3 install multiprocess

Colab 런타임 환경 설정:
Colab에서는 런타임이 /usr/lib 폴더에서 공유 라이브러리를 탐색하기 때문에, 
/usr/local/lib/ 에 설치된 라이브러리를 복사해야 합니다.
**이 과정은 Colab 환경에서 필수적인 단계 입니다.**.

.. code:: python

    !sudo cp /usr/local/lib/lib* /usr/lib/

**이 시점에서 새로 설치된 패키지를 인식하도록 런타임을 재시작하세요** 
재시작 직후 아래 단계를 실행하여 Python이 패키지의 위치를 알 수 있도록 합니다.
**런타임을 재시작한 후 항상 이 단계를 실행해야 합니다.**

.. code:: python

    import sys
    sys.path = ['', '/env/python', '/usr/local/lib/python37.zip', '/usr/local/lib/python3.7', '/usr/local/lib/python3.7/lib-dynload', '/usr/local/lib/python3.7/site-packages', './.local/lib/python3.7/site-packages']


분산 설정 (Distributed Setup)
-----------------

노트북 환경에서는 `SPMD <https://en.wikipedia.org/wiki/SPMD>`_ 프로그램을 직접 실행할 
수 없기 때문에, 여기서는 멀티프로세싱을 활용하여 이를 유사하게 구현합니다. TorchRec을 사용할 
때는 사용자가 직접 `SPMD <https://en.wikipedia.org/wiki/SPMD>`_ 실행 환경을 설정해야 합니다. 이 예시에서는 PyTorch의 
분산 통신(Distributed Communication) 백엔드가 정상적으로 동작할 수 있도록 환경 설정을 구성합니다.

.. code:: python

    import os
    import torch
    import torchrec

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

임베딩 모델 구성 (Constructing our embedding model)
--------------------------------

여기에서는 `EmbeddingBagCollection <https://github.com/facebookresearch/torchrec/blob/main/torchrec/modules/embedding_modules.py#L59>`_ 을 사용하여, 여러 개의 임베딩 테이블로 구성된 임베딩 백(embedding bag) 모델을 구축합니다.

이번 예시에서는 4개의 임베딩 백(embedding bag) 으로 구성된 EmbeddingBagCollection (EBC)를 생성합니다.
테이블은 두 가지 크기로 구분됩니다:
큰 테이블과 작은 테이블로, 각각 행 크기 4096과 1024로 구분됩니다.
모든 테이블의 임베딩 차원은 동일하게 64차원으로 설정합니다.

또한, 각 테이블에 대해 ``ParameterConstraints`` 데이터 구조를 설정합니다.
이 구조는 모델 병렬화 API가 테이블의 샤딩 및 배치 전략을 결정하는 데 도움이 되는 힌트를 제공합니다.
TorchRec에서는 다음과 같은 샤딩 방식을 지원합니다:
\* ``table-wise``: 전체 테이블을 하나의 디바이스에 배치; \*
``row-wise``: 테이블을 행 단위로 균등 분할하여 통신 그룹의 각 디바이스에 하나씩 배치; \* 
``column-wise``:
임베딩 차원을 기준으로 균등 분할하여 각 디바이스에 하나씩 배치; \* 
``table-row-wise``: NVLink와 같은 빠른 디바이스 간 연결을 활용해, 호스트 내부 통신에 최적화된 특수 샤딩 방식; \* 
``data_parallel``:모든 디바이스에 테이블 전체를 복제;

EBC를 처음 생성할 때 “meta” 디바이스에 할당하는 점에 주의하세요. 
이는 아직 실제 메모리를 할당하지 않고, 이후에 필요한 시점에 할당하도록 지시하는 설정입니다.

.. code:: python

    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.embedding_types import EmbeddingComputeKernel
    from torchrec.distributed.types import ShardingType
    from typing import Dict

    large_table_cnt = 2
    small_table_cnt = 2
    large_tables=[
      torchrec.EmbeddingBagConfig(
        name="large_table_" + str(i),
        embedding_dim=64,
        num_embeddings=4096,
        feature_names=["large_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
      ) for i in range(large_table_cnt)
    ]
    small_tables=[
      torchrec.EmbeddingBagConfig(
        name="small_table_" + str(i),
        embedding_dim=64,
        num_embeddings=1024,
        feature_names=["small_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
      ) for i in range(small_table_cnt)
    ]

    def gen_constraints(sharding_type: ShardingType = ShardingType.TABLE_WISE) -> Dict[str, ParameterConstraints]:
      large_table_constraints = {
        "large_table_" + str(i): ParameterConstraints(
          sharding_types=[sharding_type.value],
        ) for i in range(large_table_cnt)
      }
      small_table_constraints = {
        "small_table_" + str(i): ParameterConstraints(
          sharding_types=[sharding_type.value],
        ) for i in range(small_table_cnt)
      }
      constraints = {**large_table_constraints, **small_table_constraints}
      return constraints

.. code:: python

    ebc = torchrec.EmbeddingBagCollection(
        device="cuda",
        tables=large_tables + small_tables
    )

멀티프로세싱에서의 DistributedModelParallel
-------------------------------------------

이제, `SPMD <https://en.wikipedia.org/wiki/SPMD>`_ 실행 중에 각 프로세스(rank) 가 수행하는 작업을 
모방하기 위한 단일 프로세스 실행 함수를 정의합니다.

이 코드에서는 다른 프로세스들과 함께 모델을 공동으로 샤딩하고, 그에 따라 메모리를 적절히 할당합니다.
먼저 프로세스 그룹을 설정한 뒤, 플래너를 사용해 임베딩 테이블의 배치를 수행하고,
그 결과를 바탕으로 ``DistributedModelParallel`` 을 통해 샤딩된 모델을 생성합니다.

.. code:: python

    def single_rank_execution(
        rank: int,
        world_size: int,
        constraints: Dict[str, ParameterConstraints],
        module: torch.nn.Module,
        backend: str,
    ) -> None:
        import os
        import torch
        import torch.distributed as dist
        from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
        from torchrec.distributed.model_parallel import DistributedModelParallel
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.types import ModuleSharder, ShardingEnv
        from typing import cast

        def init_distributed_single_host(
            rank: int,
            world_size: int,
            backend: str,
            # pyre-fixme[11]: `ProcessGroup`이 타입(type)으로 정의되어 있지 않습니다.
        ) -> dist.ProcessGroup:
            os.environ["RANK"] = f"{rank}"
            os.environ["WORLD_SIZE"] = f"{world_size}"
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            return dist.group.WORLD

        if backend == "nccl":
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        topology = Topology(world_size=world_size, compute_device="cuda")
        pg = init_distributed_single_host(rank, world_size, backend)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
        )
        sharders = [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
        plan: ShardingPlan = planner.collective_plan(module, sharders, pg)
    
        sharded_model = DistributedModelParallel(
            module,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            sharders=sharders,
            device=device,
        )
        print(f"rank:{rank},sharding plan: {plan}")
        return sharded_model


멀티프로세싱 실행 (Multiprocessing Execution)
~~~~~~~~~~~~~~~~~~~~~~~~~

이제 여러 개의 GPU rank를 나타내는 다중 프로세스 환경에서 코드를 실행해 보겠습니다.

.. code:: python

    import multiprocess
       
    def spmd_sharing_simulation(
        sharding_type: ShardingType = ShardingType.TABLE_WISE,
        world_size = 2,
    ):
      ctx = multiprocess.get_context("spawn")
      processes = []
      for rank in range(world_size):
          p = ctx.Process(
              target=single_rank_execution,
              args=(
                  rank,
                  world_size,
                  gen_constraints(sharding_type),
                  ebc,
                  "nccl"
              ),
          )
          p.start()
          processes.append(p)
    
      for p in processes:
          p.join()
          assert 0 == p.exitcode

테이블 단위 샤딩 (Table-Wise Sharding)
~~~~~~~~~~~~~~~~~~~

이제 두 개의 GPU를 사용하여 2개의 프로세스로 코드를 실행해 보겠습니다. 출력된 plan을 보면, 
각 테이블이 GPU 간에 어떻게 샤딩되었는지를 확인할 수 있습니다. 각 노드는 큰 테이블 하나와 
작은 테이블 하나씩을 가지며, 이는 플래너가 임베딩 테이블의 로드 밸런싱을 고려하여 분배했음을 
보여줍니다. Table-wise 샤딩은 여러 개의 소형~중형 규모 테이블을 디바이스 간에 균형 있게 
분산시키기 위한 가장 일반적이고 기본적인 샤딩 방식입니다.

.. code:: python

    spmd_sharing_simulation(ShardingType.TABLE_WISE)


.. parsed-literal::

    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:0/cuda:0)])), 'large_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:0/cuda:0)])), 'small_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:1/cuda:1)]))}}
    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:0/cuda:0)])), 'large_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:0/cuda:0)])), 'small_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:1/cuda:1)]))}}

다른 샤딩 방식 살펴보기 (Explore other sharding modes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

앞서 table-wise 샤딩이 어떻게 작동하고 테이블 배치를 균형 있게 수행하는지를 살펴보았습니다.이제는 
로드 밸런싱(load balance)에 더 초점을 맞춘 다른 샤딩 방식, 즉 row-wise 샤딩을 살펴보겠습니다.
Row-wise 샤딩은 특히 임베딩 행의 수가 매우 많아 단일 디바이스 메모리에 전체 테이블을 담을 수 없는 
큰 테이블을 처리하기 위한 방식입니다. 이 방법은 모델 내의 초대형 테이블을 효율적으로 분산 배치할 수 
있게 해줍니다. 출력된 플랜 로그의 ``shard_sizes`` 섹션을 보면, 테이블이 행 단위로 절반씩 나뉘어 
두 개의 GPU에 분산된 것을 확인할 수 있습니다.

.. code:: python

    spmd_sharing_simulation(ShardingType.ROW_WISE)


.. parsed-literal::

    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)]))}}
    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)]))}}

반면, column-wise 샤딩은 임베딩 차원이 큰 테이블에서 발생하는 로드 불균형 문제를 해결하기 위한 방식입니다.
이 경우 테이블을 세로 방향(임베딩 차원 기준) 으로 분할합니다. 출력된 플랜 로그의 ``shard_sizes`` 섹션을 
보면, 테이블이 임베딩 차원 기준으로 절반씩 나뉘어 두 개의 GPU에 분산된 것을 확인할 수 있습니다.

.. code:: python

    spmd_sharing_simulation(ShardingType.COLUMN_WISE)


.. parsed-literal::

    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)]))}}
    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)]))}}

``table-row-wise`` 방식은 멀티 호스트(multi-host) 환경에서 동작하도록 설계되어 있기 때문에,
현재는 이를 시뮬레이션할 수 없습니다. 앞으로는 Python 기반의 `SPMD <https://en.wikipedia.org/wiki/SPMD>`_ 예제를 통해
``table-row-wise`` 방식을 사용하여 모델을 학습하는 방법을 소개할 예정입니다.

data-parallel 방식에서는 모든 디바이스에 동일한 테이블을 복제하여 사용합니다.

.. code:: python

    spmd_sharing_simulation(ShardingType.DATA_PARALLEL)


.. parsed-literal::

    rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'large_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None)}}
    rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'large_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None)}}

