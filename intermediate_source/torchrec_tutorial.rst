TorchRec 소개
===============

.. tip::
   이 튜토리얼을 최대한 활용하려면 이
   `Colab 버전 <https://colab.research.google.com/github/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb>`__ 을 사용하는 것이 좋습니다.
   이를 통해 아래에 제시된 정보를 실험할 수 있습니다.

아래 동영상이나 `유튜브 <https://www.youtube.com/watch?v=cjgj41dvSeQ>`__ 에서 따라해보세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/cjgj41dvSeQ" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

추천 시스템을 만들 때, 제품이나 페이지와 같은 객체를 임베딩으로 표현하고 싶은 경우가 많습니다.
Meta AI의 `딥러닝 추천 모델 <https://arxiv.org/abs/1906.00091>`__ 또는 DLRM을 예로 들 수 있습니다.
객체의 수가 증가함에 따라, 임베딩 테이블의 크기가 단일 GPU의 메모리를 초과할 수 있습니다.
일반적인 방법은 모델 병렬화의 일종으로, 임베딩 테이블을 여러 디바이스로 샤딩(shard)하는 것입니다.
이를 위해, TorchRec은 |DistributedModelParallel|_ 또는 DMP로 불리는 주요한 API를 소개합니다.
PyTorch의 DistributedDataParallel와 같이, DMP는 분산 학습을 가능하게하기 위해 모델을 포장합니다.

설치
----

요구 사항: python >= 3.7

TorchRec을 사용할 때는 CUDA를 적극 추천합니다. (CUDA를 사용하는 경우: cuda >= 11.0)


.. code:: shell

    # install pytorch with cudatoolkit 11.3
    conda install pytorch cudatoolkit=11.3 -c pytorch-nightly -y
    # install TorchTec
    pip3 install torchrec-nightly


개요
----

이 튜토리얼에서는 TorchRec의 ``nn.module`` |EmbeddingBagCollection|_, |DistributedModelParallel|_ API,
데이터 구조 |KeyedJaggedTensor|_ 3가지 내용을 다룹니다.


분산 설정
~~~~~~~

torch.distributed를 사용하여 환경을 설정합니다. 분산에 대한 자세한 내용은 이
`튜토리얼 <https://pytorch.org/tutorials/beginner/dist_overview.html>`__ 을 참고하세요.

여기서는 1개의 colab GPU에 대응하는 1개의 랭크(colab 프로세스)를 사용합니다.

.. code:: python

    import os
    import torch
    import torchrec
    import torch.distributed as dist

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # 참고 - 튜토리얼을 실행하려면 V100 또는 A100이 필요합니다!
    # colab free K80과 같은 오래된 GPU를 사용한다면,
    # 적절한 CUDA 아키텍처로 fbgemm를 컴파일하거나,
    # CPU에서 "gloo"로 실행해야 합니다.
    dist.init_process_group(backend="nccl")


EmbeddingBag에서 EmbeddingBagCollection으로
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch는 |torch.nn.Embedding|_ 와 |torch.nn.EmbeddingBag|_ 를 통해 임베딩을 나타냅니다.
EmbeddingBag은 임베딩의 풀(pool) 버전입니다.

TorchRec은 임베딩 컬렉션을 생성하여 이 모듈들을 확장합니다.
EmbeddingBag 그룹을 나타내고자 |EmbeddingBagCollection|_ 을 사용합니다.

여기서는, 2개의 EmbeddingBag을 가지는 EmbeddingBagCollection (EBC)을 생성합니다.
각 테이블 ``product_table`` 과 ``user_table`` 는 4096 크기의 64 차원 임베딩으로 표현됩니다.
“meta” 디바이스에서 EBC를 초기에 할당하는 방법에 주의하세요. EBC에게 아직 메모리가 할당되지 않았습니다.

.. code:: python

    ebc = torchrec.EmbeddingBagCollection(
        device="meta",
        tables=[
            torchrec.EmbeddingBagConfig(
                name="product_table",
                embedding_dim=64,
                num_embeddings=4096,
                feature_names=["product"],
                pooling=torchrec.PoolingType.SUM,
            ),
            torchrec.EmbeddingBagConfig(
                name="user_table",
                embedding_dim=64,
                num_embeddings=4096,
                feature_names=["user"],
                pooling=torchrec.PoolingType.SUM,
            )
        ]
    )


DistributedModelParallel
~~~~~~~~~~~~~~~~~~~~~~~~

이제 모델을 |DistributedModelParallel|_ (DMP)로 감쌀 준비가 되었습니다.
DMP의 인스턴스화는 다음과 같습니다.

1. 모델을 샤딩하는 방법을 결정합니다. DMP는 이용 가능한 ‘sharders’를 수집하고
   임베딩 테이블을 샤딩하는 최적의 방법 (즉, the EmbeddingBagCollection)의 ‘plan’을 작성합니다.
2. 모델을 샤딩합니다. 이 과정은 각 임베딩 테이블을 적절한 장치로 메모리를 할당하는 것을 포함합니다.

이 예제에서는 2개의 EmbeddingTables과 하나의 GPU가 있기 때문에,
TorchRec은 모두 단일 GPU에 배치합니다.

.. code:: python

    model = torchrec.distributed.DistributedModelParallel(ebc, device=torch.device("cuda"))
    print(model)
    print(model.plan)


입력과 오프셋이 있는 기본 nn.EmbeddingBag 질의
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``input`` 과 ``offsets`` 이 있는 |nn.Embedding|_ 과 |nn.EmbeddingBag|_ 를 질의합니다.
입력은 lookup 값을 포함하는 1-D 텐서입니다.
오프셋은 시퀀스가 각 예제에서 가져오는 값의 수의 합인 1-D 텐서입니다.

위의 EmbeddingBag을 다시 만들어보는 예는 다음과 같습니다.

::

   |------------|
   | product ID |
   |------------|
   | [101, 202] |
   | []         |
   | [303]      |
   |------------|

.. code:: python

    product_eb = torch.nn.EmbeddingBag(4096, 64)
    product_eb(input=torch.tensor([101, 202, 303]), offsets=torch.tensor([0, 2, 2]))


KeyedJaggedTensor로 미니 배치 표현하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

예제 및 기능별로 객체 ID가 임의의 수인 다양한 예제를 효율적으로 나타내야 합니다.
다양한 표현이 가능하도록, TorchRec 데이터구조 |KeyedJaggedTensor|_ (KJT)를 사용합니다.

“product” 와 “user”, 2개의 EmbeddingBag의 컬렉션을 참조하는 방법을 살펴봅니다.
미니배치가 3명의 사용자와 3개의 예제로 구성되어 있다고 가정합니다.
첫 번째는 2개의 product ID를 가지고, 두 번째는 아무것도 가지지 않고, 세 번째는 하나의 product ID를 가집니다.

::

   |------------|------------|
   | product ID | user ID    |
   |------------|------------|
   | [101, 202] | [404]      |
   | []         | [505]      |
   | [303]      | [606]      |
   |------------|------------|

질의는 다음과 같습니다.

.. code:: python

    mb = torchrec.KeyedJaggedTensor(
        keys = ["product", "user"],
        values = torch.tensor([101, 202, 303, 404, 505, 606]).cuda(),
        lengths = torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.int64).cuda(),
    )

    print(mb.to(torch.device("cpu")))


KJT 배치 크기는 ``batch_size = len(lengths)//len(keys)`` 인 것을 눈여겨봐 주세요.
위 예제에서 batch_size는 3입니다.



총정리하여, KJT 미니배치를 사용하여 분산 모델 질의하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

마지막으로 제품과 사용자의 미니배치를 사용하여 모델을 질의합니다.

결과 조회는 KeyedTensor를 포함합니다.
각 키(key) 또는 특징(feature)은 크기가 3x64 (batch_size x embedding_dim)인
2D 텐서를 포함합니다.

.. code:: python

    pooled_embeddings = model(mb)
    print(pooled_embeddings)


추가 자료
---------

자세한 내용은
`dlrm <https://github.com/pytorch/torchrec/tree/main/examples/dlrm>`__
예제를 참고하세요. 이 예제는 Meta의 `DLRM <https://arxiv.org/abs/1906.00091>`__ 을 사용하여
1테라바이트 데이터셋에 대한 멀티 노드 학습을 포함합니다.


.. |DistributedModelParallel| replace:: ``DistributedModelParallel``
.. _DistributedModelParallel: https://pytorch.org/torchrec/torchrec.distributed.html#torchrec.distributed.model_parallel.DistributedModelParallel
.. |EmbeddingBagCollection| replace:: ``EmbeddingBagCollection``
.. _EmbeddingBagCollection: https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.embedding_modules.EmbeddingBagCollection
.. |KeyedJaggedTensor| replace:: ``KeyedJaggedTensor``
.. _KeyedJaggedTensor: https://pytorch.org/torchrec/torchrec.sparse.html#torchrec.sparse.jagged_tensor.JaggedTensor
.. |torch.nn.Embedding| replace:: ``torch.nn.Embedding``
.. _torch.nn.Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
.. |torch.nn.EmbeddingBag| replace:: ``torch.nn.EmbeddingBag``
.. _torch.nn.EmbeddingBag: https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
.. |nn.Embedding| replace:: ``nn.Embedding``
.. _nn.Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
.. |nn.EmbeddingBag| replace:: ``nn.EmbeddingBag``
.. _nn.EmbeddingBag: https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
