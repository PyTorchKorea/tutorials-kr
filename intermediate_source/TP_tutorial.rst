Tensor Parallel (TP)를 활용한 대규모 트랜스포머 모델 훈련
======================================================

**저자**: `Wanchao Liang <https://github.com/wanchaol>`__, `Tianyu Liu <https://github.com/tianyu-l>`__
**번역**: `강호현 <https://github.com/stats-dev>`__

.. note::
   |edit| 이 튜토리얼은 `github <https://github.com/pytorchkorea/tutorials-kr/blob/main/intermediate_source/TP_tutorial.rst>`__ 에서 확인하고 편집하세요.

이 튜토리얼에서는 Tensor Parallel과 Fully Sharded Data Parallel를 활용하여, 수백에서 수천 개의 GPU로 대규모 트랜스포머 계열의 모델을 훈련하는 방법을 설명합니다.

사전 준비:

- CUDA/Linux 환경에서 PyTorch 2.3.0 이상 설치되어야 합니다.
-  `Tensor Parallel APIs <https://pytorch.org/docs/stable/distributed.tensor.parallel.html>`__
-  `DeviceMesh 시작하기 <https://tutorials.pytorch.kr/recipes/distributed_device_mesh.html>`__
-  `Fully Sharded Data Parallel 시작하기 <https://tutorials.pytorch.kr/intermediate/FSDP_tutorial.html>`__



Tensor Parallel은 어떻게 작동합니까?
-------------------------------------------
Tensor Parallel (TP)는 기존 `Megatron-LM <https://arxiv.org/abs/1909.08053>`__ 논문에서 제안된 방식으로, 대규모 트랜스포머(Transformer) 모델을 효율적으로 훈련하기 위한 모델 병렬처리(parallelism) 기법입니다.
이 튜토리얼에서 언급한 `Sequence Parallel <https://arxiv.org/abs/2205.05198>`__ (SP)는 Tensor Parallel의 한 변형으로, 훈련 중 활성화 메모리를 절약하기 위해 ``nn.LayerNorm`` 혹은 ``RMSNorm`` 를 시퀀스 차원으로 샤딩 합니다.
모델이 커질수록, 활성화 메모리가 병목이 되므로, Tensor Parallel 학습에서는 주로 ``LayerNorm`` 이나 ``RMSNorm`` 레이어에 시퀀스 병렬(Sequence Parallel)을 적용합니다.


.. figure:: /_static/img/distributed/megatron_lm.png
   :width: 100%
   :align: center
   :alt: Megatron-LM TP

   그림 1. 트랜스포머 모델의 MLP 및 Self-Attention 레이어에 행렬 연산이 attention/MLP에서 샤딩된 계산으로 이루어지고, 이는 Tensor Parallel 방식으로 sharding된 구조를 나타냅니다. (`이미지 출처 <https://arxiv.org/abs/1909.08053>`__)


고수준에서 PyTorch Tensor Parallel은 다음과 같이 작동합니다.

**Sharding 초기화**

* 각 레이어에 어떤 ``ParallelStyle`` 을 적용할지 결정하고, ``parallelize_module`` 을 호출해서 초기화된 모듈을 샤딩합니다.
* 병렬화된 모듈은 모델 파라미터를 DTensor로 교체하고, DTensor는 샤딩하는 연산을 사용하여 병렬화된 모듈을 실행하는 역할을 담당합니다.

**런타임 순방향/역방향**

* 사용자가 지정한 개별 ``ParallelStyle`` 의 입력/출력 DTensor 레이아웃에 따라, 입력/출력에 대한 DTensor 레이아웃을 변환하는 적절한 커뮤니케이션 동작을 실행합니다. (예를 들어, ``allreduce``, ``allgather``, ``reduce_scatter`` )
* 병렬화된 레이어( ``nn.Linear`` , ``nn.Embedding`` )은 연산 및 메모리를 절약하기 위해 샤딩된 연산을 실행합니다. 

Tensor Parallel을 적용해야 하는 시기와 이유
-----------------------------------------------
PyTorch의 Fully Sharded Data Parallel(FSDP)는 이미 모델 학습을 특정 수의 GPU로 조정할 수 있는 기능을 갖추고 있습니다. 그러나, 모델 크기와 GPU 양 측면에서 모델 학습을 더 확장하려면, 
Tensor Parallel과 FSDP의 결합이 필요한, 다음과 같은 추가적인 과제가 다수 발생할 수 있습니다.

1. GPU 수가 지나치게 커짐에 따라 (128/256 GPU 초과), FSDP 집합(예를 들어, ``allgather`` )은 ring latency에 많은 영향을 받습니다. TP/SP를 FSDP 위에 구현하여, FSDP를 호스트 간에만 적용하여 FSDP의 규모를 8개로 줄일 수 있으며, 그에 따라 지연 비용도 동일하게 줄일 수 있습니다.
2. 수렴 및 GPU 메모리 제한으로 인해 글로벌 배치 크기를 GPU 수보다 높게 설정할 수 없는 데이터 병렬 처리의 한계를 달성하려면, Tensor/Sequence Parallel이 글로벌 배치 크기를 "추정(ballpark)"하고, 더 많은 GPU로 확장하는 유일한 방법입니다.
3. 특정 유형의 모델에서는 로컬 배치 크기가 작아지면, TP/SP가 부동 소수점 연산(FLOPS)에 더 최적화된 행렬 곱 형태를 생성할 수 있습니다.

사전학습 시 이러한 한계를 경험하는 것은 흔한 일입니다. 현재로서는 수십억 혹은 수조 단위의 토큰으로 대규모 언어 모델(LLM)을 학습하려면 수천 대의 GPU를 사용하더라도 수개월이 걸릴 수 있습니다.

* LLM을 대규모로 훈련할 때는 항상 한계 1에 도달합니다. 예를 들어, Llama 2 70B 모델은 35일 동안 2천개 GPU로 훈련되었고, 2천개 규모에서는 다차원 병렬 처리가 필요합니다.
* Transformer 모델이 커지면 (예를 들면, Llama2 70B), 빠르게 한계 2에 도달할 것입니다. 메모리 및 수렴 제약 조건 때문에 로컬 ``batch_size=1`` 조건 조차도 FSDP를 단독으로 사용할 수 없습니다. 예를 들어, Llama2 글로벌 배치 크기는 1K이므로, 2K GPU에서 오직 데이터 병렬 처리만으로 사용될 수 없습니다.


Tensor Parallel을 적용하는 방법
--------------------------------------

PyTorch Tensor Parallel API는 모델의 각 개별 레이어에 대한 샤딩을 구성하기 위해 다음과 같은 모듈 수준의 이전 세트 (``ParallelStyle``)를 제공합니다.


* ``ColwiseParallel`` 및 ``RowwiseParallel`` : 열 혹은 행 방식으로 ``nn.Linear`` 과 ``nn.Embedding`` 를 공유합니다.
* ``SequenceParallel`` : ``nn.LayerNorm`` , ``nn.Dropout`` , ``RMSNormPython`` 등에서 샤딩 연산을 수행합니다.
* ``PrepareModuleInput`` 및 ``PrepareModuleOutput``: 적절한 커뮤니케이션 작업을 가진 모듈 입력/출력 샤딩 레이아웃을 구성합니다.

PyTorch 네이티브 Tensor Parallel API를 사용하는 법을 설명하기 위해, 일반적인 트랜스포머 모델을 살펴보겠습니다. 이번 튜토리얼에서는 커뮤니티에서도 널리 사용되는 최신 `Llama2 모델 <https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/llama2_model.py>`__ 을 레퍼런스 트랜스포머 모델 구현으로 사용합니다.
Tensor Parallel이 개별 tensor를 여러 디바이스에서 샤딩하기 때문에, 먼저 분산 환경(NCCL 통신기)을 설정해야 합니다.

Tensor Parallelism은 PyTorch DDP/FSDP와 유사한 단일 프로그램 멀티 데이터 (SPMD) 샤딩 알고리즘이며, 이 알고리즘은 PyTorch DTensor 내부 원리를 바탕으로 샤딩을 수행합니다. 또한 디바이스 관리 및 샤딩을 위해 DeviceMesh 추상화(내부적으로 프로세스 그룹 관리)를 활용합니다.
DeviceMesh를 활용하여 다차원 병렬화를 활용하는 방법은 `이 튜토리얼 <https://tutorials.pytorch.kr/recipes/distributed_device_mesh.html>`__ 을 참조하세요. Tensor Parallel은 일반적으로 각 호스트 내부에서 작동하므로, 먼저 호스트 내 8개의 GPU를 연결하는 DeviceMesh를 초기화해보겠습니다.

.. code-block:: python

    from torch.distributed.device_mesh import init_device_mesh

    tp_mesh = init_device_mesh("cuda", (8,))


이제 DeviceMesh를 초기화했으므로, Llama 2 모델 아키텍처를 자세히 살펴보고 Tensor Parallel 샤딩을 수행하는 방법을 살펴보겠습니다.
여기서 트랜스포머 모델이 확장하기 위해 동일한 ``TransformerBlock`` 을 쌓는 핵심 ``TransformerBlock`` 에 초점을 둡니다.

핵심 ``TransformerBlock`` 은 ``Attention`` 레이어와 ``FeedForward`` 레이어로 구성되어 있습니다. 먼저 더 간단한 ``FeedForward`` 레이어를 살펴보겠습니다.
``FeedForward`` 레이어의 경우, 세 개의 선형 레이어로 구성되어 있고, 순방향 함수를 고려해서 SwiGLU 스타일의 MLP를 수행합니다.

.. code-block:: python

    # 순전파 레이어에서 순방향으로
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

``w1`` 및 ``w3`` 행렬곱을 동시에 수행하고, 결합된 w1/w3 선형 투영 결과와 함께 ``w2`` 행렬곱을 수행합니다. 
이는 Tensor Parallelism 논문의 아이디어를 사용해서 w1/w3 선형 레이어를 열 우선 방식으로 샤딩하고, 행 우선 방식으로 ``w2`` 선형 레이어를 샤딩하여, 세 레이어 모두 끝에서 하나의 ``allreduce`` 통신만 발생하는 것을 의미합니다.
PyTorch 네이티브 Tensor Parallel을 사용하여 다음과 같이 ``FeedForward`` 레이어에 대해 ``parallelize_plan`` 을 간단히 만들 수 있습니다.


.. code-block:: python

    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    layer_tp_plan = {
        # 기본적으로 ColwiseParallel으로 입력 레이아웃이 복제됩니다
        # RowwiseParallel으로 출력 레이아웃이 복제됩니다
        "feed_foward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
    }


 
이는 단순히 PyTorch Tensor Parallel API를 이용하여 ``FeedForward`` 레이어의 샤딩을 구성하는 방식입니다. 사용자는 개별 레이어를 샤딩하는 방법만 지정하면 되고, 통신(예를 들어, ``allreduce`` )은 내부적으로 발생한다는 점을 기억합니다.
 ``Attention`` 레이어로 넘어 갑니다. 이 레이어는 ``wq`` , ``wk`` , ``wv`` 선형 레이어로 구성되어, 입력을 ``q`` / ``k`` / ``v`` 로 투영한 다음에 ``wo`` 선형 레이어로 어텐션 및 출력 투영을 수행합니다.
여기서 Tensor Parallelism은 q/k/v 투영에 대해 열 중심으로 샤딩을 수행하고, ``wo`` 선형 투영에 대해 행 중심으로 샤딩을 수행합니다. 따라서, 방금 작성한 ``tp_plan`` 에 어텐션 플랜을 추가할 수 있습니다.

.. code-block:: python

    layer_tp_plan = {
        # 기본적으로 ColwiseParallel 입력 레이아웃 반복
        # 그리고 RowwiseParallel 출력 레이아웃 반복
        "attention.wq": ColwiseParallel(use_local_output=False),
        "attention.wk": ColwiseParallel(use_local_output=False),
        "attention.wv": ColwiseParallel(use_local_output=False),
        "attention.wo": RowwiseParallel(),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
    }


이는 대체로 ``TransformerBlock`` 에 Tensor Parallel을 적용해야하는 ``layer_tp_plan`` 입니다. 그러나 알아야하는 한가지는 선형 레이어를 열 단위로 샤딩할 때, 선형 레이어의 출력이 마지막 tensor 차원에서 샤딩되고, 행 단위로 샤딩된 선형 레이어가 마지막 차원에서 샤딩된 입력을 직접 받아들인다는 것입니다.
만일 열 단위 선형과 행 단위 선형 사이에 더 많은 tensor 연산 (예를 들어, view operation) 이 있다면, 샤딩된 형태로 관련 모양의 연산을 조정해야 합니다.

Llama 모델의 경우, 어텐션 레이어에서는 형태와 관련된 여러 뷰 연산이 있습니다. 구체적으로, ``wq`` / ``wk`` / ``wv`` 선형 레이어에서 열 단위 병렬화의 경우, 활성화 tensor는  ``num_heads`` 차원에서 샤딩됩니다.

마지막으로, 각 ``TransformerBlock`` 에 대한 계획을 효과적으로 실행하려면 ``parallelize_module`` API를 호출해야 합니다. 내부적으로는 ``Attention``  및  ``FeedForward`` 레이어 내부 모델 파라미터를 DTensor에 분배하고, 필요하다면 모델 입력과 출력(각각 모듈 이전 및 이후)에 대한 통신 훅을 등록합니다.

.. code-block:: python

    for layer_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = {...}  # 예를 들어, 이전에 생성된 플랜

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

각 ``TransformerBlock`` 에 대한 샤딩 계획을 구체화했고, 보통 첫 번째 레아어에 ``nn.Embedding``가 있고, 마지막 ``nn.Linear`` 투영 레이어가 있는데, 첫 번째 ``nn.Embedding`` 에는 행 단위 혹은 열 단위 샤딩을 선택하고, 사용자가 적절한 입력 및 출력 레이아웃이 지정된 마지막 ``nn.Linear`` 투영 레이어에는 열 단위 샤딩을 선택할 수 있습니다.
다음 예시를 참고합니다.

.. code-block:: python

    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "output": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        }
    )

.. note::
    해당 모델이 너무 커서 CPU 메모리에 맞지 않는 경우, ``meta`` 장치 초기화 (예를 들어, 메타 장치에서 먼저 초기화하거나 레이어를 샤딩하고 모델을 구체화하는 등)를 사용하거나 트랜스포머 모델 초기화 중에 ``TransformerBlock`` 레이어를 레이어별로 병렬화할 수 있습니다.

``LayerNorm/RMSNorm`` 레이어에 시퀀스 병렬(Sequence Parallel) 적용하기
----------------------------------------------------------------

시퀀스 병렬(Sequence Parallel)은 앞서 설명한 Tensor Parallel 위에서 동작합니다. 기본적인 Tensor Parallel은  ``Attention`` 모듈과 ``FeedForward`` 모듈 내에서만 tensor를 샤딩하고 모듈 입력과 출력 (즉, forward pass의 활성화 및 backward pass에서 변화도)을 복제되도록 유지하는 것과 비교할 때, 시퀀스 병렬은 시퀀스 차원에서 샤딩된 상태를 유지합니다.

일반적인 ``TransformerBlock`` 에서 순방향 함수는 norm 레이어( ``LayerNorm`` 혹은 ``RMSNorm`` ), 어텐션 레이어, 순전파 레이어, residual 연결을 결합합니다. 예를 들면, 다음과 같습니다.

.. code-block:: python

    # TransformerBlock에서 순방향
    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

대부분 유즈케이스에서, 활성화 (그리고 변화도)는 ``Attention`` 및 ``FeedForward`` 모듈 외부의 ``[batch size, sequence length, hidden dimension]`` 모양입니다. DTensor의 언어로, 시퀀스 병렬은 모듈의 순방향/역방향 모두 ``Shard(1)`` 레이아웃을 사용하여 활성화 연산을 수행합니다.

이전 코드 예시에 이어서, 아래 코드는 ``TransformerBlock`` 내부의 norm 레이어에 시퀀스 병렬을 적용하는 방법을 설명합니다.

먼저 시퀀스 병렬에 필요한 의존성을 가져오겠습니다.

.. code-block:: python

    from torch.distributed.tensor.parallel import (
        PrepareModuleInput,
        SequenceParallel,
    )


다음으로  ``layer_tp_plan`` 을 수정해서 ``RMSNorm`` 레이어에 시퀀스 병렬을 가능하게 만듭니다.

.. code-block:: python

    layer_tp_plan = {
        # 이제 SequenceParallel의 입력과 출력은 Shard(1) 레이아웃을 가지고,
        # 시퀀스 차원에서 샤딩된 입력/출력 tensor를 나타냅니다
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate()),
            desired_input_layouts=(Replicate(), Replicate()),
        ),
        "attention.wq": ColwiseParallel(use_local_output=False),
        "attention.wk": ColwiseParallel(use_local_output=False),
        "attention.wv": ColwiseParallel(use_local_output=False),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }


이제 ``PrepareModuleInput`` 을 이용해서 어텐션과 순전파 레이어의 모듈 입력 레이아웃을 ``Shard(1)`` 에서 ``Replicate()`` 로 수정하고, 출력 레이아웃을 ``Shard(1)`` 으로 표시하는 것을 볼 수 있습니다.
Tensor Parallelism과 마찬가지로, 입력과 출력의 tensor 샤딩 레이아웃만 지정하면, 레이어간 통신이 자동으로 이루어집니다.

시퀀스 병렬을 활용하면, 시퀀스 차원에서 항상 ``TransformerBlock`` 의 입력과 출력이 샤딩되어, 다중 ``TransformerBlocks`` 이 원활하게 연결할 수 있다고 가정합니다.
이는 시작하는 ``nn.Embedding`` 레이어의 출력과 최종 ``nn.Linear`` 입력 레이어를 ``Shard(1)`` 으로 명시적으로 지정하여 촉진할 수 있습니다.

.. code-block:: python

    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            ),
        }
    )


손실 병렬(Loss Parallel) 적용하기
-------------------------------

손실 병렬(Loss Parallel)은 손실 함수를 계산할 때 메모리와 통신을 절약하는 관련 기술로, 일반적으로 모델 출력이 매우 크기 때문에 사용합니다. 손실 병렬에서는 모델 출력이 (자주 거대한) 어휘 차원에서 샤딩될 때, 모든 모델 출력은 매번 단일 GPU에 모으지 않고도 교차 엔트로피 손실을 효율적으로 계산할 수 있습니다. 이는 메모리 소비를 유의하게 줄일 뿐만 아니라, 통신 오버헤드를 줄이고 샤딩된 연산을 병렬로 처리하여 학습 속도를 개선합니다. 아래 그림은 손실 병렬이 샤딩된 연산을 통해 단일 GPU마다 모든 모델의 출력을 모으는 것을 피하는 방법을 간략히 보여줍니다.

.. figure:: /_static/img/distributed/loss_parallel.png
   :width: 100%
   :align: center
   :alt: loss parallel

   그림 2. 단일 GPU에서 손실이 병렬로 발생하는 교차 엔트로피 손실의 순방향 계산. 파란색은 샤딩된 tensor를 나타내고, 녹색은 복제된 tensor를 나타내며, 노란색은 부분 값을 가지는 tensor를 나타냅니다 (모두 축소될 예정입니다). 검정 화살표는 로컬 계산이고, 붉은 화살표는 GPU 간의 기능적 집합체입니다.

PyTorch Tensor Parallel API에서, 손실 병렬은 컨텍스트 관리자 ``loss_parallel`` 을 통해 사용할 수 있으며, 이를 통해 코드의 다른 부분을 수정하지 않고도 ``torch.nn.functional.cross_entropy`` 혹은 ``torch.nn.CrossEntropyLoss`` 를 직접 사용할 수 있습니다.

손실 병렬을 적용하려면, 일반적으로 ``[batch size, sequence length, vocabulary size]`` 모양의 모델 예측을 어휘 차원에서 샤딩되어야 합니다. 이는 마지막 선형 투영 레이어 출력의 출력 레이아웃을 표기하여 쉽게 수행할 수 있습니다.

.. code-block:: python

    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                # DTensor를 출력으로 사용
                use_local_output=False,
            ),
        },
    )

위 코드에서는 출력 전 norm 레이어에도 시퀀스 병렬을 적용합니다. 출력이 DTensor로 유지하고 ``loss_parallel`` 컨텍스트 관리자와 함께 작동하도록 ``use_local_output=False`` 을 적용합니다. 그 후, 다음과 같이 단순히 cross_entropy 손실 함수라고 부를 수 있습니다. 역방향 계산도 컨텍스트 내에서 이루어져야 하는 점도 유의하세요.

.. code-block:: python

    import torch.nn.functional as F
    from torch.distributed.tensor.parallel import loss_parallel

    pred = model(input_ids)
    with loss_parallel():
        # pred 및 labels는 [batch, seq, vocab] 모양으로 가정
        loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
        loss.backward()


Fully Sharded Data Parallel과 Tensor Parallel을 함께 결합하기
-----------------------------------------------------------------


이제 Tensor/Sequence Parallel을 모델에 적용하는 방법을 보여드렸으니, Tensor Parallel과 Fully Sharded Data Parallel이 어떻게 함께 작동할 수 있는지도 살펴보겠습니다.
Tensor Parallelism는 연산을 방해하는 통신을 발생하므로, NVLink와 같은 빠른 통신 채널 내에서 실행되도록 하고 싶습니다.
실제로, 일반적으로 각 호스트 내에서 Tensor Parallel을 적용하고, 호스트 간 Fully Sharded Data Parallel를 적용합니다.

.. figure:: /_static/img/distributed/fsdp_tp.png
   :width: 100%
   :align: center
   :alt: fsdp + tp

   그림 3. FSDP와 TP는 별도의 디바이스 차원에서 작동하며, FSDP 통신은 호스트 간에, TP 통신은 호스트 내에서 이루어집니다.

이 2-D 병렬 처리 패턴은 2-D DeviceMesh를 통해 쉽게 표현할 수 있으며, 각각의 "하위" DeviceMesh를 각각의 개별 병렬 처리 API로 전달하기만 하면 됩니다.

.. code-block:: python

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    from torch.distributed.fsdp import fully_shard

    # 예를 들어, 2-D mesh는 [dp, tp]이고, 8 방향 DP와 8 방향 TP를 수행하는 64개의 GPU에서 훈련합니다
    mesh_2d = init_device_mesh("cuda", (8, 8))
    tp_mesh = mesh_2d["tp"] # 호스트 내 디바이스를 연결하는 submesh
    dp_mesh = mesh_2d["dp"] # 호스트 간 디바이스를 연결하는 submesh

    model = Model(...)

    tp_plan = {...}

    # tp_mesh에서 Tensor Parallel을 호스트 내 적용
    model_tp = parallelize_module(model, tp_mesh, tp_plan)
    # dp_mesh에서 FSDP를 호스트 간 적용
    model_2d = fully_shard(model_tp, mesh=dp_mesh, ...)


이렇게 하면 각 호스트 내 (intra-host)에서 Tensor Parallel을 쉽게 적용하고 호스트 간에 (inter-hosts) FSDP를 **0 코드 변경** 으로 Llama 모델에 적용할 수 있습니다.
tensor(모델) 병렬 및 데이터 병렬 기술을 함께 결합하면 많은 GPU를 이용해서 모델 크기를 지속적으로 늘리고 효율적으로 학습할 수 있습니다.

결론
----------
이 튜토리얼은 Tensor Parallel과 Fully Sharded Data Parallel을 결합하여 수백에서 수천 개 GPU에서 대규모 트랜스포머와 유사한 모델을 학습하는 방법을 보여줍니다.
Tensor Parallel을 모델의 여러 부분에 적용하고 **코드 변경 없이** 모델 자체에 적용하는 방법을 설명합니다. Tensor Parallel은 대규모 학습을 위한 효율적인 모델 병렬화 기술입니다.

이 튜토리얼에서 설명하는 전체(end-to-end) 코드 예제를 보려면, pytorch/examples 에 있는 `Tensor Parallel 예제 <https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py>`__ 를 참고하세요.