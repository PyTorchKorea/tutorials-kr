분산 데이터 병렬(DDP)과 분산 RPC 프레임워크 결합
=================================================================
**저자**: `Pritam Damania <https://github.com/pritamdamania87>`__ and `Yi Wang <https://github.com/SciPioneer>`__

**번역**: `박다정 <https://github.com/dajeongPark-dev>`_

.. note::
   |edit| 이 튜토리얼의 소스 코드는 `GitHub <https://github.com/PyTorchKorea/tutorials-kr/blob/master/advanced_source/rpc_ddp_tutorial.rst>`__ 에서 확인하고 변경해 볼 수 있습니다.

이 튜토리얼은 간단한 예제를 사용하여 분산 데이터 병렬 처리(distributed data parallelism)와
분산 모델 병렬 처리(distributed model parallelism)를 결합하여 간단한 모델 학습시킬 때
`분산 데이터 병렬(DistributedDataParallel) <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__ (DDP)과
`분산 RPC 프레임워크(Distributed RPC framework) <https://pytorch.org/docs/master/rpc.html>`__를 결합하는 방법에 대해 설명합니다.
예제의 소스 코드는 `여기 <https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc>`__에서 확인할 수 있습니다.

이전 튜토리얼 내용이었던
`분산 데이터 병렬 시작하기 <https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html>`__와
`분산 RPC 프레임워크 시작하기 <https://tutorials.pytorch.kr/intermediate/rpc_tutorial.html>`__는
분산 데이터 병렬 및 분산 모델 병렬 학습을 각각 수행하는 방법에 대해 설명합니다.
그러나 이 두 가지 기술을 결합할 수 있는 몇 가지 학습 패러다임이 있습니다. 예를 들어:

1) 희소 부분(큰 임베딩 테이블)과 밀집 부분(FC 레이어)이 있는 모델이 있는 경우,
   매개변수 서버(parameter server)에 임베딩 테이블(embedding table)을 놓고 `분산 데이터 병렬 <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__을 사용하여
   여러 트레이너에 걸쳐 FC 레이어를 복제하는 것을 원할 수도 있습니다.
   이때 `분산 RPC 프레임워크 <https://pytorch.org/docs/master/rpc.html>`__는
   매개변수 서버에서 임베딩 찾기 작업(embedding lookup)을 수행하는 데 사용할 수 있습니다.
2) 다음은 `PipeDream <https://arxiv.org/abs/1806.03377>`__ 문서에서 설명된 하이브리드 병렬 처리 활성화하기 입니다.
   `분산 RPC 프레임워크 <https://pytorch.org/docs/master/rpc.html>`__를 사용하여
   여러 worker에 걸쳐 모델의 단계를 파이프라인(pipeline)할 수 있고
   (필요에 따라) `분산 데이터 병렬 <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__을 이용해서
   각 단계를 복제할 수 있습니다.

|
이 튜토리얼에서는 위에서 언급한 첫 번째 경우를 다룰 것입니다.
다음과 같이 총 4개의 worker가 있습니다:


1) 1개의 마스터는 매개변수 서버에 임베딩 테이블(nn.EmbeddingBag) 생성을 담당합니다.
   또한 마스터는 두 트레이너의 학습 루프를 수행합니다.
2) 1개의 매개변수 서버는 기본적으로 메모리에 임베딩 테이블을 보유하고 마스터 및 트레이너의 RPC에 응답합니다.
3) 2개의 트레이너는 `분산 데이터 병렬 <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__을
   사용하여 자체적으로 복제되는 FC 레이어(nn.Linear)를 저장합니다.
   트레이너는 또한 순방향 전달(forward pass), 역방향 전달(backward pass) 및 최적화 단계를 실행해야 합니다.

|
전체적인 학습과정은 다음과 같이 실행됩니다:

1) 마스터는 매개변수 서버에 임베딩 테이블을 담고 있는
   `원격 모듈(RemoteModule) <https://pytorch.org/docs/master/rpc.html#remotemodule>`__을 생성합니다.
2) 그런 다음 마스터는 트레이너의 학습 루프를 시작하고 원격 모듈(remote module)을 트레이너에게 전달합니다.
3) 트레이너는 먼저 마스터에서 제공하는 원격 모듈을 사용하여
   임베딩 찾기 작업(embedding lookup)을 수행한 다음 DDP 내부에 감싸진 FC 레이어를 실행하는 ``HybridModel``을 생성합니다.
4) 트레이너는 모델의 순방향 전달을 실행하고 손실을 사용하여 `분산 Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__를
   사용하여 역방향 전달을 실행합니다.
5) 역방향 전달의 일부로 FC 레이어의 변화도가 먼저 계산되고 DDP의 allreduce를 통해 모든 트레이너와 동기화됩니다.
6) 다음으로, 분산 Autograd는 매개변수 서버로 변화도를 전파하고 그곳에서 임베딩 테이블의 변화도가 업데이트됩니다.
7) 마지막으로, `분산 옵티마이저(DistributedOptimizer) <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`__는 모든 매개변수를 업데이트하는 데 사용됩니다.

.. 주의사항::

  DDP와 RPC를 결합할 때, 역방향 전달에 대해 항상
  `분산 Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__를 사용해야 합니다.


이제 각 부분을 자세히 살펴보겠습니다.
먼저 학습을 수행하기 전에 모든 작업자를 설정해야 합니다.
순위 0과 1은 트레이너, 순위 2는 마스터, 순위 3은 매개변수 서버인 4개의 프로세스를 만듭니다.

TCP init_method를 사용하여 4개의 모든 worker에서 RPC 프레임워크를 초기화합니다.
RPC 초기화가 끝나면, 마스터는 `EmbeddingBag <https://pytorch.org/docs/master/generated/torch.nn.EmbeddingBag.html>`__ 레이어를
`원격 모듈(RemoteModule) <https://pytorch.org/docs/master/rpc.html#remotemodule>`__을 사용하여
매개변수 서버에 담고 있는 원격 모듈 하나를 생성합니다.
그런 다음 마스터는 각 트레이너를 반복하고 `rpc_async <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.rpc_async>`__를
사용하여 각 트레이너에서 ``_run_trainer``를 호출하여 반복 학습을 시작합니다.
마지막으로 마스터는 종료하기 전에 모든 학습이 완료될 때까지 기다립니다.

트레이너는 `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__을 사용하여
(2개의 트레이너) world_size=2로 DDP를 위해 ``ProcessGroup``을 초기화합니다.
다음으로 TCP init_method를 사용하여 RPC 프레임워크를 초기화합니다.
여기서 주의 할 점은 RPC 초기화와 ProgressGroup 초기화에서 쓰이는 포트(port)가 다르다는 것입니다.
이는 두 프레임워크의 초기화 간에 포트 충돌을 피하기 위해서 입니다.
초기화가 완료되면 트레이너는 마스터의 ``_run_trainer` RPC를 기다리기만 하면 됩니다.

파라피터 서버는 RPC 프레임워크를 초기화하고 트레이너와 마스터의 RPC를 기다립니다.


.. literalinclude:: ../advanced_source/rpc_ddp_tutorial/main.py
  :language: py
  :start-after: BEGIN run_worker
  :end-before: END run_worker

트레이너에 대한 자세한 설명에 앞서, 트레이너가 사용하는 ``HybridModel``에 대해 설명드리겠습니다.
아래에 설명된 대로 ``HybridModel``은 매개변수 서버의 임베딩 테이블(``remote_emb_module``)과 DDP에 사용할 ``device``를 보유하는 원격 모듈을 사용하여 초기화됩니다.
모델 초기화는 DDP 내부의 `nn.Linear <https://pytorch.org/docs/master/generated/torch.nn.Linear.html>`__ 레이어를
감싸 모든 트레이너에서 이 레이어를 복제하고 동기화합니다.


모델의 순방향(forward) 함수는 꽤 간단합니다.
RemoteModule의 ``forward``를 사용하여 매개변수 서버에서 임베딩 찾기 작업(embedding lookup)을 수행하고 그 출력을 FC 레이어에 전달합니다.


.. literalinclude:: ../advanced_source/rpc_ddp_tutorial/main.py
  :language: py
  :start-after: BEGIN hybrid_model
  :end-before: END hybrid_model

다음으로 트레이너의 설정을 살펴보겠습니다.
트레이너는 먼저 매개변수 서버의 임베딩 테이블과 자체 순위를 보유하는 원격 모듈을 사용하여
위에서 설명한 ``HybridModel``을 생성합니다.

이제 `분산 옵티마이저(DistributedOptimizer) <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`__로
최적화하려는 모든 매개변수에 대한 RRef 목록을 검색해야 합니다.
매개변수 서버에서 임베딩 테이블의 매개변수를 검색하기 위해
RemoteModule의 `remote_parameters <https://pytorch.org/docs/master/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters>`__를 호출할 수 있습니다.
그리고 이것은 기본적으로 임베딩 테이블의 모든 매개변수를 살펴보고 RRef 목록을 반환합니다.
트레이너는 RPC를 통해 매개변수 서버에서 이 메서드를 호출하여 원하는 매개변수에 대한 RRef 목록을 수신합니다.
DistributedOptimizer는 항상 최적화해야 하는 매개변수에 대한 RRef 목록을 가져오기 때문에 FC 레이어의 전역 매개변수에 대해서도 RRef를 생성해야 합니다.
이것은 ``model.fc.parameters()``를 탐색하고 각 매개변수에 대한 RRef를 생성하고
``remote_parameters()``에서 반환된 목록에 추가함으로써 수행됩니다.
참고로 ``model.parameters()``는 사용할 수 없습니다. ``RemoteModule``에서 지원하지 않는 ``model.remote_emb_module.parameters()``를 재귀적으로 호출하기 때문입니다.

마지막으로 모든 RRef를 사용하여 DistributedOptimizer를 만들고 CrossEntropyLoss 함수를 정의합니다.

.. literalinclude:: ../advanced_source/rpc_ddp_tutorial/main.py
  :language: py
  :start-after: BEGIN setup_trainer
  :end-before: END setup_trainer

이제 각 트레이너에서 실행되는 기본 학습 루프를 소개하겠습니다.
``get_next_batch``는 학습을 위한 임의의 입력과 대상을 생성하는 것을 도와주는 함수일 뿐입니다.
여러 에폭(epoch)과 각 배치(batch)에 대해 학습 루프를 실행합니다:

1) 먼저 분산 Autograd에 대해
   `분산 Autograd Context <https://pytorch.org/docs/master/rpc.html#torch.distributed.autograd.context>`__를 설정합니다.
2) 모델의 순방향 전달을 실행하고 해당 출력을 검색(retrieve)합니다.
3) 손실 함수를 사용하여 출력과 목표를 기반으로 손실을 계산합니다.
4) 분산 Autograd를 사용하여 손실을 사용하여 분산 역방향 전달을 실행합니다.
5) 마지막으로 분산 옵티마이저 단계를 실행하여 모든 매개변수를 최적화합니다.

.. literalinclude:: ../advanced_source/rpc_ddp_tutorial/main.py
  :language: py
  :start-after: BEGIN run_trainer
  :end-before: END run_trainer
.. code:: python

전체 예제의 소스 코드는 `여기 <https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc>`__에서 찾을 수 있습니다.
