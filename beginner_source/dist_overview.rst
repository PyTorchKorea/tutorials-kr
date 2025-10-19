PyTorch 분산 개요
============================
**저자** `Will Constable <https://github.com/wconstab/>`_, `Wei Feng <https://github.com/weifengpy>`_
**번역** `강지현 <https://github.com/KJH622>`_
.. note::
   |edit| 이 튜토리얼을 여기서 보고 편집하세요 `github <https://github.com/pytorchkorea/tutorials-kr/blob/main/beginner_source/dist_overview.rst>`__.

이 문서는 ``torch.distributed`` 패키지의 개요 페이지입니다.
이 페이지의 목표는 문서를 주제별로 분류하고 
각 주제를 간략히 설명하는 것입니다. PyTorch로 분산 학습 애플리케이션을 처음 구축한다면,
이 문서를 참고해 사용 사례에 가장 적합한 기술로 이동하시길 권장합니다.

서론
------------

파이토치 분산 라이브러리는 여러 병렬화 모듈, 통신 계층, 그리고 대규모 학습 작업의 실행 및 디버깅을 위한 인프라로 구성됩니다.


병렬 처리 API
****************

이러한 병렬화 모듈은 고수준 기능을 제공하며 기존 모델과 조합하여 사용할 수 있습니다.

- `분산 데이터 병렬 처리 (DDP, Distributed Data-Parallel) <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
- `완전 샤딩 데이터 병렬 학습 (FSDP2, Fully Sharded Data-Parallel Training) <https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>`__
- `텐서 병렬 처리 (TP, Tensor Parallel) <https://pytorch.org/docs/stable/distributed.tensor.parallel.html>`__
- `파이프라인 병렬 처리 (PP, Pipeline Parallel) <https://pytorch.org/docs/main/distributed.pipelining.html>`__

샤딩 프리미티브(Sharding primitives)
*******************

``DTensor`` 와 ``DeviceMesh`` 는 N차원 프로세스 그룹에서 텐서를 샤딩하거나 복제하는 방식으로 병렬화를 구성할 때 사용하는 기본 구성요소입니다.

- `DTensor <https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/README.md>`__ 는 샤딩되거나/복제된 텐서를 나타내며, 연산의 요구에 따라 텐서를 재샤딩하기 위한 통신을 자동으로 수행합니다.
- `DeviceMesh <https://pytorch.org/docs/stable/distributed.html#devicemesh>`__ 는 가속기 디바이스의 커뮤니케이터(communicator)를 다차원 배열로 추상화하며, 다차원 병렬성에서 집합(collective) 통신을 수행하기 위한 하위 ``ProcessGroup`` 인스턴스들을 관리합니다. 더 알아보려면 `Device Mesh 레시피 <https://tutorials.pytorch.kr/recipes/distributed_device_mesh.html>`__ 를 직접 따라 해보세요.

통신 API
*******************

`PyTorch 분산 통신 계층 (C10D) <https://pytorch.org/docs/stable/distributed.html>`__ 은 집합 통신 API (예: `all_reduce(전체 축소) <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce>`__
   , `all_gather(전체 수집) <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather>`__)
   와 P2P 통신 API (예: `send(동기 전송) <https://pytorch.org/docs/stable/distributed.html#torch.distributed.send>`__
   , `isend(비동기 전송) <https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend>`__)를 모두 제공하며,
   이러한 API는 모든 병렬화 구현에서 내부적으로 사용됩니다.
   `PyTorch로 분산 애플리케이션 작성하기 <../intermediate/dist_tuto.html>`__ 는 C10D 통신 API 사용 예제를 보여 줍니다.

실행기(Launcher)
********

`torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__ 은 널리 쓰이는 실행기 스크립트로, 분산 PyTorch 프로그램을 실행하기 위해 로컬 및 원격 머신에서 프로세스를 생성합니다.


모델 확장을 위한 병렬화 적용
----------------------------------------

데이터 병렬화(Data Parallelism)는 널리 채택된 SPMD(single-program multiple-data) 학습 패러다임으로,
모델이 모든 프로세스에 복제되고 각 모델의 복제본이 서로 다른 입력 데이터 샘플 집합에 대해 로컬 변화도를 계산합니다.
그런 다음 각 옵티마이저 스텝 전에 데이터-병렬 통신 그룹 내에서 변화도를 평균화합니다.

모델 병렬화(Model Parallelism) 기법(또는 샤딩된 데이터 병렬화)은 모델이 GPU 메모리에 들어가지 않을 때 필요하며, 서로 결합해 다차원(N-D) 병렬화 기법을 구성할 수 있습니다.

모델에 적용할 병렬화 기법을 결정할 때는 다음의 일반적인 지침을 참고하세요.

#. 모델이 단일 GPU를 탑재할 수 있지만, 여러 GPU로 쉽게 학습을 확장하고 싶다면 
   `DistributedDataParallel (DDP, 분산 데이터 병렬화) <https://pytorch.org/docs/stable/notes/ddp.html>`__ 를 사용하세요.

   * 여러 노드를 사용하는 경우, 여러 PyTorch 프로세스를 시작하려면 `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__ 을 사용하세요.

   * 참고: `시작하기 분산 데이터 병렬(DDP) <../intermediate/ddp_tutorial.html>`__

#. 모델이 단일 GPU에 탑재되지 않을 때는 `FullyShardedDataParallel (FSDP2, 완전 샤딩 데이터 병렬화) <https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>`__ 을 사용하세요.

   * 참고: `시작하기 FSDP2 <https://tutorials.pytorch.kr/intermediate/FSDP_tutorial.html>`__

#. FSDP2로는 확장 한계에 도달한 경우, `Tensor Parallel (TP, Tensor 병렬화) <https://pytorch.org/docs/stable/distributed.tensor.parallel.html>`__ 및/또는 `Pipeline Parallel (PP, 파이프라인 병렬화) <https://pytorch.org/docs/main/distributed.pipelining.html>`__ 를 사용하세요.

   * `텐서 병렬화 튜토리얼 <https://tutorials.pytorch.kr/intermediate/TP_tutorial.html>`__ 을 확인해 보세요.

   * 참고: `TorchTitan 3D 병렬화 전체(end to end) 예제 <https://github.com/pytorch/torchtitan>`__

.. note:: 데이터 병렬 학습은 `자동 혼합 정밀도(AMP, Automatic Mixed Precision) <https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus>`__ 와 함께에서도 동작합니다.


PyTorch 분산 개발자
------------------------------

PyTorch 분산에 기여하고 싶다면 `개발자 가이드 <https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING.md>`_ 를 참고하세요.
