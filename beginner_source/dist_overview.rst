PyTorch Distributed 개요
============================
**저자**: `Shen Li <https://mrshenli.github.io/>`_
**번역**: `Jae Joong Lee <https://github.com/JaeLee18/>`_


``torch.distributed``패키지의 개요 페이지 입니다.
점점 더 많은 문서, 예제, 그리고 튜토리얼들이 각각 다른 위치에 있기 때문에 특정한 문제에 대해서
어떤 문서나 튜토리얼을 봐야할지 또는 어떤 순서로 컨텐츠들을 읽어야하는지  점점 불분명해지고 있습니다.
이 페이지의 목표는 이러한 문제를 문서들을 각기 다른 주제들로 분류하고 각 주제들을 간단히
설명하여 문제를 해결하는것 입니다.
만약 PyTorch 를 이용해서 처음으로 분산 훈련 어플리케이션을 만드는 경우라면, 이 페이지를
요구사항을 잘 만족하는 기술을 찾을 수 있는 문서로 사용하길 추천합니다.


서론
------------

PyTorch 1.6.0 버전의  ``torch.distributed`` 는 세개의 주요 부분으로 분류됩니다:


* `분산 데이터-병렬(Data-Parallel) 훈련 (DDP)<https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
  (DDP)는 단일-프로그램 다중-데이터 훈련 패러다임을 전반적으로 사용합니다.
  DDP를 이용하며 모델은 매 프로세스 마다 복제되며, 복제된 모델들은 다른 입력 데이터 샘플들의 세트에 사용됩니다.
  DDP 는 변화도 통신을 책임져서 모델 복제품들끼리 동기화되도록 하며 또한 이것을 변화도 계산과 합쳐서 학습을 더 빨리하게 합니다.

* `RPC-기반 분산 훈련 <https://pytorch.org/docs/master/rpc.html>`__
  (RPC) 는 데이터-병렬 학습, 분산 파이프라인 병렬화, 매개변수 서버 패러다임 그리고 다른 학습 패러다임과
  DDP와의 결합에 맞지 않는 일반적인 학습 구조를 지원하기 위해 개발되었습니다.
  (RPC) 는 원격 객체 수명관리와 autograd 엔진(engine)을 기계 경계를 넘어 확장하는데 도움을 줍니다.

* `단체 통신 <https://pytorch.org/docs/stable/distributed.html>`__
  (c10d) 라이브러리는 텐서를 그룹안에 있는 프로세스들에게 보내는것을 지원합니다.
  또한 단체 통신 API (예,
  `all_reduce <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce>`__
  와 `all_gather <https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather>`__)
  그리고 P2P 통신 API (e.g.,
  `send <https://pytorch.org/docs/stable/distributed.html#torch.distributed.send>`__
  와 `isend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend>`__)
  를 제공합니다.
  DDP 와 RPC (`ProcessGroup Backend <https://pytorch.org/docs/master/rpc.html#process-group-backend>`__)
  는 1.6.0 버전 기준으로 c10d를 기반으로 개발되어있으며 DDP 는 단체 통신을 그리고 RPC 는 P2P 통신을 사용합니다.
  대부분의 개발자들은 이러한 순수한(raw) 통신 API를 사용하는것이 필요하지는 않고 위의 DDP 와 RPC 특징들은
  많은 분산 훈련 시나리오들을 해결할 수 있습니다.
  그러나, 이러한 API가 아직까지 도움이 되는 경우들 또한 존재합니다.
  One example would be distributed parameter averaging, where
  applications would like to compute the average values of all model parameters
  after the backward pass instead of using DDP to communicate gradients.
  도움이 되는 경우중 한 경우는 분산 매개변수 평균내기(averaging)입니다.
  이때 어플리케이션은 역전파 후에 DDP를 사용하여 변화도를 통신하는 대신
  모든 모델 매개변수들의 평균 값을 계산할 것 입니다.
  이것은 계산에서 통신을 분리하고 무엇과 통신할지 세부적인 제어를 가능하게 해주지만 DDP에 의해 제공되는 성능 최적화를 포기합니다.
  `PyTorch를 사용한 분산 어플리케이션 <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`__
  이 c10d 통신 API 예제들을 보여줍니다.


대부분의 문서들을 DDP 나 RPC에 대해 적혀있고 이 페이지의 나머지 부분들은 이 두 구성요소들에 대해
좀 더 설명하겠습니다.


데이터 병렬 학습(Data Parallel Training)
----------------------

PyTorch 는 데이터-병렬 학습에 대한 몇가지의 선택들을 제공합니다.
점진적으로 간단한것에서 복잡한것으로 또한 프로토타입에서 제품으로 발전해나가는 어플리케이션들의
개발 절차가 갖는 공통점들은 다음과 같습니다:

1. 한개의 GPU에 데이터와 모델이 들어간다면 단일 장치 훈련이 사용되고 훈련 속도는 큰 걱정이 아닙니다.
2. 만약 다중 GPU가 서버에 있다면 서버에서 다중 GPU `DataParallel <https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html>`__,
   를 사용하며 학습속도를 최소의 코드 변화를 통해 증가시킵니다.
3. 만약 더 학습 속도를 증가 시키고 싶다면 좀 더 많은 코드의 변화를 줘서 단일 서버 다중 GPU
   `DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__,
   를 사용합니다.
4. 다중 서버 `DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__ 사용과
   만약 어플리케이션이 서버 경계를 가로질러로 조정될 필요성이 있다면 `launching script <https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md>`__,
   를 사용합니다.
5. 만약 에러(예, OOM)나 교육중에 자원들이 추가되거나 빠질것 같다면
   `torchelastic <https://pytorch.org/elastic>`__ 을 사용하여 분산 학습을 시작합니다.


.. 알림:: 또한 데이터-병렬 학습은 `Automatic Mixed Precision (AMP) <https://pytorch.org/docs/master/notes/amp_examples.html#working-with-multiple-gpus>`__
과 사용될 수 있습니다.


``torch.nn.DataParallel``
~~~~~~~~~~~~~~~~~~~~~~~~~

`DataParallel <https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html>`__
패키지는 단일-서버 다중-GPU 병렬화를 가능하게 해주며 가장 낮은 코딩 난이도를 갖고있습니다.
 It only requires a one-line change to the application code.
이 패키지는 단지 한줄의 코드만 변경을 하면 됩니다.
이 튜토리얼
`Optional: Data Parallelism <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html>`__
이 예시를 보여줍니다.
한 가지 주의할 점은 비록 ``DataParallel`` 은 굉장히 사용하기 쉽지만, 이것은 최상의 성능을 제공하지는 않습니다.
그 이유로는 ``DataParallel`` 의 코드는 매 순방향 패스일때 모델을 복제하고 이것의 단일 프로세스 다중 쓰레드 병렬화는
GIL을 만족시키기위해 자연스럽게 고통받는 구조 이기 때문입니다.
더 나은 성능을 갖기 위해서 `DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
를 사용하는것을 고려하십시오.

``torch.nn.parallel.DistributedDataParallel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`DataParallel <https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html>`__ 과 비교했을때
`DistributedDataParallel <https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html>`__
는 한가지를 더 요구하는데 이것은 `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`__ 을 호출하는것입니다.
DDP는 다중 프로세스 병렬화를 사용하기 때문에 모델 복제품들 사이에선 GIL 문제가 발생하지 않습니다.
또한 이 모델은 매 순방향 패스일때 복제가 되는게 아니라 DDP가 구축할때 나오기 때문에 학습 속도를 증가시키는데 도움이 됩니다.
DDP 는 또한 다양한 성능 최적화 기술들을 제공합니다. 더 자세한 설명을 보려면 다음을 참고해주세요.
`DDP 논문 <https://arxiv.org/abs/2006.15704>`__ (VLDB'20).


DDP 자료들은 다음과 같습니다:

1. `DDP 노트들 <https://pytorch.org/docs/stable/notes/ddp.html>`__
   은 시작할 만한 예제와 DDP 디자인과 구현에 대한 간단한 설명을 제공합니다.
   만약 DDP를 처음사용한다면 이 문서부터 보는걸 추천합니다.
2. `분산 데이터 병렬 시작하기 <../intermediate/ddp_tutorial.html>`__
   는 DDP 학습에 일어나는 흔한 문제들을 설명하는데 불균형적인 작업량, 체크포인트 그리고 다중 디바이스 모델을 포함합니다.
    Note that, DDP can be
   easily combined with single-machine multi-device model parallelism which is
   described in the
   `Single-Machine Model Parallel Best Practices <../intermediate/model_parallel_tutorial.html>`__
   tutorial.
   알아두어야할것은 DDP는 `단일-서버 모델 병렬 튜토리얼<../intermediate/model_parallel_tutorial.html>`__
   튜토리얼에서 설명한것 처럼 단일-서버 다중-디바이스 모델 병렬화와 쉽게 결합될 수 있습니다.
3. `분산 데이터 병렬 어플리케이션 시작과 구성 <https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md>`__
    은 어떻게 DDP 시작 스크립트(launching script)를 사용하는지 보여줍니다.
4. `아마존 AWS를 이용한 PyTorch 분산 학습기 <aws_distributed_training_tutorial.html>`__
   은 AWS 에서 어떻게 DDP를 사용하는지 보여줍니다.

TorchElastic
~~~~~~~~~~~~

애플리케이션의 복잡성과 규모가 증가함에 따라, 장애 복구는 필수적인 요구 사항이 되었습니다.
때때로 DPP를 사용할때 OOM과 같은 에러를 보는것은 필연적이지만 DPP 자체로는 이러한 에러를 복구할수 없으며
또한 기본적인 ``try-except`` 구문도 마찬가지입니다.
 This is because DDP requires all processes
to operate in a closely synchronized manner and all ``AllReduce`` communications
launched in different processes must match.
DDP가 모든 프로세스들이 긴밀하게 동기화되어 작동하길 요구하고 또한 다른 프로세스에서 생성된
모든 ``AllReduce`` 통신들이 반드시 동기화 되어야 하기 때문입니다.
만약 그룹에 있는 프로세스들중 하나라도 OOM 에러를 갖는다면 동기화가 되지않는 (일치되지 못하는 ``AllReduce`` 작업들)
현상을 야기하고 이것은 충돌이나 중단의 원인이 됩니다.
만약 학습 도중 실패할것 같거나 자원들이 동적으로 합류하거나 빠질수 있다면 분산 데이터-병렬 학습은
`torchelastic <https://pytorch.org/elastic>`__ 를 사용하여 실행해 주십시오.

일반적인 분산 학습
----------------------------

많은 훈련 패러다임들(매개변수 서버 패러다임, 분산된 파이프라인 병렬화, 많은 옵저버(observer)와 에이전트(agent)가 있는 강화 학습 어플리케이션)
은 데이터 병렬화에 적합하지 않습니다.
`torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`__ 는
일반적은 분산 학습 시나리오를 지원하는것을 목표로 하고있습니다.

`torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`__ 패키지는
네 개의 주된 구성이 있습니다:

  * `RPC <https://pytorch.org/docs/master/rpc.html#rpc>`__ 는 원격 워커(worker)에
  주어진 함수를 실행하는것을 지원합니다.
* `RRef <https://pytorch.org/docs/master/rpc.html#rref>`__ 는 원격 객체의 생명주기를
  관리하는것을 도와줍니다. 참조 계산 프로토콜은
  `RRef notes <https://pytorch.org/docs/master/rpc/rref.html#remote-reference-protocol>`__
  에 나와있습니다.
* `Distributed Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__
  extends the autograd engine beyond machine boundaries. Please refer to
  `Distributed Autograd Design <https://pytorch.org/docs/master/rpc/distributed_autograd.html#distributed-autograd-design>`__
  for more details.
* `분산된 Autograd <https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework>`__
  는 autograd 엔진(engine)을 기계 경계 너머로 확장합니다. 더 자세한 사항은 이곳
  `Distributed Autograd Design <https://pytorch.org/docs/master/rpc/distributed_autograd.html#distributed-autograd-design>`__
  을 참고해주세요.
* `Distributed Optimizer <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`__
  that automatically reaches out to all participating workers to update
  parameters using gradients computed by the distributed autograd engine.
* `분산된 옵티마이저 <https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim>`__
  는 자동적으로 분산된 autograd 엔진(engine)을 사용되어 계산된 변화도를
  모든 참여하고 있는 워커(worker)들에게 접근하여 매개변수를 최신화합니다.

RPC 튜토리얼은 아래와 같습니다:

1. `분산된 RPC 프레임워크 시작하기 <../intermediate/rpc_tutorial.html>`__ 튜토리얼은 처음으로 간단한 강화학습 예제를
   하여 RPC 와 PRef 를 보여줍니다. 그리고 기본적인 분산된 모델 병렬화를 RNN 예시에 적용하여 어떻게 분산된 autograd 와 분산된 옵티마이저
   를 사용하는지 보여줍니다.
2. `분산된 RPC 프레임워크를 이용한 매개변수 서버 개발하기 <../intermediate/rpc_param_server_tutorial.html>`__
    튜토리얼은 `HogWild! training <https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf>`__ 을 빌려왔으며
    이것을 비동기 매개변수 서버 학습 어플리케이션에 적용하였습니다.
3. `RPC를 이용한 분산된 파이프라인 병렬화 <../intermediate/dist_pipeline_parallel_tutorial.html>`__
   튜토리얼은 단일-서버 파이프라인 병렬 예시(`단일-서버 모델 병렬화 튜토리얼 <../intermediate/model_parallel_tutorial.html>`__)
   를 분산된 환경으로 확장하고 RPC를 사용하여 어떻게 구현할지 보여줍니다.
4. `비동기 실행을 사용한 배치 RPC 프로세싱 개발 <../intermediate/rpc_async_execution.html>`__
  튜토리얼은 어떻게 RPC 배치 프로세싱을
  `@rpc.functions.async_execution <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution>`__
  데코레이터를 사용해 개발하는지 알려줍니다. 데코레이터를 이용하는 것은 추론과 학습시간을 빠르게 해줍니다.
  앞서 소개된 튜토리얼 1과 2에서 보여준 비슷한 강화학습과 매개변수 서버를 사용합니다.
