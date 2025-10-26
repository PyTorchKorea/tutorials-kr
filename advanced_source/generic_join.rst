불균등한 입력에 대한 분산 학습을 위한 Join 컨텍스트 관리자(context manager) 사용 예시
======================================================================

**저자**\ : `Andrew Gu <https://github.com/andwgu>`_

**번역**\ : `김민엽 <https://github.com/minyeamer>`_

.. note::
    |edit| 이 튜토리얼은 `github <https://github.com/PyTorchKorea/tutorials-kr/blob/master/advanced_source/generic_join.rst>`_ 에서 확인하고 수정할 수 있습니다.

.. note:: ``Join`` 은 PyTorch 1.10에서 프로토타입 기능으로 도입되었습니다.
    이 API는 변경될 수 있습니다.

이 튜토리얼에서는 다음을 다룹니다.

- `Join`_ 컨텍스트 관리자 개요
- ``DistributedDataParallel`` 과 함께 컨텍스트 관리자를 사용하는 예시
- ``DistributedDataParallel`` 과 ``ZeroRedundancyOptimizer`` 를
  컨텍스트 관리자와 함께 사용하는 예시
- 컨텍스트 관리자에 키워드 인자를 전달하는 예시
- `Join`_ 컨텍스트 관리자의 동작 방식 심층 분석
- 예제 클래스를 컨텍스트 관리자와 호환되게 만드는
  방법

요구 사항
-------

- PyTorch 1.10 이상
- `분산 데이터 병렬 처리 시작하기`_
- `Shard Optimizer States with ZeroRedundancyOptimizer`_

``Join`` 이란?
-------------
`분산 데이터 병렬 처리 시작하기 - 기본적인 사용법 <https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html#id3>`_ 에서,
`DistributedDataParallel`_ 을 사용한 데이터 병렬 학습의 기본 구조를
살펴보았습니다. 이 방식은 각 역전파 단계에서 모든 랭크(rank) 간에 기울기(gradient)를
동기화하기 위해 all-reduce 연산을 암묵적으로 스케줄링합니다.
이러한 `집합통신 <https://pytorch.org/docs/stable/distributed.html>`_ 은
프로세스 그룹의 모든 랭크가 참여해야 하므로, 어떤 랭크의 입력이 더 적다면,
다른 랭크들은 대기하거나 에러가 발생할 수 있습니다. 일반적으로, 이러한 문제는
각 반복마다 동기식 집합통신(collective communication)을 수행하는 모든 클래스에서
지속적으로 발생합니다.

``Join`` 은 각 랭크의 학습 루프를 감싸서 불균등한 입력이 주어지는 상황에서
학습을 원활하게 해주는 컨텍스트 관리자입니다.
입력이 먼저 끝난 (즉, 먼저 *join* 된) 랭크는
아직 *join* 되지 않은 랭크가 수행하는 집합통신을 모방할 수 있게 됩니다.
이때 통신을 모방하는 방식은 훅(hook)으로 지정할 수 있습니다.

``Join`` 을 ``DistributedDataParallel`` 과 함께 사용하기
----------------------------------------------------
PyTorch의 `DistributedDataParallel`_ 은 ``Join`` 컨텍스트 관리자와 함께 바로
사용할 수 있습니다. 예시는 다음과 같습니다.

::

    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.algorithms.join import Join
    from torch.nn.parallel import DistributedDataParallel as DDP

    BACKEND = "nccl"
    WORLD_SIZE = 2
    NUM_INPUTS = 5

    def worker(rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

        model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
        # 랭크 1은 랭크 0보다 입력을 하나 더 받습니다.
        inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

        num_inputs = 0
        with Join([model]):
            for input in inputs:
                num_inputs += 1
                loss = model(input).sum()
                loss.backward()

        print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")

    def main():
        mp.spawn(worker, nprocs=WORLD_SIZE, join=True)

    if __name__ == "__main__":
        main()

이 코드는 다음과 같은 출력을 생성합니다.
(랭크 0과 랭크 1의 ``print()`` 출력 순서는 임의로 정렬될 수 있습니다.)

::

  Rank 0 has exhausted all 5 of its inputs!
  Rank 1 has exhausted all 6 of its inputs!

.. note::
    `DistributedDataParallel`_ 은 ``Join`` 컨텍스트 관리자가
    도입되기 전 자체적으로 `join()`_ 컨텍스트 관리자를 제공했습니다.
    위 예시에서 ``with Join([model]):`` 은
    ``with model.join():`` 과 동일하게 동작합니다. 기존
    ``DistributedDataParallel.join()`` 의 한 가지 제한 사항은 여러 클래스를
    동시에 지원하지 않는다는 것입니다. (예를 들어, ``DistributedDataParallel`` 과
    `ZeroRedundancyOptimizer`_ 를 함께 사용하는 것과 같습니다)

``Join`` 을 ``DistributedDataParallel`` 및 ``ZeroRedundancyOptimizer`` 과 함께 사용하기
-------------------------------------------------------------------------------
``Join`` 컨텍스트 관리자는 하나의 클래스뿐만 아니라
여러 클래스를 동시에 지원합니다. PyTorch의 ``ZeroRedundancyOptimizer`` 또한
이 컨텍스트 관리자와 호환되므로,
이전 예시를 ``DistributedDataParallel`` 과
``ZeroRedundancyOptimizer`` 를 함께 사용하도록 수정해보겠습니다.

::

    from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
    from torch.optim import Adam

    def worker(rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

        model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
        optim = ZeRO(model.parameters(), Adam, lr=0.01)
        # 랭크 1은 랭크 0보다 입력을 하나 더 받습니다.
        inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

        num_inputs = 0
        # `model` 과 `optim` 을 모두 `Join()` 에 전달합니다.
        with Join([model, optim]):
            for input in inputs:
                num_inputs += 1
                loss = model(input).sum()
                loss.backward()
                optim.step()

        print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")

이 코드는 앞선 예시와 동일한 출력을 생성합니다.
주요 차이점은 ``ZeroRedundancyOptimizer`` 인스턴스를
``Join()`` 에 추가로 전달했다는 점입니다.

키워드 인자 전달하기
---------------
클래스는 실행 중에 컨텍스트 관리자 내에서 동작을 변경할 수 있는 키워드 인자를 제공합니다.
예를 들어, ``DistributedDataParallel`` 은
``divide_by_initial_world_size`` 라는 인자를 제공하는데,
이는 기울기를 초기 프로세스 수(world size)로 나눌지,
아니면 *Join* 되지 않은 랭크(즉, 유효 프로세스 수)로 나눌지를 결정합니다.
이러한 키워드 인자는 컨텍스트 관리자에 직접 전달할 수 있습니다.

::

    with Join([model, optim], divide_by_initial_world_size=False):
        for input in inputs:
            ...

.. warning::
    컨텍스트 관리자에 전달된 키워드 인자는 모든 클래스 간에 공유됩니다.
    이는 여러 ``Joinable`` 객체가 동일한 인자에 대해
    서로 다른 설정을 필요로 하는 경우가 없을 것으로
    예상되기 때문에 제한 사항이 되지 않습니다. 하지만 이는 염두에 두어야 할 사항입니다.

``Join`` 은 어떻게 동작하나요?
-------------------------
이제 ``Join`` 컨텍스트 관리자를 어떻게 사용하는지 예시를 살펴보았으니,
내부적으로 어떻게 동작하는지 더 깊이 알아보겠습니다. 이를 통해 ``Join`` 이 제공하는
전체 기능을 이해하고, 직접 커스텀 클래스를 호환되게 만드는 데 도움이 될 것입니다.
이번에는, ``Join`` 클래스와 이를 지원하는
``Joinable``, ``JoinHook`` 클래스에 대해 설명합니다.

``Joinable``
^^^^^^^^^^^^

시작하기에 앞서, ``Join`` 컨텍스트 관리자와 호환되는 클래스는
기본 추상 클래스인 ``Joinable`` 을 상속해야 합니다. 특히, ``Joinable`` 은
다음 메소드를 구현해야 합니다.

- ``join_hook(self, **kwargs) -> JoinHook``

이 메소드는 ``Joinable`` 에 대한 ``JoinHook`` 인스턴스를 반환하며,
*join* 된 프로세스가 ``Joinable`` 이 수행하는 반복별 집합통신을
어떻게 모방해야 하는지를 결정합니다.

- ``join_device(self) -> torch.device``

이 메소드는 ``Join`` 컨텍스트 관리자가 집합통신을 수행하는 데 사용할 장치를 반환합니다.
예를 들어, ``torch.device("cuda:0")`` 또는
``torch.device("cpu")`` 가 있습니다.

- ``join_process_group(self) -> ProcessGroup``

이 메소드는 ``Join`` 컨텍스트 관리자가 집합통신을 수행하는 데
사용할 프로세스 그룹을 반환합니다.

특히, ``join_device`` 와 ``join_process_group`` 은
컨텍스트 관리자가 *join* 되었거나 그렇지 않은 프로세스 간 집합통신을 스케줄링할 수 있도록
보장하는데 필요합니다. 예를 들어,
각 반복마다 *join* 되지 않은 프로세스의 수를 all-reduce 연산으로 집계하거나,
아래에서 설명할 ``throw_on_early_termination=True``
동작을 구현할 때 사용됩니다.

``DistributedDataParallel`` 과 ``ZeroRedundancyOptimizer`` 는 이미
``Joinable`` 을 상속하고 위 메소드들을 구현하고 있으므로,
앞선 예시에서 바로 사용할 수 있었습니다.

``Joinable`` 클래스를 만들 때는 반드시 ``Joinable`` 의 생성자를 호출해야 합니다.
이 생성자는 내부적으로 ``JoinConfig`` 인스턴스를 초기화하며,
이는 컨텍스트 관리자가 올바르게 동작하기 위해 내부적으로 사용됩니다.
해당 인스턴스는 ``Joinable`` 객체의 ``_join_config`` 필드에 저장됩니다.

``JoinHook``
^^^^^^^^^^^^

다음으로, ``JoinHook`` 클래스는 컨텍스트 관리자에 진입할 수 있는
두 가지 진입점을 제공합니다.

- ``main_hook(self) -> None``

이 훅은 아직 *join* 되지 않은 랭크가 존재하는 동안, *join* 된 각 랭크에서
반복적으로 호출됩니다.
이는 각 학습 반복(순전파, 역전파, 옵티마이저 단계 등)에서
``Joinable`` 이 수행하는 집합통신을 모방하도록 설계되었습니다.

- ``post_hook(self, is_last_joiner: bool) -> None``

이 훅은 모든 랭크가 *join* 된 후 한 번 호출됩니다.
``bool`` 타입의 추가 인자로 ``is_last_joiner`` 가 전달되며, 해당 랭크가
마지막으로 *join* 된 랭크 중 하나인지 나타냅니다. 이 인자는 동기화 등에 유용하게 사용될 수 있습니다.

이런 훅에 대한 구체적인 예로,
``ZeroRedundancyOptimizer`` 의 ``main_hook`` 은
*join* 된 랭크가 여전히 자신이 담당하는 파라미터 샤드를 업데이트 및 동기화해야 하므로
일반적인 옵티마이저 단계를 수행합니다. ``DistributedDataParallel`` 의 ``post_hook`` 은
마지막으로 *join* 된 랭크 중 하나에서 최종 업데이트된 모델을 브로드캐스트하여
모든 랭크가 동일한 모델을 갖도록 합니다.

``Join``
^^^^^^^^

마지막으로, 이러한 것들이 ``Join`` 클래스 내에서 어떻게 동작하는지 살펴보겠습니다.

- ``__init__(self, joinables: List[Joinable], enable: bool = True, throw_on_early_termination: bool = False)``

이전 예시에서 보았듯이, 생성자는 학습 반복 과정에 참여하는
``Joinable`` 객체들의 리스트를 받습니다.
이들은 각 반복마다 집합통신을 수행하는 클래스여야 합니다.

``enable`` 은 ``bool`` 타입의 추가 인자이며,
불균등한 입력이 없을 것이라 확신한다면 ``False`` 로 설정할 수 있습니다.
이 경우 컨텍스트 관리자는 ``contextlib.nullcontext()`` 와 유사하게 무효화됩니다.
또한, 참여 중인 ``Joinable`` 객체에서도 *join* 관련 연산이 비활성화됩니다.

``throw_on_early_termination`` 은 ``bool`` 타입의 추가 인자이며,
불균등한 입력이 감지되는 즉시 각 랭크에서 예외를 발생시키도록
``True`` 로 설정할 수 있습니다. 이는 컨텍스트 관리자의
요구 사항을 충족하지 않는 경우에 유용합니다. 일반적으로는 여러가지
다른 클래스의 집합통신이 임의로 섞여 있는 경우가 있는데, 대표적으로
``SyncBatchNorm`` 레이어가 포함된 모델에서 ``DistributedDataParallel`` 을
사용할 때가 해당됩니다. 이런 경우 해당 인자를 ``True`` 로 설정하여 애플리케이션 로직에서
예외를 감지하고 적절히 처리할 수 있습니다.

- 핵심 로직은 ``__exit__()`` 메소드에서 동작합니다.
  이 메소드는 *join* 되지 않은 랭크가 존재하는 동안 각 ``Joinable`` 의 ``main_hook`` 을 호출하고,
  모든 랭크가 *join* 된 후에는 ``post_hook`` 을 호출합니다.
  ``main_hook`` 과 ``post_hook`` 은 ``Joinable`` 객체가
  전달된 순서대로 호출됩니다.

- 컨텍스트 관리자는 *join* 되지 않은 프로세스의 하트비트(heartbeat)를 필요로 합니다.
  따라서 각 ``Joinable`` 클래스는 각 반복마다 집합통신을 수행하기 전에 반드시
  ``Join.notify_join_context()`` 를 호출해야 합니다.
  컨텍스트 관리자는 첫 번째로 전달된 ``Joinable`` 객체만 실제로
  하트비트를 보내도록 합니다.

.. warning:: 위에서 언급한 ``throw_on_early_termination`` 관련 내용처럼,
    ``Join`` 컨텍스트 관리자는 특정 클래스 조합과는 호환되지 않을 수 있습니다.
    각 ``Joinable`` 의 ``JoinHook`` 은 각각의 훅이 완전히 실행된 후에
    다음 훅으로 넘어가야 하기 때문에 반드시 직렬화 가능해야 합니다. 즉,
    두 개의 훅이 동시에 실행되는 구조는 지원하지 않습니다. 또한, 현재는
    ``main_hook`` 과 ``post_hook`` 모두 동일한 결정론적인 순서로 반복됩니다.
    만약 이것이 큰 제약이라면, 추후 API를 수정하여
    순서를 커스터마이즈할 수 있습니다.

``Join`` 과 함께 작동하는 간단한 클래스 만들기
-------------------------------------
이전 섹션에서 여러 개념을 소개했으므로, 이제
간단한 예제를 통해 실제로 적용해보겠습니다. 여기서는 각 랭크가 *join* 되기 전까지
모든 랭크에서 본 입력의 수를 세는 클래스를 구현합니다.
이 예제는 여러분이 직접 클래스를 ``Join`` 컨텍스트 관리자와 호환되게
만드는 방법을 이해하는 데 도움이 될 것입니다.

구체적으로, 다음 코드는 각 랭크가 (1) *join* 되기 전까지 모든 랭크에서 본
입력의 개수와 (2) 모든 랭크에서 본
전체 입력 개수를 출력합니다.

::

    import os
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.distributed.algorithms.join import Join, Joinable, JoinHook

    BACKEND = "nccl"
    WORLD_SIZE = 2
    NUM_INPUTS = 5

    class CounterJoinHook(JoinHook):
        r"""
        :class:`Counter` 에 대한 *join* 훅.

        인자:
            counter (Counter): 해당 훅을 사용하는 :class:`Counter` 객체
            sync_max_count (bool): 모든 랭크가 *join* 되면 최대 개수를 동기화할지 여부
        """
        def __init__(
            self,
            counter,
            sync_max_count
        ):
            self.counter = counter
            self.sync_max_count = sync_max_count

        def main_hook(self):
            r"""
            counter 의 all-reduce 연산을 따라가기 위해, 크기가 1이고 0으로 채워진 텐서를 all-reduce 합니다.
            (이는 *join* 된 랭크가 아직 *join* 되지 않은 랭크의 집합통신을 동기화하기 위한 더미 연산입니다)
            """
            t = torch.zeros(1, device=self.counter.device)
            dist.all_reduce(t)

        def post_hook(self, is_last_joiner: bool):
            r"""
            ``sync_max_count=True`` 인 경우,
            모든 :class:`Counter`의 최대 개수를 동기화합니다.
            """
            if not self.sync_max_count:
                return
            rank = dist.get_rank(self.counter.process_group)
            common_rank = self.counter.find_common_rank(rank, is_last_joiner)
            if rank == common_rank:
                self.counter.max_count = self.counter.count.detach().clone()
            dist.broadcast(self.counter.max_count, src=common_rank)

    class Counter(Joinable):
        r"""
        :class:`Joinable` 의 예제로,
        학습 반복에 참여한 횟수를 세는 클래스입니다.
        """
        def __init__(self, device, process_group):
            super(Counter, self).__init__()
            self.device = device
            self.process_group = process_group
            self.count = torch.tensor([0], device=device).float()
            self.max_count = torch.tensor([0], device=device).float()

        def __call__(self):
            r"""
            이번 반복에서 모든 랭크가 처리한 입력의 총 개수를, 크기가 1이고 1로 채워진 텐서를
            all-reduce 연산하여 계산합니다. 그리고, 자신의 내부 카운트를 증가시킵니다.
            """
            Join.notify_join_context(self)
            t = torch.ones(1, device=self.device).float()
            dist.all_reduce(t)
            self.count += t

        def join_hook(self, **kwargs) -> JoinHook:
            r"""
            :meth:`__call__` 의 all-reduce 연산을 따라가는 *join* 훅을 반환합니다.

            이 *join* 훅은 다음 키워드 인자를 지원합니다.
                sync_max_count (bool, 선택 사항): 모든 랭크가 *join* 된 후
                    최대 개수를 모든 랭크에 동기화할지 여부. 기본값은 ``False`` 입니다.
            """
            sync_max_count = kwargs.get("sync_max_count", False)
            return CounterJoinHook(self, sync_max_count)

        @property
        def join_device(self) -> torch.device:
            return self.device

        @property
        def join_process_group(self):
            return self.process_group

        def find_common_rank(self, rank, to_consider):
            r"""
            프로세스 그룹에서 고려할 랭크들 중 최대 랭크를 반환합니다.
            """
            common_rank = torch.tensor([rank if to_consider else -1], device=self.device)
            dist.all_reduce(common_rank, op=dist.ReduceOp.MAX, group=self.process_group)
            common_rank = common_rank.item()
            return common_rank

    def worker(rank):
        assert torch.cuda.device_count() >= WORLD_SIZE
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

        counter = Counter(torch.device(f"cuda:{rank}"), dist.group.WORLD)
        inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

        with Join([counter], sync_max_count=True):
            for _ in inputs:
                counter()

        print(f"{int(counter.count.item())} inputs processed before rank {rank} joined!")
        print(f"{int(counter.max_count.item())} inputs processed across all ranks!")

    def main():
        mp.spawn(worker, nprocs=WORLD_SIZE, join=True)

    if __name__ == "__main__":
        main()

랭크 0은 5개의 입력을, 랭크 1은 6개의 입력을 처리하므로, 다음과 같은 출력을 생성합니다.

::

    10 inputs processed before rank 0 joined!
    11 inputs processed across all ranks!
    11 inputs processed before rank 1 joined!
    11 inputs processed across all ranks!

강조할 몇 가지 주요 포인트:

- ``Counter`` 인스턴스는 각 반복마다 한 번의 all-reduce 연산만 수행하므로,
  ``main_hook`` 도 이를 따라 단일 all-reduce 연산을 수행합니다.

- ``Counter`` 클래스는 ``__call__()`` 메소드 시작 부분에서
  ``Join.notify_join_context()`` 를 호출합니다.
  이는 각 반복별 집합통신(즉, all-reduce 연산) 전에 호출되어야 합니다.

- ``is_last_joiner`` 인자는 ``post_hook`` 에서 브로드캐스트 소스를
  결정하는 데 사용됩니다.

- 컨텍스트 관리자에 ``sync_max_count`` 키워드 인자를 전달하면,
  해당 인자가 ``Counter`` 의 ``join_hook`` 으로 전달됩니다.


.. _Join: https://pytorch.org/docs/master/distributed.algorithms.join.html
.. _분산 데이터 병렬 처리 시작하기: https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html
.. _분산 데이터 병렬 처리 시작하기 - 기본적인 사용법: https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html#id3
.. _Shard Optimizer States with ZeroRedundancyOptimizer: https://tutorials.pytorch.kr/recipes/zero_redundancy_optimizer.html
.. _DistributedDataParallel: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
.. _join(): https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join
.. _ZeroRedundancyOptimizer: https://pytorch.org/docs/stable/distributed.optim.html