`소개 <ddp_series_intro.html>`__ \|\| **분산 데이터 병렬 처리 (DDP) 란 무엇인가?** \|\|
`단일 노드 다중-GPU 학습 <ddp_series_multigpu.html>`__ \|\|
`결함 내성 <ddp_series_fault_tolerance.html>`__ \|\|
`다중 노드 학습 <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT 학습 <../intermediate/ddp_series_minGPT.html>`__

분산 데이터 병렬 처리 (DDP) 란 무엇인가?
=======================================

Authors: `Suraj Subramanian <https://github.com/suraj813>`__
번역: `박지은 <https://github.com/rumjie>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` What you will learn

      *  How DDP works under the hood
      *  What is ``DistributedSampler``
      *  How gradients are synchronized across GPUs


   .. grid-item-card:: :octicon:`list-unordered;1em;` Prerequisites

      * Familiarity with `basic non-distributed training  <https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html>`__ in PyTorch

아래의 영상이나 `유투브 영상 youtube <https://www.youtube.com/watch/Cvdhwx-OBBo>`__을 따라 진행하세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Cvdhwx-OBBo" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

이 튜토리얼은 파이토치에서 분산 데이터 병렬 학습을 가능하게 하는 `분산 데이터 병렬 <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__ (DDP)
에 대해 소개합니다. 데이터 병렬 처리란 더 높은 성능을 달성하기 위해
여러 개의 디바이스에서 여러 데이터 배치들을 동시에 처리하는 방법입니다. 
파이토치에서, `분산 샘플러 <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`__ 는 
각 디바이스가 서로 다른 입력 배치를 받는 것을 보장합니다.
모델은 모든 디바이스에 복제되며, 각 사본은 변화도를 계산하는 동시에 `링 올-리듀스
알고리즘 <https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/>`__ 을 사용해 다른 사본과 동기화됩니다.

`예시 튜토리얼 <https://tutorials.pytorch.kr/intermediate/dist_tuto.html#>`__ 에서 DDP 메커니즘에 대해 파이썬 관점에서 심도 있는 설명을 볼 수 있습니다. 

``데이터 병렬 DataParallel`` (DP) 보다 DDP가 나은 이유
----------------------------------------------------

`DataParallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`__
is an older approach to data parallelism. DP is trivially simple (with just one extra line of code) but it is much less performant.
DDP improves upon the architecture in a few ways:

+---------------------------------------+------------------------------+
| ``DataParallel``                      | ``DistributedDataParallel``  |
+=======================================+==============================+
| More overhead; model is replicated    | Model is replicated only     |
| and destroyed at each forward pass    | once                         |
+---------------------------------------+------------------------------+
| Only supports single-node parallelism | Supports scaling to multiple |
|                                       | machines                     |
+---------------------------------------+------------------------------+
| Slower; uses multithreading on a      | Faster (no GIL contention)   |
| single process and runs into Global   | because it uses              |
| Interpreter Lock (GIL) contention     | multiprocessing              |
+---------------------------------------+------------------------------+

Further Reading
---------------

-  `Multi-GPU training with DDP <ddp_series_multigpu.html>`__ (next tutorial in this series)
-  `DDP
   API <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`__
-  `DDP Internal
   Design <https://pytorch.org/docs/master/notes/ddp.html#internal-design>`__
-  `DDP Mechanics Tutorial <https://tutorials.pytorch.kr/intermediate/dist_tuto.html#>`__
