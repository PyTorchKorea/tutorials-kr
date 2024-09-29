`소개 <../beginner/ddp_series_intro.html>`__ \|\| `분산 데이터 병렬 처리 (DDP) 란 무엇인가? <../beginner/ddp_series_theory.html>`__ \|\| `단일
노드 다중-GPU 학습 <../beginner/ddp_series_multigpu.html>`__ \|\| `결함
내성 <../beginner/ddp_series_fault_tolerance.html>`__ \|\| **다중 노드 (Multinode)
학습** \|\| `minGPT 학습 <ddp_series_minGPT.html>`__

멀티노드(Multinode) 학습
==================

저자: `Suraj Subramanian <https://github.com/suraj813>`__
번역: `박지은 <https://github.com/rumjie>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` 이 장에서 배우는 것

      - ``torchrun`` 으로 멀티노드 학습 시작하기
      - 싱글노드에서 멀티노드 학습으로 옮기기 위한 코드 변경 (및 염두에 두어야 하는 것들)

      .. grid:: 1

         .. grid-item::

            :octicon:`code-square;1.0em;` 이 튜토리얼에 사용된 코드 참고 - `GitHub <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py>`__

   .. grid-item-card:: :octicon:`list-unordered;1em;` 필요 사항

      - `다중 GPU 학습 <../beginner/ddp_series_multigpu.html>`__ 과 `torchrun <../beginner/ddp_series_fault_tolerance.html>`__ 에 익숙할 것
      - 2개 이상의 TCP 접근이 가능한 GPU 머신 (본 튜토리얼에서는 AWS p3.2xlarge를 사용함)
      - 모든 머신에 CUDA가 설치된 `파이토치 <https://pytorch.org/get-started/locally/>`__  

아래의 영상이나 `유튜브 영상 <https://www.youtube.com/watch/KaAJtI1T2x4>`__ 을 따라 진행하세요. 

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/KaAJtI1T2x4" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

멀티노드 학습은 여러 대의 머신에 학습 작업을 실행하는 것입니다. 
실행의 두 가지 방법은 아래와 같습니다.

-  각 머신에서 동일한 rendezvous 인수로 ``torchrun`` 명령어를 실행하기 
-  SLURM 과 같은 워크로드 매니저 를 사용하여 컴퓨터 클러스터에 배포하기

이 영상에서는 싱글노드 다중 GPU 로부터 멀티노드 학습으로 옮기기 위한 (최소한의) 코드 변경을 다루고, 
위에서 언급한 두 가지 방법의 학습 스크립트를 실행할 것입니다. 

Note that multinode training is bottlenecked by inter-node communication latencies. Running a training job
on 4 GPUs on a single node will be faster than running it on 4 nodes with 1 GPU each.

Local and Global ranks
~~~~~~~~~~~~~~~~~~~~~~~~
In single-node settings, we were tracking the 
``gpu_id`` of each device running our training process. ``torchrun`` tracks this value in an environment variable ``LOCAL_RANK``
which uniquely identifies each GPU-process on a node. For a unique identifier across all the nodes, ``torchrun`` provides another variable
``RANK`` which refers to the global rank of a process.

.. warning::
   Do not use ``RANK`` for critical logic in your training job. When ``torchrun`` restarts processes after a failure or membership changes, there is no guarantee
   that the processes will hold the same ``LOCAL_RANK`` and ``RANKS``. 
 

Heteregeneous Scaling
~~~~~~~~~~~~~~~~~~~~~~
Torchrun supports *heteregenous scaling* i.e. each of your multinode machines can have different number of 
GPUs participating in the training job. In the video, I deployed the code on 2 machines where one machine has 4 GPUs and the
other used only 2 GPUs.


Troubleshooting
~~~~~~~~~~~~~~~~~~

-  Ensure that your nodes are able to communicate with each other over
   TCP.
-  Set env variable ``NCCL_DEBUG`` to ``INFO`` (using
   ``export NCCL_DEBUG=INFO``) to print verbose logs that can help
   diagnose the issue.
-  Sometimes you might need to explicitly set the network interface for
   the distributed backend (``export NCCL_SOCKET_IFNAME=eth0``). Read
   more about this
   `here <https://pytorch.org/docs/stable/distributed.html#choosing-the-network-interface-to-use>`__.


Further Reading
---------------
-  `Training a GPT model with DDP <ddp_series_minGPT.html>`__  (next tutorial in this series)
-  `Fault Tolerant distributed training <../beginner/ddp_series_fault_tolerance.html>`__ (previous tutorial in this series)
-  `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__
-  `Rendezvous
   arguments <https://pytorch.org/docs/stable/elastic/run.html#note-on-rendezvous-backend>`__
-  `Setting up a cluster on
   AWS <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/setup_pcluster_slurm.md>`__
-  `Slurm docs <https://slurm.schedmd.com/>`__
