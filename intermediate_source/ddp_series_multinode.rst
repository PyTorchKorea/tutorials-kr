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

멀티노드 학습은 노드 간 통신 지연으로 인해 병목 현상이 발생한다는 점을 유의하십시오. 싱글노드에서 4개의 GPU를 사용한 학습 작업이 
4개의 노드에서 1개의 GPU를 사용한 것보다 빠를 것입니다. 

로컬 순위와 글로벌 순위 Local and Global ranks
~~~~~~~~~~~~~~~~~~~~~~~~
싱글노드를 설정할 때, 학습 프로세스의 각 장치의
 ``gpu_id`` 가 기록되고 있었습니다. ``torchrun`` 은 이 값을 환경 변수 ``LOCAL_RANK`` 로 기록하고,
이는 노드에서 각각의 고유한 GPU 프로세스를 식별하기 위한 값입니다. For a unique identifier across all the nodes, ``torchrun`` provides another variable
``RANK`` which refers to the global rank of a process.

.. 주의사항::
   학습 시 중요한 로직에 ``순위`` 를 사용하지 마십시오. ``torchrun``의 실패 혹은 멤버십의 변경으로 인해 재시작되면 해당 프로세스에서
   같은 ``로컬 순위`` 와 ``순위`` 가 유지된다는 보장이 없습니다.

이질적 스케일링
~~~~~~~~~~~~~~~~~~~~~~
Torchrun 은 *이질적 스케일링* 을 지원합니다. 예를 들어, 각각의 멀티노드 머신이 학습에 참여하는
GPU의 개수가 달라질 수 있습니다. 이 비디오에서는 2 대의 머신에 코드를 배포하여 한 개의 머신에는 4개, 
다른 한 개의 머신에는 2개의 GPU를 사용합니다. 

문제 해결
~~~~~~~~~~~~~~~~~~

-  노드들이 TCP를 통해 서로 통신이 가능한지 확인하세요. 
-  환경 변수 ``NCCL_DEBUG`` 를 ``INFO`` 로 설정하여 (명령어: 
   ``export NCCL_DEBUG=INFO``) 이슈를 확인할 수 있는 상세 로그를 출력하세요.
-  분산 백엔드를 위해 명시적인 네트워크 인터페이스 설정이 필요할 수도 있습니다. (``export NCCL_SOCKET_IFNAME=eth0``). 
   이 `링크 <https://pytorch.org/docs/stable/distributed.html#choosing-the-network-interface-to-use>`__. 를 참조하세요. 

읽을거리
---------------
-  `DDP 로 GPT 모델 학습시키기 <ddp_series_minGPT.html>`__  (이 시리즈의 다음 튜토리얼)
-  `결함 내성 분산 학습 <../beginner/ddp_series_fault_tolerance.html>`__ (이 시리즈의 이전 튜토리얼)
-  `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__
-  `랑데부 인자 <https://pytorch.org/docs/stable/elastic/run.html#note-on-rendezvous-backend>`__
-  `AWS 에서 클러스터 셋팅하기 <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/setup_pcluster_slurm.md>`__
-  `Slurm 문서 <https://slurm.schedmd.com/>`__
