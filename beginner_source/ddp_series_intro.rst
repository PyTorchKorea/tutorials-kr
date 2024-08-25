**소개** \|\| `DDP란 무엇인가 <ddp_series_theory.html>`__ \|\|
`단일 노드 다중-GPU 학습 <ddp_series_multigpu.html>`__ \|\|
`장애 내성 <ddp_series_fault_tolerance.html>`__ \|\|
`다중 노드 학습 <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT 학습 <../intermediate/ddp_series_minGPT.html>`__

PyTorch의 분산 데이터 병렬 처리 - 비디오 튜토리얼
=====================================================

저자: `Suraj Subramanian <https://github.com/suraj813>`__

아래 비디오를 보거나 `YouTube <https://www.youtube.com/watch/-K3bZYHYHEA>`__에서 함께 시청하세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/-K3bZYHYHEA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

이 비디오 튜토리얼 시리즈는 PyTorch에서 DDP(Distributed Data Parallel)를 사용한 분산 학습에 대해 안내합니다.

이 시리즈는 단순한 비분산 학습 작업에서 시작하여, 클러스터 내 여러 기기들(multiple machines)에서 학습 작업을 배포하는 것으로 마무리됩니다. 이 과정에서 `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`__을 사용한 장애 허용(fault-tolerant) 분산 학습에 대해서도 배우게 됩니다.

이 튜토리얼은 PyTorch에서 모델 학습에 대한 기본적인 이해를 전제로 합니다.

코드 실행
--------

튜토리얼 코드를 실행하려면 여러 개의 CUDA GPU가 필요합니다. 일반적으로 여러 GPU가 있는 클라우드 인스턴스에서 이를 수행할 수 있으며, 튜토리얼에서는 4개의 GPU가 탑재된 Amazon EC2 P3 인스턴스를 사용합니다.

튜토리얼 코드는 이 `GitHub 저장소 <https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series>`__에 호스팅되어 있습니다. 저장소를 복제하고 함께 진행하세요!

튜토리얼 섹션
--------------

0. 소개 (이 페이지)
1. `DDP란 무엇인가? <ddp_series_theory.html>`__ DDP가 내부적으로 수행하는 작업에 대해 간단히 소개합니다.
2. `싱글 노드 멀티-GPU 학습 <ddp_series_multigpu.html>`__ 한 기기에서 여러 GPU를 사용하여 모델을 학습하는 방법
3. `장애 내성 분산 학습 <ddp_series_fault_tolerance.html>`__ torchrun을 사용하여 분산 학습 작업을 견고하게 만드는 방법
4. `멀티 노드 학습 <../intermediate/ddp_series_multinode.html>`__ 여러 기기에서 여러 GPU를 사용하여 모델을 학습하는 방법
5. `DDP를 사용한 GPT 모델 학습 <../intermediate/ddp_series_minGPT.html>`__ DDP를 사용한 `minGPT <https://github.com/karpathy/minGPT>`__ 모델 학습의 “실제 예시”
