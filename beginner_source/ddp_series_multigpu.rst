`Introduction <ddp_series_intro.html>`__ \|\|
`What is DDP <ddp_series_theory.html>`__ \|\|
**Single-Node Multi-GPU Training** \|\|
`Fault Tolerance <ddp_series_fault_tolerance.html>`__ \|\|
`Multi-Node training <../intermediate/ddp_series_multinode.html>`__ \|\|
`minGPT Training <../intermediate/ddp_series_minGPT.html>`__


DDP를 이용한 다중 GPU 훈련
===========================

저자: `Suraj Subramanian <https://github.com/subramen>`__
역자: `Nathan Kim <https://github.com/NK590>`__

.. grid:: 2

   .. grid-item-card:: :octicon:`mortar-board;1em;` 여기에서 배우는 것
      :class-card: card-prerequisites

      -  DDP를 이용하여 단일 GPU 학습 스크립트를 다중 GPU 학습 스크립트로 바꾸는 법
      -  분산 프로세스 그룹(distributed process group)을 설정하는 법
      -  분산 환경에서 모델을 저장 및 읽어오는 법

      .. grid:: 1

         .. grid-item::

            :octicon:`code-square;1.0em;` 이 튜토리얼에서 사용된 코드는 `GitHub <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py>`__ 에서 확인 가능

   .. grid-item-card:: :octicon:`list-unordered;1em;` 들어가기 앞서 준비할 것
      :class-card: card-prerequisites

      * `DDP가 어떻게 동작하는지 <ddp_series_theory.html>`__ 에 대한 전반적인 이해도
      * 다중 GPU를 가진 하드웨어 (이 튜토리얼에서는 AWS p3.8xlarge 인스턴스를 이용함)
      * CUDA 환경에서 `설치된 PyTorch <https://pytorch.org/get-started/locally/>`__

아래의 비디오 혹은 `유튜브 <https://www.youtube.com/watch/-LAtx9Q6DA8>`__ 도 참고해주세요.

.. raw:: html

   <div style="margin-top:10px; margin-bottom:10px;">
     <iframe width="560" height="315" src="https://www.youtube.com/embed/-LAtx9Q6DA8" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
   </div>

`이전 튜토리얼 <ddp_series_theory.html>`__ 에서, DDP가 어떻게 동작하는지에 대해 전반적으로 알아보았으므로, 이제 실제로 DDP를 어떻게 사용하는지 코드를 볼 차례입니다.
이 튜토리얼에서는, 먼저 단일 GPU 학습 스크립트에서 시작하여, 단일 노드를 가진 4개의 GPU에서 동작하게 만들 것입니다.
이 과정에서, 분산 훈련(distributed training)에 대한 중요한 개념들을 직접 코드로 구현하면서 다루게 될 것입니다.

.. note::
   만약 당신의 모델이 ``BatchNorm`` 레이어를 가지고 있다면, 해당 레이어 간 동작 상황의 동기화를 위해 이걸 모두 ``SyncBatchNorm`` 으로 바꿀 필요가 있습니다.

   도움 함수(helper function)
   `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm>`__ 를 이용하여 모델 안의 ``BatchNorm`` 레이어를 ``SyncBatchNorm`` 레이어로 바꿔주세요.

`single_gpu.py <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/single_gpu.py>`__ 와 `multigpu.py <https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py>`__ 의 차이

위 코드의 차이를 비교하면서 일반적으로 단일 GPU 학습 스크립트에서 DDP를 적용하는 법을 알 수 있습니다.

임포트
-------
-  ``torch.multiprocessing`` 은 Python의 네이티브 멀티프로세싱 모듈의 래퍼(wrapper)입니다.

-  분산 프로세스 그룹(distributed process group)은 서로 정보 교환이 가능하고 동기화가 가능한 모든 프로세스들을 포함합니다.

.. code-block:: python

   import torch
   import torch.nn.functional as F
   from utils import MyTrainDataset

   import torch.multiprocessing as mp
   from torch.utils.data.distributed import DistributedSampler
   from torch.nn.parallel import DistributedDataParallel as DDP
   from torch.distributed import init_process_group, destroy_process_group
   import os


프로세스 그룹 구성
------------------------------

-  먼저, 그룹 프로세스를 초기화하기 전에, `set_device <https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html?highlight=set_device#torch.cuda.set_device>`__ 를 호출하여
   각각의 프로세스에 GPU를 할당해주세요. 이 과정은 `GPU:0` 에 과도한 메모리 사용 혹은 멈춤 현상을 방지하기 위해 중요합니다.
-  이 프로세스 그룹은 TCP(기본) 혹은 공유 파일 시스템 등을 통하여 초기화될 수 있습니다.
   자세한 내용은 `프로세스 그룹 초기화 <https://pytorch.org/docs/stable/distributed.html#tcp-initialization>`__ 를 참고해주세요.
-  `init_process_group <https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group>`__ 으로 분산 프로세스 그룹을 초기화시킵니다.
-  추가적인 내용은 `DDP 백엔드 선택 <https://pytorch.org/docs/stable/distributed.html#which-backend-to-use>`__ 을 참고해주세요.

.. code-block:: python

   def ddp_setup(rank: int, world_size: int):
      """
      Args:
          rank: Unique identifier of each process
         world_size: Total number of processes
      """
      os.environ["MASTER_ADDR"] = "localhost"
      os.environ["MASTER_PORT"] = "12355"
      torch.cuda.set_device(rank)
      init_process_group(backend="nccl", rank=rank, world_size=world_size)



DDP 모델 구축
--------------------------

.. code-block:: python

   self.model = DDP(model, device_ids=[gpu_id])


입력 데이터 분산
--------------------------

-  `DistributedSampler <https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler>`__
   를 이용하여 모든 분산 프로세스에 입력 데이터를 나눕니다.
-  `DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__ 는 데이터셋과 샘플러를 결합하여
   주어진 데이터셋에 대한 반복 가능 객체를 제공합니다.
-  각각의 프로세스는 32개 샘플 크기의 입력 배치를 받습니다.
   이상적인 배치 크기는 ``32 * nprocs``, 혹은 4개의 GPU를 사용할 때 128입니다.

.. code-block:: python

    train_data = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=False,  # shuffle을 사용하지 않음
        sampler=DistributedSampler(train_dataset), # DistributedSampler를 여기서 사용
    )

-  매 에폭(epoch)의 시작마다 ``DistributedSampler`` 의 ``set_epoch()`` 메소드를 호출하는 것은 다수의 에폭에서 순서를 적절히 섞기 위해 필수적입니다.
   이를 사용하지 않을 경우, 매 에폭마다 같은 순서가 사용됩니다.

.. code-block:: python

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)   # 매 에폭마다 추가된 이 코드를 호출
        for source, targets in self.train_data:
          ...
          self._run_batch(source, targets)


모델 체크포인트(checkpoints) 저장
--------------------------------------
-  모델 체크포인트를 저장할 때, 하나의 프로세스에 대해서만 체크포인트를 저장하면 됩니다. 이렇게 하지 않으면,
   각각의 프로세스가 모두 동일한 상태를 저장하게 될 것입니다.
   `여기 <https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html#save-and-load-checkpoints>`__ 에서
   DDP 환경에서 모델의 저장과 읽어오기 등에 대해 자세한 내용을 확인할 수 있습니다.

.. code-block:: diff

    - ckp = self.model.state_dict()
    + ckp = self.model.module.state_dict()
    ...
    ...
    - if epoch % self.save_every == 0:
    + if self.gpu_id == 0 and epoch % self.save_every == 0:
      self._save_checkpoint(epoch)

.. warning::
   `집합 콜(Collective Calls) <https://pytorch.org/docs/stable/distributed.html#collective-functions>`__ 은 모든 분산 프로세스에서 동작하는 함수(functions)이며,
   특정 프로세스의 특정한 상태나 값을 모으기 위해 사용됩니다. 집합 콜은 집합 코드(collective code)를 실행하기 위해 모든 랭크(rank)를 필요로 합니다.
   이 예제에서, `_save_checkpoint`는 오로지 ``rank:0`` 프로세스에서만 실행되기 때문에, 어떠한 집합 콜도 가지고 있으면 안 됩니다.
   만약 집합 콜을 만들어야 된다면, ``if self.gpu_id == 0`` 확인 이전에 만들어져야 합니다.

분산 학습 작업의 실행
------------------------------------

-  새로운 인자값 ``rank`` (``device`` 를 대체)와 ``world_size`` 를 도입합니다.
-  ``rank`` 는 `mp.spawn <https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses>`__ 을 호출할 때
   DDP에 의해 자동적으로 할당됩니다.
-  ``world_size`` 는 학습 작업에 이용되는 프로세스의 개수입니다. GPU를 이용한 학습의 경우에는,
   이 값은 현재 사용중인 GPU의 개수 및 한 GPU에 할당된 프로세스의 개수에 해당합니다.

.. code-block:: diff

   - def main(device, total_epochs, save_every):
   + def main(rank, world_size, total_epochs, save_every):
   +  ddp_setup(rank, world_size)
      dataset, model, optimizer = load_train_objs()
      train_data = prepare_dataloader(dataset, batch_size=32)
   -  trainer = Trainer(model, train_data, optimizer, device, save_every)
   +  trainer = Trainer(model, train_data, optimizer, rank, save_every)
      trainer.train(total_epochs)
   +  destroy_process_group()

   if __name__ == "__main__":
      import sys
      total_epochs = int(sys.argv[1])
      save_every = int(sys.argv[2])
   -  device = 0      # shorthand for cuda:0
   -  main(device, total_epochs, save_every)
   +  world_size = torch.cuda.device_count()
   +  mp.spawn(main, args=(world_size, total_epochs, save_every,), nprocs=world_size)

코드는 다음과 같습니다:

.. code-block:: python
   def main(rank, world_size, total_epochs, save_every):
      ddp_setup(rank, world_size)
      dataset, model, optimizer = load_train_objs()
      train_data = prepare_dataloader(dataset, batch_size=32)
      trainer = Trainer(model, train_data, optimizer, rank, save_every)
      trainer.train(total_epochs)
      destroy_process_group()

   if __name__ == "__main__":
      import sys
      total_epochs = int(sys.argv[1])
      save_every = int(sys.argv[2])
      world_size = torch.cuda.device_count()
      mp.spawn(main, args=(world_size, total_epochs, save_every,), nprocs=world_size)



더 읽을거리
---------------

-  `결함 허용(fault tolerant) 분산 시스템 <ddp_series_fault_tolerance.html>`__  (본 시리즈의 다음 튜토리얼)
-  `DDP 입문 <ddp_series_theory.html>`__ (본 시리즈의 이전 튜토리얼)
-  `분산 데이터 병렬 처리(DDP) 시작하기 <https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html>`__
-  `프로세스 그룹 초기화 <https://pytorch.org/docs/stable/distributed.html#tcp-initialization>`__
