RPC를 이용한 분산 파이프라인 병렬 처리(Parallelism)
===================================================================
**저자**: `Shen Li <https://mrshenli.github.io/>`_
**번역**: `양수진 </https://github.com/musuys>`_ , `RushBsite </https://github.com/RushBsite>`_

선수과목(Prerequisites):

-  `파이토치 분산(Distributed) 개요 <../beginner/dist_overview.html>`__
-  `단일 머신(Single-Machine) 모델 병렬(Parallel) 모범 사례 <https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html>`__
-  `분산 RPC 프레임워크(Framework) <https://pytorch.org/tutorials/intermediate/rpc_tutorial.html>`__
-  RRef 도움 함수:
   `RRef.rpc_sync() <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_sync>`__,
   `RRef.rpc_async() <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_async>`__, and
   `RRef.remote() <https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.remote>`__



이 튜토리얼에서는 Resnet50 모델을 사용하여 `torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`__
APIs로 분산 파이프라인 병렬 처리를 구현합니다. `단일 머신 모델 병렬 모범 사례 <model_parallel_tutorial.html>`_ 에서 논의된 다중 GPU(multi-GPU) 파이프라인 병렬 처리의 분산 대안으로 볼 수 있습니다.


.. note:: 이 튜토리얼에서는 PyTorch v1.6.0 이상이 필요합니다.

.. note:: 이 튜토리얼의 전체 소스 코드는
    `pytorch/examples <https://github.com/pytorch/examples/tree/master/distributed/rpc/pipeline>`__ 에서 찾을 수 있습니다.

Basics
----------------


이전튜토리얼, `분산 RPC 프레임워크 시작하기 <rpc_tutorial.html>`_ 에서는 `torch.distributed.rpc <https://pytorch.org/docs/master/rpc.html>`_
를 사용하여 RNN 모델에 대한 분산 모델 병렬 처리를 구현하는 방법을 보여줍니다. 이 튜토리얼은 하나의 GPU를 사용하여 ``EmbeddingTable`` 을 호스팅하며,
제공된 코드는 정상 작동합니다. 하지만 모델이 여러 GPU에 있는 경우, 모든 GPU의 분할 사용률(amortized utilization)을 높이기 위해 몇가지 추가 단계가 필요합니다.
파이프라인 병렬 처리는 이 경우에 도움이 될 수 있는 패러다임의 한 유형입니다.

이 튜토리얼에서는, ``ResNet50`` 을  `단일 머신 모델 병렬 모범 사례 <model_parallel_tutorial.html>`_ 에서도 사용되는 예제 모델로 사용합니다.
마찬가지로, ``ResNet50`` 모델은 두개의 shard로 나뉘고 입력 배치(batch)도 여러 개의 분할로 분할되어(partitioned) 파이프라인 방식으로 두개의 모델 shard로 공급됩니다.
차이점은, CUDA 스트림을 사용하여 실행을 병렬화하는 대신에 이 튜토리얼은 비동기(asynchronous) RPC를 실행한다는 것입니다.
따라서, 이 튜토리얼에 제시된 솔루션은  머신 경계에서도 작동합니다.
이 튜토리얼의 나머지 부분에서는 구현을 4단계로 설명합니다.



Step 1: ResNet50 모델 분할
--------------------------------

이 단계는  ``ResNet50`` 을 두 개의 모델 샤드(shard)에 구현하는 준비 단계입니다.
아래 코드는
`torchvision에서의 ResNet 구현 <https://github.com/pytorch/vision/blob/7c077f6a986f05383bcb86b535aedb5a63dd5c4b/torchvision/models/resnet.py#L124>`_ 에서 차용한 것입니다.
``ResNetBase`` 모듈에는 두개의 ResNet 샤드에 대한 공통 구성 요소와 속성이 포함되어 있습니다.


.. code:: python

    import threading

    import torch
    import torch.nn as nn

    from torchvision.models.resnet import Bottleneck

    num_classes = 1000


    def conv1x1(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    class ResNetBase(nn.Module):
        def __init__(self, block, inplanes, num_classes=1000,
                    groups=1, width_per_group=64, norm_layer=None):
            super(ResNetBase, self).__init__()

            self._lock = threading.Lock()
            self._block = block
            self._norm_layer = nn.BatchNorm2d
            self.inplanes = inplanes
            self.dilation = 1
            self.groups = groups
            self.base_width = width_per_group

        def _make_layer(self, planes, blocks, stride=1):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if stride != 1 or self.inplanes != planes * self._block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * self._block.expansion, stride),
                    norm_layer(planes * self._block.expansion),
                )

            layers = []
            layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * self._block.expansion
            for _ in range(1, blocks):
                layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def parameter_rrefs(self):
            return [RRef(p) for p in self.parameters()]



이제, 우리는 두개의 모델 샤드를 정의할 준비가 되었습니다. 생성자에서는 간단하게 모든 resNet50 레이어들을
두개의 부분으로 나누고, 각 부분을 제공된 디바이스로 이동시킵니다. 각 샤드의 ``foward`` 함수는 입력 데이터의
``RRef`` 를 가져오고, 로컬로 데이터를 가져온 다음, 적절한 디바이스로 이동시킵니다. 모든 레이어의 입력을 처리한 후에,
출력을 CPU로 전달하고 반환합니다. RPC API 가 발신자(caller)와 수신자(callee)의 장치수가 맞지 않는 경우의 디바이스 에러를
방지하기 위해 tensor 가 유효한 cpu에 존재하는것을 요구하기 때문입니다.



.. code:: python

    class ResNetShard1(ResNetBase):
        def __init__(self, device, *args, **kwargs):
            super(ResNetShard1, self).__init__(
                Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

            self.device = device
            self.seq = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                self._norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(64, 3),
                self._make_layer(128, 4, stride=2)
            ).to(self.device)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x_rref):
            x = x_rref.to_here().to(self.device)
            with self._lock:
                out =  self.seq(x)
            return out.cpu()


    class ResNetShard2(ResNetBase):
        def __init__(self, device, *args, **kwargs):
            super(ResNetShard2, self).__init__(
                Bottleneck, 512, num_classes=num_classes, *args, **kwargs)

            self.device = device
            self.seq = nn.Sequential(
                self._make_layer(256, 6, stride=2),
                self._make_layer(512, 3, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
            ).to(self.device)

            self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

        def forward(self, x_rref):
            x = x_rref.to_here().to(self.device)
            with self._lock:
                out = self.fc(torch.flatten(self.seq(x), 1))
            return out.cpu()



Step 2: ResNet50 모델 샤드를 하나의 모듈로 연결
------------------------------------------------------


그다음, ``DistResNet50`` 모듈을 두개의 샤드를 조립하고 파이프 라인 병렬 로직을
수행하도록 생성합니다. 생성자에서는, 두개의``rpc.remote`` 호출을 실행해, 두개의 샤드를 각기 
다른 두개의 RPC 작업자에 배치하고, 호출된 두 모델의 ``RRef`` 파트를 각각 유지하여 포워드(foward) 패스에서
참조 가능하게 합니다. ``foward`` 함수는 입력 배치를 여러 마이크로 배치로 분할하고 파이프라인 방식으로 두 
모엘 파트에 마이크로 배치를 제공합니다. 먼저, ``rpc.rmote`` 를 호출하여 첫번째 샤드를 마이크로 배치에 적용한 다음
``RRef`` 중간 출력을 두번째 모델 샤드에 반환합니다. 그 후, 모든 마이크로 출력의 ``Future`` 를 수집하고 
루프 이후 모든 출력을 대기합니다. ``remote()`` 와 ``rpc_async()`` 모두 즉시 반환되고 비동기적으로 실행됩니다.
따라서 전체적인 루프는 차단 없이 이루어지며, 동시에 여러 rpc를 실행 가능하게 합니다. 두 모델 파트에서
마이크로 배치의 실행 순서는 중간출력 ``y_rref`` 에 의해 보존됩니다. 마이크로 배치간의 실행순서는 중요하지 않습니다.
마지막으로, 포워드 함수의 모든 마이크로 배치의 출력을 하나의 단일 tensor 로 연결하고 반환합니다.
``parameter_rrefs`` 함수는 나중에 사용될 분산 최적화 프로그램 구성을 단순화 시키는것 에 사용됩니다.


.. code:: python

    class DistResNet50(nn.Module):
        def __init__(self, num_split, workers, *args, **kwargs):
            super(DistResNet50, self).__init__()

            self.num_split = num_split

            # Put the first part of the ResNet50 on workers[0]
            self.p1_rref = rpc.remote(
                workers[0],
                ResNetShard1,
                args = ("cuda:0",) + args,
                kwargs = kwargs
            )

            # Put the second part of the ResNet50 on workers[1]
            self.p2_rref = rpc.remote(
                workers[1],
                ResNetShard2,
                args = ("cuda:1",) + args,
                kwargs = kwargs
            )

        def forward(self, xs):
            out_futures = []
            for x in iter(xs.split(self.split_size, dim=0)):
                x_rref = RRef(x)
                y_rref = self.p1_rref.remote().forward(x_rref)
                z_fut = self.p2_rref.rpc_async().forward(y_rref)
                out_futures.append(z_fut)

            return torch.cat(torch.futures.wait_all(out_futures))

        def parameter_rrefs(self):
            remote_params = []
            remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
            remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
            return remote_params



Step 3: 학습 루프 정의하기
-------------------------------


모델을 정의했으므로 , 이번에는 학습 루프를 구현해 보겠습니다. 우리는 랜덤 입력들과 라벨들을
전담하며 분산된 역방향 패스 및 최적화 단계를 컨트롤 하는 ``master`` 작업자를 사용합니다.
작업자는 먼저 ``DistResNet50`` 모듈의 인스턴스를 생성합니다. 그 다음, 각 배치에 대한 마이크로 배치의 수를
지정하고, 두 RPC 작업자의 이름도 제공합니다.(예 : "worker1" 및 "worker2") 다음으로, 손실(loss) 함수를 정의하고
``RRefs`` 의 매개변수 목록을 얻도록 ``parameter_rrefs()`` 헬퍼를 사용하여 ``DistributedOptimizer`` 를 생성합니다.
이후의 주 학습 루프는 ``dist_autograd`` 를 사용하여 시작하는 것을 제외하곤, 일반적인 로컬 학습과 매우 유사합니다. 
이는 역방향 실행 및 역방향 프로그램 모두에 대해 ``context_id`` 를 제공하고 ``step()`` 를 최적화 하기 위함입니다.


.. code:: python

    import torch.distributed.autograd as dist_autograd
    import torch.optim as optim
    from torch.distributed.optim import DistributedOptimizer

    num_batches = 3
    batch_size = 120
    image_w = 128
    image_h = 128


    def run_master(num_split):
        # put the two model parts on worker1 and worker2 respectively
        model = DistResNet50(num_split, ["worker1", "worker2"])
        loss_fn = nn.MSELoss()
        opt = DistributedOptimizer(
            optim.SGD,
            model.parameter_rrefs(),
            lr=0.05,
        )

        one_hot_indices = torch.LongTensor(batch_size) \
                            .random_(0, num_classes) \
                            .view(batch_size, 1)

        for i in range(num_batches):
            print(f"Processing batch {i}")
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, one_hot_indices, 1)

            with dist_autograd.context() as context_id:
                outputs = model(inputs)
                dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
                opt.step(context_id)



Step 4: RPC 프로세서 실행
----------------------------


마지막으로, 아래 코드는 모든 프로세스에 대한 대상 함수를 나타냅니다. 주 로직은 ``run_master`` 에
정의되어 있습니다. 작업자는 마스터의 명령을 수동적으로 기다리고 명령이 오면, ``init_rpc`` 와 ``shutdown`` 을
단순히 실행시키며, 여기서 ``shutdown`` 는 기본적으로 모든 RPC 참가자가 완료 될 때까지 차단됩니다.

.. code:: python

    import os
    import time

    import torch.multiprocessing as mp


    def run_worker(rank, world_size, num_split):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        options = rpc.ProcessGroupRpcBackendOptions(num_send_recv_threads=128)

        if rank == 0:
            rpc.init_rpc(
                "master",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=options
            )
            run_master(num_split)
        else:
            rpc.init_rpc(
                f"worker{rank}",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=options
            )
            pass

        # block until all rpcs finish
        rpc.shutdown()


    if __name__=="__main__":
        world_size = 3
        for num_split in [1, 2, 4, 8]:
            tik = time.time()
            mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
            tok = time.time()
            print(f"number of splits = {num_split}, execution time = {tok - tik}")



아래의 출력은 각 배치의 분할 수를 늘림으로써 얻은 속도 향상을 보여줍니다.

::

    $ python main.py
    Processing batch 0
    Processing batch 1
    Processing batch 2
    number of splits = 1, execution time = 16.45062756538391
    Processing batch 0
    Processing batch 1
    Processing batch 2
    number of splits = 2, execution time = 12.329529762268066
    Processing batch 0
    Processing batch 1
    Processing batch 2
    number of splits = 4, execution time = 10.164430618286133
    Processing batch 0
    Processing batch 1
    Processing batch 2
    number of splits = 8, execution time = 9.076049566268921
