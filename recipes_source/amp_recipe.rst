AUTOMATIC MIXED PRECISION(자동 혼합 정밀도)
====================

**Author**: `Michael Carilli <https://github.com/mcarilli>`_
  **번역**: `하헌진 <https://github.com/hihunjin>`_


`torch.cuda.amp`_ 는 mixed precision을 사용할 수 있는 편리한 메서드를 제공합니다. precision은 ``torch.float32`` (``float``)과, ``torch.float16`` (``half``)를 사용하는 operations들이 있습니다. Linear 층과 convolusion 층들과 같은 operations들은, ``float16`` 에서 더욱 빠릅니다. reduction과 같은 operations들은 ``float32`` 의 동적인 범주 계산을 요구하기도 합니다. Mixed precision은 각각의 operation에 맞는 적절한 데이터 타입을 맞춰줍니다. 따라서 네트워크의 실행 시간과 메모리 할당량을 줄일 수 있습니다.

보통, "automatic mixed precision training"은 `torch.cuda.amp.autocast`_ 와 `torch.cuda.amp.GradScaler`_ 를 함께 사용합니다.

이 레시피는 먼저 디폴트 precision에서 간단한 네트워크의 성능을 측정하고, 다음으로 ``autocast`` 와 ``GradScaler`` 를 추가하여, 같은 네트워크에서 향상된 성능을 살펴봅니다.

이 레시피를 파이썬 스크립트로 다운로드하고 실행해 볼 수 있습니다. 환경은 Pytorch 1.6+와 CUDA-capable GPU가 필요합니다.

Mixed precision은 원시적으로 텐서 코어를 사용할 수 있는 아키텍처(볼타, 튜링, 암페어)에서 효과를 잘 발휘합니다. 이 레시피는 이 아키텍처에서 2에서 3배의 속도 향상을 보여줍니다. 이전 버전의 아키텍처(케플러, 맥스웰, 파스칼)에서는, 완만한 속도 향상을 관찰할 수 있습니다. ``nvidia-smi`` 를 실행해, GPU 아키텍처를 확인하세요.
::

   import torch, time, gc
   
   # Timing utilities
   start_time = None
   
   def start_timer():
       global start_time
       gc.collect()
       torch.cuda.empty_cache()
       torch.cuda.reset_max_memory_allocated()
       torch.cuda.synchronize()
       start_time = time.time()
   
   def end_timer_and_print(local_msg):
       torch.cuda.synchronize()
       end_time = time.time()
       print("\n" + local_msg)
       print("Total execution time = {:.3f} sec".format(end_time - start_time))
       print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))

간단한 네트워크
------------

아래 Linear 레이어와 ReLU의 연속으로 이루어진 모델이 mixed precision으로 속도 향상이 될 것입니다.
::

   def make_model(in_size, out_size, num_layers):
       layers = []
       for _ in range(num_layers - 1):
           layers.append(torch.nn.Linear(in_size, in_size))
           layers.append(torch.nn.ReLU())
       layers.append(torch.nn.Linear(in_size, out_size))
       return torch.nn.Sequential(*tuple(layers)).cuda()

``batch_size``, ``in_size``, ``out_size``, 그리고 ``num_layers`` 를 GPU에 꽉 차도록 크게 조절할 수 있습니다. 특히, mixed precision은 GPU가 포화상태일 때, 가장 크게 속도를 증가합니다. 작은 네트워크는 CPU 바인딩 될 수 있습니다. 이 경우 mixed precision이 성능을 향상시키지 못할 수 있습니다. GPU에서 텐서 코어의 활용을 가능하도록 Linear 레이어의 사이즈는 8의 배수로 선택됩니다.

연습문제 : 레이어의 차원을 조절해서 mixed precision이 얼마나 속도를 변화하는지 확인해보세요.

::

   batch_size = 512 # 128, 256, 513로 다양하게 시도해보세요.
   in_size = 4096
   out_size = 4096
   num_layers = 3
   num_batches = 50
   epochs = 3

   # 디폴트 precision으로 데이터를 생성하세요.
   # 같은 데이터가 디폴트와 mixed precision 시행 모두에 사용됩니다.
   # 입력 데이터의 타입(dtype)을 조절하지 않는것을 추천합니다.
   data = [torch.randn(batch_size, in_size, device="cuda") for _ in range(num_batches)]
   targets = [torch.randn(batch_size, out_size, device="cuda") for _ in range(num_batches)]

   loss_fn = torch.nn.MSELoss().cuda()

디폴트 Precision
------------

``torch.cuda.amp`` 없이, 디폴트 precision(``torch.float32``)로 아래 간단한 네트워크를 실행합니다.

::

    net = make_model(in_size, out_size, num_layers)
    opt = torch.optim.SGD(net.parameters(), lr=0.001)

    start_timer()
    for epoch in range(epochs):
        for input, target in zip(data, targets):
            output = net(input)
            loss = loss_fn(output, target)
            loss.backward()
            opt.step()
            opt.zero_grad() # set_to_none=True 를 사용하면 속도를 약간 향상할 수 있습니다.
    end_timer_and_print("Default precision:")

autocast 사용하기
------------

`torch.cuda.amp.autocast`_ 의 인스턴스는 컨텍스트 매니저로써 mixed precision을 수행하는 코드 리전을 제공합니다.

이 리전에서는, 쿠다 operations이 autocast에 의해 선택된 데이터 타입으로 실행됩니다. 이는 정확도를 유지하면서 성능을 향상시킵니다. `Autocast Op Reference`_ 를 참고하여 각각의 operation마다 어떤 precision을 선택하는지 확인해보세요.

::

    for epoch in range(0): # 0 epochs, 이 섹션은 설명을 위한 스크립트입니다.
        for input, target in zip(data, targets):
            # autocast 아래에서 순전파를 진행합니다.
            with torch.cuda.amp.autocast():
                output = net(input)
                # Linear 레이어의 autocast는 float16이기에, output의 데이터 타입은 float16입니다.
                assert output.dtype is torch.float16

                loss = loss_fn(output, target)
                # loss는 float32입니다. 이유: mse_loss 레이어 때문에.
                assert loss.dtype is torch.float32

            # backward() 전에 autocast를 빠져 나옵니다.
            # 역전파시 autocast는 지양됩니다.
            # 역전파시 autocast 데이터 타입은 순전파의 데이터타입과 같습니다.
            loss.backward()
            opt.step()
            opt.zero_grad()

GradScaler 사용하기
------------

`Gradient scaling`_ 은 mixed precision 훈련 시 작은 크기의 gradient 값이 0으로 바뀌는 underflowing 현상을 막습니다.

`torch.cuda.amp.GradScaler`_ 는 gradient를 크기를 편리하게 조절합니다.

::

    # scaler를 먼저 생성합니다.
    # 네트워크가 수렴하지 않는다면 pytorch github에 issue를 생성해주세요.
    # 같은 GradScaler 인스턴스가 실행 전반에 동일하게 사용되어야 합니다.
    # 다중 수렴을 시행한다면, 각각의 시행에 맞는 GradSacler 인스턴스를 사용해야 합니다. GradScaler 인스턴스는 무겁지 않습니다.
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(0): # 0 epochs, 이 섹션은 설명을 위한 스크립트입니다.
        for input, target in zip(data, targets):
            with torch.cuda.amp.autocast():
                output = net(input)
                loss = loss_fn(output, target)

            # Scales 손실값. 스케일된 gradient를 만들기 위해, backward()를 스케일된 손실값에 적용해주세요.
            scaler.scale(loss).backward()

            # sacler.step() 은 먼저 optimizer에 할당된 파라미터들을 unscale합니다.
            # 이 gradients들이 inf나 NaN을 가지고 있지 않다면, optimizer.step()이 호출됩니다.
            # 가지고 있다면, optimizer.step()은 건너뜁니다.
            scaler.step(opt)

            # 다음 루프를 위해 scale을 업데이트 합니다.
            scaler.update()

            opt.zero_grad()

모두 합치기 : "Automatic Mixed Precision"
------------

``use_amp`` 를 통해 ``autocast`` 와 ``GradScaler`` 를 끄고 켤 수 있습니다.

::

    use_amp = True

    net = make_model(in_size, out_size, num_layers)
    opt = torch.optim.SGD(net.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_timer()
    for epoch in range(epochs):
        for input, target in zip(data, targets):
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = net(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
    end_timer_and_print("Mixed precision:")

Gradient를 검사하고 수정하기(gradient 클리핑)
------------

``scaler.scale(loss).backward()`` 로 생성된 모든 Gradient는 스케일링됩니다. ``backward()`` 와 ``scaler.step(optimizer)`` 사이에서 파라미터의 ``.grad`` 어트리뷰트로 검사하고 싶다면, `scaler.unscale_(optimizer)`_ 를 사용해 먼저 스케일링을 해제해야 합니다.

::

    for epoch in range(0): # 0 epochs, 이 섹션은 설명을 위한 스크립트입니다.
        for input, target in zip(data, targets):
            with torch.cuda.amp.autocast():
                output = net(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()

            # Gradient를 unscale합니다.
            scaler.unscale_(opt)

            # optimizer에 할당된 파라미터들의 gradient가 unscaled되었기 때문에, 평소처럼 클리핑을 합니다.
            # max_norm을 같은 값으로 사용할 수 있습니다.
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

저장하기/다시 시작하기
------------

AMP를 사용한 저장/재시작은 `scaler.state_dict`_ 와 `scaler.load_state_dict`_ 를 사용합니다.

저장을 할 때는, scaler의 state dict를 모델과 optimizer의 state dict과 함께 저장합니다. 이것을 루프 시작 전에 하거나, ``scaler.update()`` 후 루프의 끝에 합니다.

::

    checkpoint = {"model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict()}
    # torch.save(checkpoint, "filename")

재시작 할 때는, scaler의 state dict을 모델과 optimizer와 함께 가져옵니다.

::

    # checkpoint = torch.load("filename",
    #                         map_location = lambda storage, loc: storage.cuda(dev))
    net.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

만약 AMP 없이 checkpoint가 생성되었지만, AMP를 사용해 훈련을 재시작하고 싶다면, GradScaler의 새 인스턴스로 훈련할 수 있습니다.

만약 AMP가 함께 checkpoint가 저장되었고, AMP 없이 훈련을 재시작하고 싶다면, 저장된 scaler state를 무시하고 훈련할 수 있습니다.

추론/평가
------------

``autocast`` 가 추론과 평가의 순전파를 감쌀 수 있습니다. ``GradScaler`` 는 필요하지 않습니다.



.. _torch.cuda.amp: https://pytorch.org/docs/stable/amp.html
.. _torch.cuda.amp.autocast: https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast
.. _torch.cuda.amp.GradScaler: https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
.. _Autocast Op Reference: https://pytorch.org/docs/stable/amp.html#autocast-op-reference
.. _Gradient scaling: https://pytorch.org/docs/stable/amp.html#gradient-scaling
.. _scaler.unscale_(optimizer): https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.unscale_
.. _scaler.state_dict: https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.state_dict
.. _scaler.load_state_dict: https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.load_state_dict