TorchScript의 동적 병렬 처리(DYNAMIC PARALLELISM)
==================================================

이 튜토리얼에서는, 우리는 TorchScript에서 *dynamic inter-op parallelism* 를 하는 구문(syntax)을 소개합니다.
이 병렬처리에는 다음과 같은 속성이 있습니다:

* 동적(dynamic) - 생성된 병렬 작업의 수와 작업 부하는 프로그램의 제어 흐름에 따라 달라질 수 있습니다.
* inter-op - 병렬 처리는 TorchScript 프로그램 조각을 병렬로 실행하는 것과 관련이 있습니다. 이는 개별 연산자를 분할하고 연산자 작업의 하위 집합을 병렬로 실행하는 것과
관계되는 *intra-op parallelism* 와는 구별됩니다.

기본 구문
------------

dynamic 병렬 처리를 위한 두가지 중요한 API는 다음과 같습니다:

* ``torch.jit.fork(fn : Callable[..., T], *args, **kwargs) -> torch.jit.Future[T]``
* ``torch.jit.wait(fut : torch.jit.Future[T]) -> T``

예제를 통해 이러한 작동 방식을 보여주는 좋은 방법:

.. code-block:: python

    import torch

    def foo(x):
        return torch.neg(x)

    @torch.jit.script
    def example(x):
        # 병렬 처리를 사용하여 `foo`를 호출:
        # 먼저, 작업을 "fork" 합니다. 이 작업은 `x` 인자(argument)와 함께 `foo` 를 실행합니다.
        future = torch.jit.fork(foo, x)

        # 일반적으로 `foo` 호출
        x_normal = foo(x)

        # 둘째, 작업이 실행 중일 수 있으므로 우리는 작업을 "기다립니다".
        # 병렬로, 결과를 사용할 수 있을 때까지 "대기" 해야합니다.
        # "fork()" 와 "wait()" 사이에 코드 라인이 있음에 유의하십시오.
        # 주어진 Future를 호출하면, 계산을 오버랩(overlap)해서 병렬로 실행할 수 있습니다.
        x_parallel = torch.jit.wait(future)

        return x_normal, x_parallel

    print(example(torch.ones(1))) # (-1., -1.)


``fork()`` 는 호출 가능한 ``fn`` 과 해당 호출 가능한  ``args`` 및  ``kwargs`` 에 대한 인자를 취하고  ``fn`` 실행을 위한 비동기(asynchronous) 작업을 생성합니다.
``fn`` 은 함수, 메소드, 또는 모듈 인스턴스일 수 있습니다. ``fork()`` 는  ``Future`` 라고 불리는 이 실행 결과의 값에 대한 참조(reference)를 반환합니다.
``fork`` 는 비동기 작업을 생성한 직후에 반환되기 때문에,  ``fork()`` 호출 후 코드 라인이 실행될 때까지 ``fn`` 이 실행되지 않을 수 있습니다.
따라서, ``wait()`` 은 비동기 작업이 완료 될때까지 대기하고 값을 반환하는데 사용됩니다.

이러한 구조는 함수 내에서 명령문 실행을 오버랩하거나 (작업된 예제 섹션에 표시됨) 루프와 같은 다른 언어 구조로 구성 될 수 있습니다:

.. code-block:: python

    import torch
    from typing import List

    def foo(x):
        return torch.neg(x)

    @torch.jit.script
    def example(x):
        futures : List[torch.jit.Future[torch.Tensor]] = []
        for _ in range(100):
            futures.append(torch.jit.fork(foo, x))

        results = []
        for future in futures:
            results.append(torch.jit.wait(future))

        return torch.sum(torch.stack(results))

    print(example(torch.ones([])))

.. note::

    Future의 빈 리스트(list)를 초기화할때, 우리는 명시적 유형 주석을  ``futures`` 에 추가해야 했습니다.
    TorchScript에서 빈 컨테이너(container)는 기본적으로 tensor 값을 포함한다고 가정하므로
    리스트 생성자(constructor) #에  ``List[torch.jit.Future[torch.Tensor]]`` 유형의 주석을 달았습니다.

이 예제는  ``fork()`` 를 사용하여 함수  ``foo`` 의 인스턴스 100개를 시작하고, 100개의 작업이 완료 될때까지
대기한 다음, 결과를 합산하여  ``-100.0`` 을 반환합니다.

적용된 예시: 양방향(bidirectional) LSTMs의 앙상블(Ensemble)
------------------------------------------------

보다 현실적인 예시에 병렬화를 적용하고 어떤 종류의 성능을 얻을 수 있는지 살펴봅시다.
먼저, 양방향 LSTM 계층의 앙상블인 기준 모델을 정의합시다.

.. code-block:: python

    import torch, time

    # RNN 용어에서 우리가 관심 갖는 차원:
    # 시간 단계의 # (T)
    # Batch 크기 (B)
    # "channels"의 숨겨진 크기/숫자 (C)
    T, B, C = 50, 50, 1024

    # 단일 "양방향 LSTM"을 정의하는 모듈
    # 이는 단순히 동일한 시퀀스에 적용된 두 개의 LSTMs이지만 하나는 반대로 적용됩니다.
    class BidirectionalRecurrentLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cell_f = torch.nn.LSTM(input_size=C, hidden_size=C)
            self.cell_b = torch.nn.LSTM(input_size=C, hidden_size=C)

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Forward 계층
            output_f, _ = self.cell_f(x)

            # Backward 계층. 시간 차원(time dimension)(dim 0)에서 입력 flip (dim 0),
            # 계층 적용, 그리고 시간 차원에서 출력 flip
            x_rev = torch.flip(x, dims=[0])
            output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
            output_b_rev = torch.flip(output_b, dims=[0])

            return torch.cat((output_f, output_b_rev), dim=2)


    # `BidirectionalRecurrentLSTM` 모듈의 "ensemble"
    # 앙상블의 모듈은 같은 입력에서 하나하나씩 실행되고, 결과들이 누적되고 합산되어 결합된 결과를 반환합니다.
    class LSTMEnsemble(torch.nn.Module):
        def __init__(self, n_models):
            super().__init__()
            self.n_models = n_models
            self.models = torch.nn.ModuleList([
                BidirectionalRecurrentLSTM() for _ in range(self.n_models)])

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            results = []
            for model in self.models:
                results.append(model(x))
            return torch.stack(results).sum(dim=0)

    # fork/wait으로 실행할 것들의 직접 비교를 위해
    # 모듈을 인스턴스화하고 TorchScript를 통해 컴파일합시다.
    ens = torch.jit.script(LSTMEnsemble(n_models=4))

    # 일반적으로 임베딩 테이블(embedding table)에서 입력을 가져오지만,
    # 목적을 위해 이 데모에서는 랜덤 데이터를 사용하겠습니다.
    x = torch.rand(T, B, C)

    # 메모리 할당자(memory allocator)
    ens(x)

    x = torch.rand(T, B, C)

    # 얼마나 빠르게 실행되는지 봅시다!
    s = time.time()
    ens(x)
    print('Inference took', time.time() - s, ' seconds')

컴퓨터에서 네트워크가 ``2.05``초에 실행되었습니다. 우리는 더 잘 할 수 있습니다!

Forward, Backward 계층(Layer) 병렬화
-----------------------------------------

``BidirectionalRecurrentLSTM`` 내에서 forward, backward 계층들을 병렬화하는 것은 우리가 할 수 있는 아주 간단한 일입니다.
이를 위해 계산 구조는 고정되어 우리는 어떤 루프도 필요로 하지 않습니다.
``BidirectionalRecurrentLSTM``의 ``forward`` 메소드를 다음과 같이 재작성해봅시다:

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # Forward layer - fork() 이므로 이는 backward와 병렬로 실행될 수 있음.
            future_f = torch.jit.fork(self.cell_f, x)

            # Backward 계층. 시간 차원(time dimension)(dim 0)에서 입력 flip (dim 0),
            # 계층 적용, 그리고 시간 차원에서 출력 flip
            x_rev = torch.flip(x, dims=[0])
            output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
            output_b_rev = torch.flip(output_b, dims=[0])

            # forward 계층에서 출력을 검색.
            # 이는 우리가 병렬화하려는 작업 *이후*에 일어나야함을 주의.
            output_f, _ = torch.jit.wait(future_f)

            return torch.cat((output_f, output_b_rev), dim=2)

이 예시에서, ``forward()``는 ``cell_b``의 실행을 계속하는 동안 ``cell_f``를 다른 스레드로 위임합니다.
이는 두 셀(cell)들의 실행이 서로 오버랩됩니다.

이 간단한 수정과 함께 스크립트를 다시 실행하면 ``1.71``초의 런타임으로 ``17%``만큼 향상되었습니다!

Aside: 병렬화 시각화 (Visualizing Parallelism)
------------------------------

우리는 아직 모델을 최적화시키는 것을 끝내지 않았지만 성능 시각화를 위한 도구를 설명해 볼 만 합니다.
한 가지 중요한 도구는  `PyTorch 프로파일러(profiler) <https://pytorch.org/docs/stable/autograd.html#profiler>`_입니다.

Chrome 추적 내보내기 기능(trace export functionality)과 함께 프로파일러를 사용해
우리의 병렬화된 모델의 성능을 시각화해봅시다:

.. code-block:: python
    with torch.autograd.profiler.profile() as prof:
        ens(x)
    prof.export_chrome_trace('parallel.json')

이 작은 코드 조각은 ``parallel.json`` 파일을 작성합니다.
만약 당신이 Google Chrome에서 ``chrome://tracing``으로 이동하여 ``Load`` 버튼을 클릭하고
JSON 파일을 로드하면 다음과 같은 타임라인을 보게 될 겁니다:

.. image:: https://i.imgur.com/rm5hdG9.png

타임라인의 가로축은 시간을, 세로축은 실행 스레드를 나타냅니다.
보다시피 한 번에 두 개의 ``lstm``를 실행하고 있습니다.
이것은 양방향(forward, backward) 계층을 병렬화하기 위해 노력한 결과입니다.

앙상블에서의 병렬화 모델
------------------------------------

당신은 이 코드에 더 많은 병렬화 기회가 있다는 것을 눈치챘을지도 모릅니다:
우리는 ``LSTMEnsemble``에 포함된 모델들을 서로 병렬로 실행할 수도 있습니다.
이렇게 하기 위한 방법은 아주 간단합니다. 바로 ``LSTMEnsemble``의 ``forward`` 메소드를 변경하는 방법입니다.

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            # 각 모델을 위한 작업 실행
            futures : List[torch.jit.Future[torch.Tensor]] = []
            for model in self.models:
                futures.append(torch.jit.fork(model, x))

            # 실행된 작업들에서 결과 수집
            results : List[torch.Tensor] = []
            for future in futures:
                results.append(torch.jit.wait(future))

            return torch.stack(results).sum(dim=0)

또는, 만약 당신이 간결함을 중요하게 생각한다면 목록 이해력(list comprehension)를 사용할 수 있습니다.

.. code-block:: python

        def forward(self, x : torch.Tensor) -> torch.Tensor:
            futures = [torch.jit.fork(model, x) for model in self.models]
            results = [torch.jit.wait(fut) for fut in futures]
            return torch.stack(results).sum(dim=0)

인트로에서 설명했듯이, 우리는 루프를 사용해 앙상블의 각 모델들에 대한 작업을 나눴습니다.
그리고 모든 작업이 완료될 때까지 기다릴 다른 루프를 사용했습니다.
이는 더 많은 계산의 오버랩을 제공합니다.

이 작은 업데이트로 스크립트는 ``1.4``초에 실행되어 총 ``32%``만큼 속도가 향상되었습니다!
단 두 줄의 코드인 것에 비해 좋은 효과입니다.

또한 Chrome 추적기(tracer)를 다시 사용해 진행 상황을 볼 수 있습니다:

.. image:: https://i.imgur.com/kA0gyQm.png

이제 모든 ``LSTM`` 인스턴스가 완전히 병렬로 실행되는 것을 볼 수 있습니다.

결론
----------

이 튜토리얼에서 우리는 TorchScript에서 dynamic, inter-op parallelism를 수행하기 위한 기본 API인
``fork()``와 ``wait()``에 대해 배웠습니다. 이러한 함수들을 사용해 TorchScript 코드에서
함수, 메소드, 또는 ``Modules``의 실행을 병렬화하는 몇 가지 일반적인 사용 패턴도 보았습니다.
마지막으로, 이 기술을 사용해 모델을 최적화하는 예를 훑어보고, PyTorch에서 사용 가능한
성능 측정 및 시각화 도구를 살펴보았습니다.
