``torch.compile``의 컴파일 시점 캐싱
=========================================================
**저자** `Oguz Ulgen <https://github.com/oulgen>`_
**번역** `김영준 <https://github.com/YoungyoungJ>`_
Introduction
------------------

PyTorch Compiler는 컴파일 지연 시간을 줄이기 위해 여러 가지 캐싱 기능을 제공합니다.
이 레시피에서는 이러한 캐싱 기능들을 자세히 설명하고, 사용자가 자신의 활용 목적에 가장 적합한 옵션을 선택할 수 있도록 안내합니다.

캐시를 설정하는 방법은 `컴파일 시점 캐싱 설정 <https://tutorials.pytorch.kr/recipes/torch_compile_caching_configuration_tutorial.html>`__ 문서를 참고하세요.

또한 `PT CacheBench 벤치마크 <https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fpytorch&benchmarkName=TorchCache+Benchmark>`__ 에서 캐싱 성능 비교 결과도 확인할 수 있습니다.

사전 준비 사항
-------------------

이 레시피를 시작하기 전에 다음 항목을 준비했는지 확인하세요.

* ``torch.compile`` 에 대한 기본적인 이해가 필요합니다. 아래 자료를 참고하세요.

  * `torch.compiler API documentation <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler>`__
  * `Introduction to torch.compile <https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html>`__
  * `Triton language documentation <https://triton-lang.org/main/index.html>`__

* PyTorch 2.4 이상 버전

캐싱 기능
---------------------

``torch.compile`` 은 다음과 같은 캐싱 기능을 제공합니다.

* End to end caching (``Mega-Cache`` 라고도 불림)
* ``TorchDynamo``, ``TorchInductor``, ``Triton`` 모듈별 캐싱

캐시가 올바르게 동작하기 위해서는 캐시 아티팩트가 동일한 PyTorch 및 Triton 버전에서 생성된 것이어야 하며,  
디바이스가 CUDA로 설정된 경우에는 같은 GPU 환경에서 사용되어야 한다는 점에 유의해야 합니다.

``torch.compile`` end-to-end caching (``Mega-Cache``)
------------------------------------------------------------

``Mega-Cache``”로 지칭되는 엔드 투 엔드 캐싱은, 캐시 데이터를 데이터베이스에 저장해 다른 머신에서도 불러올 수 있는 이식 가능한(portable) 캐싱 솔루션을 찾는 사용자에게 이상적인 방법입니다.

``Mega-Cache`` 는 다음 두 가지 컴파일러 API를 제공합니다.

* ``torch.compiler.save_cache_artifacts()``
* ``torch.compiler.load_cache_artifacts()``

일반적인 사용 방식은 다음과 같습니다. 모델을 컴파일하고 실행한 후, 사용자는 ``torch.compiler.save_cache_artifacts()`` 함수를 호출하여 이식 가능한 형태의 컴파일러 아티팩트를 반환받습니다.  
그 후, 다른 머신에서 이 아티팩트를 ``torch.compiler.load_cache_artifacts()`` 에 전달하여 ``torch.compile`` 캐시를 미리 채워 캐시를 빠르게 초기화할 수 있습니다.

다음 예시를 살펴보세요. 먼저 모델을 컴파일하고 캐시 아티팩트를 저장합니다.

.. code-block:: python

    @torch.compile
    def fn(x, y):
        return x.sin() @ y

    a = torch.rand(100, 100, dtype=dtype, device=device)
    b = torch.rand(100, 100, dtype=dtype, device=device)

    result = fn(a, b)

    artifacts = torch.compiler.save_cache_artifacts()

    assert artifacts is not None
    artifact_bytes, cache_info = artifacts

    # 이제 artifact_bytes를 데이터베이스에 저장할 수도 있습니다.
    # cache_info는 기록할(logging)할 수 있습니다.

Later, you can jump-start the cache by the following:

.. code-block:: python

    # 데이터베이스에서 아티팩트를 다운로드하거나 불러올 수도 있습니다.
    torch.compiler.load_cache_artifacts(artifact_bytes)

이 작업은 다음 섹션에서 다룰 모든 모듈별 캐시(modular caches) 를 미리 채웁니다. 여기에는 ``PGO``, ``AOTAutograd``, ``Inductor``, ``Triton``, 그리고 ``Autotuning`` 이 포함됩니다.


``TorchDynamo``, ``TorchInductor``, 그리고 ``Triton`` 의 모듈별 캐싱
-----------------------------------------------------------

앞서 언급한 ``Mega-Cache`` 는 사용자의 별도 개입 없이 자동으로 동작하는 개별 구성요소들로 이루어져 있습니다. 기본적으로 PyTorch Compiler는 ``TorchDynamo``, ``TorchInductor``, 그리고 ``Triton`` 을 위한  
로컬 디스크 기반(on-disk) 캐시를 함께 제공합니다. 이러한 캐시에는 다음이 포함됩니다.

* ``FXGraphCache``: 파일 과정에서 사용되는 그래프 기반 중간 표현(IR, Intermediate Representation) 구성요소를 저장하는 캐시입니다.
* ``TritonCache``: 컴파일 결과를 저장하는 캐시로, ``Triton`` 에 의해 생성된 ``cubin`` 파일과 기타 캐싱 관련 아티팩트를 포함합니다.
* ``InductorCache``: ``FXGraphCache`` 와 ``Triton`` 캐시를 함께 포함하는 통합 캐시(bundled cache) 입니다.
* ``AOTAutogradCache``: 통합 그래프(joint graph) 관련 아티팩트를 저장하는 캐시입니다.
* ``PGO-cache``: 동적 입력 형태 에 대한 결정 정보를 저장하여 재컴파일 횟수를 줄이는 데 사용되는 캐시입니다.
* `AutotuningCache <https://github.com/pytorch/pytorch/blob/795a6a0affd349adfb4e3df298b604b74f27b44e/torch/_inductor/runtime/autotune_cache.py#L116>`__:
    * ``Inductor`` 는 ``Triton`` 커널을 생성하고, 가장 빠른 커널을 선택하기 위해 누가 더 빠른지, 효율적인지를 비교합니다.
    * ``torch.compile`` 에 내장된 ``AutotuningCache`` 는 이 결과를 캐싱합니다.

이 모든 캐시 아티팩트는 ``TORCHINDUCTOR_CACHE_DIR`` 경로에 저장됩니다. 기본값(default)은 ``/tmp/torchinductor_myusername`` 형태로 설정됩니다.


원격 캐싱(Remote Caching)
----------------

Redis 기반 캐시를 활용하고자 하는 사용자를 위해 원격 캐싱 옵션도 제공합니다. Redis 기반 캐싱을 활성화하는 방법에 대해서는  `컴파일 시점 캐싱 설정 <https://tutorials.pytorch.kr/recipes/torch_compile_caching_configuration_tutorial.html>`__ 문서를 참고하세요.


결론
-------------
이 레시피에서는 PyTorch Inductor의 캐싱 메커니즘이 로컬 캐시와 원격 캐시를 모두 활용하여 컴파일 지연 시간을 크게 줄일 수 있다는 점을 배웠습니다. 이러한 캐시들은 사용자의 별도 개입 없이 백그라운드에서 원활하게 작동합니다.
