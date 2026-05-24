==============================================
Intel® Advanced Matrix Extensions 활용하기
==============================================
**번역**: `김지현 <https://github.com/hijyun>`_

소개
============

Advanced Matrix Extensions(AMX)는 Intel® Advanced Matrix Extensions(Intel® AMX)라고도 부르는 x86 확장 기능입니다.
이 확장 기능은 두 가지 새로운 구성 요소를 도입합니다. 하나는 ‘tiles’라고 불리는 2차원 레지스터 파일이고, 다른 하나는 이러한 tiles에서 동작할 수 있는 Tile Matrix Multiplication(TMUL) 가속기입니다.
AMX는 행렬에서 동작하도록 설계되어 CPU에서 딥러닝 학습과 추론을 가속하며, 자연어 처리, 추천 시스템, 이미지 인식과 같은 워크로드에 이상적입니다.


Intel은 4세대 Intel® Xeon® Scalable 프로세서와 Intel® AMX를 통해 AI 기능을 발전시켜, 이전 세대 대비 3배에서 10배 더 높은 추론 및 학습 성능을 제공합니다. `Accelerate AI Workloads with Intel® AMX`_ 를 참고하세요.
Intel® Advanced Vector Extensions 512 Neural Network Instructions(Intel® AVX-512 VNNI)를 실행하는 3세대 Intel Xeon Scalable 프로세서와 비교했을 때,
Intel AMX를 실행하는 4세대 Intel Xeon Scalable 프로세서는 한 사이클당 256개의 INT8 연산이 아니라 2,048개의 INT8 연산을 수행할 수 있습니다. 또한 한 사이클당 64개의 FP32 연산과 비교해, 한 사이클당 1,024개의 BF16 연산도 수행할 수 있습니다. `Accelerate AI Workloads with Intel® AMX`_ 의 4페이지를 참고하세요. AMX에 대한 더 자세한 정보는 `Intel® AMX Overview`_ 를 참고하세요.


PyTorch에서의 AMX
==================

PyTorch는 백엔드인 oneDNN을 통해 BFloat16 기반의 연산 집약적 연산자와 INT8 기반의 양자화에 AMX를 활용하여,
AMX를 지원하는 x86 CPU에서 별도의 설정 없이 더 높은 성능을 얻을 수 있도록 합니다.
oneDNN에 대한 더 자세한 정보는 `oneDNN`_ 을 참고하세요.

이 연산은 생성된 실행 코드 경로에 따라 oneDNN이 전적으로 처리합니다. 예를 들어, AMX를 지원하는 하드웨어 플랫폼에서 oneDNN 구현으로 지원하는 연산을 실행하면, oneDNN 내부에서 AMX 명령어를 자동으로 호출합니다.
oneDNN은 PyTorch CPU의 기본 가속 라이브러리이므로, AMX 지원을 활성화하기 위해 별도의 수동 작업은 필요하지 않습니다.

AMX를 워크로드에 활용하기 위한 가이드라인
---------------------------------------------------

이 절에서는 다양한 워크로드에서 AMX를 활용하는 방법에 대한 가이드라인을 제공합니다.

- BFloat16 데이터 타입:

  - ``torch.cpu.amp`` 또는 ``torch.autocast("cpu")`` 를 사용하면 지원되는 연산자에 대해 AMX 가속을 활용할 수 있습니다.

   ::

      model = model.to(memory_format=torch.channels_last)
      with torch.cpu.amp.autocast():
         output = model(input)

.. note:: 더 나은 성능을 얻으려면 ``torch.channels_last`` 메모리 형식(memory format)을 사용하세요.

- 양자화(Quantization):

  - 양자화를 적용하면 지원되는 연산자에 대해 AMX 가속을 활용할 수 있습니다.

- torch.compile:

  - 생성된 그래프 모델이 지원되는 연산자를 사용하여 oneDNN 구현으로 실행될 때, AMX 가속이 활성화됩니다.

.. note:: AMX를 지원하는 CPU에서 PyTorch를 사용할 경우, 프레임워크는 기본적으로 AMX 사용을 자동으로 활성화합니다. 즉, PyTorch는 행렬 곱셈 연산의 속도를 높이기 위해 가능한 경우 AMX 기능을 활용하려고 시도합니다. 그러나 AMX 커널로 디스패치할지 여부는 최종적으로 PyTorch가 성능 향상을 위해 의존하는 oneDNN 라이브러리와 양자화 백엔드의 내부 최적화 전략에 따라 결정된다는 점에 유의해야 합니다. PyTorch와 oneDNN 라이브러리 내부에서 AMX 활용이 처리되는 구체적인 방식은 프레임워크의 업데이트와 개선에 따라 변경될 수 있습니다.


AMX를 활용할 수 있는 CPU 연산자:
------------------------------------

AMX를 활용할 수 있는 BF16 CPU 연산자:

- ``conv1d``
- ``conv2d``
- ``conv3d``
- ``conv_transpose1d``
- ``conv_transpose2d``
- ``conv_transpose3d``
- ``bmm``
- ``mm``
- ``baddbmm``
- ``addmm``
- ``addbmm``
- ``linear``
- ``matmul``

AMX를 활용할 수 있는 양자화 CPU 연산자:

- ``conv1d``
- ``conv2d``
- ``conv3d``
- ``conv_transpose1d``
- ``conv_transpose2d``
- ``conv_transpose3d``
- ``linear``



AMX가 활용되고 있는지 확인하기
------------------------------------

환경 변수 ``export ONEDNN_VERBOSE=1`` 을 설정하거나, ``torch.backends.mkldnn.verbose`` 를 사용하여 oneDNN이 상세 메시지를 출력하도록 활성화하세요.

::

   with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
       with torch.cpu.amp.autocast():
           model(input)

예를 들어, 다음과 같이 oneDNN의 상세 출력을 확인할 수 있습니다:

::

   onednn_verbose,info,oneDNN v2.7.3 (commit 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
   onednn_verbose,info,cpu,runtime:OpenMP,nthr:128
   onednn_verbose,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with bfloat16 and 8-bit integer support
   onednn_verbose,info,gpu,runtime:none
   onednn_verbose,info,prim_template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
   onednn_verbose,exec,cpu,reorder,simple:any,undef,src_f32::blocked:a:f0 dst_f32::blocked:a:f0,attr-scratchpad:user ,,2,5.2561
   ...
   onednn_verbose,exec,cpu,convolution,jit:avx512_core_amx_bf16,forward_training,src_bf16::blocked:acdb:f0 wei_bf16:p:blocked:ABcd16b16a2b:f0 bia_f32::blocked:a:f0 dst_bf16::blocked:acdb:f0,attr-scratchpad:user ,alg:convolution_direct,mb7_ic2oc1_ih224oh111kh3sh2dh1ph1_iw224ow111kw3sw2dw1pw1,0.628906
   ...
   onednn_verbose,exec,cpu,matmul,brg:avx512_core_amx_int8,undef,src_s8::blocked:ab:f0 wei_s8:p:blocked:BA16a64b4a:f0 dst_s8::blocked:ab:f0,attr-scratchpad:user ,,1x30522:30522x768:1x768,7.66382
   ...

BFloat16의 경우 ``avx512_core_amx_bf16`` 가 포함된 상세 출력이 나타나거나, INT8 양자화의 경우 ``avx512_core_amx_int8`` 가 포함된 상세 출력이 나타나면 AMX가 활성화되었음을 의미합니다.


결론
----------


이 튜토리얼에서는 AMX와, PyTorch에서 AMX를 활용하여 워크로드를 가속하는 방법, 그리고 AMX가 활용되고 있는지 확인하는 방법을 간략히 소개했습니다.

PyTorch와 oneDNN의 개선 및 갱신에 따라, AMX의 활용 방식도 그에 맞게 변경될 수 있습니다.

문제가 발생하거나 궁금한 점이 있다면 언제든
`포럼 <https://discuss.pytorch.org/>`_ 이나 `GitHub 이슈
<https://github.com/pytorch/pytorch/issues>`_ 를 통해 문의해 주세요.


.. _Accelerate AI Workloads with Intel® AMX: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/ai-solution-brief.html

.. _Intel® AMX Overview: https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html

.. _oneDNN: https://oneapi-src.github.io/oneDNN/index.html
