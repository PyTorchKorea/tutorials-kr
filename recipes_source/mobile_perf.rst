PyTorch 모바일 성능 레시피
=========================

소개
----
전부는 아니지만, 모바일 기기에서의 애플리케이션과 ML 모델 추론 사용 사례에
성능(지연시간)은 매우 중대한 사항입니다.

오늘날 PyTorch는 GPU, DSP, NPU와 같은 하드웨어 백엔드가 사용 가능할 때까지
CPU 백엔드에서 모델을 실행합니다.

이 레시피에서 배울 내용은:

- 모바일 기기에서 실행 시간을 줄이는데 도움이 될(성능은 높이고, 지연시간은 줄이는) 모델 최적화 방법
- 벤치마킹(최적화가 사용 사례에 도움이 되었는지 확인) 하는 방법


모델 준비
--------

모바일 기기에서 실행 시간을 줄이는데 도움이 될(성능은 높이고, 지연시간은 줄이는)
모델의 최적화를 위한 준비부터 시작합니다.


설정
^^^^

첫번째로 적어도 버전이 1.5.0 이상인 PyTorch를 conda나 pip으로 설치합니다.

::

   conda install pytorch torchvision -c pytorch

또는

::

   pip install torch torchvision

모델 코드:

::

  import torch
  from torch.utils.mobile_optimizer import optimize_for_mobile

  class AnnotatedConvBnReLUModel(torch.nn.Module):
      def __init__(self):
          super(AnnotatedConvBnReLUModel, self).__init__()
          self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
          self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
          self.relu = torch.nn.ReLU(inplace=True)
          self.quant = torch.quantization.QuantStub()
          self.dequant = torch.quantization.DeQuantStub()

      def forward(self, x):
          x = x.contiguous(memory_format=torch.channels_last)
          x = self.quant(x)
          x = self.conv(x)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dequant(x)
          return x

  model = AnnotatedConvBnReLUModel()


``torch.quantization.QuantStub`` 와 ``torch.quantization.DeQuantStub()`` 은 미사용 스텁(stub)이며, 양자화(quantization) 단계에 사용합니다.


1. ``torch.quantization.fuse_modules`` 이용하여 연산자 결합(fuse)하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fuse_modules은 양자화 패키지 내부에 있다는 것을 혼동하지 마십시오.
fuse_modules은 모든 ``torch.nn.Module`` 에서 동작합니다.

``torch.quantization.fuse_modules`` 은 모듈들의 리스트를 하나의 모듈로 결합합니다.
이것은 아래 순서의 모듈들만 결합시킵니다:

- Convolution, Batch normalization
- Convolution, Batch normalization, Relu
- Convolution, Relu
- Linear, Relu

이 스크립트는 이전에 선언된 모델에서 Convolution, Batch Normalization, Relu를 결합합니다.

::

  torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)


2. 모델 양자화하기
^^^^^^^^^^^^^^^^^

PyTorch 양자화에 대한 내용은
`the dedicated tutorial <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/>`_ 에서 찾을 수 있습니다.

모델의 양자화는 연산을 int8로 옮기면서
디스크상의 모델 크기를 줄이기도 합니다.
이런 크기 감소는 모델을 처음 읽어 들일 때 디스크 읽기 연산을 줄이는데 도움을 주고 램(RAM)의 총량도 줄입니다.
이러한 두 자원은 모바일 애플리케이션의 성능에 매우 중요할 수 있습니다.
이 코드는 모델 보정(calibration) 함수를 위해 스텁을 사용해서 양자화를 합니다. `여기 <https://tutorials.pytorch.kr/advanced/static_quantization_tutorial.html#post-training-static-quantization>`__ 에서 관련된 사항을 찾을 수 있습니다.

::

  model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
  torch.quantization.prepare(model, inplace=True)
  # 모델 보정
  def calibrate(model, calibration_data):
      # 모델 보정 코드
      return
  calibrate(model, [])
  torch.quantization.convert(model, inplace=True)



3. torch.utils.mobile_optimizer 사용하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Torch mobile_optimizer 패키지는 스크립트된 모델을 이용해서 몇 가지 최적화를 수행하고,
이러한 최적화는 conv2d와 선형 연산에 도움이 됩니다.
이 패키지는 최적화된 형식으로 모델 가중치를 우선 패키징하며(pre-packs)
다음 연산이 relu이면 위의 연산들과 relu 연산을 결합 시킵니다.

먼저 이전 단계에서부터 결과 모델을 작성합니다:

::

  torchscript_model = torch.jit.script(model)

다음은 ``optimize_for_mobile`` 을 호출하고 디스크에 모델을 저장합니다.

::

  torchscript_model_optimized = optimize_for_mobile(torchscript_model)
  torch.jit.save(torchscript_model_optimized, "model.pt")

4. Channels Last Tensor 메모리 형식 선택하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Channels Last(NHWC) 메모리 형식은 PyTorch 1.4.0에서 도입되었습니다. 이 형식은 오직 4차원 텐서만을 지원합니다. 이 메모리 형식은 대부분의 연산에, 특히 합성곱 연산에 더 나은 메모리 지역성을 제공합니다. 측정 결과는 MobileNetV2 모델에서 기본 Channels First(NCHW) 형식에 비해 3배의 속도 향상을 보여 줍니다.

이 레시피를 작성하는 시점에서는, PyTorch Android 자바 API는 Channels Last 메모리 형식으로 된 입력을 지원하지 않습니다. 하지만 모델 입력을 위해 이 메모리 형식으로 변환하면 TorchScript 모델 수준에서 사용이 가능합니다.

.. code-block:: python

  def forward(self, x):
      x = x.contiguous(memory_format=torch.channels_last)
      ...


이 변환은 입력이 Channels Last 메모리 형식이면 비용이 들지 않습니다. 결국에는 모든 연산자가 Channels Last 메모리 형식을 유지하면서 작업을 합니다.

5. Android - 순방향 전달을 위한 텐서 재사용하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

레시피에서 이 부분은 Android에만 해당합니다.

메모리는 Android 성능에 매우 중요한 자원입니다. 오래된 디바이스에선 특히나 더 중요합니다.
텐서는 상당한 양의 메모리를 필요로 할 수 있습니다.
예를 들어 표준 컴퓨터 비전 텐서는 1*3*224*224개의 요소를 포함합니다.
데이터 타입이 float이고 588kb 메모리가 필요하다고 가정한 경우입니다.

::

  FloatBuffer buffer = Tensor.allocateFloatBuffer(1*3*224*224);
  Tensor tensor = Tensor.fromBlob(buffer, new long[]{1, 3, 224, 224});


여기에선 네이티브 메모리를 ``java.nio.FloatBuffer`` 로 할당하고 저장소가 할당된 버퍼의 메모리를 가리킬 ``org.pytorch.Tensor`` 를 만듭니다.

대부분의 사용 사례에서 모델 순방향 전달을 단 한 번만 하지 않고, 일정한 빈도로 혹은 가능한 한 빨리 진행합니다.

만약 모든 모듈 순방향 전달을 위해 메모리 할당을 새로 한다면 - 그건 최적화가 아닙니다.
대신에, 이전 단계에서 할당한 동일한 메모리에 새 데이터를 채우고 모듈 순방향 전달을 동일한 텐서 객체에서 다시 실행함으로써 동일한 메모리를 재사용 할 수 있습니다.

코드가 어떤 식으로 구성이 되어 있는지는 `pytorch android application example <https://github.com/pytorch/android-demo-app/blob/master/PyTorchDemoApp/app/src/main/java/org/pytorch/demo/vision/ImageClassificationActivity.java#L174>`_ 에서 확인할 수 있습니다.

::

  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mModule == null) {
      mModule = Module.load(moduleFileAbsoluteFilePath);
      mInputTensorBuffer =
      Tensor.allocateFloatBuffer(3 * 224 * 224);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, 224, 224});
    }

    TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
        image.getImage(), rotationDegrees,
        224, 224,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
        TensorImageUtils.TORCHVISION_NORM_STD_RGB,
        mInputTensorBuffer, 0);

    Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
  }

멤버 변수 ``mModule`` , ``mInputTensorBuffer`` , ``mInputTensor`` 는 단 한 번 초기화를 하고
버퍼는 ``org.pytorch.torchvision.TensorImageUtils.imageYUV420CenterCropToFloatBuffer`` 를 이용해서 다시 채워집니다.

벤치마킹
-------

벤치마킹(최적화가 사용 사례에 도움이 되었는지 확인)하는 최고의 방법은 최적화를 하고 싶은 특정한 사용 사례를 측정하는 것입니다. 성능 측정 행위가 환경에 따라 달라질 수 있기 때문입니다.

PyTorch 배포판은 모델 순방향 전달을 실행하는 방식을 사용해서 원형 그대로의(naked) 바이너리를 벤치마킹하는 수단을 제공합니다.
이 접근법은 애플리케이션 내부에서 시험하는 방법보다 더 안정적인 측정치를 제공합니다.


Android - 벤치마킹 설정
^^^^^^^^^^^^^^^^^^^^^^

레시피에서 이 부분은 Android에만 해당합니다.

벤치마킹을 위해 먼저 벤치마크 바이너리를 빌드해야 합니다:

::

    <from-your-root-pytorch-dir>
    rm -rf build_android
    BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=arm64-v8a ./scripts/build_android.sh -DBUILD_BINARY=ON

이 곳에 arm64 바이너리가 있어야 합니다: ``build_android/bin/speed_benchmark_torch`` .
이 바이너리는 ``--model=<path-to-model>``, ``--input_dim="1,3,224,224"`` 을 입력을 위한 차원 정보로 받고  ``--input_type="float"`` 으로 입력 타입을 인자로 받습니다.

Android 디바이스를 연결한 적이 있으면,
speedbenchark_torch 바이너리와 모델을 폰으로 푸시합니다:

::

  adb push <speedbenchmark-torch> /data/local/tmp
  adb push <path-to-scripted-model> /data/local/tmp


이제 모델을 벤치마킹할 준비가 되었습니다:

::

  adb shell "/data/local/tmp/speed_benchmark_torch --model=/data/local/tmp/model.pt" --input_dims="1,3,224,224" --input_type="float"
  ----- output -----
  Starting benchmark.
  Running warmup runs.
  Main runs.
  Main run finished. Microseconds per iter: 121318. Iters per second: 8.24281


iOS - 벤치마킹 설정
^^^^^^^^^^^^^^^^^^

iOS의 경우 , 벤치마킹의 도구로 `TestApp <https://github.com/pytorch/pytorch/tree/master/ios/TestApp>`_ 을 사용합니다.

먼저 ``optimize_for_mobile`` 메소드를  `TestApp/benchmark/trace_model.py <https://github.com/pytorch/pytorch/blob/master/ios/TestApp/benchmark/trace_model.py>`_ 에 있는 파이썬 스크립트에 적용합니다. 간단히 아래와 같이 코드를 수정합니다.

::

  import torch
  import torchvision
  from torch.utils.mobile_optimizer import optimize_for_mobile

  model = torchvision.models.mobilenet_v2(pretrained=True)
  model.eval()
  example = torch.rand(1, 3, 224, 224)
  traced_script_module = torch.jit.trace(model, example)
  torchscript_model_optimized = optimize_for_mobile(traced_script_module)
  torch.jit.save(torchscript_model_optimized, "model.pt")

이제 ``python trace_model.py`` 를 실행합시다. 모든 것이 잘 작동한다면 벤치마킹 디렉토리 내부에 최적화된 모델을 생성할 수 있어야 합니다.

다음은 소스에서부터 PyTorch 라이브러리를 빌드합니다.

::

  BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh

이제 최적화된 모델과 PyTorch가 준비되었기에 XCode 프로젝트를 만들고 벤치마킹할 시간입니다. 이를 위해 XCode 프로젝트를 설정하는 무거운 작업을 수행하는 루비 스크립트 `setup.rb` 를 사용합니다.

::

  ruby setup.rb

이제 `TestApp.xcodeproj` 를 열고 iPhone을 연결하면 준비가 끝났습니다. 아래는 iPhoneX에서의 예제 결과입니다.

::

  TestApp[2121:722447] Main runs
  TestApp[2121:722447] Main run finished. Milliseconds per iter: 28.767
  TestApp[2121:722447] Iters per second: : 34.762
  TestApp[2121:722447] Done.
