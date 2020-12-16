Pytorch 모바일 성능 레시피
==================================

도입
----------------
전부는 아니더라도, 대부분의 모바일 기기에서 ML모델 추론의 애플리케이션과 유스케이스에 있어 성능(performance aka latency)은 매우 중요합니다.

오늘날 PyTorch는 CPU, DSP, NPU와 같이 다른 하드웨어 백엔드의 가용성을 기다리는 CPU 백엔드에서 모델을 실행합니다. 

이 레시피에서는 아래와 같은 것들을 학습합니다.

- 모바일 기기에서 실행시간을 줄이기 위해 (더 높은 성능, 낮은 지연시간) 모델을 최적화 하는 방법
- 벤치마킹(benchmark)하는 방법(최적화가 유스케이스에 대해 도움이 되었는지 확인)


모델 준비
-----------------

모바일 기기에서 실행 시간을 줄이기 위해 (성능 향상, 낮은 지연시간) 모델을 최적화할 준비부터 시작합니다.

설정
^^^^^^^
먼저, 버전이 1.5.0.이상인 conda 또는 pip를 사용하여 pytorch를 설치해야 합니다.

::

   conda install pytorch torchvision -c pytorch

또는

::

   pip install torch torchvision

모델 코드 :

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
          x.contiguous(memory_format=torch.channels_last)
          x = self.quant(x)
          x = self.conv(x)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dequant(x)
          return x

  model = AnnotatedConvBnReLUModel()


``torch.quantization.QuantStub`` 와 ``torch.quantization.DeQuantStub()``\은 양자화(quantization) 단계에서 사용되는 작동하지 않는 스텁입니다.

1. ``torch.quantization.fuse_modules``\를 사용하여 결합(fuse) 연산하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fuse_modules가 양자화 패키지에 있다는 것을 헷갈리지 마세요.
이것은 모두 ``torcn.nn.Module``\에서 작동합니다.

``torch.quantization.fuse_modules``\은 모듈 목록을 하나의 모듈로 통합합니다.
다음 모듈들을 결합합니다.

- Convolution, Batch normalization
- Convolution, Batch normalization, Relu
- Convolution, Relu
- Linear, Relu

이 스크립트는 이전에 선언된 모델에서 Convolution, Batch Normalization, Relu을 결합합니다.

::

  torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)


2. 모델 양자화 하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch 양자화에 대해 더 자세한 내용은 `the dedicated tutorial <https://pytorch.org/blog/introduction-to-quantization-on-pytorch/>`_ 에서 확인할 수 있습니다.

모델 양자화는 계산을 int8으로 옮길 뿐 아니라 디스크에서의 모델 크기를 줄입니다. 이러한 크기 감소는 모델을 처음 가져오는 동안 디스크를 읽는 작업을 줄이고 RAM의 총량을 줄이는데 도움이 됩니다. 두 리소스 모두 모바일 애플리케이션의 성능에 있어 중요할 수 있습니다. 이 코드는 모델 보정 함수를 위한 스텁을 사용하여 양자화를 수행하고 이것에 대해 더 자세한 내용은 `여기 <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#post-training-static-quantization>`__ 에서 확인할 수 있습니다.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Torch mobile_optimizer 패키지는 스크립트된 모델로 최적화를 수행하고, 이것은 conv2d와 선형 연산에 도움이 됩니다. 최적화된 형식으로 모델 가중치를 미리 포장하고(pre-packs), 다음 작업의 경우 위의 작업을 relu와 결합합니다.

먼저 이전 단계의 결과 모델을 스크립트 합니다.

::

  torchscript_model = torch.jit.script(model)

다음은 ``optimize_for_mobile``\을 호출하고 디스크에 모델을 저장합니다.

::

  torchscript_model_optimized = optimize_for_mobile(torchscript_model)
  torch.jit.save(torchscript_model_optimized, "model.pt")

4. Channels Last Tensor 메모리 형식 사용하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Channels Last(NHWC) 메모리 형식은 PyTorch 1.4.0에서 도입되었습니다. 이것은 4차원 텐서에만 지원됩니다. 이 메모리 형식은 대부분의 연산자, 특히 컨볼루션(convolution)에 더 나은 메모리 지역성을 제공합니다. 측정 결과 MobileNetV2 모델의 속도가 기본 Channels First(NCHW) 형식에 비해 3배 빨라졌습니다.

이 레시피를 작성하는 현재, PyTorch Android java API는 Channels Last 메모리 형식 입력 사용을 지원하지 않습니다. 하지만 모델 입력을 위해 변환(conversion)을 더하면 TorchScript 모델 레벨에서는 사용이 가능합니다.

.. code-block:: python

  def forward(self, x):
      x.contiguous(memory_format=torch.channels_last)
      ...

이 변환은 입력이 이미 Channels Last 메모리 형식인 경우 비용이 들지 않습니다. 그 후 모든 연산자는 ChannelsLast 메모리 형식을 유지하면서 작업합니다. 

5. Android - 전송을 위한 텐서 재사용
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이 레시피 부분은 Android에만 해당합니다.

특히 오래된 기기에서, 안드로이드 성능에 있어 메모리는 중요한 자원입니다. 텐서는 상당한 양의 메모리가 필요할 수 있습니다. 예를 들어 표준 컴퓨터 비전 텐서는 데이터 유형이 float 이고 588Kb의 메모리가 필요하다고 가정했을 때 1*3*224*224개 요소를 포함합니다.

::

  FloatBuffer buffer = Tensor.allocateFloatBuffer(1*3*224*224);
  Tensor tensor = Tensor.fromBlob(buffer, new long[]{1, 3, 224, 224});

여기서 우리는 네이티브 메모리를 ``java.nio.FloatBuffer``\로 할당하고 스토리지가 할당된 버퍼의 메모리를 가리킬 ``org.pytorch.Tensor``\를 생성합니다.

대부분의 유스케이스에서, 모델을 한번만 전달하는것이 아니라 일정한 빈도로 또는 가능한 빠르게 반복합니다.

우리가 모든 모듈 전송에 대해 새로운 메모리 할당을 수행한다면, 이것은 차선책이 됩니다. 그 대신에 이전 단계에서 할당했던 같은 메모리를 재사용하고 새로운 데이터를 채운 다음, 동일한 텐서 객체에 다시 모듈 전송을 수행합니다.

`pytorch android application example <https://github.com/pytorch/android-demo-app/blob/master/PyTorchDemoApp/app/src/main/java/org/pytorch/demo/vision/ImageClassificationActivity.java#L174>`_ 에서 코드를 확인할 수 있습니다. 

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

멤버 필드 ``mModule``, ``mInputTensorBuffer``, ``mInputTensor``\는 한번만 초기화 되고 버퍼는 ``org.pytorch.torchvision.TensorImageUtils.imageYUV420CenterCropToFloatBuffer``\를 사용하여 다시 채워집니다. 

벤치마킹
------------

수행 동작이 환경에 따라 달라지기 때문에, 벤치마킹(최적화가 유스케이스에 도움이 되었는지 확인)하는 가장 좋은 방법은 최적화하려는 특정 유스케이스를 측정하는 것입니다. 

PyTorch 배포는 모델전송을 수행하는 네이키드(naked) 바이너리를 벤치마크하는 방법을 제공합니다. 이 접근 방식은 애플리케이션 내부에서 테스트하는 것보다 더 안정적인 측정을 제공할 수 있습니다.

Android - 벤치마킹 설정
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이 레시피 부분은 Android에만 해당합니다.

이를 수행하기 위해 먼저 벤치마크 바이너리를 빌드해야 합니다. 

::

    <from-your-root-pytorch-dir>
    rm -rf build_android
    BUILD_PYTORCH_MOBILE=1 ANDROID_ABI=arm64-v8a ./scripts/build_android.sh -DBUILD_BINARY=ON

``build_android/bin/speed_benchmark_torch``\에 arm64 바이너리가 있어야 합니다. 이 바이너리는 ``--model=<path-to-model>``\을 취하고 입력에 대한 차원 정보로 ``--input_dim="1,3,224,224"``\를, 인자로의 입력의 타입을 ``--input_type="float"``\취합니다.

일단 안드로이드 기기가 연결되면 speedbenchark_torch 바이너리와 모델을 전화기에 푸시합니다.

::

  adb push <speedbenchmark-torch> /data/local/tmp
  adb push <path-to-scripted-model> /data/local/tmp


이제 모델을 벤치마킹할 준비가 되었습니다.

::

  adb shell "/data/local/tmp/speed_benchmark_torch --model="/data/local/tmp/model.pt" --input_dims="1,3,224,224" --input_type="float"
  ----- output -----
  Starting benchmark.
  Running warmup runs.
  Main runs.
  Main run finished. Microseconds per iter: 121318. Iters per second: 8.24281


iOS - 벤치마킹 설정
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

iOS의 경우, 벤치마킹의 도구로 `TestApp <https://github.com/pytorch/pytorch/tree/master/ios/TestApp>`_ 를 사용합니다.

먼저 `TestApp/benchmark/trace_mode.py <https://github.com/pytorch/pytorch/blob/master/ios/TestApp/benchmark/trace_model.py>`_ 에 있는 파이썬 스크립트 ``optimize_for_mobile``\를 적용합니다. 간단히 아래와 같이 코드를 수정합니다.

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

이제 ``python trace_model.py``\를 실행합니다. 전부 잘 작동한다면 벤치마크 디렉토리의 최적화된 모델을 생성할 수 있어야 합니다. 

다음으로 소스의 PyTorch 라이브러리를 빌드합니다.

::

  BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh

최적화된 모델과 PyTorch가 준비되었으므로, 이제 XCode 프로젝트를 생성하고 벤치마킹할 차례입니다. 이를 위해 XCode 프로젝트를 설정하는 무거운 작업을 수행하는 루비 스크립트 `setup.rb` 를 사용합니다.

::

  ruby setup.rb

이제 `TestApp.xcodeproj` 를 열고 iPhone에 연결하면 바로 사용이 가능합니다. 아래는 iPhoneX에서의 예제 결과입니다. 

::

  TestApp[2121:722447] Main runs
  TestApp[2121:722447] Main run finished. Milliseconds per iter: 28.767
  TestApp[2121:722447] Iters per second: : 34.762
  TestApp[2121:722447] Done.
