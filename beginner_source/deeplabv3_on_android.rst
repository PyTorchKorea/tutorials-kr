안드로이드에서의 이미지 분할 DeepLapV3
=================================================

**저자**: `Jeff Tang <https://github.com/jeffxtang>`_

**감수**: `Jeremiah Chung <https://github.com/jeremiahschung>`_
**번역**: `김현길 <https://github.com/des00>`_

소개
----

의미론적 이미지 분할(Semantic image segmentation)은 의미론적 라벨을 사용하여 입력 이미지의 특정 영역을 표시하는 컴퓨터 비전 작업입니다.
PyTorch의 의미론적 이미지 분할에 사용하는 `DeepLabV3 모델 <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_ 은 `20가지 의미론적 클래스 <http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html>`_ 가 있습니다. 예를 들어 자전거, 버스, 차, 개, 사람과 같은 것들의 이미지 영역에 라벨을 달 수 있습니다.
의미론적 이미지 분할은 자율주행이나 장면 이해(scene understanding)같은 적용 분야에 매우 유용합니다.

이 튜토리얼에서는 안드로이드에서 PyTorch DeepLabV3 모델을 준비하고 실행하는 단계별 가이드를 제공합니다. 사용하고자 하는 모델을 준비하는 시작 단계에서부터 안드로이드 앱에서 모델을 사용하는 마지막 단계까지 모두 살펴봅니다.
또한 안드로이드에서 여러분이 선호하는 사전에 학습된(pre-trained) PyTorch 모델을 사용하는 방법과 여러 함정들을 피하는 실용적이며 보편적인 팁도 다룰 예정입니다.

.. note:: 이 튜토리얼을 진행하기 앞서 `안드로이드를 위한 PyTorch 모바일 <https://pytorch.org/mobile/android/>`_ 을 확인하고, PyTorch 안드로이드 예제인 `HelloWorld <https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp>`_ 앱을 실행해 보십시오. 이 튜토리얼은 대게 처음으로 모바일에 배포하는 모델인 이미지 분류 모델을 넘어선 다음 단계를 다루고 있습니다. 이 튜토리얼을 위한 전체 코드는 `여기 <https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation>`_ 에서 확인 가능합니다.

학습 목표
---------

이 튜토리얼에서 배울 것들:

1. DeepLabV3 모델을 안드로이드 배포용으로 변환하기

2. 파이썬에서 예제 이미지를 입력하여 모델의 결과값을 얻고 안드로이드 앱에서의 결과값과 비교하기

3. 새로운 안드로이드 앱을 만들거나 안드로이드 예제 앱에 변환된 모델을 가져와서 재사용하기

4. 모델이 원하는 형식에 맞는 입력값 준비하고 모델에서 결과값 처리하기

5. UI 완성, 리팩토링, 앱 빌드 및 실행해서 이미지 분류 동작 확인하기

요구사항
----------

* PyTorch 1.6 이나 1.7

* torchvision 0.7 이나 0.8

* NDK가 설치된 Android Studio 3.5.1 혹은 그 이후 버전

단계
----

1. DeepLabV3 모델을 안드로이드 배포용으로 변환하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

안드로이드에 모델을 배포하는 첫 단계는 모델을 `TorchScript <https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html>`_ 형식으로 변환하는 것입니다.

.. note::
    현 시점에선 PyTorch 모델 중 TorchScript로 변환되지 않는 모델도 있습니다. 모델 정의에서 파이썬의 부분 집합인 TorchScript가 지원하지 않는 언어의 기능을 사용하고 있을 수 있기 때문입니다. 세부사항은 `Script and Optimize Recipe <../recipes/script_optimized.html>`_ 를 참고하세요.

스크립트된 모델 `deeplabv3_scripted.pt` 생성을 위해 아래 스크립트를 실행합니다:

::

    import torch

    # 모델 사이즈를 줄이기 위해 resnet101 대신 deeplabv3_resnet50 사용
    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scriptedm = torch.jit.script(model)
    torch.jit.save(scriptedm, "deeplabv3_scripted.pt")

생성한 `deeplabv3_scripted.pt` 모델 파일의 크기는 168MB 정도가 되어야 됩니다. 이상적으로는 모델을 안드로이드 앱에 배포하기 전에 크기 감소와 더 빠른 추론을 위해 양자화(Quantization)가 되어야 합니다. 양자화가 무엇인지 알고 싶다면 `Quantization Recipe <../recipes/quantization.html>`_ 와 이 안의 참고 링크들을 확인해 주십시오. DeepLabV3에서 어떻게 올바르게 양자화 작업 흐름, 속칭 학습 후(Post Training) `Static Quantization <https://tutorials.pytorch.kr/advanced/static_quantization_tutorial.html>`_ 을 적용할 것인지 관련하여 세부사항은 앞으로의 튜토리얼이나 레시피에서 다룰 예정입니다.

2. 파이썬에서 모델의 예제 입출력 얻기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이제 스크립트된 PyTorch 모델을 얻었으니, 안드로이드에서 모델이 잘 동작하는지 예제를 입력해 테스트를 진행합시다. 첫번째로 모델을 이용해서 추론하고 입출력을 검토하는 파이썬 스크립트를 작성해 봅시다. DeepLabV3의 예시를 들기 위해 첫번째 단계의 코드 `DeepLabV3 model hub site <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_ 를 재사용합니다. 위의 코드에 아래의 코드 조각을 덧붙입니다:

::

    from PIL import Image
    from torchvision import transforms
    input_image = Image.open("deeplab.jpg")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    print(input_batch.shape)
    print(output.shape)

`여기 <https://github.com/jeffxtang/android-demo-app/blob/new_demo_apps/ImageSegmentation/app/src/main/assets/deeplab.jpg>`_ 에서  `deeplab.jpg` 을 다운받고 위의 스크립트를 실행하면 모델 입출력의 shape를 확인할 수 있습니다:

::

    torch.Size([1, 3, 400, 400])
    torch.Size([21, 400, 400])

그래서 400x400 크기와 동일한 입력 이미지  `deeplab.jpg` 를 안드로이드의 모델에 입력하면, 모델 출력은 [21, 400, 400]의 크기를 가져야 합니다. 또한, 최소한 실제 입출력 데이터의 시작 부분만이라도 출력해서 확인을 해 봅시다. 아래의 4단계에서는 안드로이드에서 앱을 실행하는데, 이 때 모델의 실제 입출력 값과 비교하기 위함입니다.

3. 새로운 안드로이드 앱을 만들거나 안드로이드 예제 앱에 변환된 모델을 가져와서 재사용하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

첫번째로 모델을 안드로이드 스튜디오 프로젝트에서 PyTorch Mobile과 함께 쓰기 위해 `안드로이드 레시피를 위한 모델 준비 <../recipes/model_preparation_android.html#add-the-model-and-pytorch-library-on-android>`_ 를 따라해 봅니다. 이 튜토리얼의 DeepLabV3과 PyTorch HelloWorld Android 예제 내부의 MobileNet v2 둘 다 컴퓨터 비전 모델이기에, `HelloWorld 예제 저장소 <https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp>`_ 에서도 손쉽게 모델을 읽고 입출력을 처리하기 위한 코드 수정 방법을 찾을 수 있습니다. 이 단계와 4단계의 목표는 1단계에서 만들어진 `deeplabv3_scripted.pt` 모델이 안드로이드에서 확실하게 동작하는지 확인하는 것입니다.

이제 2단계에서 사용한 `deeplabv3_scripted.pt` 와 `deeplab.jpg` 를 안드로이드 스튜디오 프로젝트에 더하고 `MainActivity` 내부의 `onCreate` 메소드를 이와 유사하게 수정합니다:

.. code-block:: java

    Module module = null;
    try {
      module = Module.load(assetFilePath(this, "deeplabv3_scripted.pt"));
    } catch (IOException e) {
      Log.e("ImageSegmentation", "Error loading model!", e);
      finish();
    }

그 후 `finish()` 라인에 브레이크포인트를 설정하고 빌드 및 앱 실행을 합니다. 브레이크포인트에서 앱이 멈추지 않는다면 안드로이드에서 1단계의 스크립트된 모델을 성공적으로 읽어 들였다는 의미입니다.

4. 모델 추론을 위한 입출력 처리하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이전 단계에서 모델을 읽어들인 이후 입력값이 잘 동작하는지, 예상한대로 출력값을 생성하는지 확인해 봅시다. DeepLabV3 모델을 위한 입력은 HelloWorld 예제 내부의 MobileNet v2에서 쓰는 이미지와 동일합니다. 그래서 `MainActivity.java <https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/src/main/java/org/pytorch/helloworld/MainActivity.java>`_ HelloWorld 프로젝트의 입력 처리를 위한 코드를 재사용 합니다. `MainActivity.java` 파일의 `50번째 줄 <https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/src/main/java/org/pytorch/helloworld/MainActivity.java#L50>`_ 과 73번째 줄 사이의 코드를 아래와 같이 변경합니다:

.. code-block:: java

    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);
    final float[] inputs = inputTensor.getDataAsFloatArray();

    Map<String, IValue> outTensors =
        module.forward(IValue.from(inputTensor)).toDictStringKey();

    // 결과로 출력된 텐서의 키 "out"은 의미론적 마스크(semantic masks)를 포함
    // 링크 참고 https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101
    final Tensor outputTensor = outTensors.get("out").toTensor();
    final float[] outputs = outputTensor.getDataAsFloatArray();

    int width = bitmap.getWidth();
    int height = bitmap.getHeight();

.. note::
    모델 출력은 DeepLabV3 모델의 딕셔너리이기에 `toDictStringKey` 를 사용해서 결과를 적절히 추출합니다. 다른 모델의 출력은 단일 텐서 혹은 튜블 텐서중 하나일 수 있습니다.

위에서 보여준 코드 변경사항에서 `final float[] inputs` 와 `final float[] outputs` 뒤에 브레이크포인트를 설정할 수도 있습니다. 이러면 입출력 텐서가 float 배열에 할당되는 것을 확인하여 디버깅을 더 쉽게 할 수 있습니다.
앱 실행 후 브레이크포인트에서 정지할 때에 `inputs` 과 `outputs` 의 숫자가 2단계에서의 모델의 입출력과 매치되는지 비교하세요. 안드로이드와 파이썬에서 동작하는 모델에 동일한 입력값을 넣었으면 출력값도 동일해야 됩니다.

.. warning::
    안드로이드 에뮬레이터에서는 같은 이미지 입력값에 다른 모델 출력값을 얻는 경우도 있습니다. 이는 안드로이드 에뮬레이터의 실수 구현 이슈로 인한 것입니다. 그래서 실제 안드로이드 기기에서 테스트를 하는 것이 가장 좋습니다.

지금까지 했던 모든 것들은 파이썬에서처럼 안드로이드 앱에서도 우리의 흥미를 끄는 모델이 스크립팅되고 정상적으로 동작하는지 확인하는 것입니다.

일반적인 머신러닝 프로젝트에서 데이터 처리가 가장 힘든 부분인 것처럼, 안드로이드에서 모델을 사용하여 여기까지 밟아온 단계들이 앱 개발 기간 중 대부분은 아니지만 상당히 많은 시간을 차지합니다.

5. UI 완성, 리팩토링, 앱 빌드 및 실행
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이제 새 이미지를 처리한 결과를 확인하기 위해 앱과 UI를 완성할 준비가 되었습니다. 결과 처리 코드는 아래와 같아야 되며, 4단계의 코드 끝부분에 추가되어야 합니다:

.. code-block:: java

    int[] intValues = new int[width * height];
    // 크기가 [WIDTH, HEIGHT]인 결과값의 각 원소들을 순회하며
    // 각기 다른 classnum별로 각기 다른 색을 설정
    for (int j = 0; j < width; j++) {
        for (int k = 0; k < height; k++) {
            // maxi: 21 CLASSNUM 중에서 가장 높은 확률을 가리키는 인덱스
            int maxi = 0, maxj = 0, maxk = 0;
            double maxnum = -100000.0;
            for (int i=0; i < CLASSNUM; i++) {
                if (outputs[i*(width*height) + j*width + k] > maxnum) {
                    maxnum = outputs[i*(width*height) + j*width + k];
                    maxi = i; maxj = j; maxk= k;
                }
            }
            // 사람 (빨강), 개 (초록), 양 (파랑)을 위한 색깔 코드
            // 검은색은 배경이나 다른 클래스들을 위한 색
            if (maxi == PERSON)
                intValues[maxj*width + maxk] = 0xFFFF0000; // 빨강
            else if (maxi == DOG)
                intValues[maxj*width + maxk] = 0xFF00FF00; // 초록
            else if (maxi == SHEEP)
                intValues[maxj*width + maxk] = 0xFF0000FF; // 파랑
            else
                intValues[maxj*width + maxk] = 0xFF000000; // 검은색
        }
    }

위의 코드에서 사용한 상수는 `MainActivity` 클래스의 시작 부분에서 선언했습니다:

.. code-block:: java

    private static final int CLASSNUM = 21;
    private static final int DOG = 12;
    private static final int PERSON = 15;
    private static final int SHEEP = 17;


여기에서 구현한 것은 width*height인 입력 이미지로 [21, width, height] 크기의 텐서를 출력하는 DeepLabV3 모델에 대한 이해를 바탕으로 구현한 것입니다. width*height인 결과 행렬의 각 원소들은 0에서 20 사이의 값(소개에서 설명한 총 21개의 의미론적 라벨을 표현)을 가지며, 각각의 값은 특정한 색을 가집니다. 여기에서 설명하는 분할에서는 가장 높은 확률을 가지는 클래스의 색깔 코드(color coding)을 사용하고, 데이터셋의 모든 클래스들에 각각의 색깔 코드 설정하도록 확장도 할 수 있습니다.

결과 처리 이후, `ImageView` 에 결과를 표시하기 전에 RGB `intValues` 행렬을 비트맵 인스턴스 `outputBitmap` 으로 렌더링하고자 아래의 코드를 실행할 필요가 있을 수도 있습니다.

.. code-block:: java

    Bitmap bmpSegmentation = Bitmap.createScaledBitmap(bitmap, width, height, true);
    Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
    outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0,
        outputBitmap.getWidth(), outputBitmap.getHeight());
    imageView.setImageBitmap(outputBitmap);

이 앱의 UI는 HelloWorld의 UI와 유사하지만 이미지 분류의 결과를 보여주기 위해 `TextView` 를 필요로 하지 않습니다. 코드 저장소에서 볼 수 있는 것처럼 `Segment` and `Restart` 버튼 두 개를 추가할 수도 있습니다. 이 버튼들은 모델 추론을 실행하고 분할 결과를 보다가 원본 이미지로 되돌리기 위해 사용합니다.

이제 앱을 안드로이드 에뮬레이터나 (가능하다면) 실제 기기에서 실행하면 이런 화면들을 볼 수 있습니다:

.. image:: /_static/img/deeplabv3_android.png
   :width: 300 px
.. image:: /_static/img/deeplabv3_android2.png
   :width: 300 px


정리
--------

이 튜토리얼에서는 사전에 학습된 PyTorch DeepLabV3 모델을 안드로이드에서 사용하기 위한 변환과, 그 모델이 어떻게 안드로이드에서 성공적으로 실행되는지 보았습니다. 여기에서는 모델이 안드로이드에서도 정말 실행이 되는지 각 과정을 확인해 보면서 전체 과정을 이해하는 것에 초점을 두었습니다. 전체 코드는 `여기 <https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation>`_ 에서 확인 가능합니다.

안드로이드에서 양자화나 전이 학습(transfer learning)같은 고급 주제는 앞으로의 데모 앱이나 튜토리얼에서 다룰 예정입니다.


더 알아보기
------------

1. `PyTorch 모바일 사이트 <https://pytorch.org/mobile>`_
2. `DeepLabV3 모델 <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_
3. `DeepLabV3 논문 <https://arxiv.org/pdf/1706.05587.pdf>`_
