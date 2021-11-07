iOS에서의 이미지 분할 DeepLapV3
==============================================

**저자**: `Jeff Tang <https://github.com/jeffxtang>`_

**감수**: `Jeremiah Chung <https://github.com/jeremiahschung>`_
**번역**: `김현길 <https://github.com/des00>`_

소개
------------

의미론적 이미지 분할(Semantic image segmentation)은 의미론적 라벨을 사용하여 입력 이미지의 특정 영역을 표시하는 컴퓨터 비전 작업입니다.
PyTorch의 의미론적 이미지 분할에 사용하는 `DeepLabV3 모델 <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_ 은 `20가지 의미론적 클래스 <http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html>`_ 가 있습니다. 예를 들어 자전거, 버스, 차, 개, 사람과 같은 것들의 이미지 영역에 라벨을 달 수 있습니다.
의미론적 이미지 분할은 자율주행이나 장면 이해(scene understanding)같은 적용 분야에 매우 유용합니다.

이 튜토리얼에서는 iOS에서 PyTorch DeepLabV3 모델을 준비하고 실행하는 단계별 가이드를 제공합니다. 사용하고자 하는 모델을 준비하는 시작 단계에서부터 iOS 앱에서 모델을 사용하는 마지막 단계까지 모두 살펴봅니다.
또한 iOS에서 여러분이 선호하는 사전에 학습된(pre-trained) PyTorch 모델을 사용하는 방법과 여러 함정들을 피하는 실용적이며 보편적인 팁도 다룰 예정입니다.

.. note:: 이 튜토리얼을 진행하기 앞서 `iOS를 위한 PyTorch 모바일 <https://pytorch.org/mobile/ios/>`_ 을 확인하고, PyTorch iOS 예제인 `HelloWorld <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld>`_ 앱을 실행해 보십시오. 이 튜토리얼은 대게 처음으로 모바일에 배포하는 모델인 이미지 분류 모델을 넘어선 다음 단계를 다루고 있습니다. 이 튜토리얼을 위한 전체 코드는 `여기 <https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation>`_ 에서 확인 가능합니다.

학습 목표
-------------------

이 튜토리얼에서 배울 것들:

1. DeepLabV3 모델을 iOS 배포용으로 변환하기

2. 파이썬에서 예제 이미지를 입력하여 모델의 결과값을 얻고 iOS 앱에서의 결과값과 비교하기

3. 새로운 iOS 앱을 만들거나 iOS 예제 앱에 변환된 모델을 가져와서 재사용하기

4. 모델이 원하는 형식에 맞는 입력값 준비하고 모델에서 결과값 처리하기

5. UI 완성, 리팩토링, 앱 빌드 및 실행해서 이미지 분할 동작 확인하기

요구사항
---------------

* PyTorch 1.6 이나 1.7

* torchvision 0.7 이나 0.8

* Xcode 11 이나 12

단계
---------


1. DeepLabV3 모델을 iOS 배포용으로 변환하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

iOS에 모델을 배포하는 첫 단계는 모델을 `TorchScript <https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html>`_ 형식으로 변환하는 것입니다.

.. note::
    현 시점에선 PyTorch 모델 중 TorchScript로 변환되지 않는 모델도 있습니다. 모델 정의에서 파이썬의 부분 집합인 TorchScript가 지원하지 않는 언어의 기능을 사용하고 있을 수 있기 때문입니다. 세부사항은 `Script and Optimize Recipe <../recipes/script_optimized.html>`_ 를 참고하세요.

스크립트된 모델 `deeplabv3_scripted.pt` 생성을 위해 아래 스크립트를 실행합니다:

::

    import torch

    # 모델 사이즈를 줄이기 위해 resnet101 대신 deeplabv3_resnet50 사용
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scriptedm = torch.jit.script(model)
    torch.jit.save(scriptedm, "deeplabv3_scripted.pt")

생성한 `deeplabv3_scripted.pt` 모델 파일의 크기는 168MB 정도가 되어야 됩니다. 이상적으로는 모델을 iOS 앱에 배포하기 전에 크기 감소와 더 빠른 추론을 위해 양자화(Quantization)가 되어야 합니다. 양자화가 무엇인지 알고 싶다면 `Quantization Recipe <../recipes/quantization.html>`_ 와 이 안의 참고 링크들을 확인해 주십시오. DeepLabV3에서 어떻게 올바르게 양자화 작업 흐름, 속칭 학습 후(Post Training) `Static Quantization <https://tutorials.pytorch.kr/advanced/static_quantization_tutorial.html>`_ 을 적용할 것인지 관련하여 세부사항은 앞으로의 튜토리얼이나 레시피에서 다룰 예정입니다.

2. 파이썬에서 모델의 예제 입출력 얻기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이제 스크립트된 PyTorch 모델을 얻었으니, iOS에서 모델이 잘 동작하는지 예제를 입력해 테스트를 진행합시다. 첫번째로 모델을 이용해서 추론하고 입출력을 검토하는 파이썬 스크립트를 작성해 봅시다. DeepLabV3의 예시를 들기 위해 첫번째 단계의 코드와 `DeepLabV3 model hub site <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_ 를 재사용합니다. 위의 코드에 아래의 코드 조각을 덧붙입니다:

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

`여기 <https://github.com/pytorch/ios-demo-app/blob/master/ImageSegmentation/ImageSegmentation/deeplab.jpg>`_ 에서  `deeplab.jpg` 을 다운받고 위의 스크립트를 실행하면 모델 입출력의 shape를 확인할 수 있습니다:

::

    torch.Size([1, 3, 400, 400])
    torch.Size([21, 400, 400])

그래서 400x400 크기와 동일한 입력 이미지  `deeplab.jpg` 를 iOS의 모델에 입력하면, 모델 출력은 [21, 400, 400]의 크기를 가져야 합니다. 또한, 최소한 실제 입출력 데이터의 시작 부분만이라도 출력해서 확인을 해 봅시다. 아래의 단계 4에서는 iOS에서 앱을 실행하는데, 이 때 모델의 실제 입출력 값과 비교하기 위함입니다.

3. 새로운 iOS 앱을 만들거나 예제 앱에 변환된 모델을 가져와서 재사용하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

첫번째로 모델을 Xcode 프로젝트에서 PyTorch Mobile과 함께 쓰기 위해 `iOS 레시피를 위한 모델 준비 <../recipes/model_preparation_ios.html#add-the-model-and-pytorch-library-on-ios>`_ 의 단계 3을 따라합니다.
이 튜토리얼의 DeepLabV3 모델과 PyTorch HelloWorld iOS 예제 내부의 MobileNet v2 모델 둘 다 컴퓨터 비전 모델이기에, `HelloWorld 예제 저장소 <https://github.com/pytorch/ios-demo-app/tree/master/HelloWorld>`_ 를 모델을 읽어 들이고 입출력을 처리하는 본보기로 삼아 시작할 수도 있습니다.

이제 단계 2에서 사용한 `deeplabv3_scripted.pt` 와 `deeplab.jpg` 를 Xcode 프로젝트에 추가하고 `ViewController.swift` 를 이와 유사하게 수정합니다:

.. code-block:: swift

    class ViewController: UIViewController {
        var image = UIImage(named: "deeplab.jpg")!

        override func viewDidLoad() {
            super.viewDidLoad()
        }

        private lazy var module: TorchModule = {
            if let filePath = Bundle.main.path(forResource: "deeplabv3_scripted",
                  ofType: "pt"),
                let module = TorchModule(fileAtPath: filePath) {
                return module
            } else {
                fatalError("Can't load the model file!")
            }
        }()
    }

그 후 `return module` 라인에 브레이크포인트를 설정하고 빌드 및 앱 실행을 합니다. 앱이 브레이크포인트에서 반드시 멈춘다면 iOS에서 단계 1의 스크립트된 모델을 성공적으로 읽어 들였다는 의미입니다.

4. 모델 추론을 위한 입출력 처리하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이전 단계에서 모델을 읽어들인 이후 입력값이 잘 동작하는지, 예상한대로 출력값을 생성하는지 확인해 봅시다. DeepLabV3 모델을 위한 입력은 HelloWorld 예제 내부의 MobileNet v2에서 쓰는 이미지와 동일합니다. 그래서 `TorchModule.mm <https://github.com/pytorch/ios-demo-app/blob/master/HelloWorld/HelloWorld/HelloWorld/TorchBridge/TorchModule.mm>`_ HelloWorld 프로젝트의 입력 처리를 위한 코드를 재사용 합니다. `TorchModule.mm` 안의 `predictImage` 메소드 구현을 아래와 같이 변경합니다:

.. code-block:: objective-c

    - (unsigned char*)predictImage:(void*)imageBuffer {
        // 1. 예제 deeplab.jpg의 크기는 400x400 이며 21개의 의미론적 클래스가 있습니다
        const int WIDTH = 400;
        const int HEIGHT = 400;
        const int CLASSNUM = 21;

        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, WIDTH, HEIGHT}, at::kFloat);
        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        // 2. 디버깅을 위해 입력 텐서를 NSMutableArray로 변환합니다
        float* floatInput = tensor.data_ptr<float>();
        if (!floatInput) {
            return nil;
        }
        NSMutableArray* inputs = [[NSMutableArray alloc] init];
        for (int i = 0; i < 3 * WIDTH * HEIGHT; i++) {
            [inputs addObject:@(floatInput[i])];
        }

        // 3. 모델 출력은 문자열과 텐서의 딕셔너리이며, 자세한 설명은
        // https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101 에 있습니다
        auto outputDict = _impl.forward({tensor}).toGenericDict();

        // 4. 쉬운 디버깅을 위해 출력을 다른 NSMutableArray로 변환합니다
        auto outputTensor = outputDict.at("out").toTensor();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
          return nil;
        }
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < CLASSNUM * WIDTH * HEIGHT; i++) {
          [results addObject:@(floatBuffer[i])];
        }

        return nil;
    }

.. note::
    모델의 출력은 DeepLabV3 모델을 위한 딕셔너리여서 `toGenericDict` 를 사용해서 적절하게 결과를 추출할 수 있습니다. 다른 모델은 모델 출력이 단일 텐서나 텐서 튜플 같은 것들이 될 수도 있습니다.

위의 코드 변경에서도 보았듯이, `inputs` 과 `results` 를 만드는 두 개의 for 반복문 뒤에 브레이크포인트를 설정하여 단계 2에서의 모델의 입출력과 맞아 떨어지는지 비교할 수도 있습니다. iOS와 파이썬에서 동작하는 모델에 동일한 입력값을 넣었으면 출력값도 동일해야 됩니다.

지금까지 했던 모든 것들은 파이썬에서처럼 iOS 앱에서도 우리의 흥미를 끄는 모델이 스크립팅되고 정상적으로 동작하는지 확인하는 것입니다. 일반적인 머신러닝 프로젝트에서 데이터 처리가 가장 힘든 부분인 것처럼, iOS 앱에서 모델을 사용하여 여기까지 밟아온 단계들이 앱 개발 기간 중 대부분은 아니지만 상당히 많은 시간을 차지합니다.

5. UI 완성, 리팩토링, 앱 빌드 및 실행
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이제 새 이미지를 처리한 결과를 확인하기 위해 앱과 UI를 완성할 준비가 되었습니다. 결과 처리 코드는 아래와 같아야 되며, 단계 4에서의 `TorchModule.mm` 코드 끝부분에 추가되어야 합니다 - 먼저 `return nil;` 라인을 지우는걸 명심하세요. 코드를 빌드하고 실행하기 위해 임시로 넣은 것입니다:

.. code-block:: objective-c

    // 튜토리얼 소개에서의 20가지 의미론적 클래스 링크를 보세요
    const int DOG = 12;
    const int PERSON = 15;
    const int SHEEP = 17;

    NSMutableData* data = [NSMutableData dataWithLength:
        sizeof(unsigned char) * 3 * WIDTH * HEIGHT];
    unsigned char* buffer = (unsigned char*)[data mutableBytes];
    // go through each element in the output of size [WIDTH, HEIGHT] and
    // set different color for different classnum
    for (int j = 0; j < WIDTH; j++) {
        for (int k = 0; k < HEIGHT; k++) {
            // maxi: the index of the 21 CLASSNUM with the max probability
            int maxi = 0, maxj = 0, maxk = 0;
            float maxnum = -100000.0;
            for (int i = 0; i < CLASSNUM; i++) {
                if ([results[i * (WIDTH * HEIGHT) + j * WIDTH + k] floatValue] > maxnum) {
                    maxnum = [results[i * (WIDTH * HEIGHT) + j * WIDTH + k] floatValue];
                    maxi = i; maxj = j; maxk = k;
                }
            }
            int n = 3 * (maxj * width + maxk);
            // 사람 (빨강), 개 (초록), 양 (파랑)을 위한 색깔 코드
            // 검은색은 배경이나 다른 클래스들을 위한 색
            buffer[n] = 0; buffer[n+1] = 0; buffer[n+2] = 0;
            if (maxi == PERSON) buffer[n] = 255;
            else if (maxi == DOG) buffer[n+1] = 255;
            else if (maxi == SHEEP) buffer[n+2] = 255;
        }
    }
    return buffer;

여기에서 구현한 것은 width*height인 입력 이미지로 [21, width, height] 크기의 텐서를 출력하는 DeepLabV3 모델에 대한 이해를 바탕으로 구현한 것입니다. width*height인 결과 행렬의 각 원소들은 0에서 20 사이의 값(소개에서 설명한 총 21개의 의미론적 라벨을 표현)을 가지며, 각각의 값은 특정한 색을 가집니다. 여기에서 설명하는 분할에서는 가장 높은 확률을 가지는 클래스의 색깔 코드(color coding)을 사용하고, 데이터셋의 모든 클래스들에 각각의 색깔 코드를 설정하도록 확장도 할 수 있습니다.

결과 처리 이후, `UIImageView` 에 표시하기 위해 RGB `buffer` 를  `UIImage` 인스턴스로 변환하는 헬퍼 함수를 호출해야 할 수도 있습니다. 코드 저장소 내부의 `UIImageHelper.mm` 에 정의된 예제 코드인 `convertRGBBufferToUIImage` 를 참조할 수도 있습니다.

이 앱의 UI는 HelloWorld의 UI와 유사하지만 이미지 분류의 결과를 보여주기 위해 `UITextView` 를 필요로 하지 않습니다. 코드 저장소에서 볼 수 있는 것처럼 `Segment` and `Restart` 버튼 두 개를 추가할 수도 있습니다. 이 버튼들은 모델 추론을 실행하고 분할 결과를 보다가 원본 이미지로 되돌리기 위해 사용합니다.

앱을 실행하기 전 마지막 단계는 모든 조각들을 하나로 합치는 것입니다. `predictImage` 를 사용하기 위해 `ViewController.swift` 를 변경하십시오. `predictImage` 는 저장소에서 리팩토링되어 `segmentImage` 로 변경됩니다. 그리고 저장소에 있는 `ViewController.swift` 의 헬퍼 함수를 예제 코드에서 본 것과 같이 수정하세요. 버튼에 액션을 연결하면 바로 실행할 수 있습니다.

이제 앱을 iOS 에뮬레이터나 실제 iOS 기기에서 실행하면 이런 화면들을 볼 수 있습니다:

.. image:: /_static/img/deeplabv3_ios.png
   :width: 300 px
.. image:: /_static/img/deeplabv3_ios2.png
   :width: 300 px


정리
--------

이 튜토리얼에서는 사전에 학습된 PyTorch DeepLabV3 모델을 iOS에서 사용하기 위한 변환과, 그 모델이 어떻게 iOS에서 성공적으로 실행되는지 보았습니다. 여기에서는 모델이 iOS에서도 정말 실행이 되는지 각 과정을 확인해 보면서 전체 과정을 이해하는 것에 초점을 두었습니다. 전체 코드는 `여기 <https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation`_ 에서 확인 가능합니다.

iOS에서 양자화나 전이 학습(transfer learning)같은 고급 주제는 앞으로의 데모 앱이나 튜토리얼에서 다룰 예정입니다.

더 알아보기
------------

1. `PyTorch 모바일 사이트 <https://pytorch.org/mobile>`_
2. `DeepLabV3 모델 <https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101>`_
3. `DeepLabV3 논문 <https://arxiv.org/pdf/1706.05587.pdf>`_
