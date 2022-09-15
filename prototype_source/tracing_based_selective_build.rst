(prototype) 추적-기반 선택적 빌드 모바일 Android, iOS 인터프리터
==============================================================


*저자*: Chen Lai <https://github.com/cccclai>, Dhruv Matani <https://github.com/dhruvbird>

*역자*: Lee Jong Bub <https://github.com/bub3690>


.. 경고::
추적-기반 선택적 빌드는 라이브러리 사이즈를 줄이기 위한 프로토타입 기능입니다. 추적된 결과는 모델 입력과 추적된 환경에 의존합니다. 따라서, 만약 추적기가 모바일 인터프리터가 아닌 다른 환경에서 실행되면, 연산자 리스트가 실제 사용된 연산자 리스트와 다를 수 있고 빠진 연산자들이 오류를 발생시킬 수 있습니다.

Introduction
------------

이 튜토리얼은 모바일 인터프리터 사이즈를 더욱 더 최소화하기 위해, 모바일 인터프리터 빌드를 커스터마이즈하는 새로운 방법을 소개합니다. 컴파일된 이진 파일에 포함되는 연산자들을 목표 모델에 실제로 필요로 하는 집합만큼으로 제한합니다. 파이토치의 모바일 배포의 이진 파일 사이즈를 줄이는 기술입니다. 추적 기반 선택적 빌드는 모델을 특정 대표 입력값들과 함께 동작시킵니다, 그리고 어떤 연산자들이 호출되었는지 기록합니다. 그러면 빌드는 그 연산자들만 포함하게 됩니다. 



다음은 추적-기반 선택적 방법으로 커스텀 모바일 인터프리터를 제작하는 과정들입니다.

1. *묶인 입력과 모델을 준비합니다*

.. code:: python

    import numpy as np
    import torch
    import torch.jit
    import torch.utils
    import torch.utils.bundled_inputs
    from PIL import Image
    from torchvision import transforms

    # 단계 1. 모델을 가져옵니다.
    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    scripted_module = torch.jit.script(model)
    # 비교를 위해, jit 전체 버전 모델을 출력합니다.(가벼운 인터프리터와는 호환되지 않습니다.) 
    scripted_module.save("deeplabv3_scripted.pt")
    # 가벼운 인터프리터 버전 모델을 출력합니다.(가벼운 인터프리터와 호환됩니다.)
    # path = "<모델이 저장된 기본 위치>"

    scripted_module._save_for_lite_interpreter(f"${path}/deeplabv3_scripted.ptl")

    model_file = f"${path}/deeplabv3_scripted.ptl"

    # 단계 2. 모델에 사용될 이미지를 준비합니다.
    input_image_1 = Image.open(f"${path}/dog.jpg")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor_1 = preprocess(input_image_1)
    input_batch_1 = input_tensor_1.unsqueeze(0) # 모델에 사용될 미니 배치를 제작합니다.

    scripted_module = torch.jit.load(model_file)
    scripted_module.forward(input_batch_1) # 선택적으로, 모델을 검증하기 위해 input_batch_1과 함께 작동시킵니다.

    input_image_2 = Image.open(f"${path}/deeplab.jpg")
    input_tensor_2 = preprocess(input_image_2)
    input_batch_2 = input_tensor_2.unsqueeze(0) # 모델에 사용될 미니 배치를 제작합니다.

    scripted_module = torch.jit.load(model_file)
    scripted_module.forward(input_batch_2) # 선택적으로, 모델을 검증하기 위해 input_batch_2과 함께 작동시킵니다.

    # 단계 3. 모델과 단계2에서 준비된 입력을 같이 묶어줍니다. 가능한 많은 입력을 묶어줍니다.
    bundled_model_input = [
        (torch.utils.bundled_inputs.bundle_large_tensor(input_batch_1), ),
        (torch.utils.bundled_inputs.bundle_large_tensor(input_batch_2), )]
    bundled_model = torch.utils.bundled_inputs.bundle_inputs(scripted_module, bundled_model_input)
    bundled_model._save_for_lite_interpreter(f"${path}/deeplabv3_scripted_with_bundled_input.ptl")

2. 추적기를 제작합니다.

.. code:: shell

 MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ MAX_JOBS=16 TRACING_BASED=1 python setup.py develop

3. 추적기와 모델을 입력과 함께 동작시킵니다.

.. code:: shell

 ./build/bin/model_tracer --model_input_path ${path}/deeplabv3_scripted_with_bundled_input.ptl --build_yaml_path ${path}/deeplabv3_scripted.yaml



Android
-------

이미지 영역 분할 Android 데모 App을 가져옵니다 :  https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation

1. **Android를 위한 libtorch lite 추적-기반 빌드**: 모든 4 가지 Android abis(``armeabi-v7a``, ``arm64-v8a``, ``x86``, ``x86_64``)를 대상으로 libtorch를 빌드합니다.

.. code-block:: bash

   SELECTED_OP_LIST=${path}/deeplabv3_scripted.yaml TRACING_BASED=1  ./scripts/build_pytorch_android.sh

만약 ``x86`` 의 Pixel 4 emulator에서 테스트 된다면, cmd 에서 ``BUILD_LITE_INTERPRETER=1 ./scripts/build_pytorch_android.sh x86`` 사용하여, 빌드 시간을 절약하기 위해 abi를 명시해줍니다.

.. code-block:: bash

   SELECTED_OP_LIST=${path}/deeplabv3_scripted.yaml TRACING_BASED=1  ./scripts/build_pytorch_android.sh x86


빌드가 끝난 후, 라이브러리 경로를 보여줄 것입니다 : 

.. code-block:: bash

   BUILD SUCCESSFUL in 55s
   134 actionable tasks: 22 executed, 112 up-to-date
   + find /Users/chenlai/pytorch/android -type f -name '*aar'
   + xargs ls -lah
   -rw-r--r--  1 chenlai  staff    13M Feb 11 11:48 /Users/chenlai/pytorch/android/pytorch_android/build/outputs/aar/pytorch_android-release.aar
   -rw-r--r--  1 chenlai  staff    36K Feb  9 16:45 /Users/chenlai/pytorch/android/pytorch_android_torchvision/build/outputs/aar/pytorch_android_torchvision-release.aar

2. **이미지 영역 분할 App 소스에서 빌드된 Pytorch Android 라이브러리를 사용합니다**: 경로에 `libs` 폴더를 만들고, 경로는 root 저장소로부터 `ImageSegmentation/app/libs` 가 됩니다. `pytorch_android-release` 를 경로 ``ImageSegmentation/app/libs/pytorch_android-release.aar`` 에 복사합니다. `pytorch_android_torchvision` (다운로드 : `Pytorch Android Torchvision Nightly <https://oss.sonatype.org/#nexus-search;quick~torchvision_android/>`_)를 경로 ``ImageSegmentation/app/libs/pytorch_android_torchvision.aar`` 에 복사합니다. ``ImageSegmentation/app/build.gradle`` 에서 `dependencies` 부분을 다음 코드와 같이 수정합니다 :

.. code:: gradle

   dependencies {
       implementation 'androidx.appcompat:appcompat:1.2.0'
       implementation 'androidx.constraintlayout:constraintlayout:2.0.2'
       testImplementation 'junit:junit:4.12'
       androidTestImplementation 'androidx.test.ext:junit:1.1.2'
       androidTestImplementation 'androidx.test.espresso:espresso-core:3.3.0'


       implementation(name:'pytorch_android-release', ext:'aar')
       implementation(name:'pytorch_android_torchvision', ext:'aar')

       implementation 'com.android.support:appcompat-v7:28.0.0'
       implementation 'com.facebook.fbjni:fbjni-java-only:0.0.3'
   }

``ImageSegmentation/build.gradle`` 에서 `all projects` 파트를 다음 코드와 같이 수정합니다.


.. code:: gradle

    allprojects {
        repositories {
            google()
            jcenter()
            flatDir {
                dirs 'libs'
            }
        }
    }

3. **App 테스트하기**: Android 스튜디오에서 `ImageSegmentation` App을 빌드하고 실행합니다.


iOS
---

이미지 영역 분할 iOS 데모 App을 가져옵니다: https://github.com/pytorch/ios-demo-app/tree/master/ImageSegmentation


1. **libtorch lite iOS 빌드합니다**:

.. code-block:: bash

   SELECTED_OP_LIST=${path}/deeplabv3_scripted.yaml TRACING_BASED=1 IOS_PLATFORM=SIMULATOR ./scripts/build_ios.sh


2. **프로젝트에서 Cocoapods 제거합니다** (이 과정은 `pod install` 을 사용했을 때만 필요합니다):


.. code-block:: bash

   pod deintegrate


3.  **이미지 영역 분할 데모 App을 커스텀 라이브러리들과 링크해줍니다**:

Xcode에서 프로젝트를 열고, 목표 프로젝트의 **Build Phases - Link Binaries With Libraries** 로 가서, **+** 기호를 클릭하고 `build_ios/install/lib` 에 위치한 모든 라이브러리 파일들을 추가합니다. 프로젝트 **Build Settings** 로 이동하여, **Header Search Paths** 에서 값을 `build_ios/install/include` 로 값을 설정하고 **Library Search Paths** 를 `build_ios/install/lib` 로 값을 설정합니다.
build settings에서, **other linker flags** 를 검색합니다. `-all_load` 아래에 커스텀 링커 플래그를 추가합니다.
마지막으로, 목표를 위해 Build Settings에서 bitcode를 사용하지 못하게 선택해야 합니다, Enable Bitcode를 검색하여, **No** 값으로 설정합니다.

4. **Xcode에서 App을 빌드하고 테스트합니다**



Conclusion
----------

이 튜토리얼에서는, Android와 iOS App에서 효율적인 Pyotorch 모바일 인터프리터를 커스텀 빌드하는 새로운 방법인 추적-기반 선택적 빌드를 시연했습니다.  

이미지 영역 분할 예제를 수행하며 모델에 들어갈 입력을 어떻게 묶는지 보여주었고, 묶인 입력과 모델을 추적함으로써 연산자 리스트를 생성했고, 추적된 결과의 연산자 리스트와 소스로 커스텀 torch 라이브러리를 빌드했습니다.

커스텀 빌드는 여전히 개발중이고, 앞으로 미래에도 계속 사이즈를 개선시킬 것입니다. 그러나, API들은 미래 version에 따라 종속된다는 것을 주의하세요.

읽어주셔서 감사합니다! 언제나, 어떤 피드백이든 환영합니다. `<https://github.com/pytorch/pytorch/issues>`_ 에 이슈를 생성해주세요.

더 배우기

- Pytorch 모바일에 대해 더 배우기 위해서는, Pytorch 모바일 홈페이지 `<https://pytorch.org/mobile/home/>`_ 를 참조해주세요.

* 이미지 영역 분할에 대해 더 배우려면, Andorid Recipe의 Image Segmentation DeepLabV3를 참조해주세요 `<https://tutorials.pytorch.kr/beginner/deeplabv3_on_android.html>`_
