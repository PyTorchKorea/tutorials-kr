Real Time Inference on Raspberry Pi 4 (30 fps!)
=================================================
**저자**: `Tristan Rice <https://github.com/d4l3k>`_
**번역**: `유연승 <https://github.com/yuyeonseung>`_

파이토치는 라즈베리파이4에서 즉시 사용할수 있도록 지원합니다. 이 튜토리얼은 라즈베리파이에 파이토치를 설치하는 방법과 MobileNet v2 분류모델을 CPU에서 실시간(30 fps이상)으로 실행하는 법을 설명합니다.

이 튜토리얼은 라즈베리파이4 모델B 4GB에서 실험하였지만 성능이 더 낮은 3B모델이나 2GB모델에서도 작동해야합니다. 

.. image:: https://user-images.githubusercontent.com/909104/153093710-bc736b6f-69d9-4a50-a3e8-9f2b2c9e04fd.gif

전제조건
~~~~~~~~~~~~~~~~

이 튜토리얼을 배우기 위해선 라즈베리파이4와 라즈베리파이용 카메라,그리고 다른 표준 악세사리가 필요합니다.

* `Raspberry Pi 4 Model B 2GB+ <https://www.raspberrypi.com/products/raspberry-pi-4-model-b/>`_
* `Raspberry Pi Camera Module <https://www.raspberrypi.com/products/camera-module-v2/>`_
* Heat sinks and Fan (optional but recommended)
* 5V 3A USB-C Power Supply
* SD card (at least 8gb)
* SD card read/writer


라즈베리파이4 설치
~~~~~~~~~~~~~~~~~~~~~~~

파이토치는 ARM 64bit(aarch64)를 위한 pip 패키지만 제공합니다. 그러므로 라즈베리파이4에 OS를 설치할때 꼭 64비트 버전을 설치하여야 합니다.

https://downloads.raspberrypi.org/raspios_arm64/images/ 에서 가장 최신의 64bit 라즈베리파이 OS를 다운받을 수 있고 rpi-imager를 이용해 설치할 수 있습니다.

**32-bit Raspberry Pi OS는 동작하지 않습니다.**

.. image:: https://user-images.githubusercontent.com/909104/152866212-36ce29b1-aba6-4924-8ae6-0a283f1fca14.gif

설치는 인터넷속도와 sd카드의 속도에 따라 몇 분간 진행될것이며 성공적으로 설치가 되면 아래와 같이 창이 나타날 것입니다.

.. image:: https://user-images.githubusercontent.com/909104/152867425-c005cff0-5f3f-47f1-922d-e0bbb541cd25.png

이번엔 sd카드를 라즈베리파이에 꼽고 카메라를 연결한 뒤 부팅하세요.

.. image:: https://user-images.githubusercontent.com/909104/152869862-c239c980-b089-4bd5-84eb-0a1e5cf22df2.png


부팅이 되고 초기설정이 끝나면, 카메라 사용을 위해 ``/boot/config.txt``을 수정해줘야 합니다.

.. code:: toml

    # This enables the extended features such as the camera.
    start_x=1

    # This needs to be at least 128M for the camera processing, if it's bigger you can just leave it as is.
    gpu_mem=128

    # You need to commment/remove the existing camera_auto_detect line since this causes issues with OpenCV/V4L2 capture.
    #camera_auto_detect=1

수정이 끝나면 재부팅하세요. 재부팅하고나면 video4linux2 장치 ``/dev/video0``가 종료되어야 합니다.

파이토치와 OpenCV 설치
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

파이토치와 필요한 다른 라이브러리들은 ARM 64-bit/aarch64용이 있기 때문에 pip을 통해 쉽게 설치할 수 있으며 다른 리눅스처럼 사용할 수 있습니다.

.. code:: shell

    $ pip install torch torchvision torchaudio
    $ pip install opencv-contrib-python
    $ pip install numpy --upgrade

.. image:: https://user-images.githubusercontent.com/909104/152874260-95a7a8bd-0f9b-438a-9c0b-5b67729e233f.png


이제 우리는 모두 설치가 잘 되었는지 확인할 수 있습니다.

.. code:: shell

  $ python -c "import torch; print(torch.__version__)"

.. image:: https://user-images.githubusercontent.com/909104/152874271-d7057c2d-80fd-4761-aed4-df6c8b7aa99f.png


영상 가져오기
~~~~~~~~~~~~~~

영상을 가져오는 경우 우리는 많이 사용되는 picamera라이브러리 대신 OpenCV를 사용해 비디오 프레임을 가져올 것입니다.
`picamera` 는 64bit 라즈베리파이 OS에서 사용이 불가능하고 OpenCV보다 많이 느립니다. OpenCV는 직접 ``/dev/video0``장치에 접근해서 프레임을 가져옵니다.

우리가 사용할 (MobileNetV2) 모델은 이미지 크기를 ``224x224``로 가져옵니다. 그래서 우리는 OpenCV로부터 36fps로 요청할 수 있습니다
우리는 모델을 위해 30fps를 목표하지만 더 높은 프레임비율을 요청해서 항상 충분한 프레임을 얻을 것입니다. 

.. code:: python

  import cv2
  from PIL import Image

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
  cap.set(cv2.CAP_PROP_FPS, 36)

OpenCV는 BGR순서의 ``numpy``배열을 반환합니다. 그래서 우리는 RGB포멧으로 변환하기 위해 프레임을 읽고 순서를 바꿔줄 필요가 있습니다 

.. code:: python

    ret, image = cap.read()
    # convert opencv output from BGR to RGB
    image = image[:, :, [2, 1, 0]]

이 데이터를 읽고 처리하는데는 약 ``3.5 ms`` 가 소요됩니다.

이미지 전처리
~~~~~~~~~~~~~~~~~~~~

우리는 프레임을 가져와서 모델 예측에 필요한 포멧으로 변형시켜야합니다. 이것은 다른 기기에서 표준 토치비전 변형을 사용할 때랑 같은 진행입니다. 

.. code:: python

    from torchvision import transforms

    preprocess = transforms.Compose([
        # convert the frame to a CHW torch tensor for training
        transforms.ToTensor(),
        # normalize the colors to the range that mobilenet_v2/3 expect
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    # The model can handle multiple images simultaneously so we need to add an
    # empty dimension for the batch.
    # [3, 224, 224] -> [1, 3, 224, 224]
    input_batch = input_tensor.unsqueeze(0)

모델 선택
~~~~~~~~~~~~~~~

다양한 퍼모먼스 특징들을 사용하기 위해 선택할 수 있는 많은 모델들이 있습니다. 모든 모델이 ``qnnpack`` pretrain을 제공하는 것은 아니므로 테스트를 위해 그것을 선택해야하지만 직접 모델을 학습하고 양자화하면 어떤 것이든 사용할 수 있습니다. 

이번 튜토리얼에서는 성능과 정확도가 뛰어난 ``mobilenet_v2``을 사용할 것입니다.

라즈베리파이4 벤치마크 결과:

+--------------------+------+-----------------------+-----------------------+--------------------+
| Model              | FPS  | Total Time (ms/frame) | Model Time (ms/frame) | qnnpack Pretrained |
+====================+======+=======================+=======================+====================+
| mobilenet_v2       | 33.7 |                  29.7 |                  26.4 | True               |
+--------------------+------+-----------------------+-----------------------+--------------------+
| mobilenet_v3_large | 29.3 |                  34.1 |                  30.7 | True               |
+--------------------+------+-----------------------+-----------------------+--------------------+
| resnet18           |  9.2 |                 109.0 |                 100.3 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| resnet50           |  4.3 |                 233.9 |                 225.2 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| resnext101_32x8d   |  1.1 |                 892.5 |                 885.3 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| inception_v3       |  4.9 |                 204.1 |                 195.5 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| googlenet          |  7.4 |                 135.3 |                 132.0 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| shufflenet_v2_x0_5 | 46.7 |                  21.4 |                  18.2 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| shufflenet_v2_x1_0 | 24.4 |                  41.0 |                  37.7 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| shufflenet_v2_x1_5 | 16.8 |                  59.6 |                  56.3 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+
| shufflenet_v2_x2_0 | 11.6 |                  86.3 |                  82.7 | False              |
+--------------------+------+-----------------------+-----------------------+--------------------+

MobileNetV2: 양자화와 JIT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

최적의 수행을 위해 우리는 양자화되고 융합화된 모델을 원합니다.
양자화는 표준 float32 보다 매우 효율적인 int8을 사용하여 계산하는 것을 의미합니다.
용합화는 연속 연산이 가능한 경우 더 성능이 좋은 버전으로 융합되었음을 의미합니다.
일반적으로 활성화함수(``ReLU``)와 같은 것들은 추론하는 동안 (``Conv2d``) 전에 레이어에 병합될 수 있다.

파이토치 aarch64 버전에서는 ``qnnpack`` 엔진사용을 필요로 합니다.

.. code:: python

    import torch
    torch.backends.quantized.engine = 'qnnpack'

이번 예시에서는 우리는 사용할 미리 양자화되어있고 융합화된 버전의 바로 사용할 수 있게 제공되는 MobileNetV2를 사용할 것입니다 .

.. code:: python

    from torchvision import models
    net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)

우리는 파이썬 오버헤드를 줄이기위해 모델을 jit하고 특정 연산자를 융합하기를 원합니다. Jit은 사용하지 않았을 떄 20fps까지 제공되지만 사용하면 30fps까지 제공됩니다.

.. code:: python

    net = torch.jit.script(net)

이미지 넣고 실행하기
~~~~~~~~~~~~~~~~~~~~~~~~~

우리는 이제 이미지를 넣고 실행할 수 있습니다.

.. code:: python

    import time

    import torch
    import numpy as np
    from torchvision import models, transforms

    import cv2
    from PIL import Image

    torch.backends.quantized.engine = 'qnnpack'

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv2.CAP_PROP_FPS, 36)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
    # jit model to take it from ~20fps to ~30fps
    net = torch.jit.script(net)

    started = time.time()
    last_logged = time.time()
    frame_count = 0

    with torch.no_grad():
        while True:
            # read frame
            ret, image = cap.read()
            if not ret:
                raise RuntimeError("failed to read frame")

            # convert opencv output from BGR to RGB
            image = image[:, :, [2, 1, 0]]
            permuted = image

            # preprocess
            input_tensor = preprocess(image)

            # create a mini-batch as expected by the model
            input_batch = input_tensor.unsqueeze(0)

            # run model
            output = net(input_batch)
            # do something with output ...

            # log model performance
            frame_count += 1
            now = time.time()
            if now - last_logged > 1:
                print(f"{frame_count / (now-last_logged)} fps")
                last_logged = now
                frame_count = 0

실행하면 30fps까지 유지하는것을 보여줍니다.

.. image:: https://user-images.githubusercontent.com/909104/152892609-7d115705-3ec9-4f8d-beed-a51711503a32.png

라즈베리파이os 설정은 모두 기본으로 합니다.
만약 UI를 사용하지 않고 다른 백그라운드 기능들을 사용하지 않으면 더 높은 성능과 안정성을 얻을 수 있습니다. 

만약 ``htop`` 을 확인한다면 거의 100% 사용을 볼 수 있습니다.

.. image:: https://user-images.githubusercontent.com/909104/152892630-f094b84b-19ba-48f6-8632-1b954abc59c7.png

모델이 end to end로 동작하고 있는 것을 확인하기 위해서, 우리는 클래스들의 확률을 계산할 수 있습니다. 그리고 디텍션 결과를 출력하기 위해 `ImageNet 클래스 라벨<https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a>`들을 사용할 수 있습니다. 

.. code:: python

    top = list(enumerate(output[0].softmax(dim=0)))
    top.sort(key=lambda x: x[1], reverse=True)
    for idx, val in top[:10]:
        print(f"{val.item()*100:.2f}% {classes[idx]}")

``mobilenet_v3_large`` 실시간 실행:

.. image:: https://user-images.githubusercontent.com/909104/153093710-bc736b6f-69d9-4a50-a3e8-9f2b2c9e04fd.gif


오렌지 감지:

.. image:: https://user-images.githubusercontent.com/909104/153092153-d9c08dfe-105b-408a-8e1e-295da8a78c19.jpg


머그컵 감지:

.. image:: https://user-images.githubusercontent.com/909104/153092155-4b90002f-a0f3-4267-8d70-e713e7b4d5a0.jpg


문제해결: 성능
~~~~~~~~~~~~~~~~~

파이토치는 기본적으로 가능한 모든 코어를 사용할 수 있습니다.
만약 라즈베리파이에서 백그라운드로 실행되고 있는 것이 있다면, 모델 추론에 지연시간이 발생하는 것과 같은 충돌이 발생할 수도 있습니다.  
이를 완화시키기 위해 적은 성능 저하로 최대 지연 시간을 줄일 수 있도록 스레드의 수를 줄힐 수 있습니다.

.. code:: python

  torch.set_num_threads(2)

``shufflenet_v2_x1_5``의 경우 4개의 스레드 대신 2개의 스레드를 사용하면 최고의 경우 지연 시간이 60ms에서 72ms로 증가하지만 128ms의 지연시간 발생을 제거합니다.

다음 단계
~~~~~~~~~~~~~

당신은 직접 모델을 만들거나 존재하는 것을 finetuning 할 수 있습니다.
만약 당신이 `torchvision.models.quantized<https://pytorch.org/vision/stable/models.html#quantized-models>`_ 에서 한 모델을 finetuning한다면 융합화와 양자화가 재부분 진행 되어있어서 라즈베리파이에서 좋은 성능으로 바로 배포할 수 있습니다.

더 보기:

* `Quantization <https://pytorch.org/docs/stable/quantization.html>`_ for more information on how to quantize and fuse your model.
* `Transfer Learning Tutorial <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>`_
  for how to use transfer learning to fine tune a pre-existing model to your dataset.
