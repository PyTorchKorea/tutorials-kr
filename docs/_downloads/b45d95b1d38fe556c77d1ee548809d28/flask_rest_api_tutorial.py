# -*- coding: utf-8 -*-
"""
Flask를 사용하여 Python에서 PyTorch를 REST API로 배포하기
===========================================================
**Author**: `Avinash Sajjanshetty <https://avi.im>`_
  **번역**: `박정환 <http://github.com/9bow>`_

이 튜토리얼에서는 Flask를 사용하여 PyTorch 모델을 배포하고 모델 추론(inference)을
할 수 있는 REST API를 만들어보겠습니다. 미리 훈련된 DenseNet 121 모델을 배포하여
이미지를 인식해보겠습니다.

.. tip:: 여기서 사용한 모든 코드는 MIT 라이선스로 배포되며,
         `GitHub <https://github.com/avinassh/pytorch-flask-api>`_ 에서 확인하실 수 있습니다.

이것은 PyTorch 모델을 상용(production)으로 배포하는 튜토리얼 시리즈의 첫번째
편입니다. Flask를 여기에 소개된 것처럼 사용하는 것이 PyTorch 모델을 제공하는
가장 쉬운 방법이지만, 고성능을 요구하는 때에는 적합하지 않습니다. 그에 대해서는:

    - TorchScript에 이미 익숙하다면, 바로 `Loading a TorchScript Model in C++ <https://tutorials.pytorch.kr/advanced/cpp_export.html>`_
      를 읽어보세요.

    - TorchScript가 무엇인지 알아보는 것이 필요하다면 `TorchScript 소개 <https://tutorials.pytorch.kr/beginner/Intro_to_TorchScript_tutorial.html>`_
      부터 보시길 추천합니다.
"""


######################################################################
# API 정의
# --------------
#
# 먼저 API 엔드포인트(endpoint)의 요청(request)와 응답(response)을 정의하는 것부터
# 시작해보겠습니다. 새로 만들 API 엔드포인트는 이미지가 포함된 ``file`` 매개변수를
# HTTP POST로 ``/predict`` 에 요청합니다. 응답은 JSON 형태이며 다음과 같은 예측 결과를
# 포함합니다:
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#
#

######################################################################
# 의존성(Dependencies)
# -------------------------
#
# 아래 명령어를 실행하여 필요한 패키지들을 설치합니다:
#
# ::
#
#     $ pip install Flask==2.0.1 torchvision==0.10.0


######################################################################
# 간단한 웹 서버
# -----------------
#
# Flask의 문서를 참고하여 아래와 같은 코드로 간단한 웹 서버를 구성합니다.


from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

###############################################################################
# 위 코드를 ``app.py`` 라는 파일명으로 저장한 후, 아래와 같이 Flask 개발 서버를
# 실행합니다:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

###############################################################################
# 웹 브라우저로 ``http://localhost:5000/`` 에 접속하면, ``Hello World!`` 가
# 표시됩니다.

###############################################################################
# API 정의에 맞게 위 코드를 조금 수정해보겠습니다. 먼저, 메소드의 이름을
# ``predict`` 로 변경합니다. 엔드포인트의 경로(path)도 ``/predict`` 로 변경합니다.
# 이미지 파일은 HTTP POST 요청을 통해서 보내지기 때문에, POST 요청에만 허용하도록
# 합니다:


@app.route('/predict', methods=['POST'])
def predict():
    return 'Hello World!'

###############################################################################
# 또한, ImageNet 분류 ID와 이름을 포함하는 JSON을 회신하도록 응답 형식을 변경하겠습니다.
# 이제 ``app.py`` 는 아래와 같이 변경되었습니다:

from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


######################################################################
# 추론(Inference)
# -----------------
#
# 다음 섹션에서는 추론 코드 작성에 집중하겠습니다. 먼저 이미지를 DenseNet에 공급(feed)할 수
# 있도록 준비하는 방법을 살펴본 뒤, 모델로부터 예측 결과를 얻는 방법을 살펴보겠습니다.
#
# 이미지 준비하기
# ~~~~~~~~~~~~~~~~~~~
#
# DenseNet 모델은 224 x 224의 3채널 RGB 이미지를 필요로 합니다.
# 또한 이미지 텐서를 평균 및 표준편차 값으로 정규화합니다. 자세한 내용은
# `여기 <https://pytorch.org/vision/stable/models.html>`_ 를 참고하세요.
#
# ``torchvision`` 라이브러리의 ``transforms`` 를 사용하여 변환 파이프라인
# (transform pipeline)을 구축합니다. Transforms와 관련한 더 자세한 내용은
# `여기 <https://pytorch.org/vision/stable/transforms.html>`_ 에서
# 읽어볼 수 있습니다.

import io

import torchvision.transforms as transforms
from PIL import Image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


######################################################################
# 위 메소드는 이미지를 byte 단위로 읽은 후, 일련의 변환을 적용하고 Tensor를
# 반환합니다. 위 메소드를 테스트하기 위해서는 이미지를 byte 모드로 읽은 후
# Tensor를 반환하는지 확인하면 됩니다.  (먼저 `../_static/img/sample_file.jpeg` 을
# 컴퓨터 상의 실제 경로로 바꿔야 합니다.)

with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)

######################################################################
# 예측(Prediction)
# ~~~~~~~~~~~~~~~~~~~
#
# 미리 학습된 DenseNet 121 모델을 사용하여 이미지 분류(class)를 예측합니다.
# ``torchvision`` 라이브러리의 모델을 사용하여 모델을 읽어오고 추론을 합니다.
# 이 예제에서는 미리 학습된 모델을 사용하지만, 직접 만든 모델에 대해서도
# 이와 동일한 방법을 사용할 수 있습니다. 모델을 읽어오는 것은 이
# :doc:`튜토리얼 </beginner/saving_loading_models>` 을 참고하세요.

from torchvision import models

# 이미 학습된 가중치를 사용하기 위해 `pretrained` 에 `True` 값을 전달합니다:
model = models.densenet121(pretrained=True)
# 모델을 추론에만 사용할 것이므로, `eval` 모드로 변경합니다:
model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


######################################################################
# ``y_hat`` Tensor는 예측된 분류 ID의 인덱스를 포함합니다. 하지만 사람이 읽을 수
# 있는 분류명이 있어야 하기 때문에, 이를 위해 이름과 분류 ID를 매핑하는 것이 필요합니다.
# `이 파일 <https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json>`_
# 을 다운로드 받아 ``imagenet_class_index.json`` 이라는 이름으로 저장 후, 저장한 곳의
# 위치를 기억해두세요. (또는, 이 튜토리얼과 똑같이 진행하는 경우에는 `tutorials/_static`
# 에 저장하세요.) 이 파일은 ImageNet 분류 ID와 ImageNet 분류명의 쌍을 포함하고 있습니다.
# 이제 이 JSON 파일을 불러와 예측 결과의 인덱스에 해당하는 분류명을 가져오겠습니다.

import json

imagenet_class_index = json.load(open('../_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


######################################################################
# ``imagenet_class_index`` 사전(dictionary)을 사용하기 전에,
# ``imagenet_class_index`` 의 키 값이 문자열이므로 Tensor의 값도
# 문자열로 변환해야 합니다.
# 위 메소드를 테스트해보겠습니다:


with open("../_static/img/sample_file.jpeg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

######################################################################
# 다음과 같은 응답을 받게 될 것입니다:

['n02124075', 'Egyptian_cat']

######################################################################
# 배열의 첫번째 항목은 ImageNet 분류 ID이고, 두번째 항목은 사람이 읽을 수 있는
# 이름입니다.
#
# .. Note ::
#    ``model`` 변수가 ``get_prediction`` 메소드 내부에 있지 않은 것을 눈치채셨나요?
#    왜 모델이 전역 변수일까요? 모델을 읽어오는 것은 메모리와 계산 측면에서 비싼
#    연산일 수 있습니다. 만약 ``get_prediction`` 메소드 내부에서 모델을 불러온다면,
#    메소드가 호출될 때마다 불필요하게 불러오게 됩니다. 초당 수천번의 요청을 받을
#    지도 모르는 웹 서버를 구축하고 있기 때문에, 매번 추론을 할 때마다 모델을
#    중복으로 불러오는데 시간을 낭비해서는 안됩니다. 따라서, 모델은 메모리에
#    딱 한번만 불러옵니다. 상용 시스템(production system)에서는 대량의 요청을
#    효율적으로 처리해야 하므로, 일반적으로 요청(request)을 처리하기 전에 모델을
#    불러와둡니다.

######################################################################
# 모델을 API 서버에 통합하기
# ---------------------------------------
#
# 마지막으로 위에서 만든 Flask API 서버에 모델을 추가하겠습니다.
# API 서버는 이미지 파일을 받는 것을 가정하고 있으므로, 요청으로부터 파일을 읽도록
# ``predict`` 메소드를 수정해야 합니다:
#
# .. code-block:: python
#
#    from flask import request
#
#    @app.route('/predict', methods=['POST'])
#    def predict():
#        if request.method == 'POST':
#            # we will get the file from the request
#            file = request.files['file']
#            # convert that to bytes
#            img_bytes = file.read()
#            class_id, class_name = get_prediction(image_bytes=img_bytes)
#            return jsonify({'class_id': class_id, 'class_name': class_name})

######################################################################
# ``app.py`` 파일은 이제 완성되었습니다. 아래가 정식 버전(full version)입니다;
# 아래 <PATH/TO/.json/FILE> 경로를 json 파일을 저장해둔 경로로 바꾸면 동작합니다:
#
# .. code-block:: python
#
#   import io
#   import json
#
#   from torchvision import models
#   import torchvision.transforms as transforms
#   from PIL import Image
#   from flask import Flask, jsonify, request
#
#
#   app = Flask(__name__)
#   imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
#   model = models.densenet121(pretrained=True)
#   model.eval()
#
#
#   def transform_image(image_bytes):
#       my_transforms = transforms.Compose([transforms.Resize(255),
#                                           transforms.CenterCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(
#                                               [0.485, 0.456, 0.406],
#                                               [0.229, 0.224, 0.225])])
#       image = Image.open(io.BytesIO(image_bytes))
#       return my_transforms(image).unsqueeze(0)
#
#
#   def get_prediction(image_bytes):
#       tensor = transform_image(image_bytes=image_bytes)
#       outputs = model.forward(tensor)
#       _, y_hat = outputs.max(1)
#       predicted_idx = str(y_hat.item())
#       return imagenet_class_index[predicted_idx]
#
#
#   @app.route('/predict', methods=['POST'])
#   def predict():
#       if request.method == 'POST':
#           file = request.files['file']
#           img_bytes = file.read()
#           class_id, class_name = get_prediction(image_bytes=img_bytes)
#           return jsonify({'class_id': class_id, 'class_name': class_name})
#
#
#   if __name__ == '__main__':
#       app.run()

######################################################################
# 이제 웹 서버를 테스트해보겠습니다! 다음과 같이 실행해보세요:
#
# ::
#
#     $ FLASK_ENV=development FLASK_APP=app.py flask run

#######################################################################
# `requests <https://pypi.org/project/requests/>`_ 라이브러리를 사용하여
# POST 요청을 만들어보겠습니다:
#
# .. code-block:: python
#
#   import requests
#
#   resp = requests.post("http://localhost:5000/predict",
#                        files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg','rb')})

#######################################################################
# `resp.json()` 을 호출하면 다음과 같은 결과를 출력합니다:
#
# ::
#
#     {"class_id": "n02124075", "class_name": "Egyptian_cat"}
#

######################################################################
# 다음 단계
# --------------
#
# 지금까지 만든 서버는 매우 간단하여 상용 프로그램(production application)으로써
# 갖춰야할 것들을 모두 갖추지 못했습니다. 따라서, 다음과 같이 개선해볼 수 있습니다:
#
# - ``/predict`` 엔드포인트는 요청 시에 반드시 이미지 파일이 전달되는 것을 가정하고
#   있습니다. 하지만 모든 요청이 그렇지는 않습니다. 사용자는 다른 매개변수로 이미지를
#   보내거나, 이미지를 아예 보내지 않을수도 있습니다.
#
# - 사용자가 이미지가 아닌 유형의 파일을 보낼수도 있습니다. 여기서는 에러를 처리하지
#   않고 있으므로, 이러한 경우에 서버는 다운(break)됩니다. 예외 전달(exception throe)을
#   위한 명시적인 에러 핸들링 경로를 추가하면 잘못된 입력을 더 잘 처리할 수 있습니다.
#
# - 모델은 많은 종류의 이미지 분류를 인식할 수 있지만, 모든 이미지를 인식할 수 있는
#   것은 아닙니다. 모델이 이미지에서 아무것도 인식하지 못하는 경우를 처리하도록
#   개선합니다.
#
# - 위에서는 Flask 서버를 개발 모드에서 실행하였지만, 이는 상용으로 배포하기에는
#   적당하지 않습니다. Flask 서버를 상용으로 배포하는 것은
#   `이 튜토리얼 <https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/>`_
#   을 참고해보세요.
#
# - 또한 이미지를 가져오는 양식(form)과 예측 결과를 표시하는 페이지를 만들어
#   UI를 추가할 수도 있습니다. 비슷한 프로젝트의 `데모 <https://pytorch-imagenet.herokuapp.com/>`_
#   와 이 데모의 `소스 코드 <https://github.com/avinassh/pytorch-flask-api-heroku>`_
#   를 참고해보세요.
#
# - 이 튜토리얼에서는 한 번에 하나의 이미지에 대한 예측 결과를 반환하는 서비스를
#   만드는 방법만 살펴보았는데요, 한 번에 여러 이미지에 대한 예측 결과를 반환하도록
#   수정해볼 수 있습니다. 추가로, `service-streamer <https://github.com/ShannonAI/service-streamer>`_
#   라이브러리는 자동으로 요청을 큐에 넣은 뒤 모델에 공급(feed)할 수 있는 미니-배치로
#   샘플링합니다. `이 튜토리얼 <https://github.com/ShannonAI/service-streamer/wiki/Vision-Recognition-Service-with-Flask-and-service-streamer>`_
#   을 참고해보세요.
#
# - 마지막으로 이 문서 상단에 링크된, PyTorch 모델을 배포하는 다른 튜토리얼들을
#   읽어보는 것을 권장합니다.
