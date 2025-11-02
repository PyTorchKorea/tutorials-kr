torch.export 흐름의 설명, 일반적인 문제점과 이들을 해결하기 위한 해결책
=======================================================================================
**저자:** `Ankith Gunapal <https://github.com/agunapal>`__, `Jordi Ramon <https://github.com/JordiFB>`__, `Marcos Carranza <https://github.com/macarran>`__
**번역:** `이현준 <https://github.com/joonda>`__

`Introduction to torch.export Tutorial <https://tutorials.pytorch.kr/intermediate/torch_export_tutorial.html>`__ 에서, `torch.export <https://pytorch.org/docs/stable/export.html>`__ 를 사용하는 방법을 배웠습니다.
이 튜토리얼은 이전 튜토리얼을 확장하며, 널리 사용되는 모델들을 코드와 함께 내보내는 과정과 ``torch.export`` 사용중 마주칠 수 있는 문제들을 다룹니다.

이 튜토리얼은 다음과 같은 사용 사례에 맞게 모델을 내보내는 방법을 배웁니다.

* 영상 분류 (`MViT <https://pytorch.org/vision/main/models/video_mvit.html>`__)
* 자동 음성 인식 (`OpenAI Whisper-Tiny <https://huggingface.co/openai/whisper-tiny>`__)
* 이미지 캡셔닝 (`BLIP <https://github.com/salesforce/BLIP>`__)
* 프롬프트 기반 이미지 분할 (`SAM2 <https://ai.meta.com/sam2/>`__)

각 네 가지 모델은 ``torch.export``의 고유한 기능을 보여주고, 구현 과정에서의 실질적인 고려사항과 발생할 수 있는 문제들을 함께 다루기 위해 선정되었습니다.

전제 조건
-------------

* PyTorch 2.4 이상 버전
* ``torch.export`` 및 PyTorch Eager 추론에 대한 기본적인 이해

``torch.export``의 핵심 요구 사항: 그래프 분절(graph break) 없음
----------------------------------------------------

`torch.compile <https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html>`__ 은 JIT를 활용해 PyTorch 코드를 최적화된 커널로 컴파일함으로써 실행 속도를 향상시킵니다. 주어진 모델을 ``TorchDynamo``를 활용하여 최적화하고,
 최적화된 그래프를 만든 뒤, API에서 지정한 백엔드를 통해 하드웨어에 맞게 실행되도록 변환합니다.

TorchDynamo가 지원하지 않는 Python의 기능을 만나면, 계산 그래프는 중단하고 해당 코드는 기본 Python 인터프리터가 처리하도록 하고, 그래프 캡쳐를 이어나갑니다.
이러한 중단된 계산 그래프를 `graph break <https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html#torchdynamo-and-fx-graphs>`__ 라고 칭합니다.

``torch.export``와 ``torch.compile``의 주요한 차이점 중 하나는 ``torch.export``는 그래프 분절을 지원하지 않는다는 것입니다. 즉, 내보내려는 전체 모델 또는 모델의 일부는 단일 그래프 형태여야 합니다.
이는 그래프 분절을 처리하려면 지원되지 않는 연산을 기본 Python으로 평가해야하는데, 이러한 방식이 ``torch.export``의 설계와 호환되지 않기 때문입니다.
다양한 PyTorch 프레임워크들의 차이점에 대한 세부적인 정보는 `link <https://pytorch.org/docs/main/export.html#existing-frameworks>`__ 에서 확인할 수 있습니다.

아래의 커맨드를 사용해서 프로그램 내의 그래프 분절을 확인할 수 있습니다.

.. code:: sh

   TORCH_LOGS="graph_breaks" python <file_name>.py

프로그램 내의 그래프 분절을 제거하도록 코드를 수정해야 합니다. 문제가 해결된다면, 모델을 내보낼 준비가 된 것입니다.
PyTorch는 인기 있는 HuggingFace와 TIMM 모델에서 ``torch.compile``을 위해서 `nightly benchmarks <https://hud.pytorch.org/benchmark/compilers>`__ 를 실행합니다.
이러한 모델 대부분은 그래프 분절이 없습니다.

해당 레시피에 포함된 모델들은 그래프 분절이 없지만, `torch.export` 는 실패합니다.

영상 분류
--------------------

MViT는 `MultiScale Vision Transformers <https://arxiv.org/abs/2104.11227>`__ 을 기반으로한 모델의 클래스입니다. 이 모델은 `Kinetics-400 Dataset <https://arxiv.org/abs/1705.06950>`__ 을 사용하여 사전 훈련된 영상 분류 모델입니다.
이 모델은 적절한 데이터 셋과 함께 사용한다면, 게임 환경에서의 동작 인식에 활용할 수 있습니다.

아래의 코드는 MViT를 ``batch_size=2``로 트레이싱하여 내보내고, 이후 ``batch_size=4``로 내보낸 프로그램이 정상적으로 실행되는지 확인합니다.

.. code:: python

   import numpy as np
   import torch
   from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
   import traceback as tb

   model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

   # 2개의 비디오의 배치를 만들며, 각각의 형태는 224x224x3에 16 프레임을 가집니다.
   input_frames = torch.randn(2, 16, 224, 224, 3)
   # Transpose to get [1, 3, num_clips, height, width].
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

   # 모델을 내보냅니다.
   exported_program = torch.export.export(
       model,
       (input_frames,),
   )

   # 4개의 비디오의 배치를 만들며, 각각의 형태는 224x224x3에 16 프레임을 가집니다.
   input_frames = torch.randn(4, 16, 224, 224, 3)
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
   try:
       exported_program.module()(input_frames)
   except Exception:
       tb.print_exc()


에러: 정적 배치 크기
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

       raise RuntimeError(
   RuntimeError: Expected input at *args[0].shape[0] to be equal to 2, but got 4


기본적으로 내보내는 과정에서는 모든 입력 형태가 고정되어 있다고 가정하고 트레이스(trace) 합니다, 따라서 트레이싱(tracing)을 할 때 사용한 입력 형태와 다른 형태로 프로그램을 실행하면 오류가 발생합니다.

해결 방법
~~~~~~~~

이 오류를 해결하기 위해, 입력의 첫 번째 차원 (``batch_size``)을 동적으로 지정하고, 허용되는 ``batch_size``의 범위를 지정합니다.
아래의 수정된 예제에서는, ``batch_size``의 허용 범위를 1부터 16까지로 지정합니다.
여기서 알려드릴 세부사항은 ``min=2`` 은 버그가 아니라는 것이고 이에 대한 설명은 `The 0/1 Specialization Problem <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk>`__ 문서에서 확인할 수 있습니다.
또한 ``torch.export``의 동적 입력 형태에 대한 자세한 설명은 export 튜토리얼에서 찾아볼 수 있습니다.
아래의 코드는 동적 배치 사이즈를 사용하여 mViT를 내보내는 방법을 보여줍니다.

.. code:: python

   import numpy as np
   import torch
   from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
   import traceback as tb


   model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)

   # 2개의 비디오의 배치를 만들며, 각각의 형태는 224x224x3에 16 프레임을 가집니다.
   input_frames = torch.randn(2,16, 224, 224, 3)

   # 차원을 바꿔 [1, 3, num_clips, height, width] 형태로 변환합니다.
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))

   # 모델을 내보냅니다.
   batch_dim = torch.export.Dim("batch", min=2, max=16)
   exported_program = torch.export.export(
       model,
       (input_frames,),
       # Specify the first dimension of the input x as dynamic
       dynamic_shapes={"x": {0: batch_dim}},
   )

   # 4개의 비디오의 배치를 만들며, 각각의 형태는 224x224x3에 16 프레임을 가집니다.
   input_frames = torch.randn(4,16, 224, 224, 3)
   input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
   try:
       exported_program.module()(input_frames)
   except Exception:
       tb.print_exc()


자동 음성 인식
---------------

자동 음성 인식은 기계학습을 활용하여 음성을 텍스트로 변환하는 기술입니다.
`Whisper <https://arxiv.org/abs/2212.04356>`__ 는 OpenAI에서 개발한 인코더-디코더 구조의 트랜스포머 모델로, ASR과 음성 번역을 위해 68만 시간의 라벨링된 데이터를 사용해 학습되었습니다.
아래의 코드로 자동 음성 인식을 위한 ``whisper-tiny`` 모델을 내보낼 수 있습니다.

.. code:: python

   import torch
   from transformers import WhisperProcessor, WhisperForConditionalGeneration
   from datasets import load_dataset

   # 모델을 가져옵니다.
   model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

   # 모델 내보내기를 위한 더미 입력입니다.
   input_features = torch.randn(1,80, 3000)
   attention_mask = torch.ones(1, 3000)
   decoder_input_ids = torch.tensor([[1, 1, 1 , 1]]) * model.config.decoder_start_token_id

   model.eval()

   exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_features, attention_mask, decoder_input_ids,))



에러: TorchDynamo를 이용한 엄격한(strict) 트레이싱(tracing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: console

   torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'DynamicCache' object has no attribute 'key_cache'


기본적으로 ``torch.export``는 `TorchDynamo <https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html>`__ 라는 바이트코드 분석 엔진을 사용하여 코드를 처리합니다, 이는 코드를 심볼릭하게 분석하여 그래프를 생성합니다.
이 분석은 안전성 보장을 강화해주지만, 모든 Python 코드를 지원하는 것은 아닙니다. ``whisper-tiny`` 모델을 기본 strict 모드로 내보낼 때, Dynamo에서 지원되지 않는 기능 때문에 일반적으로 오류가 발생합니다.
Dynamo에서 이 에러가 발생하는 이유를 이해하려면, `GitHub issue <https://github.com/pytorch/pytorch/issues/144906>`__ 해당 깃허브 이슈를 참고하세요.

해결 방법
~~~~~~~~

위의 에러를 해결하기 위해, ``torch.export``는 Python 인터프리터를 사용해 프로그램을 트레이싱하는 ``non_strict`` 모드를 제공하며, 이는 PyTorch eager 실행과 유사하게 동작합니다.
유일한 차이점은 모든 ``Tensor`` 객체가 ``ProxyTensors``로 대체되며, 이는 모든 연산이 그래프에 기록된다는 것입니다.
``strict=False``을 사용하면, 프로그램에서 내보낼 수 있습니다.

.. code:: python

   import torch
   from transformers import WhisperProcessor, WhisperForConditionalGeneration
   from datasets import load_dataset

   # 모델을 가져옵니다.
   model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

   # 모델 내보내기를 위한 더미 입력입니다.
   input_features = torch.randn(1,80, 3000)
   attention_mask = torch.ones(1, 3000)
   decoder_input_ids = torch.tensor([[1, 1, 1 , 1]]) * model.config.decoder_start_token_id

   model.eval()

   exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(input_features, attention_mask, decoder_input_ids,), strict=False)

이미지 캡셔닝
----------------

**이미지 캡셔닝**은 이미지에 있는 단어의 내용을 정의하는 업무를 수행한다. 게임 환경에서 이미지 캡셔닝은 장면 내 다양한 게임 객체에 대한 텍스트 설명을 동적으로 생성하며, 게이머에게 추가적인 정보를 제공함으로써 게임 플레이 경험을 향상시키는데 활용될 수 있습니다.
`BLIP <https://arxiv.org/pdf/2201.12086>`__ 는 이미지 캡셔닝 분야에서 널리 사용되는 모델로, `released by SalesForce Research <https://github.com/salesforce/BLIP>`__ 에서 공개되었습니다.
아래 코드는 ``batch_size=1``로 BLIP를 내보내려고 시도합니다.

.. code:: python

   import torch
   from models.blip import blip_decoder

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   image_size = 384
   image = torch.randn(1, 3,384,384).to(device)
   caption_input = ""

   model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
   model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
   model.eval()
   model = model.to(device)

   exported_program: torch.export.ExportedProgram= torch.export.export(model, args=(image,caption_input,), strict=False)



에러: 동결된(frozen) 저장소를 가진 텐서를 변경할 수 없습니다.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

모델을 내보낼 때, 모델 구현에서 ``torch.export`` 에서 아직 지원하지 않는 특정 Python 연산이 포함이 될 수 있기 때문에 실패할 수 있습니다.
이 실패 사례들 중 일부는 해결 방법이 있을 수 있습니다. BLIP는 원래 모델에서 오류가 발생하는 예시이지만, 코드에 작은 수정을 하면 해결할 수 있습니다.
``torch.export``는 `ExportDB <https://pytorch.org/docs/main/generated/exportdb/index.html>`__ 에서 지원하는 연산과 지원하지 않는 연산의 일반적인 사례들을 나열하고, 코드에서 내보낼 수 있도록 수정하는 방법을 보여줍니다.

.. code:: console

   File "/BLIP/models/blip.py", line 112, in forward
       text.input_ids[:,0] = self.tokenizer.bos_token_id
     File "/anaconda3/envs/export/lib/python3.10/site-packages/torch/_subclasses/functional_tensor.py", line 545, in __torch_dispatch__
       outs_unwrapped = func._op_dk(
   RuntimeError: cannot mutate tensors with frozen storage



해결 방법
~~~~~~~~

내보내기가 실패하는 위치에 있는 `tensor <https://github.com/salesforce/BLIP/blob/main/models/blip.py#L112>`__ 를 복제합니다.

.. code:: python

   text.input_ids = text.input_ids.clone() # clone the tensor
   text.input_ids[:,0] = self.tokenizer.bos_token_id

.. note::
   This constraint has been relaxed in PyTorch 2.7 nightlies. This should work out-of-the-box in PyTorch 2.7
   이 제약은 PyTorch 2.7 nightlies에서 완화되었습니다. PyTorch 2.7에서는 별도의 설정 없이 바로 동작할 것입니다.

프롬프트 기반 이미지 분할
-----------------------------

**이미지 분할**은 디지털 이미지를 픽셀 단위의 특징에 따라 서로 다른 그룹, 즉 세그먼트로 나누는 컴퓨터 비전 기술입니다.
`Segment Anything Model (SAM) <https://ai.meta.com/blog/segment-anything-foundation-model-image-segmentation/>`__) 은 프롬프트 기반 이미지 분할을 도입한 모델로, 사용자가 원하는 객체를 지정하는 프롬프트를 입력하면 해당 객체의 마스크를 예측합니다.
`SAM 2 <https://ai.meta.com/sam2/>`__ 는 이미지와 비디오에서 객체를 분할하기 위한 최초의 통합 모델입니다. `SAM2ImagePredictor <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L20>`__ 클래스는 모델에 프롬프트를 입력할 수 있는 간편한 인터페이스를 제공합니다.
이 모델은 포인트와 박스 프롬프트는 물론, 이전 예측에서 생성된 마스크도 입력으로 받을 수 있습니다.
SAM2는 객체 추적에서 강력한 제로샷 성능을 제공하므로, 장면 내 게임 객체를 추적하는 데 활용할 수 있습니다.

`SAM2ImagePredictor <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L20>`__ 의 예측 메서드에서 발생하는 텐서 연산은 실제로 `_predict <https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py#L291>`__ 메서드 안에서 수행됩니다.
따라서 아래와 같이 내보내기를 시도합니다.

.. code:: python

   ep = torch.export.export(
       self._predict,
       args=(unnorm_coords, labels, unnorm_box, mask_input, multimask_output),
       kwargs={"return_logits": return_logits},
       strict=False,
   )


에러: 모델의 타입이 ``torch.nn.Module`` 아닙니다.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch.export`` 는 모듈이 ``torch.nn.Module`` 타입이어야 합니다. 하지만, 내보내기 하려는 모듈은 클래스 메서드이기 때문에 오류가 발생합니다.

.. code:: console

   Traceback (most recent call last):
     File "/sam2/image_predict.py", line 20, in <module>
       masks, scores, _ = predictor.predict(
     File "/sam2/sam2/sam2_image_predictor.py", line 312, in predict
       ep = torch.export.export(
     File "python3.10/site-packages/torch/export/__init__.py", line 359, in export
       raise ValueError(
   ValueError: Expected `mod` to be an instance of `torch.nn.Module`, got <class 'method'>.


해결 방법
~~~~~~~~

도우미 클래스를 작성하여 ``torch.nn.Module``을 상속하고, 클래스의 ``forward`` 메서드 안에서 ``_predict method``를 호출합니다. 전체 코드는 `here <https://github.com/anijain2305/sam2/blob/ued/sam2/sam2_image_predictor.py#L293-L311>`__ 에서 확인할 수 있습니다.

.. code:: python

   class ExportHelper(torch.nn.Module):
       def __init__(self):
           super().__init__()

       def forward(_, *args, **kwargs):
           return self._predict(*args, **kwargs)

    model_to_export = ExportHelper()
    ep = torch.export.export(
         model_to_export,
         args=(unnorm_coords, labels, unnorm_box, mask_input,  multimask_output),
         kwargs={"return_logits": return_logits},
         strict=False,
         )

결론
----------

이 튜토리얼에서는 ``torch.export``를 활용하여 다양한 대표적인 사용 사례의 모델을 내보내는 방법을 학습하였고, 올바른 설정과 간단한 코드 수정으로 발생할 수 있는 여러 문제들을 해결하는 방법도 함께 다뤘습니다.
모델을 성공적으로 내보낸 후에, 서버 환경에서는 `AOTInductor <https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html>`__ 를, 엣지 디바이스 환경에서는 `ExecuTorch <https://pytorch.org/executorch/stable/index.html>`__ 를 사용하여 ``ExportedProgram``을 하드웨어에 맞게 변환할 수 있습니다.
``AOTInductor`` (AOTI)에 대한 자세한 내용은 `AOTI tutorial <https://tutorials.pytorch.kr/recipes/torch_export_aoti_python.html>`__ 을, ``ExecuTorch``에 대한 자세한 내용은 `ExecuTorch tutorial <https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html>`__ 을 참고하세요.