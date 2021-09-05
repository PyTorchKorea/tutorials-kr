(베타) BERT 모델 동적 양자화하기
====================================================

.. tip::
   이 튜토리얼을 따라 하기 위해, 이
   `Colab 버전 <https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dynamic_quantization_bert_tutorial.ipynb>`_ 을 사용하길 권장합니다.
   그러면 아래에 설명된 정보들을 이용해 실험할 수 있습니다.

**Author**: `Jianyu Huang <https://github.com/jianyuh>`_
**Reviewed by**: `Raghuraman Krishnamoorthi <https://github.com/raghuramank100>`_
**Edited by**: `Jessica Lin <https://github.com/jlin27>`_
**번역**: `Myungha Kwon <https://github.com/kwonmha>`_


시작하기
-----------------------

이 튜토리얼에서는 `HuggingFace Transformers
<https://github.com/huggingface/transformers>`_ 예제들을 따라하면서 BERT
모델을 동적으로 양자화할 것입니다. BERT 처럼 유명하면서도 최고 성능을
내는 모델을 어떻게 동적으로 양자화된 모델로 변환하는지 한 단계씩 설명하겠습니다.

-  BERT 또는 Transformer 의 양방향 임베딩 표현(representation) 이라 불리는 방법은
   질의응답, 문장 분류 등의 여러 자연어 처리 분야(문제)에서 최고 성능을 달성한
   새로운 언어 표현 사전학습 방법입니다. 원 논문은 `여기 <https://arxiv.org/pdf/1810.04805.pdf>`_
   에서 읽을 수 있습니다.

-  PyTorch에서 지원하는 동적 양자화 기능은 부동소수점 모델의 가중치를 정적인
   int8 또는 float16 타입의 양자화된 모델로 변환하고, 활성 함수 부분은
   동적으로 양자화합니다. 가중치가 int8 타입으로 양자화 됐을 때, 활성 함수 부분은
   배치마다 int8 타입으로 동적으로 양자화 됩니다. PyTorch에는 지정된 모듈을
   동적이면서 가중치만 갖도록 양자화된 형태로 변환하고, 양자화된 모델을 만들어내는
   `torch.quantization.quantize_dynamic API <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_ 가 있습니다.

-  우리는 일반 언어 이해 평가 벤치마크 `(GLUE) <https://gluebenchmark.com/>`_ 중
   `Microsoft Research 의역 코퍼스(MRPC) <https://www.microsoft.com/en-us/download/details.aspx?id=52398>`_ 를
   대상으로 한 정확도와 추론 성능을 보여줄 것입니다. MRPC (Dolan and Brockett, 2005) 는
   온라인 뉴스로부터 자동으로 추출된 두 개의 문장들과 그 두 문장이 같은 뜻인지 사람이
   평가한 정답으로 이루어져 있습니다. 클래스의 비중이 같지 않아(같음 68%, 다름 32%),
   많이 쓰이는 `F1 점수 <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_ 를
   기록합니다. MRPC는 아래에 나온 것처럼 문장 쌍을 분류하는 자연어처리 문제에 많이 쓰입니다.

.. image:: /_static/img/bert.png


1. 준비
--------------

1.1 PyTorch, HuggingFace Transformers 설치하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

튜토리얼을 시작하기 위해 먼저 `여기 <https://github.com/pytorch/pytorch/#installation>`_ 의
PyTorch 설치 안내와 `HuggingFace 깃허브 저장소 <https://github.com/huggingface/transformers#installation>`_ 의
안내를 따라 합시다. 추가로 우리가 사용할 F1 점수를 계산하는 보조 함수가 내장된
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ 패키지를 설치합니다.


.. code:: shell

   pip install sklearn
   pip install transformers


PyTorch의 베타 기능들을 사용할 것이므로, 가장 최신 버전의 torch와 torchvision을 설치하는 것을 권해드립니다.
가장 최신 버전의 설치 안내는 `여기 <https://pytorch.org/get-started/locally/>`_ 에 있습니다.
예를 들어 Mac에 설치하려면 :


.. code:: shell

   yes y | pip uninstall torch tochvision
   yes y | pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html




1.2 필요한 모듈 불러오기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

이 단계에서는 이 튜토리얼에 필요한 파이썬 모듈들을 불러오겠습니다.

.. code:: python

    from __future__ import absolute_import, division, print_function

    import logging
    import numpy as np
    import os
    import random
    import sys
    import time
    import torch

    from argparse import Namespace
    from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                  TensorDataset)
    from tqdm import tqdm
    from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
    from transformers import glue_compute_metrics as compute_metrics
    from transformers import glue_output_modes as output_modes
    from transformers import glue_processors as processors
    from transformers import glue_convert_examples_to_features as convert_examples_to_features

    # 로깅 준비
    logger = logging.getLogger(__name__)
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.WARN)

    logging.getLogger("transformers.modeling_utils").setLevel(
                        logging.WARN)  # 로깅 줄이기

    print(torch.__version__)

쓰레드 한 개를 사용할 때의 FP32와 INT8의 성능을 비교하기 위해 쓰레드의 수를 1로 설정합니다.
이 튜토리얼의 끝부분에서는 PyTorch를 적절하게 병렬적으로 빌드하여 쓰레드 수를 다르게 설정할 수 있습니다.

.. code:: python

    torch.set_num_threads(1)
    print(torch.__config__.parallel_info())


1.3 보조 함수 알아보기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

보조 함수들은 transformers 라이브러리에 내장돼 있습니다. 우리는 주로
다음과 같은 보조 함수들을 사용할 것입니다. 하나는 텍스트 예시들을
특징 벡터들로 변환하는 함수이며, 다른 하나는 예측된 결과들에 대한
F1 점수를 계산하기 위한 함수입니다.

`Glue_convert_examples_to_features <https://github.com/huggingface/transformers/blob/master/transformers/data/processors/glue.py>`_ 함수는
텍스트를 입력 특징으로 변환합니다.


-  입력 문자열 분리하기;
-  [CLS]를 맨 앞에 삽입하기;
-  [SEP]를 첫번째 문장과 두 번째 문장 사이, 그리고 제일 마지막 위치에 넣기;
-  토큰이 첫번째 문장에 속하는지 두번째 문장에 속하는지 알려주는 토큰 타입 id 생성하기

`glue_compute_metrics <https://github.com/huggingface/transformers/blob/master/transformers/data/processors/glue.py>`_ 함수는
정밀도와 재현율의 가중 평균인 `F1 점수 <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_ 를
계산하는 행렬을 갖고 있습니다. F1 점수가 가장 좋을 때는 1이며, 가장 나쁠 때는 0입니다.
정밀도와 재현율은 F1 점수를 계산할 때 동일한 비중을 갖습니다.

-  F1 점수를 구하는 식 :

  .. math:: F1 = 2 * (\text{정밀도} * \text{재현율}) / (\text{정밀도} + \text{재현율})

1.4 데이터셋 다운로드
^^^^^^^^^^^^^^^^^^^^^^^^

MRPC 문제를 풀어보기 전에 `이 스크립트 <https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e>`_ 를
실행해 `GLUE 데이터셋 <https://gluebenchmark.com/tasks>`_ 을 다운로드 받고 ``glue_data``
폴더에 저장합니다.

.. code:: shell

   python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'


2. BERT 모델 미세조정하기
---------------------------

BERT 의 사상은 언어 표현을 사전학습하고, 문제에 특화된 매개변수들을
가능한 적게 사용하면서도, 사전학습된 양방향 표현을 많은 문제들에 맞게
미세조정하여 최고의 성능을 얻는 것입니다. 이 튜토리얼에서는 사전학습된
BERT 모델을 MRPC 문제에 맞게 미세조정하여 의미적으로 동일한 문장을
분류해보겠습니다.

사전학습된 BERT 모델(HuggingFace transformer들 중 ``bert-base-uncased`` 모델)을
MRPC 문제에 맞게 미세조정하기 위해 `예시들 <https://github.com/huggingface/transformers/tree/master/examples#mrpc>`_
의 명령을 따라 실행합니다:

.. code:: python

   export GLUE_DIR=./glue_data
   export TASK_NAME=MRPC
   export OUT_DIR=./$TASK_NAME/
   python ./run_glue.py \
       --model_type bert \
       --model_name_or_path bert-base-uncased \
       --task_name $TASK_NAME \
       --do_train \
       --do_eval \
       --do_lower_case \
       --data_dir $GLUE_DIR/$TASK_NAME \
       --max_seq_length 128 \
       --per_gpu_eval_batch_size=8   \
       --per_gpu_train_batch_size=8   \
       --learning_rate 2e-5 \
       --num_train_epochs 3.0 \
       --save_steps 100000 \
       --output_dir $OUT_DIR

MRPC 문제를 위해 미세조정한 BERT 모델을 `여기 <https://download.pytorch.org/tutorial/MRPC.zip>`_ 에 업로드 했습니다.
시간을 아끼려면 모델 파일(~400MB)을 ``$OUT_DIR`` 에 바로 다운로드할 수 있습니다.

2.1 전역 환경 설정하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
이 단계에서는 미세조정한 BERT 모델을 동적 양자화 이전, 이후에 평가하기 위한
전역 환경 설정을 진행합니다.

.. code:: python

    configs = Namespace()

    # 미세조정한 모델의 출력을 저장할 폴더, $OUT_DIR.
    configs.output_dir = "./MRPC/"

    # GLUE 벤치마크 중 MRPC 데이터가 있는 폴더, $GLUE_DIR/$TASK_NAME.
    configs.data_dir = "./glue_data/MRPC"

    # 사전학습된 모델의 이름 또는 경로.
    configs.model_name_or_path = "bert-base-uncased"
    # 입력 문장의 최대 길이
    configs.max_seq_length = 128

    # GLUE 문제 준비
    configs.task_name = "MRPC".lower()
    configs.processor = processors[configs.task_name]()
    configs.output_mode = output_modes[configs.task_name]
    configs.label_list = configs.processor.get_labels()
    configs.model_type = "bert".lower()
    configs.do_lower_case = True

    # 장비 종류, 배치 크기, 분산 학습 방식, 캐싱 방식 설정
    configs.device = "cpu"
    configs.per_gpu_eval_batch_size = 8
    configs.n_gpu = 0
    configs.local_rank = -1
    configs.overwrite_cache = False


    # 재현을 위한 랜덤 시드 설정
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    set_seed(42)


2.2 미세조정한 BERT 모델 불러오기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``configs.output_dir`` 에서 토크나이저와 미세조정한 문장 분류
BERT 모델(FP32)를 불러옵니다.

.. code:: python

    tokenizer = BertTokenizer.from_pretrained(
        configs.output_dir, do_lower_case=configs.do_lower_case)

    model = BertForSequenceClassification.from_pretrained(configs.output_dir)
    model.to(configs.device)


2.3 토큰화, 평가 함수 정의하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Huggingface <https://github.com/huggingface/transformers/blob/master/examples/run_glue.py>`_
의 토큰화 함수와 평가 함수를 사용합니다.

.. code:: python

    # coding=utf-8
    # Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
    # Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    def evaluate(args, model, tokenizer, prefix=""):
        # MNLI의 두 평가 결과(일치, 불일치)를 처리하기 위한 반복문
        eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli"
                                else (args.output_dir,)

        results = {}
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # DistributedSampler는 무작위로 표본을 추출합니다
            eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1
                            else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                            batch_size=args.eval_batch_size)

            # 다중 gpu로 평가
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # 평가 실행!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1],
                              'labels':         batch[3]}
                    if args.model_type != 'distilbert':
                        inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet']
                                                    else None
                                                    # XLM, DistilBERT and RoBERTa 모델들은 segment_ids를
                                                    # 사용하지 않습니다
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(),
                                                axis=0)

            eval_loss = eval_loss / nb_eval_steps
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        return results


    def load_and_cache_examples(args, task, tokenizer, evaluate=False):
        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # 분산 학습 프로세스들 중 처음 프로세스 한 개만 데이터를 처리하고 다른
                                         # 프로세스들은 캐시를 이용하도록 합니다.

        processor = processors[task]()
        output_mode = output_modes[task]
        # 캐시 또는 데이터셋 파일로부터 데이터 특징을 불러옵니다.
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
                # 해결책(사전학습된 RoBERTa 모델에서는 라벨 인덱스 순서가 바뀌어 있습니다.)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            examples = processor.get_dev_examples(args.data_dir) if evaluate
                        else processor.get_train_examples(args.data_dir)
            features = convert_examples_to_features(examples,
                                                    tokenizer,
                                                    label_list=label_list,
                                                    max_length=args.max_seq_length,
                                                    output_mode=output_mode,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    # xlnet의 경우 앞쪽에 패딩합니다.
                                                    pad_token=tokenizer.convert_tokens_to_ids(
                                                        [tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in
                                                                            ['xlnet'] else 0,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # 분산 학습 프로세스들 중 처음 프로세스 한 개만 데이터를 처리하고 다른
                                         # 프로세스들은 캐시를 이용하도록 합니다.

        # 텐서로 변환하고 데이터셋을 빌드합니다.
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset


3. 동적 양자화 적용하기
-------------------------------

HuggingFace BERT 모델에 동적 양자화를 적용하기 위해
``torch.quantization.quantize_dynamic`` 을 호출합니다. 구체적으로,

-  모델 중 torch.nn.Linear 모듈을 양자화하도록 지정합니다.
-  가중치들을 양자화할 때 int8로 변환하도록 지정합니다.

.. code:: python

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print(quantized_model)


3.1 모델 크기 확인하기
^^^^^^^^^^^^^^^^^^^^^^^^

먼저 모델 크기를 확인해보겠습니다. 보면, 모델 크기가 상당히 줄어든 것을
알 수 있습니다(FP32 형식의 모델 크기 : 438MB; INT8 형식의 모델 크기 : 181MB):

.. code:: python

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

    print_size_of_model(model)
    print_size_of_model(quantized_model)


이 튜토리얼에 사용된 BERT 모델(``bert-base-uncased``)은 어휘 사전의
크기(V)가 30522입니다. 임베딩 크기를 768로 하면, 단어 임베딩 행렬의
크기는 4(바이트/FP32) \* 30522 \* 768 = 90MB 입니다. 양자화를 적용한 결과,
임베딩 행렬을 제외한 모델의 크기가 350 MB (FP32 모델)에서 90 MB (INT8 모델)로
줄어들었습니다.


3.2 추론 정확도와 속도 평가하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

다음으로, 기존의 FP32 모델과 동적 양자화를 적용한 INT8 모델들의
추론 속도와 정확도를 비교해보겟습니다.

.. code:: python

    def time_model_evaluation(model, configs, tokenizer):
        eval_start_time = time.time()
        result = evaluate(configs, model, tokenizer, prefix="")
        eval_end_time = time.time()
        eval_duration_time = eval_end_time - eval_start_time
        print(result)
        print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))

    # 기존 FP32 BERT 모델 평가
    time_model_evaluation(model, configs, tokenizer)

    # 동적 양자화를 거친 INT8 BERT 모델 평가
    time_model_evaluation(quantized_model, configs, tokenizer)


맥북 프로에서 양자화하지 않았을 때, 408개의 MRPC 데이터를 모두 추론하는데
160초가 소요됩니다. 양자화 하면 90초가 걸립니다. 맥북 프로에서 실행해본
결과를 아래에 정리했습니다:


.. code::

   | 정확도  |  F1 점수  |  모델 크기  |  쓰레드 1개 |  쓰레드 4개 |
   |  FP32  |  0.9019  |   438 MB   |   160 초   |   85 초    |
   |  INT8  |  0.902   |   181 MB   |   90 초    |   46 초    |


MRPC 문제에 맞게 미세조정한 BERT 모델에 학습 후 동적 양자화를 적용한
결과, 0.6% 낮은 F1 점수가 나왔습니다. 참고로, `최근 논문 <https://arxiv.org/pdf/1910.06188.pdf>`_
(표 1)에서는 학습 후 동적 양자화를 적용했을 때, F1 점수 0.8788이 나왔고,
양자화 의식 학습을 적용했을 때는 0.8956이 나왔습니다. 우리는 Pytorch의 비대칭
양자화를 사용했지만, 참고한 논문에서는 대칭적 양자화만을 사용했다는 점이 주요한
차이입니다.

이 튜토리얼에서는 단일 쓰레드를 썼을 때의 비교를 위해 쓰레드의 개수를
1로 설정했습니다. 또한 INT8 연산자들을 각 연산자마다 병렬적으로
양자화할 수 있습니다. 사용자들은 ``torch.set_num_threads(N)`` (``N``
은 연산자 별 병렬화를 수행하는 쓰레드의 개수)을 이용하여 다중 쓰레드를
사용할 수 있습니다. 연산자 별 병렬화를 사용하려면 미리 OpenMP, Native, TBB
같이 알맞은 `백엔드 <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#build-options>`_ 를
이용하여 PyTorch를 빌드해야 합니다.
``torch.__config__.parallel_info()`` 를 사용하여 병렬화 설정을 확인할 수
있습니다. 같은 맥북 프로에서 Native 백엔드로 빌드한 PyTorch를 사용했을 때,
MRPC 데이터셋을 평가하는데 약 46초가 소요됐습니다.


3.3 양자화된 모델 직렬화하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

나중에 다시 쓸 수 있도록 `torch.jit.save` 을 사용하여 양자화된 모델을 직렬화하고 저장할 수 있습니다.

.. code:: python

    input_ids = ids_tensor([8, 128], 2)
    token_type_ids = ids_tensor([8, 128], 2)
    attention_mask = ids_tensor([8, 128], vocab_size=2)
    dummy_input = (input_ids, attention_mask, token_type_ids)
    traced_model = torch.jit.trace(quantized_model, dummy_input)
    torch.jit.save(traced_model, "bert_traced_eager_quant.pt")

양자화된 모델을 불러올 때는 `torch.jit.load` 를 사용합니다.

.. code:: python

    loaded_quantized_model = torch.jit.load("bert_traced_eager_quant.pt")


마치며
----------

이 튜토리얼은 BERT처럼 잘 알려진 자연어처리 모델을 동적으로
양자화하는 방법을 설명합니다. 동적 양자화를 통해 모델의 정확도를 크게
약화시키지 않으면서도 모델의 크기를 줄일 수 있습니다.

읽어주셔서 감사합니다. 언제나처럼 어떠한 피드백도 환영이니, 의견이
있다면 `여기 <https://github.com/pytorch/pytorch/issues>`_ 에 이슈를 제기해주세요.




참고 자료
-------------

[1] J.Devlin, M. Chang, K. Lee and K. Toutanova, `BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding (2018)
<https://arxiv.org/pdf/1810.04805.pdf>`_.

[2] `HuggingFace Transformers <https://github.com/huggingface/transformers>`_.

[3] O. Zafrir, G. Boudoukh, P. Izsak, and M. Wasserblat (2019). `Q8BERT:
Quantized 8bit BERT <https://arxiv.org/pdf/1910.06188.pdf>`_.
