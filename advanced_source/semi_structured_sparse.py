# -*- coding: utf-8 -*-
"""
(beta) 반구조적 (2:4) 희소성을 통한 BERT 가속화
=============================================
**저자**: `Jesse Cai <https://github.com/jcaip>`_
**번역**: `이채운 <https://github.com/dlcodns>`_

"""

####################################################################
# 개요
# ----
# 
# 다른 형태의 희소성(sparsity)처럼, **반구조적 희소성**은 신경망의 메모리 오버헤드와 지연 시간을 
# 줄이기 위한 모델 최적화 기법으로, 일부 모델 정확도는 희생하게 됩니다. 이 방법은 
# **세분화된 구조적 희소성** 또는 **2:4 구조적 희소성**으로도 알려져 있습니다.
# 
# 반구조적 희소성은 고유한 희소성 패턴에서 유래하며, 여기서 2n개의 요소 중 n개의 요소가 
# 가지치기(prune)됩니다. 일반적으로 n=2인 경우가 많아 2:4 희소성이라고 부릅니다. 
# 반구조적 희소성은 GPU에서 효율적으로 가속화될 수 있고, 다른 희소성 패턴만큼 모델 
# 정확도를 저하시키지 않기 때문에 특히 흥미롭습니다.
# 
# `반구조적 희소성 지원 <https://pytorch.org/docs/2.1/sparse.html#sparse-semi-structured-tensors>`_,
# 이 도입되면서, PyTorch를 벗어나지 않고도 반구조적 희소 모델을 가지치기하고 가속화할 
# 수 있습니다. 이 튜토리얼에서는 이 과정을 설명할 것입니다.
#
# .. image:: ../../_static/img/pruning_flow.jpg
# 
# 튜토리얼이 끝나면 BERT 질문-응답 모델을 2:4 희소화하여 거의 모든 F1 손실을 회복한 
# 상태(86.92의 밀집 모델 vs 86.48의 희소 모델)로 미세 조정할 것입니다. 마지막으로 
# 이 2:4 희소 모델을 추론을 위해 가속화하여 1.3배 속도 향상을 달성할 것입니다.
# 

#####################################################
# 요구사항
# --------
#
# -  PyTorch >= 2.1.
# -  반구조적 희소성을 지원하는 NVIDIA GPU(Compute Capability 8.0+)
#
# 이 튜토리얼은 초보자에게 반구조적 희소성 및 일반적인 희소성을 맞춤 설명합니다.
# 이미 2:4 희소 모델을 보유한 사용자에게는 ``to_sparse_semi_structured``를 사용하여 
# 추론을 위한 ``nn.Linear`` 레이어를 가속화하는 것이 매우 간단합니다. 다음은 그 예시입니다:
# 

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.utils.benchmark import Timer
SparseSemiStructuredTensor._FORCE_CUTLASS = True

# Linear 가중치를 2:4 희소성으로 마스킹
mask = torch.Tensor([0, 0, 1, 1]).tile((3072, 2560)).cuda().bool()
linear = torch.nn.Linear(10240, 3072).half().cuda().eval()
linear.weight = torch.nn.Parameter(mask * linear.weight)

x = torch.rand(3072, 10240).half().cuda()

with torch.inference_mode():
    dense_output = linear(x)
    dense_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # SparseSemiStructuredTensor를 통해 가속화
    linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))

    sparse_output = linear(x)
    sparse_t = Timer(stmt="linear(x)",
                    globals={"linear": linear,
                             "x": x}).blocked_autorange().median * 1e3

    # 희소 및 밀집 행렬 곱셈은 수치적으로 동일함
    # A100 80GB에서, 다음과 같은 결과를 확인: `Dense: 0.870ms Sparse: 0.630ms | Speedup: 1.382x`
    assert torch.allclose(sparse_output, dense_output, atol=1e-3)
    print(f"Dense: {dense_t:.3f}ms Sparse: {sparse_t:.3f}ms | Speedup: {(dense_t / sparse_t):.3f}x")


######################################################################
# 반구조적 희소성은 어떤 문제를 해결하는가?
# --------------------------------------
# 
# 희소성의 일반적인 목적은 간단합니다: 네트워크 내에 0이 있는 경우, 
# 해당 매개변수를 저장하거나 계산하지 않음으로써 효율성을 최적화할 수 있습니다. 
# 그러나 희소성의 구체적인 구현은 까다롭습니다. 매개변수를 0으로 만드는 것만으로는 
# 기본적으로 모델의 지연 시간 / 메모리 오버헤드에 영향을 미치지 않습니다.
# 
# 그 이유는 dense tensor가 여전히 가지치기된(0인) 요소를 포함하고 있으며, 
# 밀집 행렬 곱셈 커널이 이러한 요소에 대해 계속 연산을 수행하기 때문입니다. 성능 향상을 
# 실현하려면, 밀집 커널을 가지치기된 요소의 계산을 건너뛰는 희소 커널로 교체해야 합니다.
# 
# 이를 위해, 희소 커널은 가지치기된 요소를 저장하지 않고, 지정된 요소를 압축된 형식으로 
# 저장하는 희소 행렬을 사용합니다.
# 
# 반구조적 희소성의 경우, 원래 매개변수의 정확히 절반과 요소가 어떻게 
# 배열되었는지에 대한 압축된 메타데이터를 저장합니다.
# 
# .. image:: https://developer-blogs.nvidia.com/wp-content/uploads/2023/06/2-4-structured-sparsity-pattern.png
#    :align: center :width: 80%
# 
#    Image sourced from `NVIDIA blog post <https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/>`_ on semi-structured sparsity.
# 
# 희소 레이아웃에는 각기 다른 장점과 단점을 가진 여러 가지가 있습니다. 2:4 반구조적 
# 희소 레이아웃은 흥미로운 두 가지 이유가 있습니다.
# 
# * 이전의 희소 형식과 달리 반구조적 희소성은 GPU에서 효율적으로 가속되도록 설계되었습니다. 
#   2020년 NVIDIA는 Ampere 아키텍처를 통해 반구조적 희소성을 위한 하드웨어 지원을 도입했으며, 
#   CUTLASS `cuSPARSELt <https://docs.nvidia.com/cuda/cusparselt/index.html>`__ 
#   cuSPARSELt를 통해 빠른 희소 커널도 출시했습니다.
# 
# * 동시에 반구조적 희소성은 다른 희소 형식에 비해 모델 정확도에 미치는 영향이 덜한 경향이 있습니다.
#   특히 더 발전된 가지치기 및 미세 조정 방법을 고려할 때 그렇습니다. 
#   NVIDIA가 공개한 `백서 <https://arxiv.org/abs/2104.08378>`_
#   에서 2:4 희소성을 목표로 한 단순한 크기 기준 가지치기(magnitude pruning) 후 모델을 
#   재학습하면 거의 동일한 모델 정확도를 달성할 수 있음을 보여주었습니다.
# 
# 반구조적 희소성은 이론적으로 2배의 속도 향상을 제공하면서도 희소성 수준이 낮고(50%), 모델 
# 정확도를 유지하기에 충분히 세밀한 적절한 균형점을 제공합니다.
# 
# +---------------------+-------------+--------+------------+-------------+
# | 네트워크             | 데이터 세트  | 메트릭 | Dense FP16 | Sparse FP16 |
# +=====================+=============+========+============+=============+
# | ResNet-50           | ImageNet    | Top-1  | 76.1       | 76.2        |
# +---------------------+-------------+--------+------------+-------------+
# | ResNeXt-101_32x8d   | ImageNet    | Top-1  | 79.3       | 79.3        |
# +---------------------+-------------+--------+------------+-------------+
# | Xception            | ImageNet    | Top-1  | 79.2       | 79.2        |
# +---------------------+-------------+--------+------------+-------------+
# | SSD-RN50            | COCO2017    | bbAP   | 24.8       | 24.8        |
# +---------------------+-------------+--------+------------+-------------+
# | MaskRCNN-RN50       | COCO2017    | bbAP   | 37.9       | 37.9        |
# +---------------------+-------------+--------+------------+-------------+
# | FairSeq Transformer | EN-DE WMT14 | BLEU   | 28.2       | 28.5        |
# +---------------------+-------------+--------+------------+-------------+
# | BERT-Large          | SQuAD v1.1  | F1     | 91.9       | 91.9        |
# +---------------------+-------------+--------+------------+-------------+
# 
# 반구조적 희소성은 워크플로 관점에서 추가적인 장점이 있습니다. 희소성 수준이 50%로 고정되어 
# 있어 모델을 희소화하는 문제를 두 가지 별개의 하위 문제로 분해하기가 더 쉽습니다.
# 
# - 정확도 - 2:4 희소 가중치 세트를 찾아 모델의 정확도 저하를 최소화할 수 있는 방법은 무엇인가요?
#
# - 성능: 추론 및 메모리 오버헤드를 줄이기 위해 2:4 희소 가중치를 어떻게 가속화할 수 있는가?
#

##################################################################### 
# .. math::
# 
#    \begin{bmatrix}
#       1 & 1 & 0 & 0 \\
#       0 & 0 & 1 & 1 \\
#       1 & 0 & 0 & 0 \\
#       0 & 0 & 1 & 1 \\
#       \end{bmatrix}
# 
# 이 두 문제 사이의 자연스러운 연결점은 0으로 된 밀집 텐서입니다. 우리의 추론 솔루션은 이러한 형식의 
# 텐서를 압축하고 가속하도록 설계되었습니다. 이는 활발한 연구 분야이기 때문에 많은 사용자가 맞춤형 마스킹 
# 솔루션을 고안할 것으로 예상됩니다.
# 
# 반구조적 희소성에 대해 조금 더 배웠으니, 이제 질문 응답 작업인 SQuAD에 대해 학습된 BERT 모델에 이를 
# 적용해 봅시다.
# 
# 소개 & 설정
# -----------
# 
# 필요한 모든 패키지를 불러오는 것으로 시작하겠습니다.
# 

# 만약 Google Colab에서 실행 중이라면, 다음 명령어를 실행하세요:

# .. code-block: python
# 
#    !pip install datasets transformers evaluate accelerate pandas
#
import os
os.environ["WANDB_DISABLED"] = "true"

import collections
import datasets
import evaluate
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
from torch.ao.pruning import WeightNormSparsifier
import transformers

# ``cuSPARSELt``가 사용 불가능한 경우, 강제로 CUTLASS를 사용합니다.
SparseSemiStructuredTensor._FORCE_CUTLASS = True
torch.manual_seed(100)


######################################################################
# 우리가 다루고 있는 데이터셋/작업에 특화된 몇 가지 보조 함수도 
# 정의해야 합니다. 이러한 함수들은 Hugging Face 코스의 
# `이 자료 <https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt>`__
# 를 참고하여 수정되었습니다.
# 

def preprocess_validation_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def preprocess_train_function(examples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, (offset, answer) in enumerate(zip(offset_mapping, answers)):
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # 문맥의 시작과 끝을 찾기
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1

        # 답변이 문맥 안에 완전히 포함되지 않는 경우, (0, 0)으로 레이블 지정
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # 그렇지 않으면 시작 및 끝 토큰 위치를 지정
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    # 예를 들어 ``tqdm``(examples)에서:
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # 해당 예제와 관련된 모든 특성(feature)을 반복
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 문맥에 완전히 포함되지 않은 답변은 건너뜀
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # 길이가 0보다 작거나
                    # max_answer_length보다 큰 답변은 건너뜀
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # 가장 높은 점수를 가진 답변을 선택
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


######################################################################
# 이제 이러한 함수들이 정의되었으므로, 모델의 벤치마크를 도와줄 추가적인 보조 함수 하나만 더 필요합니다.
# 

def measure_execution_time(model, batch_sizes, dataset):
    dataset_for_model = dataset.remove_columns(["example_id", "offset_mapping"])
    dataset_for_model.set_format("torch")
    batch_size_to_time_sec = {}
    for batch_size in batch_sizes:
        batch = {
            k: dataset_for_model[k][:batch_size].cuda()
            for k in dataset_for_model.column_names
        }

        with torch.no_grad():
            baseline_predictions = model(**batch)
            timer = benchmark.Timer(
                stmt="model(**batch)", globals={"model": model, "batch": batch}
            )
            p50 = timer.blocked_autorange().median * 1000
            batch_size_to_time_sec[batch_size] = p50

            model_c = torch.compile(model, fullgraph=True)
            timer = benchmark.Timer(
                stmt="model(**batch)", globals={"model": model_c, "batch": batch}
            )
            p50 = timer.blocked_autorange().median * 1000
            batch_size_to_time_sec[f"{batch_size}_compile"] = p50
            new_predictions = model_c(**batch)

    return batch_size_to_time_sec



######################################################################
# 모델과 토크나이저를 로드한 후, 데이터셋을 설정하면서 시작하겠습니다.
# 

# 모델 불러오기
model_name = "bert-base-cased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
print(f"Loading tokenizer: {model_name}")
print(f"Loading model: {model_name}")

# 학습 및 검증 데이터셋 설정
squad_dataset = datasets.load_dataset("squad")
tokenized_squad_dataset = {}
tokenized_squad_dataset["train"] = squad_dataset["train"].map(
    lambda x: preprocess_train_function(x, tokenizer), batched=True
)
tokenized_squad_dataset["validation"] = squad_dataset["validation"].map(
    lambda x: preprocess_validation_function(x, tokenizer),
    batched=True,
    remove_columns=squad_dataset["train"].column_names,
)
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)


######################################################################
# 기준 성능 설정
# =======================
# 
# 다음으로, SQuAD 데이터셋에서 모델의 빠른 기준 성능을 학습시켜 보겠습니다. 이 작업은 모델이 주어진 
# 문맥(위키피디아 기사)에서 주어진 질문에 대한 답변이 되는 텍스트의 범위 또는 구간을 식별하도록 
# 요구합니다. 다음 코드를 실행하면 F1 점수는 86.9가 나옵니다. 이는 보고된 NVIDIA 점수와 매우 가깝고, 
# 차이는 아마도 BERT-base와 BERT-large 또는 미세 조정 하이퍼파라미터 때문일 것입니다.
# 

training_args = transformers.TrainingArguments(
    "trainer",
    num_train_epochs=1,
    lr_scheduler_type="constant",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=256,
    logging_steps=50, 
    # 튜토리얼 실행을 위한 최대 단계 제한. 보고된 정확도 수치를 보려면 아래 줄을 삭제하세요.
    max_steps=500,
    report_to=None,
)

trainer = transformers.Trainer(
    model,
    training_args,
    train_dataset=tokenized_squad_dataset["train"],
    eval_dataset=tokenized_squad_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# 평가를 위한 비교 배치 크기
batch_sizes = [4, 16, 64, 256]
# 2:4 희소성은 fp16을 필요로 하므로, 공정한 비교를 위해 여기에서 캐스팅함
with torch.autocast("cuda"):
    with torch.no_grad():
        predictions = trainer.predict(tokenized_squad_dataset["validation"])
        start_logits, end_logits = predictions.predictions
        fp16_baseline = compute_metrics(
            start_logits,
            end_logits,
            tokenized_squad_dataset["validation"],
            squad_dataset["validation"],
        )
        fp16_time = measure_execution_time(
            model,
            batch_sizes,
            tokenized_squad_dataset["validation"],
        )

print("fp16", fp16_baseline)
print("cuda_fp16 time", fp16_time)

import pandas as pd
df = pd.DataFrame(trainer.state.log_history)
df.plot.line(x='step', y='loss', title="Loss vs. # steps", ylabel="loss")


######################################################################
# BERT를 2:4 희소성으로 가지치기
# -----------------------------
# 
# 이제 기준 성능을 설정했으니, BERT를 가지치기할 차례입니다. 가지치기에는 여러 가지 전략이 있지만, 
# 가장 일반적인 방법 중 하나는 **크기 기반 가지치기**로, 이는 L1 norm이 가장 
# 낮은 가중치를 제거하는 방법입니다. NVIDIA는 모든 결과에서 크기 기반 가지치기를 사용했으며, 이는 
# 일반적인 기준 방법입니다.
# 
# 이를 위해 우리는 ``torch.ao.pruning`` 패키지를 사용할 것입니다. 이 패키지에는 가중치-norm 희소화 
# 도구가 포함되어 있습니다. 이러한 희소화 도구는 모델의 가중치 텐서에 마스크 매개변수화를 적용하여 
# 작동합니다. 이를 통해 가지치기된 가중치를 마스킹하여 희소성을 시뮬레이션할 수 있습니다.
# 
# 또한, 모델의 어느 레이어에 희소성을 적용할지 결정해야 합니다. 이 경우에는 각각의 테스크 헤드 출력을 
# 제외한 모든 ``nn.Linear`` 레이어에 적용합니다. 이는 반구조적 희소성(semi-structured sparsity)이 
# `형상 제약 <https://pytorch.org/docs/2.1/sparse.html#constructing-sparse-semi-structured-tensors>`_
# 을 가지기 때문이며, 각각의 task ``nn.Linear`` 레이어는 이러한 제약을 충족하지 않기 때문입니다.
# 

sparsifier = WeightNormSparsifier(
    # 모든 블록에 희소성 적용
    sparsity_level=1.0,
    # 4개의 요소가 하나의 블록의 형태
    sparse_block_shape=(1, 4),
    # 4개의 블록마다 두 개의 0이 포함됨
    zeros_per_block=2
)

# BERT 모델에 ``nn.Linear``가 있는 경우 설정에 추가
sparse_config = [
    {"tensor_fqn": f"{fqn}.weight"}
    for fqn, module in model.named_modules()
    if isinstance(module, nn.Linear) and "layer" in fqn
]


######################################################################
# 모델의 매개변수화는 첫 번째 단계는 모델의 가중치를 마스킹하기 위한 매개변수화를 삽입하는 것입니다. 
# 이는 준비 단계에서 수행됩니다. 이렇게 하면 ``.weight``에 접근할 때마다 대신 ``mask * weight``를 
# 얻게 됩니다.
# 

# 모델을 준비하고, 학습을 위한 가짜 희소성 매개변수수화를 삽입합니다.
sparsifier.prepare(model, sparse_config)
print(model.bert.encoder.layer[0].output)


######################################################################
# 그 다음, 단일 가지치기 단계를 수행합니다. 모든 가지치기 도구(pruner)는 가지치기 도구의 구현 
# 논리에 따라 마스크를 업데이트하는 ``update_mask()`` 메서드를 구현합니다. 이 단계 메서드는 
# 희소성 설정(sparse config)에서 지정된 가중치에 대해 이 ``update_mask`` 함수를 호출합니다.
# 
# 또한 모델을 평가하여 미세 조정/재학습 없이 가지치기(zero-shot) 또는 가지치기의 정확도 저하를 보여줄 것입니다.
# 

sparsifier.step()
with torch.autocast("cuda"):
    with torch.no_grad():
        predictions = trainer.predict(tokenized_squad_dataset["validation"])
    pruned = compute_metrics(
        *predictions.predictions,
        tokenized_squad_dataset["validation"],
        squad_dataset["validation"],
    )
print("pruned eval metrics:", pruned)


######################################################################
# 이 상태에서 모델을 미세 조정(fine-tuning)하여 가지치기되지 않는 요소들을 업데이트하고, 정확도 손실을 
# 보완할 수 있습니다. 만족할 만한 상태에 도달하면, ``squash_mask``를 호출하여 마스크와 
# 가중치를 하나로 결합할 수 있습니다. 이렇게 하면 매개변수화가 제거되고, 0으로 된 2:4 밀집 
# 모델이 남게 됩니다.
# 

trainer.train()
sparsifier.squash_mask()
torch.set_printoptions(edgeitems=4)
print(model.bert.encoder.layer[0].intermediate.dense.weight[:8, :8])

df["sparse_loss"] = pd.DataFrame(trainer.state.log_history)["loss"]
df.plot.line(x='step', y=["loss", "sparse_loss"], title="Loss vs. # steps", ylabel="loss")


######################################################################
# 추론을 위한 2:4 희소 모델 가속화
# --------------------------------------------
# 
# 이제 이 형식의 모델을 얻었으므로, QuickStart 가이드에서처럼 추론을 위해 가속할 수 있습니다.
# 

model = model.cuda().half()
# 희소성을 위해 가속화
for fqn, module in model.named_modules():
    if isinstance(module, nn.Linear) and "layer" in fqn:
        module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))

with torch.no_grad():
    predictions = trainer.predict(tokenized_squad_dataset["validation"])
start_logits, end_logits = predictions.predictions
metrics_sparse = compute_metrics(
    start_logits,
    end_logits,
    tokenized_squad_dataset["validation"],
    squad_dataset["validation"],
)
print("sparse eval metrics: ", metrics_sparse)
sparse_perf = measure_execution_time(
    model,
    batch_sizes,
    tokenized_squad_dataset["validation"],
)
print("sparse perf metrics: ", sparse_perf)


######################################################################
# 크기 기반 가지치기 후 모델을 재학습한 결과, 가지치기 시 손실되었던 F1 
# 점수의 거의 대부분이 회복되었습니다. 동시에 ``bs=16``의 배치 크기에서 1.28배의 속도 향상을 
# 달성했습니다. 하지만 모든 형상이 성능 향상에 적합한 것은 아닙니다. 배치 크기가 작고 계산에 
# 사용되는 시간이 제한적일 때는 희소 커널이 밀집 커널보다 더 느릴 수 있습니다.
# 
# 반구조적 희소성(semi-structured sparsity)은 텐서의 하위 클래스(subclass)로 구현되어 있기 때문에, 
# ``torch.compile``과 호환됩니다. ``to_sparse_semi_structured``와 함께 사용하면 BERT에서 총 
# 2배의 속도 향상을 얻을 수 있습니다.
#
# .. table::
#
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Metrics            | fp16   | 2:4 희소성    | 변화 / 속도 증가율 | 컴파일됨   |
#     +====================+========+==============+===================+===========+
#     | 정확도 일치  (%)    | 78.53  | 78.44        | -0.09             |           |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | F1 (%)             | 86.93  | 86.49        | -0.44             |           |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=4)        | 11.10  | 15.54        | 0.71x             | no        |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=16)       | 19.35  | 15.74        | 1.23x             | no        |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=64)       | 72.71  | 59.41        | 1.22x             | no        |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=256)      | 286.65 | 247.63       | 1.14x             | no        |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=4)        | 7.59   | 7.46         | 1.02x             | yes       |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=16)       | 11.47  | 9.68         | 1.18x             | yes       |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=64)       | 41.57  | 36.92        | 1.13x             | yes       |
#     +--------------------+--------+--------------+-------------------+-----------+
#     | Time (bs=256)      | 159.22 | 142.23       | 1.12x             | yes       |
#     +--------------------+--------+--------------+-------------------+-----------+
# 
# 결론
# ==========
# 
# 이 튜토리얼에서는 BERT를 2:4 희소성으로 가지치기하는 방법과 2:4 희소 모델을 추론용으로 가속하는 
# 방법을 보여주었습니다. ``SparseSemiStructuredTensor`` 하위 클래스를 활용하여 fp16 기준 성능에 
# 비해 1.3배의 속도 향상을 달성했으며, ``torch.compile``을 사용하면 최대 2배까지 속도 향상을 이룰 
# 수 있었습니다. 또한, BERT를 미세 조정하여 손실된 F1 점수(밀집 모델: 86.92 vs 희소 모델: 86.48)를 
# 회복하는 과정에서 2:4 희소성의 이점을 입증했습니다.
# 
