"""
wav2vec2을 이용한 강제 정렬
==============================

**저자**: `Moto Hira <moto@meta.com>`_
**번역**: `김규진 <http://github.com/kimgyujin/>`_

이번 튜토리얼에서는 `CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition <https://arxiv.org/abs/2007.09127>`__ 에서 
설명한 CTC 분할 알고리즘을 이용하여 torchaudio를 가지고 정답 스크립트를 음성에 맞추는 방법에 대해 설명합니다.

.. note::
   이 튜토리얼은 원래 Wav2Vec2의 사용 사례를 설명하기 위해 작성되었습니다.

   TorchAudio에는 강제 정렬을 위해 설계된 API가 있습니다.
   `CTC forced alignment API tutorial <./ctc_forced_alignment_api_tutorial.html>`__ 은 핵심 API인 
   :py:func:`torchaudio.functional.forced_align` 의 사용법에 대해 보여주고 있습니다.

   만약 본인만의 코퍼스에 대해 강제 정렬하려는 경우, :py:class:`torchaudio.pipelines.Wav2Vec2FABundle` 를 사용하는 것을 추천합니다.
   이는 강제 정렬을 위해 특별히 훈련된 사전 훈련 모델과 함께 :py:func:`~torchaudio.functional.forced_align` 및 여러 함수를 결합하여 사용할 수 있게 합니다. 
   사용법에 대한 자세한 내용은 다국어 데이터를 위한 강제 정렬을 설명하는 `Forced alignment for multilingual data <forced_alignment_for_multilingual_data_tutorial.html>`__ 를 참조하세요.

"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


######################################################################
# 개요
# --------
#
# 정렬 과정은 다음과 같습니다.
#
# 1. 오디오 파형으로부터 프레임별 라벨 확률을 추정한다.
# 2. 각 시간 별로 정렬된 라벨의 확률을 나타내는 trellis 행렬을 생성한다.
# 3. trellis 행렬로부터 가장 가능성이 높은 경로를 찾는다.
#
# 이번 예시에는 음성 특징 추출을 위해 torchaudio의 wav2vec2 모델을 사용합니다.
#

######################################################################
# 준비
# -----------
#
# 먼저 필요한 패키지를 임포트하고, 작업할 데이터를 불러옵니다.
#

from dataclasses import dataclass

import IPython
import matplotlib.pyplot as plt

torch.random.manual_seed(0)

SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")


######################################################################
# 프레임 별 라벨 확률 생성 
# -------------------------------------
#
# 첫번째 과정은 각 오디오 프레임 별 라벨 클래스 확률을 생성하는 것입니다.
# ASR(음성 인식)용으로 학습된 wav2vec2 모델을 사용할 수 있습니다.
# 여기서는 :py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H` 를 사용합니다.
#
# ``torchaudio``는 연관된 라벨과 함께 미리 학습된 모델에 쉽게 접근할 수 있게 합니다.
#
#
# .. note::
#
#    여기서는 수치적인 불안정성을 피하고자 로그 도메인에서 확률을 계산할 것입니다. 
#    이렇게 하기 위해 ``torch.log_softmax()`` 를 사용하여 ``출력 확률`` 을 정규화합니다.
#

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

print(labels)

################################################################################
# 시각화
# ~~~~~~~~~~~~~


def plot():
    fig, ax = plt.subplots()
    img = ax.imshow(emission.T)
    ax.set_title("Frame-wise class probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()


plot()

######################################################################
# 정렬 확률 생성 (trellis)
# ----------------------------------------
#
# 다음, 출력 행렬로부터 각 프레임에서 정답 스크립트의 라벨들이 등장할 수 있는 확률을 나타내는 trellis를 생성합니다.
# Trellis는 시간 축과 라벨 축을 가지고 있는 2D 행렬입니다. 라벨 축은 정렬하려는 정답 스크립트를 나타냅니다. 
# :math:`t` 를 시간 축에서의 인덱스로 나타내는 데 사용하고, :math:`j` 를 라벨 축에서의 인덱스로 나타내는 데 사용합니다. 
# :math:`c_j` 는 라벨 인덱스 :math:`j` 에서의 라벨을 나타냅니다.
#
#
# :math:`t+1` 시점에서의 확률을 생성하기 위해, :math:`t` 시점에서의 trellis와 :math:`t+1` 시점에서의 출력을 봅니다. 
# :math:`t+1` 시점에서 :math:`c_{j+1}` 라벨에 도달할 수 있는 2개의 경로가 있습니다. 
# 첫번째는 :math:`t` 시점에서 라벨은 :math:`c_{j+1}` 였으며 :math:`t` 에서 :math:`{t+1}` 으로 바뀔 때 라벨 변화는 없는 경우입니다. 
# 다른 경우는 :math:`t` 시점에서 라벨은 :math:`c_j` 였으며 :math:`t+1` 시점에서는 다음 라벨 :math:`c_{j+1}` 로 전이된 경우입니다.
#
# 아래 그림은 2가지 전이를 나타내고 있습니다.
#
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/ctc-forward.png
#
# 가장 가능성 있는 전이를 찾기 위해, :math:`k_{(t+1, j+1)}` 의 값의 가장 가능성 있는 경로를 택합니다. 
# 이는 아래에 나와 있는 식으로 나타낼 수 있습니다.
#
# :math:`k_{(t+1, j+1)} = max( k_{(t, j)} p(t+1, c_{j+1}), k_{(t, j+1)} p(t+1, repeat) )`
#
# 
# :math:`k` 는 trellis 행렬을 나타내며, :math:`p(t, c_j)` 는 :math:`t` 시점에서의 라벨 :math:`c_j` 의 확률을 나타냅니다. 
# repeat는 CTC 식에서의 블랭크 토큰을 나타냅니다. (CTC 알고리즘에 대한 자세한 설명은 'Sequence Modeling with CTC'를 참고하세요.) [`distill.pub <https://distill.pub/2017/ctc/>`__])
#   


# SOS와 EOS를 나타내는 space 토큰을 가지고 정답 스크립트를 둘러쌈.
transcript = "|I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"
dictionary = {c: i for i, c in enumerate(labels)}

tokens = [dictionary[c] for c in transcript]
print(list(zip(transcript, tokens)))


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # 같은 토큰에 머무르고 있는 경우의 점수
            trellis[t, 1:] + emission[t, blank_id],
            # 다음 토큰으로 바뀌는 경우에 대한 점수
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


trellis = get_trellis(emission, tokens)

################################################################################
# 시각화
# ~~~~~~~~~~~~~


def plot():
    fig, ax = plt.subplots()
    img = ax.imshow(trellis.T, origin="lower")
    ax.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
    ax.annotate("+ Inf", (trellis.size(0) - trellis.size(1) / 5, trellis.size(1) / 3))
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()


plot()

######################################################################
# 위 시각화된 그림에서, 행렬을 대각선으로 가로지르는 높은 확률의 추적(trace)를 볼 수 있습니다.
#


######################################################################
# 가장 가능성 높은 경로 찾기 (백트래킹)
# ----------------------------------------
#
# trellis가 한번 생성되면, 높은 확률을 가지는 요소를 따라 이를 탐색할 수 있습니다.
#
# 가장 높은 확률을 가지는 시간 단계에서 마지막 라벨 인덱스로부터 시작합니다. 
# 그 후에, 이전 전이 확률 :math:`k_{t, j} p(t+1, c_{j+1})` 또는
# :math:`k_{t, j+1} p(t+1, repeat)`에 기반하여 머무를지 (:math:`c_j \rightarrow c_j`) 또는 전이할지
# (:math:`c_j \rightarrow c_{j+1}`)를 시간 역순으로 진행합니다.
#
# 라벨이 한번 시작 부분에 도달하게 되면, 전이가 수행됩니다.
#
# trellis 행렬은 경로를 찾기 위해 사용되지만, 각 분할의 최종 확률에 대해서는 출력 행렬에서의 프레임별 확률을 사용합니다.
#


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # 발생하지 않는 경우지만, 혹시 몰라 예외 처리함.
        assert t > 0

        # 1. 현재 위치가 stay인지 또는 change인지를 판단함.
        # stay 대 change의 프레임 별 점수 계산
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # stay 대 change의 문맥을 고려한 점수 계산
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # 위치 갱신
        t -= 1
        if changed > stayed:
            j -= 1

        # 프레임별 확률을 이용하여 경로를 저장함.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # 지금 j == 0이라면 이는, SoS를 도달했다는 것을 의미함.
    # 시각화를 위해 나머지 부분을 채움.
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


path = backtrack(trellis, emission, tokens)
for p in path:
    print(p)


################################################################################
# 시각화
# ~~~~~~~~~~~~~


def plot_trellis_with_path(trellis, path):
    # 경로와 함께 trellis를 그리기 위해, 'nan' 값을 이용합니다.
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.title("The path found by backtracking")
    plt.tight_layout()


plot_trellis_with_path(trellis, path)

######################################################################
# 좋습니다.

######################################################################
# 경로 분할
# ----------------
# 지금 이 경로는 같은 라벨의 반복이 포함되어 있기 때문에 이를 병합하여 원본 정답 스크립트와 가깝게 만들어봅시다.
#
# 다수의 경로 지점들을 병합할 때 단순하게, 병합된 분할의 평균 확률을 취합니다.
#


# 라벨을 병합함
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


segments = merge_repeats(path)
for seg in segments:
    print(seg)


################################################################################
# 시각화
# ~~~~~~~~~~~~~


def plot_trellis_with_segments(trellis, segments, transcript):
    # 경로와 함께 trellis를 그리기 위해, 'nan' 값을 이용합니다.
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start : seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    ax2.set_title("Label probability with and without repetation")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07))
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.grid(True, axis="y")
    ax2.set_ylim(-0.1, 1.1)
    fig.tight_layout()


plot_trellis_with_segments(trellis, segments, transcript)


######################################################################
# 좋습니다.

######################################################################
# 여러 분할을 단어로 병합
# -----------------------------
# 지금 단어로 병합해봅시다. wav2vec2 모델은 ``'|'`` 을 단어 경계로 사용합니다. 
# 그래서 ``'|'`` 이 등장할 때마다 분할을 병합합니다.
#
# 그러고 나서 최종적으로 원본 오디오를 분할된 오디오로 분할하고 이를 들어 분할이 옳게 되었는지 확인합니다.
#

# 단어 병합
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


word_segments = merge_words(segments)
for word in word_segments:
    print(word)


################################################################################
# 시각화
# ~~~~~~~~~~~~~
def plot_alignments(trellis, segments, word_segments, waveform, sample_rate=bundle.sample_rate):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start : seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1)

    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")
    ax1.set_facecolor("lightgray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        ax1.axvspan(word.start - 0.5, word.end - 0.5, edgecolor="white", facecolor="none")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    # 원본 waveform
    ratio = waveform.size(0) / sample_rate / trellis.size(0)
    ax2.specgram(waveform, Fs=sample_rate)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, facecolor="none", edgecolor="white", hatch="/")
        ax2.annotate(f"{word.score:.2f}", (x0, sample_rate * 0.51), annotation_clip=False)

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, sample_rate * 0.55), annotation_clip=False)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    fig.tight_layout()


plot_alignments(
    trellis,
    segments,
    word_segments,
    waveform[0],
)


################################################################################
# 오디오 샘플
# -------------
#


def display_segment(i):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=bundle.sample_rate)


######################################################################
#

# 각 분할에 해당하는 오디오 생성
print(transcript)
IPython.display.Audio(SPEECH_FILE)


######################################################################
#

display_segment(0)

######################################################################
#

display_segment(1)

######################################################################
#

display_segment(2)

######################################################################
#

display_segment(3)

######################################################################
#

display_segment(4)

######################################################################
#

display_segment(5)

######################################################################
#

display_segment(6)

######################################################################
#

display_segment(7)

######################################################################
#

display_segment(8)

######################################################################
# 결론
# ----------
#
# 이번 튜토리얼에서, torchaudio의 wav2vec2 모델을 사용하여 강제 정렬을 위한 CTC 분할을 수행하는 방법을 살펴보았습니다.
