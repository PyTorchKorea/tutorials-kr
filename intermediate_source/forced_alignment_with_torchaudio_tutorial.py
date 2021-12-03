"""
Wav2Vec2를 사용한 강제 정렬
==============================

**Author** `Moto Hira <moto@fb.com>`__

이 튜토리얼은 스크립트를 음성으로 정렬하는 방법을 보여줍니다.
`CTC-Segmentation of Large Corpora for German End-to-end Speech
Recognition <https://arxiv.org/abs/2007.09127>`__.
에 설명된 CTC 분할 알고리즘 사용-``torchaudio``

"""


######################################################################
# 개요
# --------
# 
# 정렬 과정은 다음과 같습니다.
# 
# 1. 오디오 파형에서 프레임별 레이블 확률 추정
# 2. 시간 단계에서 레이블이 정렬될 확률을 나타내는 격자 행렬을 생성합니다.
# 3. 격자 행렬에서 가장 가능성이 높은 경로를 찾습니다.
# 
# 이 예에서는 음향 특징 추출을 위해 ``torchaudio`` 의 ``Wav2Vec2`` 모델을 사용합니다.
# 


######################################################################
# 준비
# -----------
# 
# 먼저 필요한 패키지를 가져오고 작업할 데이터를 가져옵니다.
# 

# %matplotlib inline

import os
from dataclasses import dataclass

import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import IPython

matplotlib.rcParams['figure.figsize'] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(torchaudio.__version__)
print(device)

SPEECH_URL = 'https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
SPEECH_FILE = '_assets/speech.wav'

if not os.path.exists(SPEECH_FILE):
  os.makedirs('_assets', exist_ok=True)
  with open(SPEECH_FILE, 'wb') as file:
    file.write(requests.get(SPEECH_URL).content)

######################################################################
# 프레임별 레이블 확률 생성
# -------------------------------------
# 
# 첫 번째 단계는 각 Aduio 프레임의 레이블 클래스 이식성을 생성하는 것입니다. 
# ASR에 대해 훈련된 Wav2Vec2 모델을 사용할 수 있습니다. 
# 여기서 우리가 사용하는 것 :py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`.
# 
# ``torchaudio`` 는 관련 레이블이 있는 사전 훈련된 모델에 쉽게 액세스할 수 있도록 합니다.
# 
# .. note::
#
#    다음 섹션에서는 수치적 불안정성을 피하기 위해 로그 영역에서 확률을 계산할 것입니다. 
#    이를 위해 우리는 ``emission`` 을 :py:func:`torch.log_softmax`로 정규화합니다.
# 

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
  waveform, _ = torchaudio.load(SPEECH_FILE)
  emissions, _ = model(waveform.to(device))
  emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

################################################################################
# 시각화
################################################################################
print(labels)
plt.imshow(emission.T)
plt.colorbar()
plt.title("Frame-wise class probability")
plt.xlabel("Time")
plt.ylabel("Labels")
plt.show()


######################################################################
# 정렬 확률 생성(격자)
# ----------------------------------------
# 
# 방출 행렬에서 다음으로 각 시간 프레임에서 성적표 레이블이 발생할 확률을 나타내는 격자를 생성합니다.
# 
# 격자는 시간 축과 레이블 축이 있는 2D 행렬입니다. 레이블 축은 우리가 정렬하고 있는 성적표를 나타냅니다. 
# 다음에서 :math:`t` 를 사용하여 시간 축의 인덱스를 나타내고 :math:`j` 를 사용하여 레이블 축의 인덱스를 나타냅니다. 
# :math:`c_j` 는 레이블 인덱스 :math:`j` 의 레이블을 나타냅니다.
# 
# 시간 단계 :math:`t+1` 의 확률을 생성하기 위해 시간 단계 :math:`t` 의 격자와 시간 단계 :math:`t+1` 의 방출을 봅니다. 
# 레이블이 :math:`c_{j+1}` 인 시간 단계 :math:`t+1` 에 도달하는 두 가지 경로가 있습니다. 
# 첫 번째는 레이블이 :math:`c_{j+1}` at :math:`t` 이고 레이블이 :math:`t` 에서 :math:`t+1` 로 변경되지 않은 경우입니다.
# 다른 경우는 레이블이 :math:`c_j` 에서 :math:`t` 이고 다음 레이블인 :math:`c_{j+1}` 에서 :math:`t+1` 로 전환된 경우입니다.
# 
# 다음 다이어그램은 이러한 전환을 보여줍니다.
# 
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/ctc-forward.png
# 
# 가장 가능성 있는 전환을 찾고 있기 때문에 :math:`k_{(t+1, j+1)}` 값에 대해 더 가능성이 높은 경로를 선택합니다. 즉,
# 
# :math:`k_{(t+1, j+1)} = max( k_{(t, j)} p(t+1, c_{j+1}), k_{(t, j+1)} p(t+1, repeat) )`
# 
# 여기서 :math:`k` 는 격자 행렬을 나타내고, :math:`p(t, c_j)` 는 시간 단계 :math:`t` 에서 레이블 :math:`c_j` 의 확률을 나타냅니다. 
# :math:`repeat` 는 CTC 공식의 빈 토큰을 나타냅니다. 
# (CTC 알고리즘에 대한 자세한 내용은 *CTC를 사용한 시퀀스 모델링* [`distill.pub <https://distill.pub/2017/ctc/>`__] 참조)
# 

transcript = 'I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT'
dictionary  = {c: i for i, c in enumerate(labels)}

tokens = [dictionary[c] for c in transcript]
print(list(zip(transcript, tokens)))

def get_trellis(emission, tokens, blank_id=0):
  num_frame = emission.size(0)
  num_tokens = len(tokens)

  # Trellis에는 시간 축과 토큰 모두에 대한 추가 치수가 있습니다.
  # 토큰에 대한 추가 dim은 <SoS>(문장 시작)를 나타냅니다.
  # 시간 축에 대한 추가 dim은 코드를 단순화하기 위한 것입니다.
  trellis = torch.full((num_frame+1, num_tokens+1), -float('inf'))
  trellis[:, 0] = 0
  for t in range(num_frame):
    trellis[t+1, 1:] = torch.maximum(
        # Score for staying at the same token
        trellis[t, 1:] + emission[t, blank_id],
        # Score for changing to the next token
        trellis[t, :-1] + emission[t, tokens],
    )
  return trellis

trellis = get_trellis(emission, tokens)

################################################################################
# 시각화
################################################################################
plt.imshow(trellis[1:, 1:].T, origin='lower')
plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
plt.colorbar()
plt.show()

######################################################################
# 위의 시각화에서 우리는 행렬을 대각선으로 가로지르는 높은 확률의 흔적이 있음을 알 수 있습니다.
# 


######################################################################
# 가장 가능성 있는 경로 찾기(역추적)
# ----------------------------------------
# 
# 격자가 생성되면 높은 확률로 요소를 따라 이동합니다.
# 
# 확률이 가장 높은 시간 단계로 마지막 레이블 인덱스부터 시작합니다.
# 그 다음 우리는 시간을 거슬러 올라가, picking stay
# (:math:`c_j \rightarrow c_j` ) or transition
# (:math:`c_j \rightarrow c_{j+1}` ), 전환 후 확률에 기반하여
# :math:`k_{t, j} p(t+1, c_{j+1})` or
# :math:`k_{t, j+1} p(t+1, repeat)`.
# 
# 레이블이 시작 부분에 도달하면 전환이 완료됩니다.
# 
# trellis matrix는 path-finding에 사용되지만 각 
# segment의 최종 확률은 emission matrix에서 frame-wise 확률을 취한다.
# 

@dataclass
class Point:
  token_index: int
  time_index: int
  score: float


def backtrack(trellis, emission, tokens, blank_id=0):
  # Note:
  # j와 t는 처음에 시간과 토큰에 대한 추가 차원이 있는 격자에 대한 인덱스입니다.
  # 격자의 시간 프레임 인덱스 'T' 를 참조할 때 해당 방출 인덱스는 'T-1' 입니다.
  # 마찬가지로 격자에서 토큰 인덱스 'JS'를 참조할 때 transcript 해당 인덱스는 'J-1' 입니다.
  j = trellis.size(1) - 1
  t_start = torch.argmax(trellis[:, j]).item()

  path = []
  for t in range(t_start, 0, -1):
    # 1. Figure out if the current position was stay or change
    # Note (again):
    # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
    # Score for token staying the same from time frame J-1 to T.
    stayed = trellis[t-1, j] + emission[t-1, blank_id]
    # Score for token changing from C-1 at T-1 to J at T.
    changed = trellis[t-1, j-1] + emission[t-1, tokens[j-1]]

    # 2. Store the path with frame-wise probability.
    prob = emission[t-1, tokens[j-1] if changed > stayed else 0].exp().item()
    # Return token index and time index in non-trellis coordinate.
    path.append(Point(j-1, t-1, prob))

    # 3. Update the token
    if changed > stayed:
      j -= 1
      if j == 0:
        break
  else:
    raise ValueError('Failed to align')
  return path[::-1]

path = backtrack(trellis, emission, tokens)
print(path)

################################################################################
# 시각화
################################################################################
def plot_trellis_with_path(trellis, path):
  # To plot trellis with path, we take advantage of 'nan' value
  trellis_with_path = trellis.clone()
  for i, p in enumerate(path):
    trellis_with_path[p.time_index, p.token_index] = float('nan')
  plt.imshow(trellis_with_path[1:, 1:].T, origin='lower')

plot_trellis_with_path(trellis, path)
plt.title("The path found by backtracking")
plt.show()

######################################################################
# 좋습니다. 이제 이 경로에는 동일한 레이블에 대한 반복이 포함되어 있으므로 
# 이를 병합하여 원본 대본에 가깝게 만들겠습니다.
# 
# 여러 경로 포인트를 병합할 때 병합된 segment에 대한 평균 확률을 취합니다.
# 

# 레이블 병합
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
    segments.append(Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2-1].time_index + 1, score))
    i1 = i2
  return segments

segments = merge_repeats(path)
for seg in segments:
  print(seg)

################################################################################
# 
################################################################################
def plot_trellis_with_segments(trellis, segments, transcript):
  # To plot trellis with path, we take advantage of 'nan' value
  trellis_with_path = trellis.clone()
  for i, seg in enumerate(segments):
    if seg.label != '|':
      trellis_with_path[seg.start+1:seg.end+1, i+1] = float('nan')

  fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
  ax1.set_title("Path, label and probability for each label")
  ax1.imshow(trellis_with_path.T, origin='lower')
  ax1.set_xticks([])

  for i, seg in enumerate(segments):
    if seg.label != '|':
      ax1.annotate(seg.label, (seg.start + .7, i + 0.3), weight='bold')
      ax1.annotate(f'{seg.score:.2f}', (seg.start - .3, i + 4.3))

  ax2.set_title("Label probability with and without repetation")
  xs, hs, ws = [], [], []
  for seg in segments:
    if seg.label != '|':
      xs.append((seg.end + seg.start) / 2 + .4)
      hs.append(seg.score)
      ws.append(seg.end - seg.start)
      ax2.annotate(seg.label, (seg.start + .8, -0.07), weight='bold')
  ax2.bar(xs, hs, width=ws, color='gray', alpha=0.5, edgecolor='black')

  xs, hs = [], []
  for p in path:
    label = transcript[p.token_index]
    if label != '|':
      xs.append(p.time_index + 1)
      hs.append(p.score)
  
  ax2.bar(xs, hs, width=0.5, alpha=0.5)
  ax2.axhline(0, color='black')
  ax2.set_xlim(ax1.get_xlim())
  ax2.set_ylim(-0.1, 1.1)

plot_trellis_with_segments(trellis, segments, transcript)
plt.tight_layout()
plt.show()


######################################################################
# 이제 단어를 병합해 보겠습니다. 
# Wav2Vec2 모델은 단어 경계로 사용하므로 각 발생 전에 segment를 병합합니다.
# ``'|'`` . ``'|'``
# 
# 그런 다음 마지막으로 원본 오디오를 분할된 오디오로 분할하고 듣고 분할이 올바른지 확인합니다.
# 

# Merge words
def merge_words(segments, separator='|'):
  words = []
  i1, i2 = 0, 0
  while i1 < len(segments):
    if i2 >= len(segments) or segments[i2].label == separator:
      if i1 != i2:
        segs = segments[i1:i2]
        word = ''.join([seg.label for seg in segs])
        score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
        words.append(Segment(word, segments[i1].start, segments[i2-1].end, score))
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
################################################################################
def plot_alignments(trellis, segments, word_segments, waveform):
  trellis_with_path = trellis.clone()
  for i, seg in enumerate(segments):
    if seg.label != '|':
      trellis_with_path[seg.start+1:seg.end+1, i+1] = float('nan')

  fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))

  ax1.imshow(trellis_with_path[1:, 1:].T, origin='lower')
  ax1.set_xticks([])
  ax1.set_yticks([])

  for word in word_segments:
    ax1.axvline(word.start - 0.5)
    ax1.axvline(word.end - 0.5)

  for i, seg in enumerate(segments):
    if seg.label != '|':
      ax1.annotate(seg.label, (seg.start, i + 0.3))
      ax1.annotate(f'{seg.score:.2f}', (seg.start , i + 4), fontsize=8)

  # 원래 파형
  ratio = waveform.size(0) / (trellis.size(0) - 1)
  ax2.plot(waveform)
  for word in word_segments:
    x0 = ratio * word.start
    x1 = ratio * word.end
    ax2.axvspan(x0, x1, alpha=0.1, color='red')
    ax2.annotate(f'{word.score:.2f}', (x0, 0.8))

  for seg in segments:
    if seg.label != '|':
      ax2.annotate(seg.label, (seg.start * ratio, 0.9))
  xticks = ax2.get_xticks()
  plt.xticks(xticks, xticks / bundle.sample_rate)
  ax2.set_xlabel('time [second]')
  ax2.set_yticks([])
  ax2.set_ylim(-1.0, 1.0)
  ax2.set_xlim(0, waveform.size(-1))

plot_alignments(trellis, segments, word_segments, waveform[0],)
plt.show()

# 생성된 파일에 결과 오디오를 포함하는 트릭입니다.
# 'IPython.display.Audio' 는 셀의 마지막 호출이어야 하며,
# 그리고 오직 하나의 call par cell만 있어야 합니다.
def display_segment(i):
  ratio = waveform.size(1) / (trellis.size(0) - 1)
  word = word_segments[i]
  x0 = int(ratio * word.start)
  x1 = int(ratio * word.end)
  filename = f"_assets/{i}_{word.label}.wav"
  torchaudio.save(filename, waveform[:, x0:x1], bundle.sample_rate)
  print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
  return IPython.display.Audio(filename)

######################################################################
# 

# 각 segment에 대한 오디오 생성
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
# 이 튜토리얼에서 우리는 토치오디오의 Wav2Vec2 모델을 사용하여 
# 강제 정렬을 위한 CTC 분할을 수행하는 방법을 살펴보았습니다.
# 
