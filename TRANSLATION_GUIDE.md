# 일반 규칙

* 번역된 문서만으로도 내용을 이해할 수 있도록 문서를 번역해야 합니다.
  * 기계적인 번역이나 피상적인 리뷰는 지양해주세요.
  * 일반 명사와 Class 이름은 구분하여 번역을 하거나 원문을 표기합니다. (예. 데이터셋과 Dataset)

* 반드시 1:1로 번역하지 않아도 됩니다.
  * 이해를 돕기 위한 (약간의) 의역이나 설명을 추가해도 좋습니다.
    * 단, 원문의 의미가 다르게 해석될 여지가 있는 경우에는 자제해주세요.
  * 문장 단위는 쉬운 유지보수를 위해 가급적 지켜주시기를 요청드립니다.
    * 하지만 문장이 여러 줄에 걸쳐 조각나 있는 경우 등에는 한 줄에 하나의 문장으로 모아주셔도 됩니다.

* 의미없는 주어는 생략해주세요.
  * 예를 들어, `we`는 강조의 의미가 있지 않는 이상 번역하지 않고 생략합니다.

* 기본적인 reStructuredText 문법은 숙지해주세요.
  * [Quick reStructredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html) 등의 문서를 참고하여 문법을 익혀주세요.
  * 이미 번역된 문서들을 참고하셔도 좋습니다. (예. \` 뒤에 한글 작성 시 공백 또는 \\이 필요합니다.)
  * 번역 후에는 `make html-noplot` 등의 명령어로 문법 오류를 확인해주세요.
    * 번역 결과물에 \`, * 또는 _ 등의 기호를 검색하면 자주 실수하는 문법 오류를 쉽게 찾을 수 있습니다.

* 번역된 문장만으로 의미를 전달하기 어려울 때에는 `한글(영어)`와 같이 작성합니다.
  * 제목과 본문에 각각 사용되는 경우 첫번째로 해당 용어가 출현하였을 때 매번 함께 작성합니다.
    * 예. including transposing, indexing, ... => 전치(transposing), 인덱싱(indexing), ...
  * 가급적 한 번씩만 함께 작성하는 것을 원칙으로 하지만 번역자가 임의로 여러번 함께 작성할 수 있습니다.
    * 예. 직렬화(Serialize)

* 소스 코드, 논문 제목, 출력 결과 등은 가급적 번역하지 않습니다.
  * 단, 소스 코드에 포함된 주석은 가급적 번역합니다.
  * 원문을 함께 찾아볼 필요가 있는 논문 제목 등은 번역 시 전체 원문을 함께 작성합니다.
  * 명령어의 출력 결과, 로그(log) 등은 결과 비교를 위해 번역하지 않습니다.

* 줄바꿈 및 공백은 가급적 원문과 동일하게 유지합니다.
  * 이후 원본 문서에 추가적인 변경이 발생할 때 유지보수를 돕습니다.
  * 너무 긴 문장은 reStructuredText 문법을 지키는 선에서 줄바꿈을 추가해도 좋습니다.

# 용어 사용 규칙

1. 아래 용어가 적절하면 가급적 아래 표의 용어를 사용합니다.
1. 지정된 용어가 없다면 아래 사이트를 참고하여 사용합니다.
  * http://www.ktword.co.kr/
  * https://github.com/keunwoochoi/machine_learning_eng2kor/blob/master/dictionary.md
1. 적절한 용어가 없으면 적절한 단어를 새로 사용하고, 아래 목록에 내용을 추가합니다.

|영문|한글|작성자|추가 설명|
|---|---|:---:|---|
|Acknowledgements|감사의 말|박정환||
|API endpoint|API 엔드포인트|박정환|음차 표기|
|argument|인자|박정환||
|Audio|오디오|박정환|ToC의 분류명입니다.|
|augmentation|증강|이재복||
|autograd|Autograd|황성수|번역안함|
|Batch Normalization|배치 정규화|박정환||
|bias|편향|이하람||
|convolution|합성곱|김현길||
|Dropout|드롭아웃|김태형|음차 표기|
|dataset|데이터셋|박정환|음차 표기|
|deep neural network|심층 신경망|박정환||
|derivative|도함수|박정환||
|Drop-out|Drop-out|황성수|번역안함|
|epoch|에폭|박정환|음차 표기|
|evaluation mode|평가 모드|박정환||
|feature|특징|백선희||
|feed data through model|데이터를 모델에 제공|||
|Feed-forward network|순전파 신경망|박정환||
|Generative|생성 모델|박정환|ToC의 분류명입니다.|
|Getting Started tutorial|시작하기 튜토리얼|박정환|ToC의 Getting Started를 뜻합니다.|
|gradient|변화도|박정환||
|Hyperparameter|하이퍼파라미터|김태영|음차 표기|
|Image|이미지|박정환|ToC의 분류명입니다.|
|in-place|제자리|허남규||
|instance|인스턴스|박정환|음차 표기|
|instantiate|생성하다|박정환||
|interpreter|인터프리터|이종법|음차 표기|
|Layer|계층|박정환||
|learning rate, lr|학습률|박정환||
|loss|손실|박정환||
|matrix|행렬|박정환||
|mean-squared error|평균제곱오차|허남규||
|MelScale|MelScale|||
|memory footprint|메모리 전체 사용량|최흥준|
|method|메소드|장효영|[음차 표기](https://terms.tta.or.kr/dictionary/dictionaryView.do?word_seq=090780-1)|
|mini-batch|미니 배치|박정환|음차 표기|
|momentum|모멘텀|박정환|음차 표기|
|normalize|정규화|허남규||
|NumPy|NumPy|박정환|번역하지 않음|
|One-Hot|One-Hot|황성수|번역안함|
|Optimizer|옵티마이저|박정환|음차 표기|
|output|출력|박정환||
|over-fitting|과적합|황성수||
|parameter|매개변수|박정환||
|placeholder|플레이스홀더|박정환|음차 표기|
|plotting|도식화|황성수||
|Production (environment, use)|Production|허남규|번역하지 않음|
|rank 0|0-순위|박정환||
|Read later|더 읽을거리|박정환||
|recap|요약|박정환||
|resample|리샘플|||
|resizing|크기 변경|박정환||
|requirements|요구 사항|장보윤||
|sampling rate|샘플링 레이트|||
|scenario|시나리오|박정환|음차 표기|
|shape|shape|허남규|번역하지 않음|
|size|크기|박정환||
|Tensor / Tensors|Tensor|박정환|번역하지 않음|
|Text|텍스트|박정환|ToC의 분류명입니다.|
|track (computation) history|연산 기록을 추적하다|박정환||
|training|학습|이하람||
|warmstart|빠르게 시작하기|박정환|Warmstarting Model = 빠르게 모델 시작하기|
|weight|가중치|박정환||
|wrapper|래퍼|박정환|음차 표기|
