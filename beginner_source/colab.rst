Google Colab에서 튜토리얼 실행하기
=====================================

Google Colab에서 튜토리얼을 실행할 때, 튜토리얼이 제대로 동작하기
위해서 충족해야 하는 추가적인 추가 요구 사항과 종속성(dependancy)이
있을 수 있습니다. 이 섹션에서는 Google Colab에서 파이토치(PyTorch) 튜토리얼을
성공적으로 실행하기 위해 다양한 설정을 구성하는 방법에 대해 설명합니다.

Google Colab의 PyTorch 버전
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

공개(release)된지 얼마되지 않은 PyTorch 버전을 사용하는 튜토리얼을 실행하는 경우,
해당 버전이 아직 Google Colab에 반영되지 않았을 수 있습니다.
필요한 ``torch`` 와 호환되는 도메인 라이브러리(domain library)가 설치되어 있는지
확인하려면 ``!pip list`` 를 실행하세요.

만약 필요한 PyTorch 버전보다 낮은 버전이 설치되어 있는 경우,
다음 명령어를 실행하여 제거한 뒤, 다시 설치하세요:

.. code-block:: python
   !pip3 uninstall --yes torch torchaudio torchvision torchtext torchdata
   !pip3 install torch torchaudio torchvision torchtext torchdata


Colab에서 Google Drive의 튜토리얼 데이터 사용하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

사용자가 Google Colab에서 튜토리얼과 관련된 노트북을 열 수 있도록 하는 새로운
기능이 튜토리얼에 추가되었습니다. 이 때, 보다 복잡한 튜토리얼을 실행하려면
사용자의 Google Drive 계정에 데이터를 복사해야 할 수도 있습니다.

이 예제에서는 챗봇(Chatbot) 튜토리얼을 Colab에서 동작하도록 변경하는 방법을
설명하겠습니다. 이를 위해서, 먼저 Google Drive에 로그인이 되어 있어야 합니다.
(Colab에서 데이터에 접근하는 방법에 대한 자세한 설명은
`여기 <https://colab.research.google.com/notebooks/io.ipynb#scrollTo=XDg9OBaYqRMd>`__
에서 예제 노트북을 통해 볼 수 있습니다.)

시작하기 전에 `챗봇 튜토리얼 <https://tutorials.pytorch.kr/beginner/chatbot_tutorial.html>`__
을 브라우저에 띄워주세요.

페이지 상단에 **Run in Google Colab** 을 클릭합니다.

Colab에서 파일이 열리게 됩니다.

**Runtime** 을 선택한 뒤 **Run All** 을 선택하면 파일을 찾을 수 없다(the file can't be found)는
에러가 발생합니다.

이를 해결하기 위해, 필요한 파일들을 Google Drive에 복사하겠습니다.

1. Google Drive에 로그인합니다.
2. Google Drive에서 **data** 라는 이름의 폴더 및 이 아래에 **cornell** 라는 하위
   폴더도 생성합니다.
3. Cornell Movie Dialogs Corpus에 방문하여 movie-corpus ZIP 파일을 내려받습니다.
4. 로컬 머신에 압축을 풉니다.
5. **utterances.jsonl** 파일을 Google Drive에 생성한 **data/cornell** 폴더 안에 복사합니다.

이제 Google Drive 상의 파일을 가르키도록 Colab의 파일을 편집해야 합니다.

Colab에서 *corpus\_name* 으로 시작하는 코드 섹션의 윗 부분에 다음 내용을 추가합니다:

::

    from google.colab import drive
    drive.mount('/content/gdrive')


이제 다음과 같이 2줄을 변경하세요:

1. **corpus\_name** 값을 **"cornell"** 로 변경합니다.
2. **corpus** 로 시작하는 줄을 아래처럼 변경합니다:

::

    corpus = os.path.join("/content/gdrive/My Drive/data", corpus_name)

이제 Google Drive에 업로드한 파일을 가리키고 있습니다.

이제 코드 섹션의 **Run cell** 버튼을 클릭하게 되면 Google Drive에 인증하라는
메시지가 표시되고 인증 코드를 받게 됩니다. 인증 코드를 Colab에 붙여넣으면
설정이 됩니다.

노트북의 **Runtime** / **Run All** 메뉴 명령을 다시 실행하면, 진행 상황을 볼 수
있습니다. (챗봇 튜토리얼은 실행하는데 시간이 오래 걸리니 참고하세요.)

이 예제가 Coalb에서 보다 복잡한 튜토리얼을 실행하는데 있어서 좋은 시작점이 되길
바랍니다. PyTorch 튜토리얼 사이트에서 Colab을 더 활용하여 사용자들이 더 쉽게
사용할 수 있는 방법을 찾아보겠습니다.
