# PyTorch 한국어 튜토리얼 기여하기

PyTorch 한국어 튜토리얼 저장소에 방문해주셔서 감사합니다. 이 문서는 PyTorch 한국어 튜토리얼에 기여하는 방법을 안내합니다.

## 기여하기 개요

[본 저장소](https://github.com/9bow/PyTorch-tutorials-kr)는 [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)을 번역하는 프로젝트를 위한 곳으로,
[공식 튜토리얼 저장소](https://github.com/pytorch/tutorials)의 내용을 비정기적으로 반영하고, 번역 및 개선합니다.

크게 다음과 같은 기여 방법이 있습니다.

* [1. 오탈자를 수정하거나 번역을 개선하는 기여](#1-오탈자를-수정하거나-번역을-개선하는-기여)
  * [한국어 튜토리얼 사이트](http://tutorials.pytorch.kr/)에서 발견한 오탈자를 [한국어 튜토리얼 저장소](https://github.com/9bow/PyTorch-tutorials-kr)에서 고치는 기여입니다.
* [2. 번역되지 않은 튜토리얼을 번역하는 기여](#2-번역되지-않은-튜토리얼을-번역하는-기여)
  * [한국어 튜토리얼 사이트](http://tutorials.pytorch.kr/)에 아직 번역되지 않은 튜토리얼 번역하는 기여입니다.
* [3. 2로 번역된 문서를 리뷰하는 기여](#3-2로-번역된-문서를-리뷰하는-기여) :star:
  * [본 저장소에 Pull Request된 튜토리얼 문서](https://github.com/9bow/PyTorch-tutorials-kr/pulls)를 리뷰하는 기여입니다.
* ~[4. 공식 튜토리얼 내용을 반영하는 기여](#4-공식-튜토리얼-내용을-반영하는-기여)~
  * ~[공식 튜토리얼 저장소](https://github.com/pytorch/tutorials)의 내용을 [한국어 튜토리얼 저장소](https://github.com/9bow/PyTorch-tutorials-kr)에 반영하는 기여입니다.~

## 기여 결과물의 라이선스 동의

PyTorch 한국어 튜토리얼은 [공식 튜토리얼 저장소](https://github.com/pytorch/tutorials)와 동일한 [BSD 3항 라이선스](https://github.com/9bow/PyTorch-tutorials-kr/blob/master/LICENSE)를 따릅니다. \
따라서 기여하신 모든 내용에 [BSD 3항 라이선스](https://github.com/9bow/PyTorch-tutorials-kr/blob/master/LICENSE)가 적용됨을 인지하시고 동의하시는 경우에만 아래 문서 내용과 같이 기여해주세요.

## 기여하기 절차

모든 기여는 [본 저장소에 이슈](https://github.com/9bow/PyTorch-tutorials-kr/issues/new)를 남긴 후 [Pull Request를 보내는 것](https://github.com/9bow/PyTorch-tutorials-kr/pulls)으로 합니다. \
이 과정을 통해 Pull Request를 위한 Commit을 만들기 전에 이슈를 통해 해당 내용에 기여가 필요한지 여부를 확인하고 협의하셔야 합니다. \
(물론 이슈를 남기셨다고 해서 반드시 해당 문제를 개선하셔야 하는 것은 아니니, 마음 편히 이슈를 남겨주세요. :))

## (기여 종류에 따른) 기여 방법

### 1. 오탈자를 수정하거나 번역을 개선하는 기여

[한국어 튜토리얼 사이트](http://tutorials.pytorch.kr/)에서 발견한 오탈자를 고치는 기여 방법입니다.

#### 1-1. 이슈 남기기

(매우 낮은 확률로) 해당 오탈자가 의도한 것일 수 있으니, 해당 문제점을 고친 Pull Request를 생성하기 전에 [본 저장소 이슈](https://github.com/9bow/PyTorch-tutorials-kr/issues)를 검색하거나 새로 남겨주세요.

해당 문제점에 대한 개선 사항이 **이미 논의되었거나 진행 중인 Pull Request를 통해 해결 중일 수 있으니, 새로 이슈를 만드시기 전, 먼저 검색**을 해주시기를 부탁드립니다. \
이후, 새로 남겨주신 이슈에서 저장소 관리자 및 다른 방문자들이 함께 문제점에 대해 토의하실 수 있습니다. (또는 이미 관련 이슈가 존재하지만 해결 중이지 않은 경우에는 덧글을 통해 기여를 시작함을 알려주세요.)

#### 1-2. 저장소 복제하기

오탈자를 수정하기 위해 저장소를 복제합니다. \
저장소 복제가 처음이시라면 [GitHub의 저장소 복제 관련 도움말](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)을 참조해주세요.

#### 1-3. 원본 경로 / 문서 찾기

[한국어 튜토리얼 사이트의 소스 코드](https://github.com/9bow/PyTorch-tutorials-kr/tree/master/docs)는 튜토리얼 원본 문서를 빌드한 결과물입니다. \
따라서 오탈자 수정을 위해서는 [한국어 튜토리얼 사이트](http://tutorials.pytorch.kr/)에서 튜토리얼 원본의 경로와 문서명을 확인하고 수정해야 합니다. \
예를 들어 튜토리얼의 주소가 `https://tutorials.pytorch.kr/beginner/deep_learning_60min_blitz.html`라면, 튜토리얼 경로는 `beginner`이고, 문서명은 `deep_learning_60min_blitz`입니다. \
해당 문서의 원본은 [한국어 튜토리얼 저장소](https://github.com/9bow/PyTorch-tutorials-kr)에 `경로명_source` 경로에 `문서명.rst` 또는 `문서명.py` 파일입니다. \
(위 예시 경우 원본 문서는 `beginner_source` 경로의 `deep_learning_60min_blitz.rst` 파일입니다.)

#### 1-4. 오탈자 수정하기

위에서 찾은 원본 튜토리얼 문서를 [reStructuredText 문법](http://docutils.sourceforge.net/docs/user/rst/quickref.html)에 맞춰 수정합니다. \
reStructuredText 문법에 익숙하지 않은 경우, 다른 튜토리얼의 원본 문서와 빌드 결과물(HTML)을 비교해보면서 빌드 결과물을 예상할 수 있습니다.

#### 1-5. (내 컴퓨터에서) 결과 확인하기

저장소의 최상위 경로에서 `make html-noplot` 명령어를 이용하면 코드 실행 없이 reStructuredText 문서들의 HTML 빌드 결과물을 빠르게 확인하실 수 있습니다. \
이 과정에서 수정한 문서 상에서 발생하는 오류가 있다면 [reStructuredText 문법](http://docutils.sourceforge.net/docs/user/rst/quickref.html)을 참고하여 올바르게 고쳐주세요. \
빌드 결과물은 `_build/html` 디렉토리 아래의 경로 / 문서에서 확인하실 수 있습니다.

#### 1-6. Pull Request 남기기

수정을 완료한 내용을 복제한 저장소에 Commit 및 Push하고, Pull Request를 남깁니다. \
Pull Request가 처음이시라면 [GitHub의 Pull Request 소개 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) 및 [복제한 저장소로부터 Pull Request 만들기 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)을 참조해주세요.

### 2. 번역되지 않은 튜토리얼을 번역하는 기여

[한국어 튜토리얼 사이트](http://tutorials.pytorch.kr/)에 아직 번역되지 않은 튜토리얼을 번역하는 기여 방법입니다.

#### 2-1. 이슈 남기기

(매우 낮은 확률로) 해당 튜토리얼이 번역 중일 수 있으니, 번역 전에 Pull Request를 생성하기 전에 [본 저장소 이슈](https://github.com/9bow/PyTorch-tutorials-kr/issues)를 검색하거나 새로 남겨주세요.

해당 튜토리얼에 대한 **번역이 이미 논의되었거나 Pull Request를 통해 진행 중일 수 있으니, 새로 이슈를 만드시기 전, 먼저 검색**을 해주시기를 부탁드립니다. \
이후, 새로 남겨주신 이슈에서 저장소 관리자 및 다른 방문자들이 함께 번역 진행에 대해 토의하실 수 있습니다. \
(또는 이미 관련 이슈가 존재하지만 번역 중이지 않은 것처럼 보이는 경우에는 덧글을 통해 기여를 시작함을 알려주세요.)

#### 2-2. 저장소 복제하기

신규 튜토리얼을 번역하기 위해 저장소를 복제합니다. \
저장소 복제가 처음이시라면 [GitHub의 저장소 복제 관련 도움말](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)을 참조해주세요.

#### 2-3. 원본 경로 / 문서 찾기

[한국어 튜토리얼 사이트의 소스 코드](https://github.com/9bow/PyTorch-tutorials-kr/tree/master/docs)는 튜토리얼 원본 문서를 빌드한 결과물입니다. \
따라서 튜토리얼 번역을 위해서는 [한국어 튜토리얼 사이트](http://tutorials.pytorch.kr/)에서 튜토리얼 원본의 경로와 문서명을 확인하고 번역해야 합니다. \
예를 들어 튜토리얼의 주소가 `https://tutorials.pytorch.kr/beginner/deep_learning_60min_blitz.html`라면, 튜토리얼 경로는 `beginner`이고, 문서명은 `deep_learning_60min_blitz`입니다. \
해당 문서의 원본은 [한국어 튜토리얼 저장소](https://github.com/9bow/PyTorch-tutorials-kr)에 `경로명_source` 경로에 `문서명.rst` 또는 `문서명.py` 파일입니다. \
(위 예시 경우 원본 문서는 `beginner_source` 경로의 `deep_learning_60min_blitz.rst` 파일입니다.)

#### 2-4. 튜토리얼 번역하기

위에서 찾은 원본 튜토리얼 문서를 [reStructuredText 문법](http://docutils.sourceforge.net/docs/user/rst/quickref.html)에 맞춰 번역합니다. \
번역 중 번역 용어에 대해서는 다른 튜토리얼을 참조하시거나, `2-1`에서 남긴 이슈의 덧글을 통해 토의하실 수 있습니다. \
reStructuredText 문법에 익숙하지 않은 경우, 다른 튜토리얼의 원본 문서와 빌드 결과물(HTML)을 비교해보면서 빌드 결과물을 예상할 수 있습니다.

#### 2-5. (내 컴퓨터에서) 결과 확인하기

저장소의 최상위 경로에서 `make html-noplot` 명령어를 이용하면 코드 실행 없이 reStructuredText 문서들의 HTML 빌드 결과물을 빠르게 확인하실 수 있습니다. \
이 과정에서 수정한 문서 상에서 발생하는 오류가 있다면 [reStructuredText 문법](http://docutils.sourceforge.net/docs/user/rst/quickref.html)을 참고하여 올바르게 고쳐주세요. \
빌드 결과물은 `_build/html` 디렉토리 아래의 경로 / 문서에서 확인하실 수 있습니다.

#### 2-6. Pull Request 남기기

번역을 완료한 내용을 복제한 저장소에 Commit 및 Push하고, Pull Request를 남깁니다. \
Pull Request가 처음이시라면 [GitHub의 Pull Request 소개 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) 및 [복제한 저장소로부터 Pull Request 만들기 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)을 참조해주세요.

### 3. 2로 번역된 문서를 리뷰하는 기여

[본 저장소에 Pull Request된 튜토리얼 문서](https://github.com/9bow/PyTorch-tutorials-kr/pulls)를 리뷰하는 기여입니다.

Pull Request된 문서의 오탈자 수정, reStructuredText 문법 오류 또는 잘못 번역된 내용을 개선하는 기여로, 가장 기다리고 있는 기여 방식입니다. :pray: \
PR 리뷰가 익숙하지 않으시다면 [GitHub의 Pull Request 리뷰 관련 도움말](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-request-reviews)을 참조해주세요.

### 4. 공식 튜토리얼 내용을 반영하는 기여

[공식 튜토리얼 저장소](https://github.com/pytorch/tutorials)의 변경 내용을 [한국어 튜토리얼 저장소](https://github.com/9bow/PyTorch-tutorials-kr)에 반영하는 기여입니다. \
이 때에는 한국어 튜토리얼 저장소에 반영된 마지막 Commit ID와 공식 튜토리얼 저장소의 최선 Commit ID 사이의 변경 사항들을 [일괄적으로 확인](https://help.github.com/en/github/committing-changes-to-your-project/comparing-commits-across-time)하고 반영해야 합니다. \
이 종류의 기여는 **저장소 관리자가 일괄적으로 진행하며, 아직 방문하신 분들의 기여를 받고 있지는 않습니다.**
