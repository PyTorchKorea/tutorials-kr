PyTorch 사전 빌드된 라이브러리를 사용하는 네이티브 Android 애플리케이션 만들기
==============================================================================

**저자**: `Ivan Kobzarev <https://github.com/IvanKobzarev>`_

**번역**: `김현길 <https://github.com/des00>`_,  `Ajin Jeong <https://github.com/ajin-jng>`_

이 레시피에서 배울 내용은:

 - 네이티브 코드 (C++) 에서 LibTorch API를 사용하여 Android 애플리케이션을 만드는 방법.

 - 이 애플리케이션에서 사용자 정의 연산자로 TorchScript 모델을 사용하는 방법.

이 앱의 전체 설정은 `PyTorch Android Demo Application Repository <https://github.com/pytorch/android-demo-app/tree/master/NativeApp>`_ 에서 찾을 수 있습니다.

설정
~~~~~

다음과 같은 패키지(및 종속성)가 설치된 Python 3 환경이 필요합니다:

- PyTorch 1.6

안드로이드 개발을 위해 설치할 것들:

- Android NDK

::

  wget https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
  unzip android-ndk-r19c-linux-x86_64.zip
  export ANDROID_NDK=$(pwd)/android-ndk-r19c


- Android SDK

::

  wget https://dl.google.com/android/repository/sdk-tools-linux-3859397.zip
  unzip sdk-tools-linux-3859397.zip -d android_sdk
  export ANDROID_HOME=$(pwd)/android_sdk



- Gradle 4.10.3

Gradle은 Android 애플리케이션을 위해 가장 많이 사용되는 빌드 시스템이며, 우리가 만들 애플리케이션을 빌드하기 위해 필요합니다. ``Gradle`` 을 command line에서 사용하기 위해 다운로드하고 path에 추가하십시오.

.. code-block:: shell

  wget https://services.gradle.org/distributions/gradle-4.10.3-bin.zip
  unzip gradle-4.10.3-bin.zip
  export GRADLE_HOME=$(pwd)/gradle-4.10.3
  export PATH="${GRADLE_HOME}/bin/:${PATH}"

- JDK

Gradle은 JDK가 필요하기에, JDK를 설치하고 환경 변수 ``JAVA_HOME`` 을 설정하여 JDK 위치를 가리키도록 합니다.
예를 들어 OpenJDK를 설치한다면 `이 설명 <https://openjdk.java.net/install/>`_ 을 따릅니다.

- Android용 OpenCV SDK

사용자 지정 연산자는 OpenCV 라이브러리를 사용하여 구현합니다. OpenCV를 Android를 위해 사용하려면 사전 빌드된 Android용 OpenCV 라이브러리를 다운로드해야 됩니다.
`OpenCV releases page <https://opencv.org/releases/>`_ 에서 다운로드합니다. 압축을 풀고 환경 변수 ``OPENCV_ANDROID_SDK`` 를 설정합니다.


사용자 지정 C++ 연산자로 TorchScript Model 준비하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchScript는 사용자 정의 C++ 연산자를 사용하는 것을 허용하며, 사용자 정의 연산 관련된 세부사항은
`the dedicated tutorial <https://tutorials.pytorch.kr/advanced/torch_script_custom_ops.html>`_ 여기에서 읽을 수 있습니다.

결과적으로 사용자 지정 연산자를 사용하는 모델을 스크립팅 할 수 있습니다. 이 사용자 지정 연산자는 OpenCV의 ``cv::warpPerspective`` 함수를 사용합니다.

.. code-block:: python

  import torch
  import torch.utils.cpp_extension

  print(torch.version.__version__)
  op_source = """
  #include <opencv2/opencv.hpp>
  #include <torch/script.h>

  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data_ptr<float>());
    cv::Mat warp_mat(/*rows=*/warp.size(0),
                     /*cols=*/warp.size(1),
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data_ptr<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{64, 64});

    torch::Tensor output =
      torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{64, 64});
    return output.clone();
  }

  static auto registry =
    torch::RegisterOperators("my_ops::warp_perspective", &warp_perspective);
  """

  torch.utils.cpp_extension.load_inline(
      name="warp_perspective",
      cpp_sources=op_source,
      extra_ldflags=["-lopencv_core", "-lopencv_imgproc"],
      is_python_module=False,
      verbose=True,
  )

  print(torch.ops.my_ops.warp_perspective)


  @torch.jit.script
  def compute(x, y):
      if bool(x[0][0] == 42):
          z = 5
      else:
          z = 10
      x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
      return x.matmul(y) + z


  compute.save("compute.pt")


이 코드 조각은  ``compute.pt`` 파일을 생성합니다. 이 파일은 사용자 지정 연산자인  ``my_ops.warp_perspective`` 을 사용하는 TorchScript 모델입니다.

실행하려면 개발용 OpenCV를 설치해야 합니다.
리눅스 시스템은 다음 명령어를 통해 설치할 수 있습니다:
CentOS:

.. code-block:: shell

  yum install opencv-devel

Ubuntu:

.. code-block:: shell

  apt-get install libopencv-dev

Android 애플리케이션 만들기
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``compute.pt`` 를 만들었으면 Android 애플리케이션 내에서 이 TorchScript 모델을 사용하겠습니다. Java API를 이용해서 Android상에서 일반적인 TorchScript 모델(사용자 지정 연산자 없이)을 사용하고자 한다면 `여기 <https://pytorch.org/mobile/android/>`_ 를 살펴 보십시오. 이 예제에서는 사용자 지정 연산자(``my_ops.warp_perspective``)를 사용해서 위와 같은 방밥을 사용할 수 없습니다. 기본 TorchScript 실행이 이 사용자 지정 연산자를 찾지 못하기 때문입니다.

연산자 동륵은 PyTorch Java API에 노출이 되지 않기에, Android 애플리케이션 네이티브 부분(C++)을 빌드하고, Android용 동일한 사용자 지정 연산자를 LibTorch C++ API를 이용해서 구현하고 등록해야 합니다. 연산자가 OpenCV 라이브러리를 사용하기에, 사전 빌드된 OpenCV Android 라이브러리와 OpenCV의 동일한 함수를 이용합니다.

Android 애플리케이션을 ``NativeApp`` 폴더 내에서 생성해 봅시다.

.. code-block:: shell

  mkdir NativeApp
  cd NativeApp

Android 애플리케이션 빌드 설정하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Android 애플리케이션 빌드는 메인 gradle 부분과 네이티브 빌드 Cmake 부분으로 이루어집니다.
여기 나열된 목록은 전체 파일 목록입니다. 그래서 전체 구조를 새로이 만들고자 한다면
코드를 별도로 추가하지 않아도 결과로 나온 Android 애플리케이션을 빌드하고 설치할 수 있습니다.

Gradle 빌드 설정하기
-------------------
이러한 gradle 설정 파일울 추가해야 합니다: build.gradle, gradle.properties, settings.gradle.
추가적인 Android Gradle 빌드 설정은 `여기 <https://developer.android.com/studio/build>`_ 에서 찾을 수 있습니다.

``NativeApp/settings.gradle``

.. code-block:: gradle

  include ':app'


``NativeApp/gradle.properties``

.. code-block:: gradle

  android.useAndroidX=true
  android.enableJetifier=true


``NativeApp/build.gradle``

.. code-block:: gradle

  buildscript {
      repositories {
          google()
          jcenter()
      }
      dependencies {
          classpath 'com.android.tools.build:gradle:3.5.0'
      }
  }

  allprojects {
      repositories {
          google()
          jcenter()
      }
  }


``NativeApp/build.gradle`` 안에서 Android gradle 플러그인 버전을 `3.5.0`으로 명기합니다. 이 버전이 최신 버전은 아닙니다. 그럼에도, PyTorch Android gradle 빌드가 이 버전을 사용해서 우리도 이 버전을 사용합니다.

``NativeApp/settings.gradle`` 이 보여주듯이 이 프로젝트는 ``app`` 이라는 모듈 하나만 포함하며, 이 모듈이 Android 애플리케이션이 됩니다.

.. code-block:: shell

    mkdir app
    cd app


``NativeApp/app/build.gradle``

.. code-block:: gradle

  apply plugin: 'com.android.application'

  repositories {
    jcenter()
    maven {
      url "https://oss.sonatype.org/content/repositories/snapshots"
    }
  }

  android {
    configurations {
      extractForNativeBuild
    }
    compileSdkVersion 28
    buildToolsVersion "29.0.2"
    defaultConfig {
      applicationId "org.pytorch.nativeapp"
      minSdkVersion 21
      targetSdkVersion 28
      versionCode 1
      versionName "1.0"
      externalNativeBuild {
        cmake {
          arguments "-DANDROID_STL=c++_shared"
        }
      }
    }
    buildTypes {
      release {
        minifyEnabled false
      }
    }
    externalNativeBuild {
      cmake {
        path "CMakeLists.txt"
      }
    }
    sourceSets {
      main {
        jniLibs.srcDirs = ['src/main/jniLibs']
      }
    }
  }

  dependencies {
    implementation 'com.android.support:appcompat-v7:28.0.0'

    implementation 'org.pytorch:pytorch_android:1.6.0-SNAPSHOT'
    extractForNativeBuild 'org.pytorch:pytorch_android:1.6.0-SNAPSHOT'
  }

  task extractAARForNativeBuild {
    doLast {
      configurations.extractForNativeBuild.files.each {
        def file = it.absoluteFile
        copy {
          from zipTree(file)
          into "$buildDir/$file.name"
          include "headers/**"
          include "jni/**"
        }
      }
    }
  }

  tasks.whenTaskAdded { task ->
    if (task.name.contains('externalNativeBuild')) {
      task.dependsOn(extractAARForNativeBuild)
    }
  }

이 gradle 빌드 스크립트는 pytorch_android 스냅샷에 대한 종속성을 등록합니다. 이러한 스냅샷은 nightly 채널에 게시됩니다.

이러한 스냅샷은 nexus sonatype 저장소에 게시되므로, 해당 저장소를 등록해 줍니다:
``https://oss.sonatype.org/content/repositories/snapshots``.

애플리케이션 내부의 네이티브 빌드 부분에서는 LibTorch C++ API를 사용해야 합니다. 이를 위해 사전 빌드된 바이너리와 헤더에 접근을 해야 합니다. 바이너리와 헤더는 PyTorch Android 빌드에 미리 패키징되어 있습니다. 이러한 것들은 Maven 저장소에 올라가 있습니다.

gradle 종속성(aar 파일들)에서 사전 빌드된 PyTorch Android 라이브러리를 사용하기 위해
``extractForNativeBuild`` 에 대한 설정을 추가로 등록해야 합니다. 이 설정을 종속성에 추가하고 설정의 정의를 마지막 부분에 넣어 줍니다.

``extractForNativeBuild`` 태스크는 pytorch_android aar을 gradle 빌드 디렉토리에 압축을 푸는 ``extractAARForNativeBuild`` 태스크를 호출합니다.

Pytorch_android aar은 ``headers`` 폴더 안에 LibTorch 헤더를 포함하고
``jni`` 폴더 내부에는 다른 Android ABI들을 위한 사전 빌드된 라이브러리들을 포함합니다:
``$ANDROID_ABI/libpytorch_jni.so``, ``$ANDROID_ABI/libfbjni.so``.
이런 것들을 네이티브 빌드에 이용해 봅시다.

네이티브 빌드는 ``build.gradle`` 내부에 아래와 같은 코드 라인들로 등록이 되어 있습니다:

.. code-block:: gradle

  android {
    ...
    externalNativeBuild {
      cmake {
        path "CMakeLists.txt"
      }
  }
  ...
  defaultConfig {
    externalNativeBuild {
      cmake {
        arguments "-DANDROID_STL=c++_shared"
      }
    }
  }

네이티브 빌드를 위해 ``CMake`` 설정을 사용합니다. 우리가 사용하는 다양한 라이브러리들이 있기 때문에, STL과 동적으로 연결시키는 것도 명기합니다. 이에 대한 자세한 내용은 `여기 <https://developer.android.com/ndk/guides/cpp-support>`_ 에서 찾을 수 있습니다.


네이티브 빌드 CMake 설정하기
--------------------------

네이티브 빌드는 ``NativeApp/app/CMakeLists.txt`` 에서 설정합니다:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.4.1)
  set(TARGET pytorch_nativeapp)
  project(${TARGET} CXX)
  set(CMAKE_CXX_STANDARD 14)

  set(build_DIR ${CMAKE_SOURCE_DIR}/build)

  set(pytorch_testapp_cpp_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)
  file(GLOB pytorch_testapp_SOURCES
    ${pytorch_testapp_cpp_DIR}/pytorch_nativeapp.cpp
  )

  add_library(${TARGET} SHARED
      ${pytorch_testapp_SOURCES}
  )

  file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
  file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

  target_compile_options(${TARGET} PRIVATE
    -fexceptions
  )

  set(BUILD_SUBDIR ${ANDROID_ABI})

  find_library(PYTORCH_LIBRARY pytorch_jni
    PATHS ${PYTORCH_LINK_DIRS}
    NO_CMAKE_FIND_ROOT_PATH)
  find_library(FBJNI_LIBRARY fbjni
    PATHS ${PYTORCH_LINK_DIRS}
    NO_CMAKE_FIND_ROOT_PATH)

  # OpenCV
  if(NOT DEFINED ENV{OPENCV_ANDROID_SDK})
    message(FATAL_ERROR "Environment var OPENCV_ANDROID_SDK is not set")
  endif()

  set(OPENCV_INCLUDE_DIR "$ENV{OPENCV_ANDROID_SDK}/sdk/native/jni/include")

  target_include_directories(${TARGET} PRIVATE
   "${OPENCV_INCLUDE_DIR}"
    ${PYTORCH_INCLUDE_DIRS})

  set(OPENCV_LIB_DIR "$ENV{OPENCV_ANDROID_SDK}/sdk/native/libs/${ANDROID_ABI}")

  find_library(OPENCV_LIBRARY opencv_java4
    PATHS ${OPENCV_LIB_DIR}
    NO_CMAKE_FIND_ROOT_PATH)

  target_link_libraries(${TARGET}
    ${PYTORCH_LIBRARY}
    ${FBJNI_LIBRARY}
    ${OPENCV_LIBRARY}
    log)

여기에서는 소스 파일 ``pytorch_nativeapp.cpp`` 하나만 등록합니다.

이전 단계인 ``NativeApp/app/build.gradle`` 에서, ``extractAARForNativeBuild`` 태스크를 사용해서 헤더와 네이티브 라이브러리를 빌드 디렉토리로 추출했습니다. 이것들을 위해 ``PYTORCH_INCLUDE_DIRS`` 와 ``PYTORCH_LINK_DIRS`` 를 설정해 줍니다.

그 이후 ``libpytorch_jni.so`` 와 ``libfbjni.so`` 라이브러리를 찾아 우리의 목표에 연결해 줍니다.

OpenCV 함수를 이용해서 사용자 지정 연산자 ``my_ops::warp_perspective`` 를 구현할 계획이었기에 ``libopencv_java4.so`` 를 연결해 줘야 합니다. 이 라이브러리는 설정 단계에서 다운로드한 Android용 OpenCV SDK에 패키징되어 있습니다.
이 설정 내부에서는 환경 변수 ``OPENCV_ANDROID_SDK`` 으로 찾을 수 있습니다.

또한 Android LogCat으로 로그를 남길 수 있도록 ``log`` 라이브러리도 연결합니다.

OpenCV Android SDK의 ``libopencv_java4.so`` 도 연결했기 때문에, 이 라이브러리를 ``NativeApp/app/src/main/jniLibs/${ANDROID_ABI}`` 에도 복사를 해야 합니다.

.. code-block:: shell

  cp -R $OPENCV_ANDROID_SDK/sdk/native/libs/* NativeApp/app/src/main/jniLibs/


애플리케이션에 모델 파일 추가하기
----------------------------------------

애플리케이션 내부에 TorschScript 모델인 ``compute.pt`` 를 패키징하려면 모델 파일을 assets 폴더로 복사를 해야 합니다:

.. code-block:: shell

  mkdir -p NativeApp/app/src/main/assets
  cp compute.pt NativeApp/app/src/main/assets


Android 애플리케이션 매니페스트(Manifest)
------------------------------------------

모든 Android 애플리케이션은 매니페스트가 있습니다.
여기에 애플리케이션 이름, 패키지, 메인 액티비티를 명기합니다.

``NativeApp/app/src/main/AndroidManifest.xml``:

.. code-block:: default

  <?xml version="1.0" encoding="utf-8"?>
  <manifest xmlns:android="http://schemas.android.com/apk/res/android"
      package="org.pytorch.nativeapp">

      <application
          android:allowBackup="true"
          android:label="PyTorchNativeApp"
          android:supportsRtl="true"
          android:theme="@style/Theme.AppCompat.Light.DarkActionBar">

          <activity android:name=".MainActivity">
              <intent-filter>
                  <action android:name="android.intent.action.MAIN" />
                  <category android:name="android.intent.category.LAUNCHER" />
              </intent-filter>
          </activity>
      </application>
  </manifest>


소스코드
-------

Java 코드
---------

이제 MainActivity를 아래 파일에서 구현할 준비가 되었습니다

``NativeApp/app/src/main/java/org/pytorch/nativeapp/MainActivity.java``:

.. code-block:: java

  package org.pytorch.nativeapp;

  import android.content.Context;
  import android.os.Bundle;
  import android.util.Log;
  import androidx.appcompat.app.AppCompatActivity;
  import java.io.File;
  import java.io.FileOutputStream;
  import java.io.IOException;
  import java.io.InputStream;
  import java.io.OutputStream;

  public class MainActivity extends AppCompatActivity {

    private static final String TAG = "PyTorchNativeApp";

    public static String assetFilePath(Context context, String assetName) {
      File file = new File(context.getFilesDir(), assetName);
      if (file.exists() && file.length() > 0) {
        return file.getAbsolutePath();
      }

      try (InputStream is = context.getAssets().open(assetName)) {
        try (OutputStream os = new FileOutputStream(file)) {
          byte[] buffer = new byte[4 * 1024];
          int read;
          while ((read = is.read(buffer)) != -1) {
            os.write(buffer, 0, read);
          }
          os.flush();
        }
        return file.getAbsolutePath();
      } catch (IOException e) {
        Log.e(TAG, "Error process asset " + assetName + " to file path");
      }
      return null;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
      super.onCreate(savedInstanceState);
      final String modelFileAbsoluteFilePath =
          new File(assetFilePath(this, "compute.pt")).getAbsolutePath();
      NativeClient.loadAndForwardModel(modelFileAbsoluteFilePath);
    }
  }


이전 단계에서 ``compute.pt`` 을 ``NativeApp/app/src/main/assets`` 으로 복사했기 때문에, 이 파일은 Android 애플리케이션에 포함이 되는 asset이 되었습니다. Android 시스템은 그 파일에 접근할 수 있는 스트림만 제공합니다.
이 모듈을 LibTorch에서 사용하고자 한다면, 디스크 상에서 파일로 만들어야(materialize) 합니다. ``assetFilePath`` 함수는 asset 입력 스트림에서부터 데이터를 복사해서 디스크에 기록한 다음 파일의 절대 경로를 반환합니다.

액티비티의 ``OnCreate`` 메소드는 액티비티 생성 직후 호출됩니다. 이 메소드 안에서 ``assertFilePath`` 를 호출하고, 앞서 생성한 파일을 JNI 호출을 통해 네이티브 코드로 전달하는 ``NativeClient`` 클래스를 호출합니다.

``NativeClient`` 는 내부에 private 클래스인 ``NativePeer`` 가 있는 헬퍼 클래스로서, 애플리케이션의 네이티브 부분과 같이 동작하는 클래스입니다. 이 클래스의 static 블록에선 이전 단계에서 추가했던 ``CMakeLists.txt`` 와 함께 빌드하는 ``libpytorch_nativeapp.so`` 를 읽습니다. static 블록은 ``NativePeer`` 를 처음 참조할 때에 같이 실행이 됩니다. ``NativeClient#loadAndForwardModel`` 안에서 일어납니다.

``NativeApp/app/src/main/java/org/pytorch/nativeapp/NativeClient.java``:

.. code-block:: java

  package org.pytorch.nativeapp;

  public final class NativeClient {

    public static void loadAndForwardModel(final String modelPath) {
      NativePeer.loadAndForwardModel(modelPath);
    }

    private static class NativePeer {
      static {
        System.loadLibrary("pytorch_nativeapp");
      }

      private static native void loadAndForwardModel(final String modelPath);
    }
  }

``NativePeer#loadAndForwardModel`` 은 ``native`` 로 선언이 되어 있는데, Java를 위한 정의는 아닙니다. 이 메소드를 호출하면 JNI를 통해 ``NativeApp/app/src/main/cpp/pytorch_nativeapp.cpp`` 내부에 있는 ``libpytorch_nativeapp.so`` 의 C++ 메소드를 다시 가져옵니다.

네이티브 코드
-------------

이제 애플리케이션의 네이티브 부분을 작성할 준비가 되었습니다.

``NativeApp/app/src/main/cpp/pytorch_nativeapp.cpp``:

.. code-block:: cpp

  #include <android/log.h>
  #include <cassert>
  #include <cmath>
  #include <pthread.h>
  #include <unistd.h>
  #include <vector>
  #define ALOGI(...)                                                             \
    __android_log_print(ANDROID_LOG_INFO, "PyTorchNativeApp", __VA_ARGS__)
  #define ALOGE(...)                                                             \
    __android_log_print(ANDROID_LOG_ERROR, "PyTorchNativeApp", __VA_ARGS__)

  #include "jni.h"

  #include <opencv2/opencv.hpp>
  #include <torch/script.h>

  namespace pytorch_nativeapp {
  namespace {
  torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
    cv::Mat image_mat(/*rows=*/image.size(0),
                      /*cols=*/image.size(1),
                      /*type=*/CV_32FC1,
                      /*data=*/image.data_ptr<float>());
    cv::Mat warp_mat(/*rows=*/warp.size(0),
                     /*cols=*/warp.size(1),
                     /*type=*/CV_32FC1,
                     /*data=*/warp.data_ptr<float>());

    cv::Mat output_mat;
    cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});

    torch::Tensor output =
        torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8});
    return output.clone();
  }

  static auto registry =
      torch::RegisterOperators("my_ops::warp_perspective", &warp_perspective);

  template <typename T> void log(const char *m, T t) {
    std::ostringstream os;
    os << t << std::endl;
    ALOGI("%s %s", m, os.str().c_str());
  }

  struct JITCallGuard {
    torch::autograd::AutoGradMode no_autograd_guard{false};
    torch::AutoNonVariableTypeMode non_var_guard{true};
    torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
  };
  } // namespace

  static void loadAndForwardModel(JNIEnv *env, jclass, jstring jModelPath) {
    const char *modelPath = env->GetStringUTFChars(jModelPath, 0);
    assert(modelPath);
    JITCallGuard guard;
    torch::jit::Module module = torch::jit::load(modelPath);
    module.eval();
    torch::Tensor x = torch::randn({4, 8});
    torch::Tensor y = torch::randn({8, 5});
    log("x:", x);
    log("y:", y);
    c10::IValue t_out = module.forward({x, y});
    log("result:", t_out);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
  }
  } // namespace pytorch_nativeapp

  JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
      return JNI_ERR;
    }

    jclass c = env->FindClass("org/pytorch/nativeapp/NativeClient$NativePeer");
    if (c == nullptr) {
      return JNI_ERR;
    }

    static const JNINativeMethod methods[] = {
        {"loadAndForwardModel", "(Ljava/lang/String;)V",
         (void *)pytorch_nativeapp::loadAndForwardModel},
    };
    int rc = env->RegisterNatives(c, methods,
                                  sizeof(methods) / sizeof(JNINativeMethod));

    if (rc != JNI_OK) {
      return rc;
    }

    return JNI_VERSION_1_6;
  }


이 목록은 꽤 긴데다 이런저런 것들이 혼합되어 있기 때문에 이 코드가 어떻게 동작하는지 이해하기 위해 제어 흐름을 따라가 보겠습니다.
제어 흐름이 처음 도착하는 곳은 ``JNI_OnLoad`` 입니다.
이 함수는 라이브러리를 읽어들인 이후 호출되며, ``NativePeer#loadAndForwardModel`` 가 호출되었을 때 네이티브 메소드를 등록할 책임을 가집니다. 여기에서는  ``pytorch_nativeapp::loadAndForwardModel`` 함수입니다.

``pytorch_nativeapp::loadAndForwardModel`` 은 인자로 모델의 경로를 받습니다.
첫째로 인자의 ``const char*`` 값을 추출해서 ``torch::jit::load`` 로 모듈을 읽어 들입니다.

모바일용 TorchScript을 읽으려면, 가드(guards)를 설정해야 됩니다. 모바일 빌드는 더 작은 빌드 크기를 위한 autograd같은 기능을 지원하지 않기 때문입니다.
이 예제에서는 ``struct JITCallGuard`` 에 있는 autograd 기능입니다.
향후 변경될 수 있습니다. 최신 변경 사항을 추적하고자 한다면 아래를 확인하세요.
`PyTorch GitHub 내부 소스 <https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp>`_.

메소드 ``warp_perspective`` 구현과 등록은  `tutorial for desktop build <https://tutorials.pytorch.kr/advanced/torch_script_custom_ops.html>`_
내부에 있는 내용과 정확히 동일합니다.

앱 빌드하기
---------------

gradle에 Android SDK와 Android NDK를 명기하기 위해 ``NativeApp/local.properties`` 를 작성합니다.

.. code-block:: shell

  cd NativeApp
  echo "sdk.dir=$ANDROID_HOME" >> NativeApp/local.properties
  echo "ndk.dir=$ANDROID_NDK" >> NativeApp/local.properties


결과 ``apk`` 파일을 빌드하기 위해 실행합니다:

.. code-block:: shell

  cd NativeApp
  gradle app:assembleDebug

연결된 디바이스에 앱을 설치하기:

.. code-block:: shell

  cd NativeApp
  gradle app::installDebug

그러고 나면 PyTorchNativeApp 아이콘을 클릭하여 앱을 디바이스에서 실행할 수 있습니다.
또는 command line에서 실행할 수도 있습니다:

.. code-block:: shell

  adb shell am start -n org.pytorch.nativeapp/.MainActivity

Android logcat을 확인하려면:

.. code-block:: shell

  adb logcat -v brief | grep PyTorchNativeApp


'PyTorchNativeApp' 태그의 로그에서 x, y, 모델 순전파의 결과를 확인할 수 있어야 합니다. 이러한 로그는 ``NativeApp/app/src/main/cpp/pytorch_nativeapp.cpp`` 내부의 ``log`` 함수에서 출력합니다.

::

  I/PyTorchNativeApp(26968): x: -0.9484 -1.1757 -0.5832  0.9144  0.8867  1.0933 -0.4004 -0.3389
  I/PyTorchNativeApp(26968): -1.0343  1.5200 -0.7625 -1.5724 -1.2073  0.4613  0.2730 -0.6789
  I/PyTorchNativeApp(26968): -0.2247 -1.2790  1.0067 -0.9266  0.6034 -0.1941  0.7021 -1.5368
  I/PyTorchNativeApp(26968): -0.3803 -0.0188  0.2021 -0.7412 -0.2257  0.5044  0.6592  0.0826
  I/PyTorchNativeApp(26968): [ CPUFloatType{4,8} ]
  I/PyTorchNativeApp(26968): y: -1.0084  1.8733  0.5435  0.1087 -1.1066
  I/PyTorchNativeApp(26968): -1.9926  1.1047  0.5311 -0.4944  1.9178
  I/PyTorchNativeApp(26968): -1.5451  0.8867  1.0473 -1.7571  0.3909
  I/PyTorchNativeApp(26968):  0.4039  0.5085 -0.2776  0.4080  0.9203
  I/PyTorchNativeApp(26968):  0.3655  1.4395 -1.4467 -0.9837  0.3335
  I/PyTorchNativeApp(26968): -0.0445  0.8039 -0.2512 -1.3122  0.6543
  I/PyTorchNativeApp(26968): -1.5819  0.0525  1.5680 -0.6442 -1.3090
  I/PyTorchNativeApp(26968): -1.6197 -0.0773 -0.5967 -0.1105 -0.3122
  I/PyTorchNativeApp(26968): [ CPUFloatType{8,5} ]
  I/PyTorchNativeApp(26968): result:  16.0274   9.0330   6.0124   9.8644  11.0493
  I/PyTorchNativeApp(26968):   8.7633   6.9657  12.3469  10.3159  12.0683
  I/PyTorchNativeApp(26968):  12.4529   9.4559  11.7038   7.8396   6.9716
  I/PyTorchNativeApp(26968):   8.5279   9.1780  11.3849   8.4368   9.1480
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968):  10.0000  10.0000  10.0000  10.0000  10.0000
  I/PyTorchNativeApp(26968): [ CPUFloatType{8,5} ]



이 앱의 전체 설정은 `PyTorch Android 데모 애플리케이션 저장소 <https://github.com/pytorch/android-demo-app/tree/master/NativeApp>`_ 에서 찾을 수 있습니다.
