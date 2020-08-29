PyTorch 사전 빌드 된 라이브러리를 사용하는 네이티브 Android 애플리케이션 만들기
==============================================================================

**Author**: `Ivan Kobzarev <https://github.com/IvanKobzarev>`_ 

**번역**: `Ajin Jeong <https://github.com/ajin-jng>`

이 레시피에서는 다음을 학습합니다.

 - 네이티브 코드 (C ++)에서 LibTorch API를 사용하는 Android 애플리케이션을 만드는 방법.

 - 이 응용 프로그램에서 사용자 정의 연산자와 함께 TorchScript 모델을 사용하는 방법.

이 앱의 전체 설정은 `PyTorch Android Demo Application Repository <https://github.com/pytorch/android-demo-app/tree/master/NativeApp>`_ 에서 찾을 수 있습니다.


설정
~~~~~

다음 패키지 (및 해당 종속성)가 설치된 Python 3 환경이 필요합니다.

 - PyTorch 1.6

Android 개발의 경우 다음을 설치해야합니다.

 - 안드로이드 NDK

::

  wget https://dl.google.com/android/repository/android-ndk-r19c-linux-x86_64.zip
  unzip android-ndk-r19c-linux-x86_64.zip
  export ANDROID_NDK=$(pwd)/android-ndk-r19c


- 안드로이드 SDK

::

  wget https://dl.google.com/android/repository/sdk-tools-linux-3859397.zip
  unzip sdk-tools-linux-3859397.zip -d android_sdk
  export ANDROID_HOME=$(pwd)/android_sdk



- Gradle 4.10.3

Gradle은 Android 애플리케이션에 가장 널리 사용되는 빌드 시스템이며 애플리케이션을 빌드하는 데 필요합니다. 그것을 다운로드하고 경로에 추가하여 command line 에서 ``gradle '' 을 사용하십시오.

.. code-block:: shell

  wget https://services.gradle.org/distributions/gradle-4.10.3-bin.zip
  unzip gradle-4.10.3-bin.zip
  export GRADLE_HOME=$(pwd)/gradle-4.10.3
  export PATH="${GRADLE_HOME}/bin/:${PATH}"

- JDK

Gradle에는 JDK가 필요하므로이를 설치하고이를 가리 키도록 환경 변수``JAVA_HOME``을 설정해야합니다.
예를 들어 `<https://openjdk.java.net/install/>`_에 따라 OpenJDK를 설치할 수 있습니다.

- Android 용 OpenCV SDK

사용자 지정 연산자는 OpenCV 라이브러리를 사용하여 구현됩니다. Android 용으로 사용하려면 사전 빌드 된 라이브러리가있는 Android 용 OpenCV SDK를 다운로드해야합니다.
`OpenCV 릴리스 페이지 <https://opencv.org/releases/>`_ 에서 다운로드하여 압축을 풀고 환경 변수``OPENCV_ANDROID_SDK``를 설정하십시오.


사용자 지정 C ++ 연산자로 TorchScript 모델 준비
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~

TorchScript는 사용자 정의 C ++ 연산자를 사용하여 읽을 수있는 세부 정보로 읽을 수 있습니다.
`전용 튜토리얼 <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_.

결과적으로 OpenCV``cv :: warpPerspective ''함수를 사용하는 커스텀 op를 사용하는 모델을 스크립팅 할 수 있습니다.

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

 
이 snippet 은 사용자 지정 op ``my_ops.warp_perspective`` 를 사용하는 TorchScript 모델 인 ``compute.pt '' 파일을 생성합니다.

실행하려면 개발 용 OpenCV를 설치해야합니다.
다음 명령을 사용하여 수행 할 수있는 Linux 시스템의 경우 :
CentOS :

.. code-block:: shell

  yum install opencv-devel

Ubuntu :

.. code-block:: shell

  apt-get install libopencv-dev

Android 애플리케이션 만들기
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``compute.pt`` 를 갖는 데 성공한 후 Android 애플리케이션 내에서 이 TorchScript 모델을 사용하려고 합니다. Java API를 사용하여 Android에서 일반 TorchScript 모델 (사용자 지정 연산자 없음)을 사용하면`here <https://pytorch.org/mobile/android/>`_에서 찾을 수 있습니다. 모델이 사용자 정의 연산자 ( ``my_ops.warp_perspective '' )를 사용하기 때문에 이 방법을 사용할 수 없습니다. 기본 TorchScript 실행은 이를 찾지 못합니다.

ops 등록은 PyTorch Java API에 노출되지 않으므로 네이티브 파트 (C ++)로 Android 애플리케이션을 빌드하고 LibTorch C ++ API를 사용하여 Android 용 동일한 사용자 지정 연산자를 구현하고 등록해야합니다. 운영자가 OpenCV 라이브러리를 사용하므로 미리 빌드 된 OpenCV Android 라이브러리를 사용하고 OpenCV의 동일한 기능을 사용합니다.

``NativeApp ''폴더에서 Android 애플리케이션 생성을 시작하겠습니다.

.. code-block:: shell
  
  mkdir NativeApp
  cd NativeApp

Android 애플리케이션 빌드 설정
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Android 애플리케이션 빌드는 기본 gradle 부분과 기본 빌드 CMake 부분으로 구성됩니다.
여기에있는 모든 목록은 전체 파일 목록입니다. 전체 구조를 다시 만들려면
코드 추가없이 결과 Android 애플리케이션을 빌드하고 설치할 수 있습니다.

Gradle 빌드 설정
------------------
gradle 설정 파일 (build.gradle, gradle.properties, settings.gradle)을 추가해야합니다.
Android Gradle 빌드 구성에 대한 자세한 내용은`여기 <https://developer.android.com/studio/build>`_에서 찾을 수 있습니다.
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

``NativeApp / build.gradle`` 에서 Android gradle 플러그인 버전 '3.5.0'을 지정합니다. 이 버전은 최신 버전이 아님에도 불구하고  PyTorch android gradle 빌드가이 버전을 사용하므로 이를 사용합니다.

``NativeApp / settings.gradle`` 은 out 프로젝트에 Android 애플리케이션이 될``app ''모듈이 하나만 포함되어 있음을 보여줍니다.

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

이 gradle 빌드 스크립트는 야간 채널에 게시 된 pytorch_android 스냅 샷에 대한 종속성을 등록합니다.

nexus sonatype 저장소에 게시되므로 해당 저장소를 등록해야합니다.
``https : // oss.sonatype.org / content / repositories / snapshots``.

우리의 애플리케이션에서 우리는 애플리케이션 네이티브 빌드 부분에서 LibTorch C ++ API를 사용해야합니다. 이를 위해서는 미리 빌드 된 바이너리와 헤더에 대한 액세스가 필요합니다. Maven 저장소에 게시된 PyTorch Android 빌드에 미리 포장되어 있습니다.

gradle 종속성 (aar 파일)에서 PyTorch Android 사전 빌드 라이브러리를 사용하려면-
구성``extractForNativeBuild `` 에 대한 등록을 추가해야합니다.
이 구성을 종속성에 추가하고 그 정의를 끝에 넣으십시오.

``extractForNativeBuild `` 작업은 pytorch_android aar를 gradle 빌드 디렉터리로 압축 해제하는``extractAARForNativeBuild `` 작업을 호출합니다.

Pytorch_android aar에는``headers `` 폴더의 LibTorch 헤더와``jni `` 폴더의 여러 Android abis 용 사전 빌드된 라이브러리가 포함되어 있습니다.
``$ ANDROID_ABI / libpytorch_jni.so``,``$ ANDROID_ABI / libfbjni.so``.
기본 빌드에 사용합니다.

네이티브 빌드는 이 ``build.gradle`` 에 다음과 같이 등록 됩니다:

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

네이티브 빌드에``CMake`` 구성을 사용합니다. 여기에는 여러 라이브러리가 있으므로 STL과 동적으로 링크하도록 지정합니다. 이에 대한 자세한 내용은`여기 <https://developer.android.com/ndk/guides/cpp-support>`_에서 찾을 수 있습니다.


네이티브 빌드 CMake 설정
------------------------

네이티브 빌드는 ``NativeApp/app/CMakeLists.txt`` 에서 구성됩니다. :

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

여기서는 소스 파일 ``pytorch_nativeapp.cpp`` 를 하나만 등록합니다.
 
``NativeApp / app / build.gradle`` 의 이전 단계에서 ``extractAARForNativeBuild `` 작업은 헤더와 네이티브 라이브러리를 추출하여 디렉터리를 빌드합니다. ``PYTORCH_INCLUDE_DIRS `` 및 ``PYTORCH_LINK_DIRS`` 를 설정합니다.

그 후 ``libpytorch_jni.so `` 및 ``libfbjni.so `` 라이브러리를 찾아서 타겟 링크에 추가합니다.

OpenCV 함수를 사용하여 사용자 지정 연산자 ``my_ops :: warp_perspective `` 를 구현할 계획이므로 ``libopencv_java4.so`` 에 링크해야합니다. 설정 단계에서 다운로드 한 Android 용 OpenCV SDK에 패키지되어 있습니다.
이 구성에서는 환경 변수 ``OPENCV_ANDROID_SDK `` 로 찾을 수 있습니다.

또한 결과를 Android LogCat에 기록 할 수 있도록 ``log `` 라이브러리와 연결합니다.

OpenCV Android SDK의 ``libopencv_java4.so`` 에 링크 할 때 이를 ``NativeApp / app / src / main / jniLibs / $ {ANDROID_ABI}`` 에 복사해야합니다.

.. code-block:: shell

  cp -R $OPENCV_ANDROID_SDK/sdk/native/libs/* NativeApp/app/src/main/jniLibs/

애플리케이션에 모델 파일 추가
----------------------------------------

애플리케이션 내에서 TorschScript 모델``compute.pt``를 패키징하려면 해당 모델을 assets 폴더에 복사해야합니다.

.. code-block:: shell

  mkdir -p NativeApp/app/src/main/assets
  cp compute.pt NativeApp/app/src/main/assets


Android 애플리케이션 매니페스트
----------------------------

모든 Android 애플리케이션에는 매니페스트가 있습니다.
여기에서 애플리케이션 이름, 패키지, 주요 활동을 지정합니다.

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


소스 코드
-------

자바 코드
---------

이제 MainActivity를 구현할 준비가되었습니다.

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

이전 단계에서 ``compute.pt`` 를 ``NativeApp / app / src / main / assets`` 에 복사했을 때 해당 파일은 Android 애플리케이션에 한 부분이 되었으며 애플리케이션에 압축됩니다. Android 시스템은 스트림 액세스만 제공합니다.
LibTorch에서 이 모듈을 사용하려면 디스크에 파일로 구체화해야합니다. ``assetFilePath `` 함수는 asset 입력 스트림에서 데이터를 복사하고 디스크에 쓰고 이에 대한 절대 파일 경로를 반환합니다.

Activity의 ``OnCreate `` 메서드는 Activity 생성 직후에 호출됩니다. 이 메서드에서는 ``assertFilePath `` 를 호출하고 JNI 호출을 통해 네이티브 코드로 전달하는 ``NativeClient `` 클래스를 호출합니다.

``NativeClient``는 내부 개인 클래스 ``NativePeer`` 가있는 도우미 클래스로, 애플리케이션의 기본 부분을 담당합니다. 이전 단계에서 추가 한 ``CMakeLists.txt`` 로 빌드되는``libpytorch_nativeapp.so`` 를로드하는 정적 블록이 있습니다. 정적 블록은 ``NativePeer `` 클래스의 첫 번째 참조로 실행됩니다. ``NativeClient # loadAndForwardModel`` 에서 발생합니다.

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

``NativePeer#loadAndForwardModel`` is declared as ``native``, it does not have definition in Java. Call to this method will be re-dispatched through JNI to C++ method in our ``libpytorch_nativeapp.so``, in ``NativeApp/app/src/main/cpp/pytorch_nativeapp.cpp``.

네이티브 코드
-----------

이제 우리는 앱의 네이티브 부분를 쓸 준비가 되었습니다:

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


이 목록은 꽤 길고 여기에 몇 가지가 혼합되어 있으므로 제어 흐름을 따라이 코드가 작동하는 방식을 이해할 것입니다.
제어 흐름이 도착하는 첫 번째 위치는``JNI_OnLoad``입니다.
이 함수는 라이브러리를로드 한 후 호출됩니다. ``NativePeer # loadAndForwardModel`` 이 호출 될 때 호출되는 네이티브 메서드를 등록하는 역할을 합니다. 여기서는 ``pytorch_nativeapp :: loadAndForwardModel`` 함수입니다.

``pytorch_nativeapp :: loadAndForwardModel`` 은 인수 모델 경로로 사용됩니다.
먼저 ``const char *`` 값을 추출하고 ``torch :: jit :: load`` 로 모듈을 로드합니다.

모바일 용 TorchScript 모델을로드하려면 모바일 빌드가 지원하지 않기 때문에 이러한 가드를 설정해야합니다.
이 예제에서는 ``struct JITCallGuard '' 에 배치 된 더 작은 빌드 크기를위한 autograd와 같은 기능입니다.
향후 변경 될 수 있습니다. 최신 변경 사항을 추적할 수 있습니다.
`PyTorch GitHub의 소스 <https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp>`_.

``warp_perspective `` 메소드 구현 및 등록은 다음과 완전히 동일합니다.
`데스크톱 빌드 자습서 <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_에 있습니다.

앱 빌드
----------------

Android SDK 및 Android NDK가 어디에 있는지 지정하려면 ``NativeApp / local.properties`` 를 채워야합니다.

.. code-block:: shell

  cd NativeApp
  echo "sdk.dir=$ANDROID_HOME" >> NativeApp/local.properties
  echo "ndk.dir=$ANDROID_NDK" >> NativeApp/local.properties


결과 ``apk`` 파일을 빌드하기 위해서 다음을 실행합니다:

.. code-block:: shell

  cd NativeApp
  gradle app:assembleDebug

연결된 디바이스에 앱을 설치하기 위해서 다음을 참고하십시오:

.. code-block:: shell

  cd NativeApp
  gradle app::installDebug

그 후 PyTorch Native App 아이콘을 클릭하여 장치에서 앱을 실행할 수 있습니다.
또는 command line 에서 수행 할 수 있습니다.

.. code-block:: shell

  adb shell am start -n org.pytorch.nativeapp/.MainActivity

android logcat를 체크할 시 다음과 같은 결과가 나와야 합니다:

.. code-block:: shell

  adb logcat -v brief | grep PyTorchNativeApp


"PyTorchNativeApp"라는 태그 로그가 나타나고 x, y 및 모델의 결과가 전송됩니다. 이것은 "NativeApp / app / src / main / cpp / pytorch_nativeapp.cpp"의 "log"함수에서 출력합니다. 

.. code-block::

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



이 앱의 전체 설정은`PyTorch Android 데모 애플리케이션 저장소 <https://github.com/pytorch/android-demo-app/tree/master/NativeApp>`_에서 찾을 수 있습니다.
