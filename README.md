# Overview
This is a basic example of object detection on i.MXRT1060 evaluation kit ([MIMXRT1060-EVK](https://www.nxp.com/design/development-boards/i-mx-evaluation-and-development-boards/i-mx-rt1060-evaluation-kit:MIMXRT1060-EVK)).

It is heavily based on example named `tensorflow_lite_label_image` coming from SDK 2.6.0. Please be aware that it is not state-of-the-art SDK (the current one is 2.10.1, at the day of writing this guide, i.e. 22.09.2021) however I do still have some troubles with running object detection models on newer SDKs, for further reference please see NXP Community topic [here](https://community.nxp.com/t5/i-MX-RT/RT1062-run-tensorflowlite-cifar10-sample-code-error-message/m-p/1342158#M16383).

> **Additional info**
> * I really recommend checking special NXP Community referring to [eIQ Machine Learning Software](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/bd-p/eiq) where you can interact with other users playing around machine learning running on NXP platforms.

It is based also on example named `emwin_slide_show` which shows how to use emWin graphics library.

# Hardware
* [MIMXRT1060-EVK](https://www.nxp.com/design/development-boards/i-mx-evaluation-and-development-boards/i-mx-rt1060-evaluation-kit:MIMXRT1060-EVK) with attached [RK043FN02H-CT](https://www.nxp.com/design/development-boards/i-mx-evaluation-and-development-boards/i-mx-rt1060-evaluation-kit:MIMXRT1060-EVK#buy) 480x272 LCD display

# Software
* MCUXpresso IDE v11.4.0 [Build 6224] [2021-07-15]
* SDK_2.x_EVK-MIMXRT1060 version 2.6.0 (it is still normally available through [MCUXpresso SDK Builder](https://mcuxpresso.nxp.com/en/dashboard))

# Workflow
## 🧾 Import required examples from SDK
1. Import SDK to your MCUXpresso
2. From `QuickStart Panel` import SDK example `tensorflow_lite_label_image`
    * in `Project options` change `SDK Debug Console` to `UART` to use external debug console via UART
3. From `QuickStart Panel` import SDK example `emwin_slide_show`
    *  in `Project options` change `SDK Debug Console` to `UART` to use external debug console via UART (should be checked by default)
4. To keep imported examples not changed, in `Project Explorer` copy and paste `tensorflow_lite_label_image` project and change its name to for example `evkmimxrt1060_OD_TFLite`
5. You should get as follows:<br>
![1_import_SDK_examples.PNG](/doc/github_readme_images/1_import_SDK_examples.PNG)

## 📚 Add emWin graphics library
1. Select `evkmimxrt1060_OD_TFLite` project, right mouse click, choose `SDK Management` and then `Manage SDK Components`
2. Go to the `Middleware` tab and check `emWin graphics library`
3. Hit `OK` and accept incomming changes
4. Open `emwin_slide_show` project and copy below files to corresponding directories in `evkmimxrt1060_OD_TFLite` project (you can do this within `Project Explorer`):
    * `/board/pin_mux.h`
    * `/board/pin_mux.c`
    * `/drivers/fsl_elcdif.h`
    * `/drivers/fsl_elcdif.c`
    * `/drivers/fsl_gpt.h`
    * `/drivers/fsl_gpt.c`
    * `/drivers/fsl_lpi2c.h`
    * `/drivers/fsl_lpi2c.c`    
    * `/touchpanel/fsl_ft5406_rt.h` to `/drivers/fsl_ft5406_rt.h`
    * `/touchpanel/fsl_ft5406_rt.c` to `/drivers/fsl_ft5406_rt.c`
    * `/board/emwin_support.h` to `/emwin/emwin_support.h`
    * `/board/emwin_support.c` to `/emwin/emwin_support.c`
5. Update `#include` section in `evkmimxrt1060_OD_TFLite/source/label_image.cpp` with:
    ```cpp
    #include "GUI.h"
    #include "emwin_support.h"
    #include "fsl_gpt.h"
    ```
6. Copy functions from `evkmimxrt1060_emwin_slide_show/source/emwin_slide_show.c` and paste them in `evkmimxrt1060_OD_TFLite/source/label_image.cpp`: 
    * `void BOARD_InitLcd(void)`
    * `void BOARD_InitLcdifPixelClock(void)`
    * `void BOARD_InitGPT(void)`
7. In `evkmimxrt1060_OD_TFLite/source/label_image.cpp` update `int main(void)` function to be as follows:
    ```cpp
    int main(void)
    {
    tflite::label_image::Settings s;

    /* Init board hardware */
    BOARD_ConfigMPU();
    BOARD_InitPins();
    BOARD_InitI2C1Pins();
    BOARD_InitSemcPins();
    BOARD_BootClockRUN();
    BOARD_InitLcdifPixelClock();
    BOARD_InitDebugConsole();
    BOARD_InitLcd();
    BOARD_InitGPT();

    InitTimer();

    GUI_Init();

    std::cout << "Label image example using a TensorFlow Lite model\r\n";

    GUI_DispStringAt("emWin TEST!", 0, 0);

    tflite::label_image::RunInference(&s);

    std::flush(std::cout);

    for (;;) {}
    }
    ```
8. Hit `Debug` and on the display you should see black background and white string **emWin TEST!**

## 🏭 Prepare object detection model and list of classes
The pretrained model used in this example is COCO SSD MobileNetV1. It is trained to detect 90 classes of objects. The list of classes can be found [here](https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt).
1. Download object detection model from [TensorFlow Lite Object Detection example](https://www.tensorflow.org/lite/examples/object_detection/overview), direct link [here](https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite)
2. Go to `evkmimxrt1060_OD_TFLite/doc` and place here downloaded `ssd_mobilenet_v1_1_metadata_1.tflite` model 
3. Open `Command Prompt` with `cmd` and convert model to `*.h` array with command:
    ```console
    xxd -i ssd_mobilenet_v1_1_metadata_1.tflite > ssd_mobilenet_v1_1_metadata_1.h
    ```
    * `xxd` is part of [Vim](https://www.vim.org/) text editor, and provides a method to dump a binary file (in this case `*.tflite`) to hex, therefore please install Vim to be able to make necessary conversions
    * if `xxd -version` does not work straight away after Vim installation, please double check Vim installation directory (in my case: `"C:\Program Files (x86)\Vim\vim82\xxd.exe"`) and use the complete path instead of `xxd`
4. Move converted `ssd_mobilenet_v1_1_metadata_1.h` to `evkmimxrt1060_OD_TFLite/source`
5. In MCUXpresso, open `evkmimxrt1060_OD_TFLite/source/ssd_mobilenet_v1_1_metadata_1.h` and change `unsigned char` to `const char`
6. Update `evkmimxrt1060_OD_TFLite/source/labels.h` with list of classes available [here](https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt)

## ⏱️ Prepare example images to work on
1. Example images are provided with this repository, in `evkmimxrt1060_OD_TFLite/doc` 
    * they are `*.bmp` files
        * `stopwatch_224x272.bmp`
        * `apple_banana_orange_480x268.bmp`
    * they are adjusted in advance to resolution of the display, therefore they are no bigger than 480x272 pixels
2. To use the images in the project, they need to be converted to `*.h` files, using the command as described in previous section:
    ```console
    xxd -i apple_banana_orange_480x268.bmp > apple_banana_orange_480x268.h
    ```
    ```console
    xxd -i stopwatch_224x272.bmp > stopwatch_224x272.h
    ```
3. Move converted images to `evkmimxrt1060_OD_TFLite/source`
4. In MCUXpresso, open each image and change `unsigned char` to `const char`

## 🧠 Update the code to perform object detection
The detailed changes in `evkmimxrt1060_OD_TFLite/source/label_image.cpp` can be tracked as [commits history](https://github.com/marcin-ch/IMXRT1060_Object_Detection_COCO_SSD_MobileNetV1_TFLite/commits/master/source/label_image.cpp), especially [this one](https://github.com/marcin-ch/IMXRT1060_Object_Detection_COCO_SSD_MobileNetV1_TFLite/commit/e3e3eb04416bc572687733d65257d154f0ab2c40#diff-a4a8c43845907e4cf63e9f30da1574b2c712239c4e01ab017d4b438afcf58e3e).

Basically, you need to:
1. Provide new model to perform object detection
2. Provide new image to work on
3. Update machine learning code to get and interpret outputs from the model
    * these changes are based on this [Github repo](https://github.com/YijinLiu/tf-cpu/blob/master/benchmark/obj_detect_lite.cc)
    > **Additional info**
    > * [TensorFlow Lite Object Detection example](https://www.tensorflow.org/lite/examples/object_detection/overview) contains useful info how to interpret model's outputs.
4. Draw bounding boxes, recognized classes and confidence levels

Once all changes are done, hit `Debug` and, after flashing the board, you should see on the display image of fruits with bounding boxes, recognized classes and confidence levels.

Use terminal such as Tera Term to get bit more detailed output from an application, especially inference time.

# Summary
![2_imxrt1060_object_detection_resized.jpg](/doc/github_readme_images/2_imxrt1060_object_detection_resized.jpg)

Inference time is approximately 6 seconds, so it limits possible use cases.

What is more, detection of `dining table` with score `51%` in the image of fruits is bit surprising. This can be fixed with making the value of `threshold` bit higher (now it has been set to `0.48f`) and detection with scores below `threshold` will be eliminated. However, in this case, detection of `banana` (which is totally fine) with score `48%` will be eliminated as well.

Perhaps, using different object detection models can bring more accurate results.

Nevertheless, object detection can be realised on resource constrained platforms such as MCU.

There is another image provided in the example, named `stopwatch_224x272.bmp` (and `stopwatch_224x272.h`). To use it, please update the code in `evkmimxrt1060_OD_TFLite/source/label_image.cpp`:
```cpp
#define CURRENT_IMAGE     stopwatch_224x272_bmp
#define CURRENT_IMAGE_LEN stopwatch_224x272_bmp_len
```

## How to use this repo (source code)
1. Clone the repo or download as `*.zip` to your local disc drive
    * when clonning please use below command:
    ```console
    git clone https://github.com/marcin-ch/IMXRT1060_Object_Detection_COCO_SSD_MobileNetV1_TFLite.git
    ```
2. Open MCUXpresso, you will be asked for choosing existing or creating new workspace, I recommend creating new workspace for testing purposes
3. From `QuickStart Panel` choose `Import project(s) from file system` and then select either unpacked repo (in case you clonned the repo) or zipped repo (in case you downloaded the archive)
4. Make sure `Copy projects into workspace` in `Options` is checked
5. Hit `Finish`
6. Select imported project in `Project Explorer` and hit `Debug` in `QuickStart Panel`, the application should be up and running
7. You can now remove clonned or downloaded repo, as it now exists in your workspace

## How to use this repo (binary file)
If you just want to check how the project looks like running on the board, you can flash binary file available in `evkmimxrt1060_OD_TFLite/doc/evkmimxrt1060_OD_TFLite.bin`. As i.MXRT1060 evaluation kit enumerates as MSD (Mass Storage Device) when connected to PC through USB cable, you can simply drag-n-drop binary file to your board. Wait few moments when flashing is in progress, reset the board and you should see application working.