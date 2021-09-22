/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "board.h"
#include "pin_mux.h"
#include "clock_config.h"
#include "timer.h"

#include "GUI.h"
#include "emwin_support.h"
#include "fsl_gpt.h"

#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#include "label_image.h"
#include "bitmap_helpers.h"
#include "get_top_n.h"

#include "apple_banana_orange_480x268.h"
#include "stopwatch_224x272.h"
#include "ssd_mobilenet_v1_1_metadata_1.h"
#include "labels.h"

#define CURRENT_IMAGE     apple_banana_orange_480x268_bmp
#define CURRENT_IMAGE_LEN apple_banana_orange_480x268_bmp_len

/* Initialize the LCD_DISP. */
void BOARD_InitLcd(void)
{
    volatile uint32_t i = 0x100U;

    gpio_pin_config_t config = {
        kGPIO_DigitalOutput,
        0,
    };

    /* Reset the LCD. */
    GPIO_PinInit(LCD_DISP_GPIO, LCD_DISP_GPIO_PIN, &config);

    GPIO_WritePinOutput(LCD_DISP_GPIO, LCD_DISP_GPIO_PIN, 0);

    while (i--)
    {
    }

    GPIO_WritePinOutput(LCD_DISP_GPIO, LCD_DISP_GPIO_PIN, 1);

    /* Backlight. */
    config.outputLogic = 1;
    GPIO_PinInit(LCD_BL_GPIO, LCD_BL_GPIO_PIN, &config);

    /*Clock setting for LPI2C*/
    CLOCK_SetMux(kCLOCK_Lpi2cMux, LPI2C_CLOCK_SOURCE_SELECT);
    CLOCK_SetDiv(kCLOCK_Lpi2cDiv, LPI2C_CLOCK_SOURCE_DIVIDER);
}

void BOARD_InitLcdifPixelClock(void)
{
    /*
     * The desired output frame rate is 60Hz. So the pixel clock frequency is:
     * (480 + 41 + 4 + 18) * (272 + 10 + 4 + 2) * 60 = 9.2M.
     * Here set the LCDIF pixel clock to 9.3M.
     */

    /*
     * Initialize the Video PLL.
     * Video PLL output clock is OSC24M * (loopDivider + (denominator / numerator)) / postDivider = 93MHz.
     */
    clock_video_pll_config_t config = {
        .loopDivider = 31,
        .postDivider = 8,
        .numerator   = 0,
        .denominator = 0,
    };

    CLOCK_InitVideoPll(&config);

    /*
     * 000 derive clock from PLL2
     * 001 derive clock from PLL3 PFD3
     * 010 derive clock from PLL5
     * 011 derive clock from PLL2 PFD0
     * 100 derive clock from PLL2 PFD1
     * 101 derive clock from PLL3 PFD1
     */
    CLOCK_SetMux(kCLOCK_LcdifPreMux, 2);

    CLOCK_SetDiv(kCLOCK_LcdifPreDiv, 4);

    CLOCK_SetDiv(kCLOCK_LcdifDiv, 1);
}

void BOARD_InitGPT(void)
{
    gpt_config_t gptConfig;

    GPT_GetDefaultConfig(&gptConfig);

    gptConfig.enableFreeRun = true;
    gptConfig.divider       = 3000;

    /* Initialize GPT module */
    GPT_Init(EXAMPLE_GPT, &gptConfig);
    GPT_StartTimer(EXAMPLE_GPT);
}

#define LOG(x) std::cout

namespace tflite {
namespace label_image {

template<typename T>
T* TensorData(TfLiteTensor* tensor, int batch_index);

template<>
float* TensorData(TfLiteTensor* tensor, int batch_index) {
    int nelems = 1;
    for (int i = 1; i < tensor->dims->size; i++) nelems *= tensor->dims->data[i];
    switch (tensor->type) {
        case kTfLiteFloat32:
            return tensor->data.f + nelems * batch_index;
        default:
            LOG(FATAL) << "Should not reach here!\r\n";
    }
    return nullptr;
}

template<>
uint8_t* TensorData(TfLiteTensor* tensor, int batch_index) {
    int nelems = 1;
    for (int i = 1; i < tensor->dims->size; i++) nelems *= tensor->dims->data[i];
    switch (tensor->type) {
        case kTfLiteUInt8:
            return tensor->data.uint8 + nelems * batch_index;
        default:
            LOG(FATAL) << "Should not reach here!";
    }
    return nullptr;
}

/* Loads a list of labels, one per line, and returns a vector of the strings.
   It pads with empty strings so the length of the result is a multiple of 16,
   because the model expects that. */
TfLiteStatus ReadLabels(const string& labels,
                        std::vector<string>* result,
                        size_t* found_label_count) {
  std::istringstream stream(labels);
  result->clear();
  string line;
  while (std::getline(stream, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void RunInference(Settings* s) {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromBuffer(ssd_mobilenet_v1_1_metadata_1_tflite, ssd_mobilenet_v1_1_metadata_1_tflite_len);
  if (!model) {
    LOG(FATAL) << "Failed to load model\r\n";
    return;
  }
  model->error_reporter();

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\r\n";
    return;
  }

  if (s->verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\r\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\r\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\r\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\r\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\r\n";
    }
  }

  int image_width = 128;
  int image_height = 128;
  int image_channels = 3;
  uint8_t* in = read_bmp(CURRENT_IMAGE, CURRENT_IMAGE_LEN, &image_width, &image_height,
                         &image_channels, s);
  
  int input = interpreter->inputs()[0];
  if (s->verbose) LOG(INFO) << "input: " << input << "\r\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (s->verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\r\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\r\n";
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  if (s->verbose) PrintInterpreterState(interpreter.get());

  /* Get input dimension from the input tensor metadata
     assuming one input only */
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      s->input_floating = true;
      resize<float>(interpreter->typed_tensor<float>(input), in, image_height,
                    image_width, image_channels, wanted_height, wanted_width,
                    wanted_channels, s);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in,
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, s);
      break;
    default:
      LOG(FATAL) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }

  auto start_time = GetTimeInUS();
  for (int i = 0; i < s->loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\r\n";
    }
  }
  auto end_time = GetTimeInUS();
  LOG(INFO) << "Average time: " << (end_time - start_time) / 1000 << " ms\r\n";

  const float threshold = 0.48f;

  TfLiteTensor* output_locations_ = interpreter->tensor(interpreter->outputs()[0]);
  TfLiteTensor* output_classes_ = interpreter->tensor(interpreter->outputs()[1]);
  TfLiteTensor* output_scores_ = interpreter->tensor(interpreter->outputs()[2]);
  TfLiteTensor* num_detections_ = interpreter->tensor(interpreter->outputs()[3]);

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabels(labels_txt, &labels, &label_count) != kTfLiteOk)
    return;

  const float* detection_locations = TensorData<float>(output_locations_, 0);
  const float* detection_classes = TensorData<float>(output_classes_, 0);
  const float* detection_scores = TensorData<float>(output_scores_, 0);
  const int num_detections = *TensorData<float>(num_detections_, 0);

  GUI_SetBkColor(GUI_WHITE);
  GUI_BMP_Draw(CURRENT_IMAGE, 0, 0);
  GUI_SetBkColor(GUI_GRAY_D0); // to make light grey background for printing classes/class indexes
  GUI_SetPenSize(2);

  for (int i = 0; i < num_detections; i++){
	  const std::string cls = labels[detection_classes[i]];
	  const float score = detection_scores[i];

	  const int ymin = detection_locations[4 * i] * image_height;
	  const int xmin = detection_locations[4 * i + 1] * image_width;
	  const int ymax = detection_locations[4 * i + 2] * image_height;
	  const int xmax = detection_locations[4 * i + 3] * image_width;

	  int n = cls.length();
	  char class_name[n+1];
	  strcpy(class_name, cls.c_str()); // convert string to char to display it on LCD

	  char print_buf_score[10];

	  if(score > threshold) {
		  LOG(INFO) << "Detected " << cls << " with score " << (int)(score*100) << " [" << xmin << "," << ymin << ":" << xmax << "," << ymax << "]\r\n";

		  if(i==0){
			  GUI_SetColor(GUI_BLUE);
		  }
		  else{
			  GUI_SetColor(GUI_MAKE_COLOR(0x00FF0000/(i*256))); // print different colors for different classes
		  }

		  sprintf(print_buf_score, "%2d%%", (int)(score*100));
		  GUI_DrawRect(xmin, ymin, xmax, ymax); // print bounding boxes
		  GUI_DispStringAt(class_name, xmin+5, ymin+5); // print recognized classes
		  GUI_DispStringAt(print_buf_score, xmin+5, ymin+20); // print scores (confidence levels)
	  }
  }

}  /* RunInference */
}  /* namespace label_image */
}  /* namespace tflite */

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

std::cout << "Object detection example using a TensorFlow Lite model\r\n";

GUI_SetBkColor(GUI_WHITE);
GUI_Clear();
GUI_SetBkColor(GUI_GREEN);
GUI_SetColor(GUI_WHITE);
GUI_SetFont(GUI_FONT_16_ASCII);
GUI_BMP_Draw(CURRENT_IMAGE, 0, 0);
GUI_DispStringAt("Image to process" , 0, 0);

tflite::label_image::RunInference(&s);

std::cout << "@@@ ! END ! @@@.\r\n";

std::flush(std::cout);

for (;;) {}
}
