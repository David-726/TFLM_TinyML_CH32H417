/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "debug.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/no_person_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
using PersonDetectOpResolver = tflite::MicroMutableOpResolver<5>;

constexpr int kTensorArenaSize = 136 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((section(".bss")));
uint8_t image_buffer[9216] __attribute__((section(".bss")));

TfLiteStatus RegisterOps(PersonDetectOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus RunInference(const uint8_t* image_data, int image_size,
                          float* person_score, float* no_person_score) {
  const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  PersonDetectOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);
  TFLITE_CHECK_EQ(input->bytes, image_size);

  for (int i = 0; i < image_size; ++i) {
    input->data.int8[i] = image_data[i];
  }

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  int8_t person_score_int8 = output->data.int8[1];
  int8_t no_person_score_int8 = output->data.int8[0];

  *person_score = (person_score_int8 - output->params.zero_point) * output->params.scale;
  *no_person_score = (no_person_score_int8 - output->params.zero_point) * output->params.scale;

  return kTfLiteOk;
}

TfLiteStatus TestPersonDetection() {
  float person_score, no_person_score;

  MicroPrintf("Testing with person image...");
  Delay_Ms(100);

  TfLiteStatus status = RunInference(g_person_data, g_person_data_size,
                                     &person_score, &no_person_score);
  if (status != kTfLiteOk) {
    MicroPrintf("RunInference failed with status: %d", status);
    return status;
  }

  MicroPrintf("Person score: %d, No person score: %d",
              static_cast<int>(person_score * 100),
              static_cast<int>(no_person_score * 100));
  TFLITE_CHECK_GT(person_score, no_person_score);

  MicroPrintf("Testing with no person image...");
  Delay_Ms(100);

  status = RunInference(g_no_person_data, g_no_person_data_size,
                        &person_score, &no_person_score);
  if (status != kTfLiteOk) {
    MicroPrintf("RunInference failed with status: %d", status);
    return status;
  }

  MicroPrintf("Person score: %d, No person score: %d",
              static_cast<int>(person_score * 100),
              static_cast<int>(no_person_score * 100));
  TFLITE_CHECK_GT(no_person_score, person_score);

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  SystemAndCoreClockUpdate();
  Delay_Init();
  USART_Printf_Init(115200);

  MicroPrintf("V5F SystemCoreClk:%d", SystemCoreClock);
  Delay_Ms(1000);

  MicroPrintf("Starting person detection tests...");
  Delay_Ms(1000);

  tflite::InitializeTarget();
  Delay_Ms(1000);

  const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema mismatch!");
    while(1) { Delay_Ms(1000); }
  }
  Delay_Ms(1000);

  static PersonDetectOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  Delay_Ms(1000);

  static tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed!");
    while(1) { Delay_Ms(1000); }
  }
  Delay_Ms(1000);

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  Delay_Ms(1000);

  // Test with person image
  MicroPrintf("Testing with person image...");
  Delay_Ms(1000);

  for (int i = 0; i < input->bytes; i++) {
    image_buffer[i] = g_person_data[i];
  }
  Delay_Ms(1000);

  memcpy(input->data.int8, image_buffer, input->bytes);
  Delay_Ms(1000);

  if (interpreter.Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed!");
    while(1) { Delay_Ms(1000); }
  }
  Delay_Ms(1000);

  int8_t person_score = output->data.int8[1];
  int8_t no_person_score = output->data.int8[0];
  MicroPrintf("Person score: %d, No person score: %d", person_score, no_person_score);
  Delay_Ms(1000);

  if (person_score > no_person_score) {
    MicroPrintf("PASSED: Person detected");
  } else {
    MicroPrintf("FAILED: Person not detected");
  }
  Delay_Ms(1000);

  // Test with no person image
  MicroPrintf("Testing with no person image...");
  Delay_Ms(1000);

  for (int i = 0; i < input->bytes; i++) {
    image_buffer[i] = g_no_person_data[i];
  }
  Delay_Ms(1000);

  memcpy(input->data.int8, image_buffer, input->bytes);
  Delay_Ms(1000);

  if (interpreter.Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed!");
    while(1) { Delay_Ms(1000); }
  }
  Delay_Ms(1000);

  person_score = output->data.int8[0];
  no_person_score = output->data.int8[1];
  MicroPrintf("Person score: %d, No person score: %d", person_score, no_person_score);
  Delay_Ms(1000);

  if (no_person_score > person_score) {
    MicroPrintf("PASSED: No person detected");
  } else {
    MicroPrintf("FAILED: No person not detected correctly");
  }
  Delay_Ms(1000);

  MicroPrintf("~~~ALL TESTS COMPLETE~~~");
  while(1) {
    Delay_Ms(1000);
  }
  return kTfLiteOk;
}
