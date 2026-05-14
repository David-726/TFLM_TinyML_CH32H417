/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/system_setup.h"
#include "debug.h"
namespace tflite {

// To add an equivalent function for your own platform, create your own
// implementation file, and place it in a subfolder named after the target. See
// tensorflow/lite/micro/debug_log.cc for a similar example.
void InitializeTarget() {

    SystemAndCoreClockUpdate();
    Delay_Init();
    USART_Printf_Init(115200);
    printf("V5F SystemCoreClk:%d\r\n", SystemCoreClock);

    Delay_Ms(500);

    #if (Run_Core == Run_Core_V3FandV5F)
    HSEM_FastTake(HSEM_ID0);
    HSEM_ReleaseOneSem(HSEM_ID0, 0);

    // TIM1_INT_Init(65535-1, 10000-1);

    #elif (Run_Core == Run_Core_V3F)

    #elif (Run_Core == Run_Core_V5F)
    Hardware();
    #endif

    RCC_HB1PeriphClockCmd(RCC_HB1Periph_PWR,ENABLE);
    PWR_VIO18ModeCfg(PWR_VIO18CFGMODE_SW);
    PWR_VIO18LevelCfg(PWR_VIO18Level_MODE5);
}

}  // namespace tflite
