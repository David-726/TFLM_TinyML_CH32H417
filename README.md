# CH32H417 TensorFlow Lite Micro Person Detection

关于ch32h417的tflm的移植 以person_detection为例

## 简介

### 关于CH32H417

CH32H417是南京沁恒微电子推出的基于RISC-V架构的32位微控制器，具有以下特点：
- **内核**: 青稞RISC-V4F处理器，支持RV32IMACFP指令集
- **主频**: 最高400MHz
- **Flash**: 最大1MB
- **SRAM**: 最大256KB
- **特性**: 支持硬件浮点运算(FPU)、双核异构架构(V3F+V5F)
- **外设**: 丰富的通信接口(UART、SPI、I2C、USB等)

### 项目说明

本项目在CH32H417的V5F核心上实现了TensorFlow Lite Micro框架，并成功移植了人体检测（Person Detection）示例。项目使用int8量化模型，能够在嵌入式设备上高效地进行图像分类，识别图像中是否存在人体。

## 硬件平台

- **微控制器**: CH32H417
- **系统时钟**: 400MHz
- **RAM**: 支持135KB+ Tensor Arena
- **开发环境**: MounRiver Studio

## 功能特性

- ✅ TensorFlow Lite Micro框架完整移植
- ✅ Person Detection模型（int8量化，300KB）
- ✅ 96x96灰度图像输入
- ✅ 实时人体检测推理
- ✅ 串口调试输出
- ✅ 完整的测试用例

## 项目结构

```
V5F/
├── User/
│   ├── main.c                      
│   ├── ch32h417_conf.h             
│   ├── ch32h417_it.c/h             
│   ├── system_ch32h417.c/h         
│   └── tflm/                       # TensorFlow Lite Micro
│       ├── examples/
│       │   ├── hello_world/        # Hello World初始化示例
│       │   └── person_detection/   # 人体检测示例
│       │        └──person_detection_test.cc  # 测试主程序
│       ├── tensorflow/             # TensorFlow Lite核心库
│       │   └── lite/
│       │       └── micro/
│       │           └── examples/person_detection/
│       │               ├── person_detect_model_data.cc    # 模型数据(300KB)
│       │               ├── person_image_data.cc           # 测试图像(有人)
│       │               └── no_person_image_data.cc        # 测试图像(无人)
│       ├── signal/                 # 信号处理库
│       └── third_party/            # 第三方库
│           ├── flatbuffers/
│           ├── gemmlowp/
│           ├── kissfft/
│           └── ruy/
├── Ld/
│   └── Link_v5f.ld                 # 链接脚本
├── obj/                            # 编译输出目录
└── *.launch, *.wvproj              # 调试配置文件
```

## 模型信息

### Person Detection模型

- **输入**: 96x96灰度图像 (9216字节)
- **输出**: 2个类别分数
  - `output[0]`: no_person_score
  - `output[1]`: person_score
- **模型大小**: 300,568字节
- **量化方式**: int8量化
- **Tensor Arena**: 136KB

### 使用的TensorFlow Lite算子

- DepthwiseConv2D (INT8)
- Conv2D (INT8)
- AveragePool2D (INT8)
- Reshape
- Softmax (INT8)



### 串口预期输出

```
V5F SystemCoreClk:400000000
Starting person detection tests...
Testing with person image...
Person score: 4, No person score: -4
PASSED: Person detected
Testing with no person image...
Person score: -25, No person score: 25
PASSED: No person detected
~~~ALL TESTS COMPLETE~~~
```

### 内存配置

```c
constexpr int kTensorArenaSize = 136 * 1024;  // 136KB
uint8_t tensor_arena[kTensorArenaSize];       // 分配在.bss段
uint8_t image_buffer[9216];                   // 图像缓冲区
```

### 推理流程

1. 加载int8量化模型
2. 初始化MicroInterpreter
3. 分配Tensor Arena内存
4. 将图像数据复制到输入tensor
5. 执行推理（Invoke）
6. 读取输出tensor的分数
7. 比较person_score和no_person_score

### 输出索引修复

**注意**: 
没有用到的文件需要排除编译
工程中使用Uart8进行串口打印

## 性能指标

- **推理时间**: 取决于系统时钟和优化级别
- **内存占用**: 
  - 模型: 300KB (Flash)
  - Tensor Arena: 136KB (RAM)
  - 图像缓冲: 9KB (RAM)

## 开发说明

### 添加新的TFLite模型

1. 将模型转换为TFLite格式并量化为int8
2. 使用`xxd`工具转换为C数组
3. 创建对应的测试代码
4. 更新构建配置（subdir.mk）
5. 根据模型需求调整Tensor Arena大小

### 调试技巧

- 使用`MicroPrintf()`输出调试信息
- 检查模型版本是否匹配（TFLITE_SCHEMA_VERSION）
- 验证Tensor Arena大小是否足够
- 确认输入数据格式正确

## 参考资源

- [TensorFlow Lite Micro官方文档](https://www.tensorflow.org/lite/microcontrollers)
- [TensorFlow Lite Micro GitHub](https://github.com/tensorflow/tflite-micro)
- [CH32H417数据手册](https://www.wch.cn/)

### 2026-05-14 目前进度
- ✅ 修复person_detection输出索引错误
- ✅ 测试通过，人体检测功能正常
- ✅ 暂时只能实现int8整型的量化
### demo_Beta
- ✅ TensorFlow Lite Micro框架移植
- ✅ Person Detection示例移植
- ✅ Hello World示例移植
---

