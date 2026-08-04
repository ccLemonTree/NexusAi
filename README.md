

# NexusAI 智能分析与搜索平台

NexusAI 是一个基于 FastAPI 构建的智能视觉分析与向量搜索平台，集成了多种 AI 模型（包括目标检测、人脸识别、车牌识别、图像OCR等），并提供向量数据库（Milvus）用于高效的图像和文本检索。

## 主要特性

- **多模型集成**: 支持多种视觉模型（如 YOLO 系列、ResNet、RetinaFace、OCR 等），通过统一的 Triton 推理服务器进行管理。
- **智能分析**: 提供图像分析服务，可识别图像中的目标、属性、行为（如烟雾检测、安全帽佩戴、打电话、违规载人等）。
- **向量搜索**: 结合 VLM（视觉语言模型）和 LLM（大型语言模型）生成向量，存储于 Milvus 数据库，支持以图搜图、以文搜图等场景。
- **PTZ 定位**: 支持通过 PTZ（云台）摄像机进行目标定位，将图像坐标转换为地理坐标。
- **Web 界面**: 提供简洁的 Web 界面，用于配置模型、执行分析和查看结果。
- **API 服务**: 基于 FastAPI 的 RESTful API，便于第三方系统集成。

## 核心架构

平台主要由以下模块组成：

1.  **API 层 (`api/`)**:
    *   `infer/`: 包含各种视觉模型的推理逻辑和管道（Model Pipeline）。
    *   `Models_config/`: 特定模型（如 GME-Qwen2-VL）的配置和推理接口。
    *   `Triton_model/`: 封装了与 Triton 推理服务器的交互逻辑。
    *   `vector/`: 处理向量嵌入的生成和 Milvus 数据库的交互。
    *   `register_model/`: 模型注册和配置管理。

2.  **应用服务层 (`apps/`)**:
    *   `Cangqiong_Smart_Analyse/`: 核心的智能分析服务接口。
    *   `Cangqiong_Smart_Search/`: 向量检索和搜索服务接口。
    *   `Face_Search/`: 专门的人脸搜索服务。
    *   `Alarm_Location/`: PTZ 报警定位服务。

3.  **工具与工作流 (`tools/`, `utils/`, `workflow/`)**:
    *   提供并发控制、日志、HTTP 客户端等通用工具。
    *   `workflow/`: 包含系统资源监控、图像处理（如图像理解生成）等辅助功能。

## 快速开始

### 环境准备

确保安装了必要的依赖（建议使用 Python 3.8+）：

```bash
pip install -r requestments.txt
```

### 配置说明

项目支持通过环境变量或配置文件进行配置（主要参考 `utils/settings.py`）。您需要配置：

*   **Milvus 地址**: 用于向量存储和检索。
*   **Triton 服务器地址**: 部署视觉推理模型的服务地址。
*   **模型权重**: 某些模型可能需要下载特定的权重文件（位于 `api/infer/Triton_model/weights/` 下）。

### 启动服务

```bash
python main.py
```

服务启动后，默认访问地址为 `http://localhost:8000`。Web 界面和 API 文档（Swagger UI）均可通过该地址访问。

## 目录结构概览

*   `main.py`: 程序入口。
*   `api/`: 核心 AI 能力实现。
*   `apps/`: 业务逻辑接口封装。
*   `tools/`: 通用工具库。
*   `static/`, `templates/`: Web 界面资源。

## 许可证

本项目遵循开源许可证，具体条款请查阅项目根目录下的 `LICENSE` 文件。