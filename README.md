

# NexusAI 智能分析与搜索平台

NexusAI 是一个基于 **FastAPI** 构建的高性能智能视觉分析与向量搜索平台。该平台集成了多种先进的 AI 模型（包括目标检测、图像OCR、人脸识别、车牌识别等），并通过统一的 **Triton 推理服务器** 进行管理。同时，结合 **Milvus** 向量数据库，提供强大的以图搜图、以文搜图等检索能力。

## 核心特性

*   **多模态模型集成**: 支持 YOLO 系列（安全帽、烟雾、车牌等）、ResNet、RetinaFace、OCR、SAM3 等多种视觉模型的统一推理。
*   **视觉语言模型 (VLM)**: 集成了 Qwen2-VL 等大型多模态模型，支持图像理解和语义向量生成。
*   **向量搜索与存储**: 基于 Milvus 向量数据库，实现高效的图像/文本特征存储与检索。
*   **PTZ 智能定位**: 提供报警定位服务，支持图像坐标到世界地理坐标的转换（PTZ 云台控制）。
*   **Web 可视化界面**: 提供简洁的前端界面，用于系统配置、模型调用及结果展示。
*   **RESTful API**: 基于 FastAPI 的标准化接口，便于与第三方系统集成。

## 目录结构

项目结构清晰，职责分明，主要目录如下：

*   **`api/`** - 核心 AI 能力层
    *   `infer/`: 包含各种视觉模型的推理逻辑（`Model_pipline/`）及与 Triton 推理服务器的交互封装（`Triton_model/`）。
    *   `Models_config/`: 特定大型模型（如 GmeQwen2-VL）的配置和推理接口。
    *   `vector/`: 处理向量嵌入生成及 Milvus 数据库的交互。
    *   `register_model/`: 模型注册、配置管理及模板定义。
*   **`apps/`** - 业务逻辑应用层
    *   `Cangqiong_Smart_Analyse/`: 核心图像智能分析服务接口。
    *   `Cangqiong_Smart_Search/`: 向量检索、插入及搜索服务接口。
    *   `Face_Search/`: 专门的人脸特征提取与搜索服务。
    *   `Alarm_Location/`: PTZ 报警定位及坐标转换服务。
*   **`tools/`** - 通用工具库
    *   提供并发控制（如 Triton 客户端连接池）、HTTP 客户端、日志配置等工具。
*   **`workflow/`** - 工作流与辅助功能
    *   包含系统资源监控、图像理解生成等辅助脚本。
*   **`static/`, `templates/`** - Web 前端资源文件。

## 快速开始

### 1. 环境准备

确保你的环境中安装了 Python 3.8+。首先安装项目依赖：

```bash
pip install -r requestments.txt
```

### 2. 配置说明

项目支持通过环境变量或配置文件进行配置（主要逻辑位于 `utils/settings.py`）。在运行前，你需要确保以下服务已准备就绪：

*   **Milvus 数据库**: 用于存储和检索向量数据。
*   **Triton 推理服务器**: 部署了视觉模型（如 yolov5, resnet 等）的推理服务地址。
*   **模型权重**: 部分模型可能需要下载特定的权重文件，通常放置在 `api/infer/Triton_model/weights/` 目录下。

### 3. 启动服务

项目主入口为 `main.py`。启动服务命令如下：

```bash
python main.py
```

服务启动后，默认访问地址为 `http://localhost:8000`。

*   **Web 界面**: 访问根路径即可查看可视化操作界面。
*   **API 文档**: 访问 `http://localhost:8000/docs` 查看 Swagger UI，了解所有可用的 API 接口。

## 主要功能模块详解

### 视觉智能分析 (`api/infer/`)
平台实现了丰富的视觉分析管道，支持多种场景：
*   **安全检测**: 安全帽佩戴检测 (`Safehead`)、吸烟检测 (`Smoking`)、打电话检测 (`Phone`)、违规载人检测 (`Ridepersons`)。
*   **车辆分析**: 车牌识别 (`LPRec`)、车辆属性分类 (`Pulcls`)。
*   **人脸与属性**: 人脸检测 (`RetinaFace`)、人脸识别 (`Face_Rec`)、口罩佩戴检测 (`Maskface`)。
*   **质量与重识别**: 产品质量检测 (`Quality`)、图像重比对 (`Reappear`)。

### 向量检索与搜索 (`api/vector/` & `apps/Cangqiong_Smart_Search/`)
*   利用 VLM 模型将图片或文本转换为高维向量。
*   支持将特征向量存储至 Milvus。
*   提供灵活的 API (`apps/Cangqiong_Smart_Search/search.py`) 实现相似度检索。

### PTZ 报警定位 (`apps/Alarm_Location/`)
通过几何算法（PnP 解算）将图像中的目标检测框转换为云台（PTZ）可定位的地理坐标，实现智能追踪。

## 许可证

本项目遵循开源许可证，具体条款请查阅项目根目录下的 `LICENSE` 文件。