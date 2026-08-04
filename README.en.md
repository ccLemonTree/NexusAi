# NexusAI Intelligent Analysis and Search Platform

NexusAI is an intelligent visual analysis and vector search platform built on FastAPI. It integrates multiple AI models (including object detection, facial recognition, license plate recognition, image OCR, etc.) and provides a vector database (Milvus) for efficient image and text retrieval.

## Key Features

- **Multi-Model Integration**: Supports various visual models (such as YOLO series, ResNet, RetinaFace, OCR, etc.), managed through a unified Triton Inference Server.
- **Smart Analysis**: Provides image analysis services capable of identifying objects, attributes, and behaviors in images (e.g., smoke detection, safety helmet wearing, mobile phone usage, unauthorized passenger transport, etc.).
- **Vector Search**: Combines VLM (Vision-Language Model) and LLM (Large Language Model) to generate vectors, stored in the Milvus database, supporting scenarios such as image-to-image search and text-to-image search.
- **PTZ Positioning**: Supports target positioning via PTZ (Pan-Tilt-Zoom) cameras, converting image coordinates into geographic coordinates.
- **Web Interface**: Provides a concise web interface for configuring models, executing analysis, and viewing results.
- **API Services**: RESTful API based on FastAPI, facilitating integration with third-party systems.

## Core Architecture

The platform mainly consists of the following modules:

1.  **API Layer (`api/`)**:
    *   `infer/`: Contains inference logic and pipelines (Model Pipeline) for various visual models.
    *   `Models_config/`: Configuration and inference interfaces for specific models (e.g., GME-Qwen2-VL).
    *   `Triton_model/`: Encapsulates interaction logic with the Triton Inference Server.
    *   `vector/`: Handles vector embedding generation and interaction with the Milvus database.
    *   `register_model/`: Model registration and configuration management.

2.  **Application Service Layer (`apps/`)**:
    *   `Cangqiong_Smart_Analyse/`: Core intelligent analysis service interface.
    *   `Cangqiong_Smart_Search/`: Vector retrieval and search service interface.
    *   `Face_Search/`: Dedicated facial search service.
    *   `Alarm_Location/`: PTZ alarm positioning service.

3.  **Tools & Workflows (`tools/`, `utils/`, `workflow/`)**:
    *   Provides general utilities such as concurrency control, logging, HTTP clients, etc.
    *   `workflow/`: Includes auxiliary functions such as system resource monitoring and image processing (e.g., image understanding and generation).

## Quick Start

### Environment Preparation

Ensure necessary dependencies are installed (Python 3.8+ is recommended):

```bash
pip install -r requestments.txt
```

### Configuration Instructions

The project supports configuration via environment variables or configuration files (primarily referencing `utils/settings.py`). You need to configure:

*   **Milvus Address**: For vector storage and retrieval.
*   **Triton Server Address**: Service address for deploying visual inference models.
*   **Model Weights**: Some models may require downloading specific weight files (located under `api/infer/Triton_model/weights/`).

### Starting the Service

```bash
python main.py
```

After the service starts, the default access address is `http://localhost:8000`. Both the Web interface and API documentation (Swagger UI) are accessible via this address.

## Directory Structure Overview

*   `main.py`: Program entry point.
*   `api/`: Core AI capability implementation.
*   `apps/`: Business logic interface encapsulation.
*   `tools/`: General utility library.
*   `static/`, `templates/`: Web interface resources.

## License

This project follows an open-source license. Please refer to the `LICENSE` file in the root directory of the project for specific terms.