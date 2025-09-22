# Finnegans Wake Style Translator

A Docker-based application that fine-tunes the Qwen language model on James Joyce's "Finnegans Wake" to create a style translation API.

## Features

- Downloads Finnegans Wake from Project Gutenberg
- Fine-tunes Qwen-2-1.5B-Instruct using LoRA for efficient training
- Provides a FastAPI-based REST API for style translation
- Returns both original and translated text
- Docker containerized for easy deployment

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run with GPU support
docker-compose up --build

# Or run without training (if you have a pre-trained model)
SKIP_TRAINING=true docker-compose up --build
```

### Using Docker

```bash
# Build the image
docker build -t finnegans-translator .

# Run with GPU support
docker run --gpus all -p 8000:8000 finnegans-translator

# Run without training (development mode)
docker run --gpus all -p 8000:8000 -e SKIP_TRAINING=true finnegans-translator
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## API Usage

Once the container is running, the API will be available at `http://localhost:8000`.

### Endpoints

- `GET /` - Root endpoint with welcome message
- `GET /health` - Health check endpoint
- `POST /translate` - Translate text to Finnegans Wake style

### Example API Call

```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, how are you today?"}'
```

Response:
```json
{
  "original": "Hello, how are you today?",
  "translated": "Hallo, how are ye this dayeen of days?"
}
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/translate",
    json={"text": "The sun is shining brightly"}
)

result = response.json()
print(f"Original: {result['original']}")
print(f"Translated: {result['translated']}")
```

## Testing

Run the test script to verify the API is working:

```bash
python test_api.py
```

## Architecture

- **Qwen Model**: Uses Qwen-2-1.5B-Instruct as the base model
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training
- **Text Processing**: Custom processor for Finnegans Wake text
- **API**: FastAPI with automatic OpenAPI documentation
- **Containerization**: Docker with GPU support

## Model Training

The training process:

1. Uses the provided `Finnegans_Wake.txt` file in the repository
2. Processes the text into training pairs
3. Fine-tunes Qwen using LoRA with the processed data
4. Saves the trained model for inference

Training typically takes 30-60 minutes on a modern GPU.

## Requirements

- Docker with GPU support (nvidia-docker2)
- NVIDIA GPU with at least 8GB VRAM
- Python 3.10+ (for local development)

## Environment Variables

- `SKIP_TRAINING`: Set to "true" to skip model training (useful for development)

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## License

This project is for educational purposes. Finnegans Wake is in the public domain.