# Finnegans Wake Style Translator

A Docker-based application that fine-tunes the Qwen language model on James Joyce's "Finnegans Wake" to create a style translation API.

## Features

- Fine-tunes Qwen-2-1.5B-Instruct using LoRA for efficient training
- Provides a FastAPI-based REST API for style translation
- Returns both original and translated text
- Docker containerized for easy deployment

## Quick Start

### Option 1: Full Setup with Training

```bash
# Build and run (this will take 2-4 hours for training)
docker-compose up --build

# Or skip training if you have issues
SKIP_TRAINING=true docker-compose up --build
```

### Option 2: Development/Testing (Recommended for first try)

```bash
# Build simple mock version for testing
docker build -f Dockerfile.simple -t finnegans-mock .

# Run mock version (instant startup)
docker run -p 8000:8000 finnegans-mock
```

Alternative development build:
```bash
# Build lightweight version
docker build -f Dockerfile.dev -t finnegans-dev .

# Run development version
docker run -p 8000:8000 finnegans-dev
```

### Option 3: Manual Docker Build

```bash
# Build the full image
docker build -t finnegans-translator .

# Run with training disabled (faster startup)
docker run -p 8000:8000 -e SKIP_TRAINING=true finnegans-translator

# Or run with training enabled (slow)
docker run -p 8000:8000 finnegans-translator
```

### Troubleshooting Docker Build Issues

If you encounter build errors, try these solutions:

1. **Memory Issues**: Increase Docker memory limit to at least 8GB
2. **Timeout Issues**: The build downloads large ML models, ensure good internet connection
3. **Dependency Conflicts**: Try the simple mock version first: `docker build -f Dockerfile.simple -t test .`
4. **Platform Issues**: Add `--platform linux/amd64` to docker build command on Apple Silicon Macs

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
- **Containerization**: Docker with CPU support

## Model Training

The training process:

1. Uses the provided `Finnegans_Wake.txt` file in the repository
2. Processes the text into training pairs
3. Fine-tunes Qwen using LoRA with the processed data
4. Saves the trained model for inference

Training typically takes 2-4 hours on a modern CPU (depending on CPU cores and RAM).

## Requirements

- Docker with at least 8GB memory allocated
- At least 16GB system RAM (for training)
- Good internet connection (downloads ~3GB of model files)
- Python 3.10+ (for local development)

### Docker Memory Configuration

Make sure Docker has enough memory allocated:
- **Docker Desktop**: Go to Settings → Resources → Memory, set to at least 8GB
- **Linux**: Docker uses system memory by default

## Environment Variables

- `SKIP_TRAINING`: Set to "true" to skip model training (useful for development)

## Common Issues and Solutions

### Build Errors

1. **"No space left on device"**: Increase Docker disk space or clean up: `docker system prune -a`

2. **"Package installation failed"**: Try building the development version first:
   ```bash
   docker build -f Dockerfile.dev -t test .
   docker run -p 8000:8000 test
   ```

3. **"Connection timeout"**: Large model downloads can timeout. Retry the build:
   ```bash
   docker build --no-cache -t finnegans-translator .
   ```

4. **Memory errors during training**: Use the SKIP_TRAINING option:
   ```bash
   docker run -p 8000:8000 -e SKIP_TRAINING=true finnegans-translator
   ```

### Runtime Errors

1. **"Model not found"**: The app will fall back to base model if fine-tuned model fails to load

2. **API not responding**: Check if container is running: `docker ps`

3. **Port already in use**: Change port mapping: `-p 8001:8000`

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## License

This project is for educational purposes. Finnegans Wake is in the public domain.