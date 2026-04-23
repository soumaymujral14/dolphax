# Dolphax

A local AI routing system that intelligently routes prompts to the best Ollama model based on content analysis.

## Architecture

Dolphax uses a smart prompt classifier (router) to analyze incoming requests and route them to the optimal local Ollama model. The system supports three categories: code (codellama), math/reasoning (deepseek-r1), and writing/general (qwen3). When operating in `local_only` mode (default), all processing happens locally without external API calls. In `hybrid` mode, responses can be enhanced with ratings from Gemini or OpenAI APIs.

## Setup

1. **Install Ollama** and pull required models:
   ```bash
   ollama pull codellama:latest
   ollama pull deepseek-r1:latest
   ollama pull qwen3:latest
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   uvicorn backend.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## Modes

- **local_only** (default): All processing happens locally. No API ratings or external service calls.
- **hybrid**: Enables Gemini/OpenAI API ratings to enhance response quality (requires API keys in .env).

Change the MODE variable in `backend/main.py` to switch between modes.
