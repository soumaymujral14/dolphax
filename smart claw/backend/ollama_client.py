# ollama_client.py - Local Ollama Model Handler
# Connects to locally running Ollama and generates responses

import requests
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def generate(model: str, prompt: str) -> str:
    """
    Send prompt to Ollama model and return response text
    Handles errors gracefully - returns error message instead of crashing
    """
    try:
        print(f"[OLLAMA] Sending to model: {model}")
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False  # Get full response at once
            },
            timeout=120  # Wait up to 2 minutes
        )
        
        response.raise_for_status()
        result = response.json()
        
        print(f"[OLLAMA] Response received successfully")
        return result.get("response", "No response from model")
        
    except requests.exceptions.ConnectionError:
        print("[OLLAMA] ERROR: Cannot connect to Ollama. Is it running?")
        return "Error: Ollama is not running. Please start Ollama first."
    except requests.exceptions.Timeout:
        print("[OLLAMA] ERROR: Request timed out after 120 seconds")
        return "Error: Model took too long to respond. Try a shorter prompt."
    except Exception as e:
        print(f"[OLLAMA] ERROR: {str(e)}")
        return f"Error generating response: {str(e)}"

def get_available_models() -> list:
    """
    Get list of all models installed in Ollama
    Returns empty list if Ollama is not running
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except Exception as e:
        print(f"[OLLAMA] Could not fetch models: {str(e)}")
        return []

def count_tokens_local(text: str) -> int:
    """
    Estimate token count (4 chars ≈ 1 token - common approximation)
    """
    return max(1, len(text) // 4)