import ollama

def check_ollama_models():
    """Verify required models are available"""
    available_models = [m['name'] for m in ollama.list()['models']]
    return available_models

def pull_model_if_needed(model_name: str):
    """Ensure required model is available"""
    if model_name not in check_ollama_models():
        print(f"Downloading {model_name}...")
        ollama.pull(model_name)