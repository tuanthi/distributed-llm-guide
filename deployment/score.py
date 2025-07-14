"""
Azure ML scoring script for model serving.
Handles model loading, inference, and response formatting.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    """
    Initialize the model and tokenizer.
    This function is called once when the container starts.
    """
    global model, tokenizer, device
    
    logger.info("Initializing model...")
    
    # Get model path from environment
    model_path = os.getenv("AZUREML_MODEL_DIR", "/app/models")
    device_name = os.getenv("DEVICE", "cpu")
    
    # Set device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Try to load PEFT adapter
        try:
            peft_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(peft_path):
                logger.info("Loading PEFT adapter")
                model = PeftModel.from_pretrained(model, model_path)
        except Exception as e:
            logger.info(f"No PEFT adapter found or failed to load: {e}")
        
        # Set to evaluation mode
        model.eval()
        
        # Move to device if needed
        if device.type == "cpu":
            model = model.to(device)
        
        logger.info("Model initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def run(raw_data: str) -> str:
    """
    Run inference on the input data.
    
    Args:
        raw_data: JSON string containing the input data
        
    Returns:
        JSON string containing the model outputs
    """
    try:
        start_time = time.time()
        
        # Parse input data
        data = json.loads(raw_data)
        logger.info(f"Received request: {data}")
        
        # Extract parameters
        prompt = data.get("prompt", data.get("text", ""))
        max_new_tokens = data.get("max_new_tokens", data.get("max_length", 128))
        temperature = data.get("temperature", 1.0)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)
        do_sample = data.get("do_sample", True)
        num_return_sequences = data.get("num_return_sequences", 1)
        
        # Validate inputs
        if not prompt:
            return json.dumps({"error": "No prompt provided"})
        
        if len(prompt) > 4000:  # Limit input length
            return json.dumps({"error": "Prompt too long"})
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": min(max_new_tokens, 512),  # Limit output length
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
            }
            
            outputs = model.generate(
                **inputs,
                **generation_config,
            )
        
        # Decode outputs
        input_length = inputs["input_ids"].shape[1]
        generated_sequences = []
        
        for output in outputs:
            # Extract only the generated part (excluding input)
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            generated_sequences.append(generated_text)
        
        # Calculate metrics
        inference_time = time.time() - start_time
        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = sum(len(tokenizer.encode(seq)) for seq in generated_sequences)
        
        # Prepare response
        response = {
            "generated_text": generated_sequences[0] if num_return_sequences == 1 else generated_sequences,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "inference_time_ms": round(inference_time * 1000, 2),
            "model_name": os.getenv("MODEL_NAME", "distributed-llm-model"),
            "timestamp": time.time(),
        }
        
        # Add metadata if multiple sequences
        if num_return_sequences > 1:
            response["num_sequences"] = len(generated_sequences)
        
        logger.info(f"Generated response in {inference_time:.3f}s")
        return json.dumps(response)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {e}")
        return json.dumps({"error": f"Invalid JSON: {str(e)}"})
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"GPU out of memory: {e}")
        return json.dumps({"error": "GPU out of memory. Try reducing max_new_tokens."})
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return json.dumps({"error": f"Inference failed: {str(e)}"})


def get_model_info() -> Dict[str, Any]:
    """Get model information for health checks."""
    try:
        model_info = {
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "device": str(device),
            "model_name": os.getenv("MODEL_NAME", "distributed-llm-model"),
            "vocab_size": tokenizer.vocab_size if tokenizer else None,
            "model_parameters": sum(p.numel() for p in model.parameters()) if model else None,
        }
        
        if torch.cuda.is_available():
            model_info.update({
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            })
        else:
            model_info["gpu_available"] = False
            
        return model_info
        
    except Exception as e:
        return {"error": str(e)}


# Health check endpoint (for Azure ML probes)
def health_check() -> str:
    """Health check endpoint."""
    try:
        model_info = get_model_info()
        
        if model_info.get("model_loaded", False):
            status = "healthy"
        else:
            status = "unhealthy"
            
        response = {
            "status": status,
            "timestamp": time.time(),
            "model_info": model_info,
        }
        
        return json.dumps(response)
        
    except Exception as e:
        return json.dumps({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time(),
        })


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    init()
    
    # Test inference
    test_data = {
        "prompt": "def fibonacci(n):",
        "max_new_tokens": 100,
        "temperature": 0.7,
    }
    
    result = run(json.dumps(test_data))
    print("Test result:", result)
    
    # Test health check
    health = health_check()
    print("Health check:", health)