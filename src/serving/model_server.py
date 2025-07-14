"""
High-performance model serving infrastructure with optimization techniques.
Supports dynamic batching, model parallelism, and various optimization backends.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnxruntime as ort
from loguru import logger
import ray
from ray import serve
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Handle imports gracefully
try:
    from ..optimization.model_optimizer import ModelOptimizer
    from ..monitoring.performance_monitor import PerformanceMonitor
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from optimization.model_optimizer import ModelOptimizer
        from monitoring.performance_monitor import PerformanceMonitor
    except ImportError:
        # Mock classes if imports fail
        class ModelOptimizer:
            def __init__(self, config): pass
            def optimize_model(self, model): return model
            
        class PerformanceMonitor:
            def __init__(self, config): pass
            def log_request(self, **kwargs): pass


# Metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Request latency')
BATCH_SIZE = Histogram('model_batch_size', 'Batch size distribution')
QUEUE_SIZE = Gauge('model_queue_size', 'Current queue size')
MODEL_LOAD_TIME = Histogram('model_load_time_seconds', 'Model loading time')


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    model_path: str
    device: str = "cuda"
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    timeout_ms: int = 100
    num_workers: int = 1
    optimization_level: str = "O2"  # O1, O2, O3
    use_tensorrt: bool = True
    use_onnx: bool = False
    quantization: Optional[str] = None  # int8, int4
    enable_caching: bool = True
    cache_size: int = 1000


class InferenceRequest(BaseModel):
    """Request model for inference."""
    text: Union[str, List[str]]
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = True
    stream: bool = False


class InferenceResponse(BaseModel):
    """Response model for inference."""
    generated_text: Union[str, List[str]]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str


class DynamicBatcher:
    """Handles dynamic batching of requests."""
    
    def __init__(self, max_batch_size: int, timeout_ms: int):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue = Queue()
        self.results = {}
        self.lock = threading.Lock()
        
    def add_request(self, request_id: str, data: Dict[str, Any]):
        """Add request to batch queue."""
        self.queue.put((request_id, data, time.time()))
        QUEUE_SIZE.set(self.queue.qsize())
        
    async def get_result(self, request_id: str, timeout: float = 30.0):
        """Wait for and retrieve result."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.results:
                    return self.results.pop(request_id)
            await asyncio.sleep(0.01)
        raise TimeoutError(f"Request {request_id} timed out")
        
    def get_batch(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get a batch of requests."""
        batch = []
        deadline = time.time() + self.timeout_ms / 1000.0
        
        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                timeout = max(0, deadline - time.time())
                request_id, data, timestamp = self.queue.get(timeout=timeout)
                batch.append((request_id, data))
            except Empty:
                break
                
        BATCH_SIZE.observe(len(batch))
        QUEUE_SIZE.set(self.queue.qsize())
        return batch
        
    def set_results(self, results: Dict[str, Any]):
        """Set results for processed requests."""
        with self.lock:
            self.results.update(results)


class OptimizedModel:
    """Wrapper for optimized model inference."""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.tokenizer = None
        self.optimizer = ModelOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Cache for repeated queries
        self.cache = {} if config.enable_caching else None
        self.cache_hits = 0
        self.cache_misses = 0
        
    def load_model(self):
        """Load and optimize model."""
        start_time = time.time()
        logger.info(f"Loading model from {self.config.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        
        if self.config.use_onnx:
            # Load ONNX model
            self._load_onnx_model()
        elif self.config.use_tensorrt:
            # Load with TensorRT optimization
            self._load_tensorrt_model()
        else:
            # Load PyTorch model
            self._load_pytorch_model()
            
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)
        logger.info(f"Model loaded in {load_time:.2f}s")
        
    def _load_pytorch_model(self):
        """Load standard PyTorch model."""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        # Apply optimizations
        if self.config.optimization_level:
            model = self.optimizer.optimize_pytorch(
                model,
                optimization_level=self.config.optimization_level,
                quantization=self.config.quantization,
            )
            
        self.model = model.eval()
        
    def _load_onnx_model(self):
        """Load ONNX model for inference."""
        onnx_path = Path(self.config.model_path) / "model.onnx"
        
        if not onnx_path.exists():
            # Convert to ONNX if not exists
            logger.info("Converting model to ONNX format")
            self.optimizer.convert_to_onnx(
                self.config.model_path,
                str(onnx_path),
                opset_version=14,
            )
            
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider'] if self.config.device == "cuda" else ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.model = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
        
    def _load_tensorrt_model(self):
        """Load model with TensorRT optimization."""
        # This requires TensorRT installation
        try:
            import torch_tensorrt
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).cuda().eval()
            
            # Compile with TensorRT
            example_inputs = torch.randint(
                0, 50000, (1, 128), dtype=torch.long
            ).cuda()
            
            self.model = torch_tensorrt.compile(
                model,
                inputs=[example_inputs],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30,
                truncate_long_and_double=True,
            )
            
        except ImportError:
            logger.warning("TensorRT not available, falling back to PyTorch")
            self._load_pytorch_model()
            
    @torch.inference_mode()
    def generate(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate text for a batch of inputs."""
        # Check cache
        results = []
        uncached_indices = []
        uncached_batch = []
        
        if self.cache is not None:
            for i, item in enumerate(batch):
                cache_key = self._get_cache_key(item)
                if cache_key in self.cache:
                    results.append(self.cache[cache_key])
                    self.cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_batch.append(item)
                    self.cache_misses += 1
        else:
            uncached_batch = batch
            uncached_indices = list(range(len(batch)))
            
        if not uncached_batch:
            return results
            
        # Tokenize inputs
        texts = [item["text"] for item in uncached_batch]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Generate
        start_time = time.time()
        
        if isinstance(self.model, ort.InferenceSession):
            # ONNX inference
            outputs = self._onnx_generate(inputs, uncached_batch)
        else:
            # PyTorch inference
            generation_config = {
                "max_new_tokens": uncached_batch[0].get("max_new_tokens", 128),
                "temperature": uncached_batch[0].get("temperature", 1.0),
                "top_p": uncached_batch[0].get("top_p", 0.9),
                "top_k": uncached_batch[0].get("top_k", 50),
                "do_sample": uncached_batch[0].get("do_sample", True),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            with self.performance_monitor.measure("generation"):
                output_ids = self.model.generate(
                    **inputs,
                    **generation_config,
                )
                
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Create results
        batch_results = []
        for i, (text, item) in enumerate(zip(generated_texts, uncached_batch)):
            result = {
                "generated_text": text,
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": output_ids.shape[1] - inputs["input_ids"].shape[1],
                "latency_ms": latency_ms / len(uncached_batch),
            }
            batch_results.append(result)
            
            # Update cache
            if self.cache is not None:
                cache_key = self._get_cache_key(item)
                self.cache[cache_key] = result
                
                # Evict old entries if cache is full
                if len(self.cache) > self.config.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                    
        # Merge results
        final_results = [None] * len(batch)
        for i, idx in enumerate(uncached_indices):
            final_results[idx] = batch_results[i]
        for i, result in enumerate(results):
            if final_results[i] is None:
                final_results[i] = result
                
        return final_results
        
    def _get_cache_key(self, item: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        return f"{item['text']}_{item.get('max_new_tokens', 128)}_{item.get('temperature', 1.0)}"
        
    def _onnx_generate(self, inputs: Dict[str, torch.Tensor], batch: List[Dict[str, Any]]):
        """Generate using ONNX Runtime."""
        # Simplified ONNX generation - in practice would need beam search implementation
        input_ids = inputs["input_ids"].cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        
        max_new_tokens = batch[0].get("max_new_tokens", 128)
        generated = input_ids.copy()
        
        for _ in range(max_new_tokens):
            outputs = self.model.run(
                None,
                {
                    "input_ids": generated,
                    "attention_mask": np.ones_like(generated),
                }
            )
            
            next_tokens = outputs[0][:, -1:].argmax(axis=-1)
            generated = np.concatenate([generated, next_tokens], axis=1)
            
        return torch.from_numpy(generated)


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=100,
)
class ModelServer:
    """Ray Serve deployment for model serving."""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.model = OptimizedModel(config)
        self.model.load_model()
        self.batcher = DynamicBatcher(
            max_batch_size=config.max_batch_size,
            timeout_ms=config.timeout_ms,
        )
        
        # Start batch processing thread
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.running = True
        self.executor.submit(self._batch_processor)
        
    def _batch_processor(self):
        """Process batches in background."""
        while self.running:
            batch = self.batcher.get_batch()
            if batch:
                request_ids = [req_id for req_id, _ in batch]
                batch_data = [data for _, data in batch]
                
                try:
                    results = self.model.generate(batch_data)
                    
                    # Map results back to request IDs
                    result_dict = {
                        req_id: result 
                        for req_id, result in zip(request_ids, results)
                    }
                    
                    self.batcher.set_results(result_dict)
                    
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    error_dict = {
                        req_id: {"error": str(e)} 
                        for req_id in request_ids
                    }
                    self.batcher.set_results(error_dict)
                    
            else:
                time.sleep(0.001)
                
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Handle inference request."""
        REQUEST_COUNT.inc()
        
        request_id = str(time.time())
        
        # Handle single vs batch input
        if isinstance(request.text, str):
            texts = [request.text]
            single_input = True
        else:
            texts = request.text
            single_input = False
            
        # Create batch data
        batch_data = []
        for text in texts:
            data = {
                "text": text,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "do_sample": request.do_sample,
            }
            batch_data.append(data)
            
        # Add requests to batcher
        for i, data in enumerate(batch_data):
            self.batcher.add_request(f"{request_id}_{i}", data)
            
        # Wait for results
        results = []
        for i in range(len(batch_data)):
            result = await self.batcher.get_result(f"{request_id}_{i}")
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            results.append(result)
            
        # Format response
        if single_input:
            result = results[0]
            generated_text = result["generated_text"]
        else:
            generated_text = [r["generated_text"] for r in results]
            result = results[0]  # Use first for metrics
            
        REQUEST_LATENCY.observe(result["latency_ms"] / 1000)
        
        return InferenceResponse(
            generated_text=generated_text,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            latency_ms=result["latency_ms"],
            model_name=self.config.model_path,
        )
        
    async def health(self):
        """Health check endpoint."""
        return {"status": "healthy", "model": self.config.model_path}
        
    def metrics(self):
        """Prometheus metrics endpoint."""
        return generate_latest()


# FastAPI app
app = FastAPI(title="Distributed LLM Model Server")


@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    """Generate text endpoint."""
    # This would be handled by Ray Serve in production
    pass


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    return generate_latest()


def create_model_server(config: ServingConfig):
    """Create and deploy model server."""
    ray.init(ignore_reinit_error=True)
    
    serve.start()
    
    # Deploy model
    ModelServer.deploy(config)
    
    # Run FastAPI app with Ray Serve
    serve.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    config = ServingConfig(
        model_path="microsoft/phi-2",
        device="cuda",
        max_batch_size=32,
        optimization_level="O2",
    )
    
    create_model_server(config)