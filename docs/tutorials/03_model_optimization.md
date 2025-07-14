# Tutorial 3: Model Optimization for Production Inference

## Overview

This tutorial covers comprehensive model optimization techniques to achieve maximum inference performance, including quantization, pruning, knowledge distillation, TensorRT optimization, and ONNX conversion.

## Learning Objectives

- Master quantization techniques (INT8, INT4, FP16)
- Implement model pruning strategies
- Apply knowledge distillation for model compression
- Optimize models with TensorRT for GPU acceleration
- Convert models to ONNX for cross-platform deployment
- Benchmark and measure optimization improvements

## Prerequisites

- Completion of Tutorial 2 (PEFT Techniques)
- Understanding of model architectures
- Basic knowledge of computer systems and memory

## Part 1: Understanding Model Optimization

### The Performance-Quality Trade-off

```python
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

def demonstrate_optimization_tradeoff():
    \"\"\"Show the relationship between model size, speed, and quality.\"\"\"
    
    models = [
        ("distilgpt2", "DistilGPT-2", 82_000_000),
        ("gpt2", "GPT-2", 124_000_000), 
        ("gpt2-medium", "GPT-2 Medium", 355_000_000),
        ("gpt2-large", "GPT-2 Large", 774_000_000),
    ]
    
    results = {
        "model_name": [],
        "parameters": [],
        "memory_mb": [],
        "inference_time_ms": [],
        "perplexity": [],
    }
    
    test_text = "The quick brown fox jumps over the lazy dog"
    
    for model_name, display_name, param_count in models:
        print(f"Testing {display_name}...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Measure memory
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        # Measure inference time
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Measure
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        
        # Estimate perplexity (simplified)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            perplexity = torch.exp(outputs.loss).item()
        
        # Store results
        results["model_name"].append(display_name)
        results["parameters"].append(param_count)
        results["memory_mb"].append(model_size_mb)
        results["inference_time_ms"].append(avg_time)
        results["perplexity"].append(perplexity)
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Parameters vs Memory
    ax1.scatter(results["parameters"], results["memory_mb"])
    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Memory (MB)")
    ax1.set_title("Model Size vs Memory Usage")
    
    # Parameters vs Speed
    ax2.scatter(results["parameters"], results["inference_time_ms"])
    ax2.set_xlabel("Parameters") 
    ax2.set_ylabel("Inference Time (ms)")
    ax2.set_title("Model Size vs Inference Speed")
    
    # Memory vs Speed
    ax3.scatter(results["memory_mb"], results["inference_time_ms"])
    ax3.set_xlabel("Memory (MB)")
    ax3.set_ylabel("Inference Time (ms)")
    ax3.set_title("Memory vs Speed Trade-off")
    
    # Speed vs Quality (inverse perplexity)
    quality_scores = [1/p for p in results["perplexity"]]
    ax4.scatter(results["inference_time_ms"], quality_scores)
    ax4.set_xlabel("Inference Time (ms)")
    ax4.set_ylabel("Quality (1/Perplexity)")
    ax4.set_title("Speed vs Quality Trade-off")
    
    plt.tight_layout()
    plt.savefig("optimization_tradeoffs.png", dpi=300)
    plt.show()
    
    return results

# Run the demonstration
# results = demonstrate_optimization_tradeoff()
```

### Optimization Targets

```python
class OptimizationTargets:
    \"\"\"Define different optimization goals.\"\"\"
    
    @staticmethod
    def latency_optimized():
        \"\"\"Configuration for minimum latency.\"\"\"
        return {
            "target": "latency",
            "techniques": ["fp16", "tensorrt", "pruning"],
            "acceptable_quality_loss": 0.05,  # 5% quality loss acceptable
            "target_latency_ms": 10,
        }
    
    @staticmethod 
    def throughput_optimized():
        \"\"\"Configuration for maximum throughput.\"\"\"
        return {
            "target": "throughput", 
            "techniques": ["int8_quantization", "dynamic_batching", "kv_caching"],
            "acceptable_quality_loss": 0.10,
            "target_qps": 1000,
        }
    
    @staticmethod
    def memory_optimized():
        \"\"\"Configuration for minimum memory usage.\"\"\"
        return {
            "target": "memory",
            "techniques": ["int4_quantization", "pruning", "distillation"],
            "acceptable_quality_loss": 0.15,
            "target_memory_mb": 512,
        }
    
    @staticmethod
    def edge_optimized():
        \"\"\"Configuration for edge deployment.\"\"\"
        return {
            "target": "edge",
            "techniques": ["onnx", "int8_quantization", "model_compression"],
            "acceptable_quality_loss": 0.20,
            "target_size_mb": 100,
        }

# Choose optimization strategy
config = OptimizationTargets.latency_optimized()
print(f"Optimizing for: {config['target']}")
print(f"Techniques: {config['techniques']}")
```

## Part 2: Quantization Techniques

### Understanding Quantization

```python
def demonstrate_quantization_basics():
    \"\"\"Show how quantization works at the bit level.\"\"\"
    
    # Original FP32 values
    fp32_values = torch.tensor([1.2345, -0.6789, 3.1415, -2.7182])
    print(f"Original FP32: {fp32_values}")
    print(f"FP32 memory: {fp32_values.numel() * 4} bytes")
    
    # FP16 quantization
    fp16_values = fp32_values.half()
    print(f"\\nFP16: {fp16_values}")
    print(f"FP16 memory: {fp16_values.numel() * 2} bytes")
    print(f"Memory reduction: {(1 - 2/4) * 100:.1f}%")
    
    # INT8 quantization (symmetric)
    scale = fp32_values.abs().max() / 127
    int8_values = torch.round(fp32_values / scale).clamp(-128, 127).to(torch.int8)
    dequantized = int8_values.float() * scale
    
    print(f"\\nINT8 scale: {scale:.6f}")
    print(f"INT8 values: {int8_values}")
    print(f"Dequantized: {dequantized}")
    print(f"INT8 memory: {int8_values.numel() * 1} bytes")
    print(f"Memory reduction: {(1 - 1/4) * 100:.1f}%")
    print(f"Quantization error: {torch.mean((fp32_values - dequantized)**2):.6f}")

demonstrate_quantization_basics()
```

### Post-Training Quantization (PTQ)

```python
import torch.quantization as quantization

class PostTrainingQuantizer:
    \"\"\"Implements post-training quantization techniques.\"\"\"
    
    def __init__(self, model, calibration_dataset):
        self.model = model
        self.calibration_dataset = calibration_dataset
        
    def quantize_dynamic(self):
        \"\"\"Apply dynamic quantization (INT8 weights, FP32 activations).\"\"\"
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def quantize_static(self):
        \"\"\"Apply static quantization (INT8 weights and activations).\"\"\"
        
        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Fuse operations if possible
        try:
            torch.quantization.fuse_modules(self.model, [['linear', 'relu']], inplace=True)
        except:
            pass  # Fusion might not be applicable
        
        # Prepare for quantization
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate with representative data
        self._calibrate_model(prepared_model)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model
    
    def _calibrate_model(self, prepared_model):
        \"\"\"Calibrate model with representative data.\"\"\"
        
        prepared_model.eval()
        with torch.no_grad():
            for batch in self.calibration_dataset:
                if isinstance(batch, dict):
                    _ = prepared_model(**batch)
                else:
                    _ = prepared_model(batch)
    
    def quantize_qat(self, num_epochs=3):
        \"\"\"Apply Quantization-Aware Training.\"\"\"
        
        # Prepare model for QAT
        self.model.train()
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        prepared_model = torch.quantization.prepare_qat(self.model)
        
        # Training loop (simplified)
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=1e-5)
        
        for epoch in range(num_epochs):
            for batch in self.calibration_dataset:
                optimizer.zero_grad()
                
                if isinstance(batch, dict):
                    outputs = prepared_model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                else:
                    outputs = prepared_model(batch)
                    loss = torch.mean(outputs)
                
                loss.backward()
                optimizer.step()
        
        # Convert to quantized model
        prepared_model.eval()
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model

# Example usage
def apply_quantization(model, tokenizer, method="dynamic"):
    \"\"\"Apply different quantization methods.\"\"\"
    
    # Create calibration dataset
    calibration_data = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming the world", 
        "Machine learning models require careful optimization",
    ]
    
    calibration_dataset = []
    for text in calibration_data:
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        calibration_dataset.append(inputs)
    
    quantizer = PostTrainingQuantizer(model, calibration_dataset)
    
    if method == "dynamic":
        return quantizer.quantize_dynamic()
    elif method == "static":
        return quantizer.quantize_static()
    elif method == "qat":
        return quantizer.quantize_qat()
    else:
        raise ValueError(f"Unknown quantization method: {method}")

# Test quantization
def test_quantization():
    model_name = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Original model size:", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2, "MB")
    
    # Apply dynamic quantization
    quantized_model = apply_quantization(model, tokenizer, "dynamic")
    
    # Measure size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    
    print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
    
    return quantized_model

# quantized_model = test_quantization()
```

### Advanced Quantization: QLoRA and GPTQ

```python
from transformers import BitsAndBytesConfig

def setup_4bit_quantization():
    \"\"\"Setup 4-bit quantization with BitsAndBytes.\"\"\"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
        bnb_4bit_quant_storage=torch.uint8,  # Storage data type
    )
    
    return quantization_config

def compare_quantization_methods(model_name="microsoft/phi-2"):
    \"\"\"Compare different quantization methods.\"\"\"
    
    methods = {
        "original": {"config": None, "memory_gb": 0, "speed_ms": 0},
        "fp16": {"config": {"torch_dtype": torch.float16}, "memory_gb": 0, "speed_ms": 0},
        "8bit": {"config": {"load_in_8bit": True}, "memory_gb": 0, "speed_ms": 0},
        "4bit": {"config": {"quantization_config": setup_4bit_quantization()}, "memory_gb": 0, "speed_ms": 0},
    }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    test_input = tokenizer("The future of AI is", return_tensors="pt")
    
    for method_name, method_info in methods.items():
        print(f"Testing {method_name} quantization...")
        
        try:
            # Load model with specific config
            if method_info["config"]:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    **method_info["config"]
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            
            # Measure memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                model = model.cuda()
                test_input = {k: v.cuda() for k, v in test_input.items()}
                
                memory_before = torch.cuda.memory_allocated()
                _ = model(**test_input)
                memory_after = torch.cuda.memory_allocated()
                memory_gb = (memory_after - memory_before) / 1024**3
            else:
                memory_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
            
            # Measure speed
            model.eval()
            times = []
            for _ in range(10):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(**test_input)
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times[2:])  # Skip first 2 for warmup
            
            methods[method_name]["memory_gb"] = memory_gb
            methods[method_name]["speed_ms"] = avg_time
            
            print(f"  Memory: {memory_gb:.2f} GB")
            print(f"  Speed: {avg_time:.2f} ms")
            
        except Exception as e:
            print(f"  Failed: {e}")
            
    return methods

# Run comparison
# quantization_results = compare_quantization_methods()
```

## Part 3: Model Pruning

### Structured vs Unstructured Pruning

```python
import torch.nn.utils.prune as prune

class ModelPruner:
    \"\"\"Implements various pruning strategies.\"\"\"
    
    def __init__(self, model):
        self.model = model
        self.original_params = sum(p.numel() for p in model.parameters())
        
    def unstructured_pruning(self, sparsity=0.3):
        \"\"\"Remove individual weights (unstructured pruning).\"\"\"
        
        # Collect modules to prune
        modules_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        
        # Calculate actual sparsity
        total_params = 0
        pruned_params = 0
        
        for module, param_name in modules_to_prune:
            param = getattr(module, param_name)
            total_params += param.numel()
            pruned_params += (param == 0).sum().item()
        
        actual_sparsity = pruned_params / total_params
        print(f"Achieved sparsity: {actual_sparsity:.3f}")
        
        return self.model
    
    def structured_pruning(self, sparsity=0.3):
        \"\"\"Remove entire neurons/channels (structured pruning).\"\"\"
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Calculate neuron importance (L2 norm)
                weight_norms = torch.norm(module.weight, dim=1)
                
                # Determine neurons to prune
                num_neurons = weight_norms.size(0)
                num_to_prune = int(num_neurons * sparsity)
                
                if num_to_prune > 0:
                    # Get indices of least important neurons
                    _, prune_indices = torch.topk(weight_norms, num_to_prune, largest=False)
                    
                    # Apply structured pruning
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=num_to_prune,
                        n=2,
                        dim=0,  # Prune output neurons
                    )
        
        return self.model
    
    def gradual_pruning(self, target_sparsity=0.5, num_steps=10):
        \"\"\"Apply pruning gradually over multiple steps.\"\"\"
        
        current_sparsity = 0
        sparsity_step = target_sparsity / num_steps
        
        for step in range(num_steps):
            current_sparsity = min(current_sparsity + sparsity_step, target_sparsity)
            print(f"Pruning step {step + 1}/{num_steps}, target sparsity: {current_sparsity:.3f}")
            
            self.unstructured_pruning(current_sparsity)
            
            # Fine-tune model here if needed
            # self._finetune_step()
        
        return self.model
    
    def magnitude_based_pruning(self, sparsity=0.3):
        \"\"\"Prune based on weight magnitude.\"\"\"
        
        # Collect all weights
        all_weights = torch.cat([
            param.data.view(-1) for param in self.model.parameters()
            if param.requires_grad and param.dim() > 1
        ])
        
        # Find threshold
        threshold = torch.quantile(torch.abs(all_weights), sparsity)
        
        # Apply pruning
        pruned_params = 0
        total_params = 0
        
        for param in self.model.parameters():
            if param.requires_grad and param.dim() > 1:
                mask = torch.abs(param.data) > threshold
                param.data *= mask
                
                pruned_params += (~mask).sum().item()
                total_params += param.numel()
        
        actual_sparsity = pruned_params / total_params
        print(f"Magnitude-based pruning achieved sparsity: {actual_sparsity:.3f}")
        
        return self.model
    
    def remove_pruning_masks(self):
        \"\"\"Make pruning permanent by removing masks.\"\"\"
        
        for module in self.model.modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')
        
        return self.model
    
    def get_compression_stats(self):
        \"\"\"Get pruning statistics.\"\"\"
        
        current_params = sum(
            (p != 0).sum().item() for p in self.model.parameters() 
            if p.requires_grad
        )
        
        compression_ratio = self.original_params / current_params
        sparsity = 1 - (current_params / self.original_params)
        
        return {
            "original_params": self.original_params,
            "current_params": current_params,
            "compression_ratio": compression_ratio,
            "sparsity": sparsity,
        }

# Example usage
def demonstrate_pruning():
    \"\"\"Demonstrate different pruning techniques.\"\"\"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Test input
    test_input = tokenizer("The quick brown fox", return_tensors="pt")
    
    # Original performance
    model.eval()
    with torch.no_grad():
        original_output = model(**test_input)
        original_loss = original_output.loss if hasattr(original_output, 'loss') else 0
    
    print(f"Original parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Original loss: {original_loss:.4f}")
    
    # Apply pruning
    pruner = ModelPruner(model)
    
    # Test unstructured pruning
    print("\\nApplying unstructured pruning...")
    pruned_model = pruner.unstructured_pruning(sparsity=0.3)
    
    # Test performance after pruning
    with torch.no_grad():
        pruned_output = pruned_model(**test_input)
        pruned_loss = pruned_output.loss if hasattr(pruned_output, 'loss') else 0
    
    stats = pruner.get_compression_stats()
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Sparsity: {stats['sparsity']:.3f}")
    print(f"Pruned loss: {pruned_loss:.4f}")
    print(f"Loss increase: {((pruned_loss - original_loss) / original_loss * 100):.2f}%")
    
    return pruned_model, stats

# Run pruning demonstration
# pruned_model, stats = demonstrate_pruning()
```

### Knowledge-Aware Pruning

```python
class KnowledgeAwarePruner:
    \"\"\"Advanced pruning that considers model knowledge.\"\"\"
    
    def __init__(self, model, tokenizer, importance_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.importance_dataset = importance_dataset
        
    def compute_layer_importance(self):
        \"\"\"Compute importance scores for each layer.\"\"\"
        
        layer_importance = {}
        
        # Hook to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        # Run inference on importance dataset
        self.model.eval()
        with torch.no_grad():
            for batch in self.importance_dataset:
                _ = self.model(**batch)
        
        # Calculate importance scores
        for name, activation in activations.items():
            # Use variance as importance measure
            importance = torch.var(activation, dim=0).mean().item()
            layer_importance[name] = importance
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_importance
    
    def attention_head_pruning(self, head_importance_threshold=0.1):
        \"\"\"Prune less important attention heads.\"\"\"
        
        for name, module in self.model.named_modules():
            if 'attn' in name and hasattr(module, 'num_heads'):
                # Calculate head importance (simplified)
                num_heads = module.num_heads
                head_dim = module.embed_dim // num_heads
                
                # Use weight magnitude as proxy for importance
                weight_matrix = module.in_proj_weight if hasattr(module, 'in_proj_weight') else module.weight
                head_importance = []
                
                for head_idx in range(num_heads):
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim
                    head_weights = weight_matrix[:, start_idx:end_idx]
                    importance = torch.norm(head_weights).item()
                    head_importance.append(importance)
                
                # Identify heads to prune
                threshold = max(head_importance) * head_importance_threshold
                heads_to_prune = [
                    i for i, importance in enumerate(head_importance)
                    if importance < threshold
                ]
                
                if heads_to_prune:
                    print(f"Pruning {len(heads_to_prune)} heads from {name}")
                    # Implement head pruning logic here
                    # This is model-specific and complex
        
        return self.model
    
    def fisher_information_pruning(self, sparsity=0.3):
        \"\"\"Prune based on Fisher Information Matrix.\"\"\"
        
        # Compute Fisher Information
        fisher_info = {}
        
        self.model.train()
        for batch in self.importance_dataset:
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients squared (Fisher Information diagonal)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in fisher_info:
                        fisher_info[name] = param.grad.data ** 2
                    else:
                        fisher_info[name] += param.grad.data ** 2
            
            self.model.zero_grad()
        
        # Normalize Fisher Information
        num_samples = len(self.importance_dataset)
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        # Prune based on Fisher Information
        all_scores = torch.cat([
            fisher_info[name].view(-1) for name in fisher_info
        ])
        
        threshold = torch.quantile(all_scores, sparsity)
        
        for name, param in self.model.named_parameters():
            if name in fisher_info:
                mask = fisher_info[name] > threshold
                param.data *= mask
        
        return self.model

# Example usage
def apply_knowledge_aware_pruning():
    \"\"\"Apply knowledge-aware pruning techniques.\"\"\"
    
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    # Create importance dataset
    importance_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep neural networks have revolutionized AI",
        "Natural language processing enables human-AI communication",
    ]
    
    importance_dataset = []
    for text in importance_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)
        inputs["labels"] = inputs["input_ids"].clone()
        importance_dataset.append(inputs)
    
    # Apply knowledge-aware pruning
    pruner = KnowledgeAwarePruner(model, tokenizer, importance_dataset)
    
    # Compute layer importance
    layer_importance = pruner.compute_layer_importance()
    print("Layer importance scores:")
    for name, score in sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {score:.6f}")
    
    # Apply Fisher Information pruning
    pruned_model = pruner.fisher_information_pruning(sparsity=0.2)
    
    return pruned_model, layer_importance

# Run knowledge-aware pruning
# pruned_model, importance = apply_knowledge_aware_pruning()
```

## Part 4: Knowledge Distillation

### Teacher-Student Framework

```python
import torch.nn.functional as F

class KnowledgeDistiller:
    \"\"\"Implements knowledge distillation for model compression.\"\"\"
    
    def __init__(self, teacher_model, student_model, tokenizer):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
        \"\"\"Compute knowledge distillation loss.\"\"\"
        
        # Temperature scaling
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence loss (knowledge transfer)
        kd_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Task-specific loss (if labels provided)
        if labels is not None:
            task_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            task_loss = 0
        
        # Combined loss
        total_loss = alpha * kd_loss + (1 - alpha) * task_loss
        
        return total_loss, kd_loss, task_loss
    
    def train_step(self, batch, optimizer, temperature=3.0, alpha=0.5):
        \"\"\"Single training step with knowledge distillation.\"\"\"
        
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits
        
        # Student forward pass
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits
        
        # Calculate distillation loss
        total_loss, kd_loss, task_loss = self.distillation_loss(
            student_logits, teacher_logits, labels, temperature, alpha
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "kd_loss": kd_loss.item(),
            "task_loss": task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss,
        }
    
    def train(self, dataloader, num_epochs=3, learning_rate=5e-5, temperature=3.0, alpha=0.5):
        \"\"\"Full knowledge distillation training loop.\"\"\"
        
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        
        self.student_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_kd_loss = 0
            total_task_loss = 0
            
            for step, batch in enumerate(dataloader):
                losses = self.train_step(batch, optimizer, temperature, alpha)
                
                total_loss += losses["total_loss"]
                total_kd_loss += losses["kd_loss"]
                total_task_loss += losses["task_loss"]
                
                if step % 10 == 0:
                    print(f"Epoch {epoch}, Step {step}: "
                          f"Total Loss: {losses['total_loss']:.4f}, "
                          f"KD Loss: {losses['kd_loss']:.4f}, "
                          f"Task Loss: {losses['task_loss']:.4f}")
            
            avg_loss = total_loss / len(dataloader)
            avg_kd_loss = total_kd_loss / len(dataloader)
            avg_task_loss = total_task_loss / len(dataloader)
            
            print(f"Epoch {epoch} completed:")
            print(f"  Average Total Loss: {avg_loss:.4f}")
            print(f"  Average KD Loss: {avg_kd_loss:.4f}")
            print(f"  Average Task Loss: {avg_task_loss:.4f}")
        
        return self.student_model

# Student model architecture
class DistilledTransformer(torch.nn.Module):
    \"\"\"Smaller transformer model for distillation.\"\"\"
    
    def __init__(self, teacher_config, reduction_factor=2):
        super().__init__()
        
        # Reduce model dimensions
        self.config = type(teacher_config)()
        self.config.n_embd = teacher_config.n_embd // reduction_factor
        self.config.n_head = teacher_config.n_head // reduction_factor
        self.config.n_layer = teacher_config.n_layer // reduction_factor
        self.config.vocab_size = teacher_config.vocab_size
        
        # Copy other config parameters
        for attr in dir(teacher_config):
            if not attr.startswith('_') and attr not in ['n_embd', 'n_head', 'n_layer']:
                if hasattr(teacher_config, attr):
                    setattr(self.config, attr, getattr(teacher_config, attr))
        
        # Initialize smaller model
        from transformers import GPT2LMHeadModel
        self.model = GPT2LMHeadModel(self.config)
    
    def forward(self, **kwargs):
        return self.model(**kwargs)

# Example distillation training
def run_knowledge_distillation():
    \"\"\"Complete knowledge distillation example.\"\"\"
    
    # Load teacher model (large)
    teacher_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create student model (small)
    student_model = DistilledTransformer(teacher_model.config, reduction_factor=2)
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    print(f"Compression ratio: {sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.2f}x")
    
    # Prepare dataset
    texts = [
        "The future of artificial intelligence is bright",
        "Machine learning algorithms are becoming more sophisticated",
        "Natural language processing has many applications",
        "Deep learning models require significant computational resources",
    ]
    
    dataset = []
    for text in texts:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True, 
            padding="max_length"
        )
        inputs["labels"] = inputs["input_ids"].clone()
        dataset.append(inputs)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize distiller
    distiller = KnowledgeDistiller(teacher_model, student_model, tokenizer)
    
    # Train student model
    distilled_model = distiller.train(
        dataloader, 
        num_epochs=3, 
        learning_rate=1e-4,
        temperature=4.0,
        alpha=0.7
    )
    
    return distilled_model, teacher_model

# Run distillation
# distilled_model, teacher_model = run_knowledge_distillation()
```

### Advanced Distillation Techniques

```python
class AdvancedDistiller(KnowledgeDistiller):
    \"\"\"Advanced knowledge distillation techniques.\"\"\"
    
    def attention_transfer_loss(self, student_attentions, teacher_attentions):
        \"\"\"Transfer attention patterns from teacher to student.\"\"\"
        
        total_loss = 0
        num_layers = min(len(student_attentions), len(teacher_attentions))
        
        for i in range(num_layers):
            student_attn = student_attentions[i]
            teacher_attn = teacher_attentions[i]
            
            # Reshape if necessary (handle different number of heads)
            if student_attn.shape != teacher_attn.shape:
                # Average over heads if different numbers
                student_attn = student_attn.mean(dim=1, keepdim=True)
                teacher_attn = teacher_attn.mean(dim=1, keepdim=True)
            
            # MSE loss between attention matrices
            attn_loss = F.mse_loss(student_attn, teacher_attn)
            total_loss += attn_loss
        
        return total_loss / num_layers
    
    def hidden_states_loss(self, student_hidden, teacher_hidden):
        \"\"\"Transfer intermediate representations.\"\"\"
        
        # Project student hidden states to teacher dimension if different
        if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
            projection = torch.nn.Linear(
                student_hidden.shape[-1], 
                teacher_hidden.shape[-1]
            ).to(student_hidden.device)
            student_hidden = projection(student_hidden)
        
        return F.mse_loss(student_hidden, teacher_hidden)
    
    def train_step_advanced(self, batch, optimizer, temperature=3.0, alpha=0.5, beta=0.1, gamma=0.1):
        \"\"\"Advanced training step with multiple loss components.\"\"\"
        
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        
        # Teacher forward pass with attention outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
        
        # Student forward pass with attention outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        
        # Standard distillation loss
        total_loss, kd_loss, task_loss = self.distillation_loss(
            student_outputs.logits, 
            teacher_outputs.logits, 
            labels, 
            temperature, 
            alpha
        )
        
        # Attention transfer loss
        if teacher_outputs.attentions and student_outputs.attentions:
            attn_loss = self.attention_transfer_loss(
                student_outputs.attentions,
                teacher_outputs.attentions
            )
            total_loss += beta * attn_loss
        else:
            attn_loss = 0
        
        # Hidden states loss (use last hidden state)
        if teacher_outputs.hidden_states and student_outputs.hidden_states:
            hidden_loss = self.hidden_states_loss(
                student_outputs.hidden_states[-1],
                teacher_outputs.hidden_states[-1]
            )
            total_loss += gamma * hidden_loss
        else:
            hidden_loss = 0
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "kd_loss": kd_loss.item(),
            "task_loss": task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss,
            "attn_loss": attn_loss.item() if isinstance(attn_loss, torch.Tensor) else attn_loss,
            "hidden_loss": hidden_loss.item() if isinstance(hidden_loss, torch.Tensor) else hidden_loss,
        }

# Progressive distillation
class ProgressiveDistiller:
    \"\"\"Implements progressive knowledge distillation.\"\"\"
    
    def __init__(self, teacher_model, tokenizer):
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.intermediate_models = []
    
    def create_intermediate_models(self, num_stages=3):
        \"\"\"Create intermediate models with gradually reduced size.\"\"\"
        
        original_config = self.teacher_model.config
        
        for stage in range(num_stages):
            reduction_factor = 2 ** (stage + 1)
            intermediate_model = DistilledTransformer(original_config, reduction_factor)
            self.intermediate_models.append(intermediate_model)
        
        return self.intermediate_models
    
    def progressive_train(self, dataloader, num_epochs_per_stage=2):
        \"\"\"Train models progressively from large to small.\"\"\"
        
        current_teacher = self.teacher_model
        
        for stage, student_model in enumerate(self.intermediate_models):
            print(f"\\nStage {stage + 1}: Training intermediate model...")
            print(f"Teacher params: {sum(p.numel() for p in current_teacher.parameters()):,}")
            print(f"Student params: {sum(p.numel() for p in student_model.parameters()):,}")
            
            # Create distiller for this stage
            distiller = KnowledgeDistiller(current_teacher, student_model, self.tokenizer)
            
            # Train student
            trained_student = distiller.train(
                dataloader, 
                num_epochs=num_epochs_per_stage,
                learning_rate=1e-4,
                temperature=3.0,
                alpha=0.6
            )
            
            # Student becomes teacher for next stage
            current_teacher = trained_student
            current_teacher.eval()
            for param in current_teacher.parameters():
                param.requires_grad = False
        
        return self.intermediate_models[-1]  # Return smallest model

# Example progressive distillation
def run_progressive_distillation():
    \"\"\"Run progressive knowledge distillation.\"\"\"
    
    teacher_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create progressive distiller
    progressive_distiller = ProgressiveDistiller(teacher_model, tokenizer)
    intermediate_models = progressive_distiller.create_intermediate_models(num_stages=2)
    
    # Prepare simple dataset
    texts = ["Artificial intelligence is transforming technology."] * 20
    dataset = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
        inputs["labels"] = inputs["input_ids"].clone()
        dataset.append(inputs)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    
    # Run progressive training
    final_model = progressive_distiller.progressive_train(dataloader, num_epochs_per_stage=1)
    
    return final_model

# Run progressive distillation
# final_distilled_model = run_progressive_distillation()
```

## Part 5: Hardware-Specific Optimization

### TensorRT Optimization

```python
import tensorrt as trt

class TensorRTOptimizer:
    \"\"\"Optimize models with NVIDIA TensorRT.\"\"\"
    
    def __init__(self, precision="fp16"):
        self.precision = precision
        self.logger = trt.Logger(trt.Logger.WARNING)
        
    def build_engine(self, onnx_path, engine_path, max_batch_size=32, max_seq_length=512):
        \"\"\"Build TensorRT engine from ONNX model.\"\"\"
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Set precision
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed here
        
        # Set optimization profiles for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Input shape: [batch_size, sequence_length]
        profile.set_shape(
            "input_ids",
            (1, 1),  # min
            (max_batch_size // 2, max_seq_length // 2),  # opt
            (max_batch_size, max_seq_length),  # max
        )
        
        config.add_optimization_profile(profile)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return None
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {engine_path}")
        return engine_path
    
    def benchmark_tensorrt(self, engine_path, test_data, num_runs=100):
        \"\"\"Benchmark TensorRT engine performance.\"\"\"
        
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate GPU memory
        input_shape = test_data.shape
        output_shape = (input_shape[0], input_shape[1], 50257)  # Vocab size for GPT-2
        
        # Allocate memory
        input_gpu = cuda.mem_alloc(test_data.nbytes)
        output_gpu = cuda.mem_alloc(output_shape[0] * output_shape[1] * output_shape[2] * 4)
        
        # Copy input data to GPU
        cuda.memcpy_htod(input_gpu, test_data)
        
        # Set input shape
        context.set_binding_shape(0, input_shape)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            
            # Execute inference
            context.execute_v2([int(input_gpu), int(output_gpu)])
            cuda.Context.synchronize()
            
            times.append((time.perf_counter() - start) * 1000)
        
        return {
            "mean_latency_ms": np.mean(times),
            "p95_latency_ms": np.percentile(times, 95),
            "p99_latency_ms": np.percentile(times, 99),
            "throughput_qps": 1000 / np.mean(times),
        }

# ONNX Optimization
class ONNXOptimizer:
    \"\"\"Optimize models with ONNX Runtime.\"\"\"
    
    def __init__(self):
        pass
    
    def optimize_onnx_model(self, model_path, optimized_path):
        \"\"\"Apply ONNX graph optimizations.\"\"\"
        
        import onnx
        from onnxruntime.tools import optimizer
        
        # Load ONNX model
        model = onnx.load(model_path)
        
        # Apply optimizations
        optimized_model = optimizer.optimize_model(
            model_path,
            model_type='gpt2',  # or appropriate model type
            num_heads=12,       # adjust based on model
            hidden_size=768,    # adjust based on model
            optimization_options=optimizer.OptimizationOptions.ALL,
        )
        
        # Save optimized model
        onnx.save(optimized_model, optimized_path)
        
        return optimized_path
    
    def benchmark_onnx(self, model_path, test_data, providers=None):
        \"\"\"Benchmark ONNX model performance.\"\"\"
        
        import onnxruntime as ort
        
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Create session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Prepare input
        input_name = session.get_inputs()[0].name
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            
            outputs = session.run(None, {input_name: test_data})
            
            times.append((time.perf_counter() - start) * 1000)
        
        return {
            "mean_latency_ms": np.mean(times),
            "p95_latency_ms": np.percentile(times, 95),
            "throughput_qps": 1000 / np.mean(times),
        }

# Complete optimization pipeline
def complete_optimization_pipeline(model_name="distilgpt2"):
    \"\"\"Run complete optimization pipeline.\"\"\"
    
    print(f"Starting optimization pipeline for {model_name}")
    
    # Load original model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test data
    test_text = "The future of artificial intelligence"
    test_input = tokenizer(test_text, return_tensors="pt", max_length=64, padding="max_length")
    
    results = {}
    
    # 1. Baseline performance
    print("\\n1. Measuring baseline performance...")
    times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**test_input)
        times.append((time.perf_counter() - start) * 1000)
    
    results["baseline"] = {
        "latency_ms": np.mean(times),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    
    # 2. Apply quantization
    print("\\n2. Applying quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            _ = quantized_model(**test_input)
        times.append((time.perf_counter() - start) * 1000)
    
    results["quantized"] = {
        "latency_ms": np.mean(times),
        "model_size_mb": sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**2,
        "speedup": results["baseline"]["latency_ms"] / np.mean(times),
    }
    
    # 3. Apply pruning
    print("\\n3. Applying pruning...")
    pruner = ModelPruner(model)
    pruned_model = pruner.unstructured_pruning(sparsity=0.3)
    pruner.remove_pruning_masks()
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            _ = pruned_model(**test_input)
        times.append((time.perf_counter() - start) * 1000)
    
    stats = pruner.get_compression_stats()
    results["pruned"] = {
        "latency_ms": np.mean(times),
        "compression_ratio": stats["compression_ratio"],
        "sparsity": stats["sparsity"],
        "speedup": results["baseline"]["latency_ms"] / np.mean(times),
    }
    
    # Print results
    print("\\nOptimization Results:")
    print("-" * 60)
    print(f"{'Method':<15} {'Latency (ms)':<12} {'Speedup':<10} {'Size Reduction':<15}")
    print("-" * 60)
    
    baseline_latency = results["baseline"]["latency_ms"]
    baseline_size = results["baseline"]["model_size_mb"]
    
    for method, metrics in results.items():
        if method == "baseline":
            speedup = 1.0
            size_reduction = "0%"
        else:
            speedup = metrics.get("speedup", 1.0)
            if "model_size_mb" in metrics:
                size_reduction = f"{(1 - metrics['model_size_mb']/baseline_size)*100:.1f}%"
            else:
                size_reduction = f"{(1 - 1/metrics.get('compression_ratio', 1))*100:.1f}%"
        
        print(f"{method:<15} {metrics['latency_ms']:<12.2f} {speedup:<10.2f} {size_reduction:<15}")
    
    return results

# Run complete optimization
# optimization_results = complete_optimization_pipeline()
```

## Part 6: Exercises and Best Practices

### Exercise: Custom Optimization Pipeline

**Challenge**: Create a custom optimization pipeline that combines multiple techniques for your specific use case.

```python
class CustomOptimizationPipeline:
    \"\"\"
    Create your custom optimization pipeline.
    
    Your task:
    1. Choose appropriate optimization techniques based on target (latency/memory/throughput)
    2. Implement automatic hyperparameter tuning for optimization parameters
    3. Add quality assessment at each step
    4. Create rollback mechanism if quality degrades too much
    5. Generate optimization report with recommendations
    \"\"\"
    
    def __init__(self, model, tokenizer, optimization_target="latency"):
        self.model = model
        self.tokenizer = tokenizer
        self.optimization_target = optimization_target
        self.optimization_history = []
        
    def analyze_model(self):
        \"\"\"Analyze model characteristics to choose optimization strategy.\"\"\"
        # TODO: Implement model analysis
        pass
    
    def auto_optimize(self, quality_threshold=0.95):
        \"\"\"Automatically apply optimizations while monitoring quality.\"\"\"
        # TODO: Implement automatic optimization with quality monitoring
        pass
    
    def generate_report(self):
        \"\"\"Generate comprehensive optimization report.\"\"\"
        # TODO: Create detailed report with recommendations
        pass

# Implement and test your pipeline
def test_custom_pipeline():
    # TODO: Test your custom optimization pipeline
    pass
```

### Best Practices Summary

1. **Optimization Strategy Selection**:
   ```python
   optimization_strategies = {
       "latency_critical": ["fp16", "tensorrt", "graph_optimization"],
       "memory_constrained": ["int8_quantization", "pruning", "distillation"], 
       "throughput_focused": ["dynamic_batching", "kv_caching", "pipeline_parallelism"],
       "edge_deployment": ["int4_quantization", "onnx", "model_compression"],
   }
   ```

2. **Quality Monitoring**:
   ```python
   def monitor_quality(original_model, optimized_model, test_dataset):
       \"\"\"Monitor quality degradation during optimization.\"\"\"
       
       original_perplexity = evaluate_perplexity(original_model, test_dataset)
       optimized_perplexity = evaluate_perplexity(optimized_model, test_dataset)
       
       quality_retention = original_perplexity / optimized_perplexity
       
       return {
           "quality_retention": quality_retention,
           "acceptable": quality_retention > 0.95,  # 5% degradation threshold
       }
   ```

3. **Performance Benchmarking**:
   ```python
   def comprehensive_benchmark(model, test_data, num_runs=100):
       \"\"\"Comprehensive performance benchmark.\"\"\"
       
       metrics = {
           "latency": measure_latency(model, test_data, num_runs),
           "throughput": measure_throughput(model, test_data),
           "memory": measure_memory_usage(model, test_data),
           "accuracy": measure_accuracy(model, test_data),
       }
       
       return metrics
   ```

## Summary

You've learned:
- ✅ Quantization techniques from FP16 to INT4
- ✅ Model pruning strategies (structured and unstructured)
- ✅ Knowledge distillation for model compression
- ✅ Hardware-specific optimization (TensorRT, ONNX)
- ✅ Complete optimization pipelines
- ✅ Quality monitoring and benchmarking

## Next Steps

- Try [Tutorial 4: Production Serving](04_production_serving.md)
- Explore [Tutorial 5: Monitoring & Observability](05_monitoring_observability.md)
- Learn about [Tutorial 6: Azure ML Integration](06_azure_ml_integration.md)