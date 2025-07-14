# Tutorial 2: Parameter-Efficient Fine-Tuning (PEFT) Techniques

## Overview

This tutorial covers advanced Parameter-Efficient Fine-Tuning techniques including LoRA, QLoRA, AdaLoRA, and other methods that enable efficient training of large language models with minimal computational resources.

## Learning Objectives

- Master LoRA (Low-Rank Adaptation) implementation and theory
- Understand QLoRA for 4-bit quantized training
- Implement AdaLoRA with adaptive rank allocation
- Compare different PEFT methods
- Apply PEFT to real-world scenarios

## Prerequisites

- Completion of Tutorial 1 (Distributed Training)
- Understanding of linear algebra and matrix decomposition
- Familiarity with PyTorch and Transformers

## Part 1: Understanding PEFT - Theory and Motivation

### The Parameter Efficiency Problem

Modern LLMs have billions of parameters, making full fine-tuning expensive:

```python
import torch
from transformers import AutoModel

# Calculate fine-tuning costs
def calculate_training_cost(model_name, num_parameters):
    \"\"\"Estimate training costs for full fine-tuning.\"\"\"
    
    # Memory requirements (rough estimates)
    model_memory = num_parameters * 4 / 1024**3  # 4 bytes per parameter (FP32)
    optimizer_memory = num_parameters * 8 / 1024**3  # Adam needs 2x model params
    gradients_memory = num_parameters * 4 / 1024**3
    
    total_memory_gb = model_memory + optimizer_memory + gradients_memory
    
    # Cost estimates (approximate)
    a100_cost_per_hour = 3.0  # USD
    hours_for_training = 24  # Example
    
    print(f"Model: {model_name}")
    print(f"Parameters: {num_parameters:,}")
    print(f"Model memory: {model_memory:.1f} GB")
    print(f"Total training memory: {total_memory_gb:.1f} GB")
    print(f"Estimated cost: ${a100_cost_per_hour * hours_for_training:.2f}")
    print(f"GPUs needed: {total_memory_gb / 80:.1f} A100s")  # 80GB per A100
    
# Examples
calculate_training_cost("GPT-2 Small", 124_000_000)
calculate_training_cost("GPT-2 XL", 1_500_000_000)
calculate_training_cost("LLaMA 7B", 7_000_000_000)
calculate_training_cost("LLaMA 70B", 70_000_000_000)
```

### PEFT Solution

Instead of updating all parameters, PEFT methods update only a small subset:

```python
# Traditional fine-tuning
total_params = sum(p.numel() for p in model.parameters())
trainable_params = total_params  # 100% of parameters

# PEFT fine-tuning
peft_params = total_params * 0.001  # Only 0.1% of parameters!
memory_reduction = 0.99  # 99% memory reduction
cost_reduction = 0.95    # 95% cost reduction
```

## Part 2: LoRA (Low-Rank Adaptation) Deep Dive

### Mathematical Foundation

LoRA decomposes weight updates into low-rank matrices:

```
W_new = W_original + ΔW
ΔW = B × A

where:
- W_original: frozen pre-trained weights (d × k)
- A: trainable matrix (r × k) 
- B: trainable matrix (d × r)
- r << min(d, k): rank constraint
```

### Implementing LoRA from Scratch

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    \"\"\"
    LoRA implementation for linear layers.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        r: Rank of adaptation (smaller = more efficient)
        alpha: Scaling factor
        dropout: Dropout probability
    \"\"\"
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 16, 
        alpha: int = 32, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward pass: x @ (A^T @ B^T) * scaling\"\"\"
        result = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return result * self.scaling

class LoRALinear(nn.Module):
    \"\"\"Linear layer with LoRA adaptation.\"\"\"
    
    def __init__(self, original_layer: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        
        # Freeze original layer
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # Add LoRA adaptation
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features, 
            r=r, 
            alpha=alpha
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward pass: original + LoRA adaptation\"\"\"
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        return original_output + lora_output

# Example usage
def add_lora_to_model(model, target_modules=["q_proj", "v_proj"], r=16):
    \"\"\"Add LoRA to specific modules in a model.\"\"\"
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                parent_module = model.get_submodule(parent_name) if parent_name else model
                lora_layer = LoRALinear(module, r=r)
                setattr(parent_module, child_name, lora_layer)
                
                print(f"Added LoRA to: {name}")
    
    return model

# Test LoRA implementation
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
print(f"Original trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

model_with_lora = add_lora_to_model(model, r=16)
print(f"LoRA trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### LoRA Parameter Analysis

```python
def analyze_lora_parameters(model, r_values=[4, 8, 16, 32, 64]):
    \"\"\"Analyze parameter count for different LoRA ranks.\"\"\"
    
    # Count target modules
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    linear_layers = []
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            linear_layers.append((name, module.in_features, module.out_features))
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total model parameters: {total_params:,}")
    print(f"Target linear layers: {len(linear_layers)}")
    print()
    
    for r in r_values:
        lora_params = 0
        for name, in_feat, out_feat in linear_layers:
            # LoRA adds: in_feat * r + r * out_feat parameters
            lora_params += in_feat * r + r * out_feat
        
        percentage = (lora_params / total_params) * 100
        print(f"Rank {r:2d}: {lora_params:,} parameters ({percentage:.3f}% of total)")

# Example analysis
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
analyze_lora_parameters(model)
```

## Part 3: QLoRA - 4-bit Quantized LoRA

QLoRA combines LoRA with 4-bit quantization for extreme efficiency:

### Understanding Quantization

```python
import torch

def demonstrate_quantization():
    \"\"\"Show memory savings from quantization.\"\"\"
    
    # Original FP32 weights
    fp32_weights = torch.randn(1024, 1024, dtype=torch.float32)
    fp32_size = fp32_weights.numel() * 4  # 4 bytes per float32
    
    # FP16 quantization
    fp16_weights = fp32_weights.half()
    fp16_size = fp16_weights.numel() * 2  # 2 bytes per float16
    
    # 8-bit quantization (simulated)
    int8_size = fp32_weights.numel() * 1  # 1 byte per int8
    
    # 4-bit quantization (simulated)
    int4_size = fp32_weights.numel() * 0.5  # 0.5 bytes per int4
    
    print("Quantization Memory Comparison:")
    print(f"FP32: {fp32_size / 1024**2:.1f} MB (100%)")
    print(f"FP16: {fp16_size / 1024**2:.1f} MB ({fp16_size/fp32_size*100:.1f}%)")
    print(f"INT8: {int8_size / 1024**2:.1f} MB ({int8_size/fp32_size*100:.1f}%)")
    print(f"INT4: {int4_size / 1024**2:.1f} MB ({int4_size/fp32_size*100:.1f}%)")

demonstrate_quantization()
```

### Implementing QLoRA

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_qlora_model(model_name: str, r: int = 16):
    \"\"\"Setup model for QLoRA training.\"\"\"
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_use_double_quant=True,  # Double quantization
    )
    
    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

# Example usage
qlora_model = setup_qlora_model("microsoft/phi-2", r=16)
```

### QLoRA Training Script

```python
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_qlora_model():
    \"\"\"Complete QLoRA training example.\"\"\"
    
    # Setup model and tokenizer
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = setup_qlora_model(model_name, r=16)
    
    # Load and prepare dataset
    dataset = load_dataset("Abirate/english_quotes", split="train[:1000]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["quote"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.add_column(
        "labels", tokenized_dataset["input_ids"]
    )
    
    # Training arguments optimized for QLoRA
    training_args = TrainingArguments(
        output_dir="./qlora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        optim="paged_adamw_32bit",  # Efficient optimizer for quantized models
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        fp16=False,  # Use bf16 instead with quantization
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Can cause issues with quantized models
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save LoRA adapter
    model.save_pretrained("./qlora_adapter")
    tokenizer.save_pretrained("./qlora_adapter")
    
    return model, tokenizer

# Run training
# model, tokenizer = train_qlora_model()
```

## Part 4: AdaLoRA - Adaptive Low-Rank Adaptation

AdaLoRA dynamically adjusts the rank during training:

### AdaLoRA Theory

```python
class AdaLoRAConfig:
    \"\"\"Configuration for AdaLoRA training.\"\"\"
    
    def __init__(
        self,
        init_r: int = 128,      # Initial rank
        target_r: int = 8,      # Target rank after pruning
        tinit: int = 0,         # Start pruning step
        tfinal: int = 1000,     # End pruning step  
        deltaT: int = 10,       # Pruning frequency
        beta1: float = 0.85,    # Exponential moving average coefficient
        beta2: float = 0.85,    # Exponential moving average coefficient
    ):
        self.init_r = init_r
        self.target_r = target_r
        self.tinit = tinit
        self.tfinal = tfinal
        self.deltaT = deltaT
        self.beta1 = beta1
        self.beta2 = beta2

def calculate_rank_schedule(config: AdaLoRAConfig, global_step: int) -> int:
    \"\"\"Calculate current rank based on training step.\"\"\"
    
    if global_step < config.tinit:
        return config.init_r
    elif global_step > config.tfinal:
        return config.target_r
    else:
        # Linear decay from init_r to target_r
        progress = (global_step - config.tinit) / (config.tfinal - config.tinit)
        current_r = config.init_r - progress * (config.init_r - config.target_r)
        return int(current_r)

# Example rank schedule
config = AdaLoRAConfig(init_r=128, target_r=8, tinit=100, tfinal=1000)
steps = [0, 100, 500, 1000, 1500]

for step in steps:
    rank = calculate_rank_schedule(config, step)
    print(f"Step {step}: Rank {rank}")
```

### AdaLoRA Implementation

```python
from peft import AdaLoraConfig, get_peft_model

def setup_adalora_model(model_name: str):
    \"\"\"Setup model for AdaLoRA training.\"\"\"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # AdaLoRA configuration
    adalora_config = AdaLoraConfig(
        init_r=128,                    # Start with high rank
        target_r=8,                    # Reduce to low rank
        tinit=200,                     # Start pruning after 200 steps
        tfinal=1000,                   # Finish pruning at 1000 steps
        deltaT=50,                     # Prune every 50 steps
        beta1=0.85,
        beta2=0.85,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    
    # Apply AdaLoRA
    model = get_peft_model(model, adalora_config)
    model.print_trainable_parameters()
    
    return model

# Usage
adalora_model = setup_adalora_model("microsoft/DialoGPT-medium")
```

### AdaLoRA Training with Custom Trainer

```python
from transformers import Trainer

class AdaLoRATrainer(Trainer):
    \"\"\"Custom trainer that handles AdaLoRA rank updates.\"\"\"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def training_step(self, model, inputs):
        \"\"\"Override to add AdaLoRA updates.\"\"\"
        
        # Standard training step
        loss = super().training_step(model, inputs)
        
        # Update AdaLoRA budget allocation
        if hasattr(model, 'update_and_prune'):
            model.update_and_prune(self.state.global_step)
            
        return loss
    
    def log(self, logs):
        \"\"\"Add AdaLoRA metrics to logs.\"\"\"
        
        # Add current rank information
        if hasattr(self.model, 'get_rank_info'):
            rank_info = self.model.get_rank_info()
            logs.update(rank_info)
            
        super().log(logs)

# Training with AdaLoRA
def train_adalora():
    model = setup_adalora_model("microsoft/DialoGPT-small")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./adalora_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        logging_steps=50,
        save_steps=500,
    )
    
    # Use custom trainer
    trainer = AdaLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return model
```

## Part 5: Other PEFT Methods

### Prompt Tuning

```python
from peft import PromptTuningConfig, PromptTuningInit

def setup_prompt_tuning(model_name: str, num_virtual_tokens: int = 20):
    \"\"\"Setup prompt tuning.\"\"\"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prompt_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text="Classify the following text:",
        tokenizer_name_or_path=model_name,
    )
    
    model = get_peft_model(model, prompt_config)
    return model
```

### Prefix Tuning

```python
from peft import PrefixTuningConfig

def setup_prefix_tuning(model_name: str):
    \"\"\"Setup prefix tuning.\"\"\"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prefix_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=30,
        encoder_hidden_size=768,  # Adjust based on model
        prefix_projection=True,
    )
    
    model = get_peft_model(model, prefix_config)
    return model
```

### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

```python
from peft import IA3Config

def setup_ia3(model_name: str):
    \"\"\"Setup IA³ adaptation.\"\"\"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    ia3_config = IA3Config(
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "v_proj", "down_proj"],
        feedforward_modules=["down_proj"],
    )
    
    model = get_peft_model(model, ia3_config)
    return model
```

## Part 6: PEFT Method Comparison

### Comprehensive Comparison Script

```python
import time
from typing import Dict, Any
import matplotlib.pyplot as plt

def compare_peft_methods(
    model_name: str = "microsoft/DialoGPT-small",
    methods: List[str] = ["lora", "adalora", "prompt_tuning", "ia3"]
) -> Dict[str, Any]:
    \"\"\"Compare different PEFT methods.\"\"\"
    
    results = {}
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    total_params = sum(p.numel() for p in base_model.parameters())
    
    for method in methods:
        print(f"Testing {method}...")
        
        # Setup method
        if method == "lora":
            config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
        elif method == "adalora":
            config = AdaLoraConfig(init_r=64, target_r=8)
        elif method == "prompt_tuning":
            config = PromptTuningConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=20,
            )
        elif method == "ia3":
            config = IA3Config(target_modules=["k_proj", "v_proj"])
        
        # Apply PEFT
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = get_peft_model(model, config)
        
        # Calculate metrics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_efficiency = trainable_params / total_params
        
        # Memory test
        model.train()
        dummy_input = torch.randint(0, 1000, (4, 128))
        
        start_time = time.time()
        for _ in range(10):
            outputs = model(dummy_input, labels=dummy_input)
            loss = outputs.loss
            loss.backward()
            model.zero_grad()
        inference_time = time.time() - start_time
        
        results[method] = {
            "trainable_params": trainable_params,
            "param_efficiency": param_efficiency,
            "inference_time": inference_time / 10,
            "memory_footprint": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        }
        
    return results

def plot_comparison(results: Dict[str, Any]):
    \"\"\"Plot PEFT method comparison.\"\"\"
    
    methods = list(results.keys())
    efficiencies = [results[m]["param_efficiency"] * 100 for m in methods]
    times = [results[m]["inference_time"] * 1000 for m in methods]  # Convert to ms
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameter efficiency
    ax1.bar(methods, efficiencies)
    ax1.set_ylabel("Trainable Parameters (%)")
    ax1.set_title("Parameter Efficiency")
    ax1.set_yscale("log")
    
    # Inference time
    ax2.bar(methods, times)
    ax2.set_ylabel("Inference Time (ms)")
    ax2.set_title("Inference Speed")
    
    plt.tight_layout()
    plt.savefig("peft_comparison.png", dpi=300)
    plt.show()

# Run comparison
results = compare_peft_methods()
plot_comparison(results)

# Print results table
print("\\nPEFT Method Comparison:")
print("-" * 80)
print(f"{'Method':<15} {'Trainable %':<12} {'Time (ms)':<10} {'Memory (MB)':<12}")
print("-" * 80)

for method, metrics in results.items():
    print(f"{method:<15} {metrics['param_efficiency']*100:<12.3f} "
          f"{metrics['inference_time']*1000:<10.2f} "
          f"{metrics['memory_footprint']/1024**2:<12.1f}")
```

## Part 7: Advanced PEFT Techniques

### Mixed-Precision LoRA

```python
class MixedPrecisionLoRA(nn.Module):
    \"\"\"LoRA with mixed precision for efficiency.\"\"\"
    
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        
        # Use different precisions for A and B
        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=torch.float16)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=torch.float32)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Mixed precision computation
        x_16 = x.half()
        a_out = self.lora_A(x_16)
        b_out = self.lora_B(a_out.float())
        return b_out * self.scaling
```

### LoRA with Structured Pruning

```python
class PrunedLoRA(nn.Module):
    \"\"\"LoRA with automatic rank pruning.\"\"\"
    
    def __init__(self, in_features, out_features, r=16, alpha=32, prune_threshold=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.prune_threshold = prune_threshold
        self.scaling = alpha / r
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Importance scores
        self.register_buffer("importance_scores", torch.ones(r))
        
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Standard LoRA forward
        result = self.lora_B(self.lora_A(x))
        return result * self.scaling
    
    def update_importance(self):
        \"\"\"Update importance scores based on gradient magnitudes.\"\"\"
        if self.lora_A.weight.grad is not None:
            # Calculate importance as gradient magnitude
            grad_norm_A = torch.norm(self.lora_A.weight.grad, dim=0)
            grad_norm_B = torch.norm(self.lora_B.weight.grad, dim=1)
            
            # Combine importance scores
            combined_importance = grad_norm_A * grad_norm_B
            
            # Exponential moving average
            self.importance_scores = 0.9 * self.importance_scores + 0.1 * combined_importance
    
    def prune_ranks(self):
        \"\"\"Prune least important ranks.\"\"\"
        # Find ranks to prune
        prune_mask = self.importance_scores < self.prune_threshold
        keep_indices = (~prune_mask).nonzero().squeeze()
        
        if len(keep_indices) > 0:
            # Create new smaller matrices
            new_r = len(keep_indices)
            
            new_lora_A = nn.Linear(self.lora_A.in_features, new_r, bias=False)
            new_lora_B = nn.Linear(new_r, self.lora_B.out_features, bias=False)
            
            # Copy weights for kept ranks
            new_lora_A.weight.data = self.lora_A.weight.data[keep_indices]
            new_lora_B.weight.data = self.lora_B.weight.data[:, keep_indices]
            
            # Replace layers
            self.lora_A = new_lora_A
            self.lora_B = new_lora_B
            self.r = new_r
            self.scaling = self.alpha / new_r
            
            print(f"Pruned LoRA rank from {self.r} to {new_r}")
```

## Part 8: Real-World Application: Code Generation

### Setup Code-Specific LoRA

```python
def setup_code_generation_lora():
    \"\"\"Setup LoRA specifically for code generation tasks.\"\"\"
    
    # Load code-specific model
    model_name = "microsoft/CodeGPT-small-py"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Code-specific LoRA config
    lora_config = LoraConfig(
        r=32,  # Higher rank for code complexity
        lora_alpha=64,
        target_modules=[
            "c_attn",  # Attention layers
            "c_proj",  # Projection layers  
            "c_fc",    # Feed-forward layers
        ],
        lora_dropout=0.05,  # Lower dropout for code
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def create_code_dataset():
    \"\"\"Create dataset for code generation training.\"\"\"
    
    # Example code patterns
    code_examples = [
        "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class Calculator:\\n    def add(self, a, b):\\n        return a + b\\n    def multiply(self, a, b):\\n        return a * b",
        "import numpy as np\\n\\ndef matrix_multiply(a, b):\\n    return np.dot(a, b)",
        # Add more examples...
    ]
    
    # Create training examples with prompts
    training_data = []
    for code in code_examples:
        # Extract function signature as prompt
        lines = code.split("\\n")
        signature = lines[0] + ":"
        completion = "\\n".join(lines[1:])
        
        training_data.append({
            "prompt": signature,
            "completion": completion,
            "full_text": code,
        })
    
    return training_data

# Train code generation model
def train_code_generation():
    model, tokenizer = setup_code_generation_lora()
    dataset = create_code_dataset()
    
    # Training with code-specific settings
    training_args = TrainingArguments(
        output_dir="./code_lora",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=50,
        save_steps=100,
        logging_steps=10,
        remove_unused_columns=False,
    )
    
    # Custom data collator for code
    def code_data_collator(examples):
        texts = [ex["full_text"] for ex in examples]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=code_data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return model, tokenizer
```

### Testing Code Generation

```python
def test_code_generation(model, tokenizer):
    \"\"\"Test the trained code generation model.\"\"\"
    
    test_prompts = [
        "def bubble_sort(arr):",
        "class BinaryTree:",
        "def quick_sort(arr):",
        "import requests\\n\\ndef fetch_data(url):",
    ]
    
    model.eval()
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated:\\n{generated}\\n")
        print("-" * 50)

# Run the test
# model, tokenizer = train_code_generation()
# test_code_generation(model, tokenizer)
```

## Part 9: Production Deployment of PEFT Models

### Merging LoRA Weights

```python
def merge_lora_weights(peft_model, save_path: str):
    \"\"\"Merge LoRA weights back into base model for deployment.\"\"\"
    
    # Merge and unload LoRA
    merged_model = peft_model.merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(save_path)
    
    # Verify no PEFT components remain
    trainable_params = sum(p.numel() for p in merged_model.parameters() if p.requires_grad)
    print(f"Merged model trainable parameters: {trainable_params}")
    
    return merged_model

def deploy_peft_model(model_path: str):
    \"\"\"Deploy PEFT model for inference.\"\"\"
    
    from transformers import pipeline
    
    # Load merged model
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1,
    )
    
    return generator

# Example deployment
# merged_model = merge_lora_weights(peft_model, "./merged_code_model")
# generator = deploy_peft_model("./merged_code_model")
# 
# result = generator("def factorial(n):", max_length=100)
# print(result[0]["generated_text"])
```

## Part 10: Exercise and Best Practices

### Exercise: Custom PEFT Implementation

**Challenge**: Implement your own PEFT method that combines LoRA with attention head pruning.

```python
class AttentionPrunedLoRA(nn.Module):
    \"\"\"
    Custom PEFT method combining LoRA with attention head pruning.
    
    Your task:
    1. Implement attention head importance scoring
    2. Prune less important heads during training
    3. Apply LoRA only to remaining heads
    4. Compare efficiency with standard LoRA
    \"\"\"
    
    def __init__(self, attention_layer, num_heads, r=16):
        super().__init__()
        # TODO: Implement initialization
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass with head pruning + LoRA
        pass
    
    def update_head_importance(self):
        # TODO: Update attention head importance scores
        pass
    
    def prune_heads(self, threshold=0.1):
        # TODO: Prune attention heads below threshold
        pass

# Test your implementation
def test_custom_peft():
    # TODO: Test your custom PEFT method
    # Compare with standard LoRA on parameter efficiency and performance
    pass
```

### Best Practices Summary

1. **Choose the Right Method**:
   - LoRA: General purpose, good balance
   - QLoRA: Extreme efficiency, some quality loss
   - AdaLoRA: Adaptive, best for long training
   - Prompt Tuning: Task-specific, very efficient

2. **Hyperparameter Selection**:
   ```python
   # LoRA rank selection guidelines
   rank_guidelines = {
       "small_models": {"r": 8, "alpha": 16},    # <1B params
       "medium_models": {"r": 16, "alpha": 32},   # 1-7B params  
       "large_models": {"r": 32, "alpha": 64},    # 7B+ params
   }
   ```

3. **Target Module Selection**:
   ```python
   # Common target modules for different architectures
   target_modules = {
       "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
       "gpt2": ["c_attn", "c_proj", "c_fc"],
       "bert": ["query", "value", "key", "dense"],
       "t5": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
   }
   ```

4. **Memory Optimization**:
   ```python
   # Gradient checkpointing + PEFT
   model.gradient_checkpointing_enable()
   
   # Use appropriate batch sizes
   effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
   ```

## Summary

You've learned:
- ✅ LoRA theory and implementation from scratch
- ✅ QLoRA for extreme efficiency with 4-bit quantization  
- ✅ AdaLoRA for adaptive rank allocation
- ✅ Comparison of different PEFT methods
- ✅ Real-world application to code generation
- ✅ Production deployment strategies

## Next Steps

- Try [Tutorial 3: Model Optimization](03_model_optimization.md)
- Explore [Tutorial 4: Production Serving](04_production_serving.md)
- Learn about [Tutorial 5: Monitoring & Observability](05_monitoring_observability.md)