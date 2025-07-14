# Tutorial 1: Distributed Training for Large Language Models

## Overview

This tutorial walks you through setting up and running distributed training for large language models using PyTorch, DeepSpeed, and Azure ML. You'll learn how to scale training across multiple GPUs and nodes efficiently.

## Learning Objectives

By the end of this tutorial, you will:
- Understand different distributed training strategies (DDP, FSDP, DeepSpeed)
- Set up multi-GPU training on Azure ML
- Optimize memory usage and training speed
- Monitor training progress and debug issues

## Prerequisites

- Basic understanding of PyTorch and Transformers
- Azure subscription with ML workspace
- Familiarity with Python and command line

## Part 1: Understanding Distributed Training

### Why Distributed Training?

Modern language models have billions of parameters that won't fit on a single GPU:

```python
# Example: GPT-3 style model memory requirements
model_params = 175_000_000_000  # 175B parameters
bytes_per_param = 4  # FP32
model_size_gb = (model_params * bytes_per_param) / (1024**3)
print(f"Model size: {model_size_gb:.1f} GB")
# Output: Model size: 651.9 GB
```

A single A100 GPU has only 80GB memory, so we need distributed training strategies.

### Distributed Training Strategies

1. **Data Parallel (DP)**: Replicate model on each GPU, split data
2. **Distributed Data Parallel (DDP)**: More efficient version of DP
3. **Fully Sharded Data Parallel (FSDP)**: Shard parameters across GPUs
4. **DeepSpeed ZeRO**: Advanced sharding with optimizer states

Let's implement each approach:

## Part 2: Setting Up Your Environment

### Step 1: Install Dependencies

```bash
# Create conda environment
conda create -n distributed-training python=3.9
conda activate distributed-training

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install training frameworks
pip install transformers[torch] datasets accelerate deepspeed
pip install azure-ai-ml wandb loguru rich
```

### Step 2: Verify GPU Setup

```python
import torch
import torch.distributed as dist

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
```

## Part 3: Data Parallel Training (Single Node, Multiple GPUs)

### Step 1: Basic DDP Setup

Create `train_ddp.py`:

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from loguru import logger

def setup_distributed():
    """Initialize distributed training."""
    # Environment variables set by torchrun
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize process group
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def create_dataloader(dataset, tokenizer, batch_size, rank, world_size):
    """Create distributed dataloader."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    
    # Create distributed sampler
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
    )
    
    return dataloader

def main():
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(local_rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:1000]")
    dataloader = create_dataloader(dataset, tokenizer, batch_size=4, rank=rank, world_size=world_size)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(dataloader) * 3  # 3 epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    model.train()
    for epoch in range(3):
        dataloader.sampler.set_epoch(epoch)  # Important for proper shuffling
        
        total_loss = 0
        for step, batch in enumerate(dataloader):
            # Move batch to GPU
            input_ids = batch["input_ids"].to(local_rank)
            attention_mask = batch["attention_mask"].to(local_rank)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if rank == 0 and step % 10 == 0:
                logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Save model (only on rank 0)
    if rank == 0:
        model.module.save_pretrained("./ddp_model")
        tokenizer.save_pretrained("./ddp_model")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
```

### Step 2: Run DDP Training

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py

# Monitor with nvidia-smi
watch -n 1 nvidia-smi
```

## Part 4: Fully Sharded Data Parallel (FSDP)

FSDP shards parameters across GPUs, enabling training of larger models:

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def setup_fsdp_model(model, local_rank):
    """Setup model with FSDP."""
    
    # Auto wrap policy for transformer layers
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={GPT2Block},
        recurse=True,
    )
    
    # Mixed precision policy
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy="FULL_SHARD",  # Shard parameters and gradients
        cpu_offload=CPUOffload(offload_params=False),  # Keep on GPU for speed
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        sync_module_states=True,
    )
    
    return model

# Usage in training loop
def train_with_fsdp():
    rank, local_rank, world_size = setup_distributed()
    
    # Load larger model
    model_name = "microsoft/DialoGPT-medium"  # 355M parameters
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Setup FSDP
    model = setup_fsdp_model(model, local_rank)
    
    # Training continues as before...
```

## Part 5: DeepSpeed ZeRO Training

DeepSpeed provides the most advanced memory optimization:

### Step 1: Create DeepSpeed Config

Create `deepspeed_config.json`:

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,
  
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 5e-5,
      "warmup_num_steps": 100
    }
  },
  
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 1e6,
    "stage3_prefetch_bucket_size": 1e6,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

### Step 2: DeepSpeed Training Script

```python
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

def train_with_deepspeed():
    # Parse DeepSpeed arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    args = parser.parse_args()
    
    # Load model
    model_name = "microsoft/DialoGPT-large"  # 774M parameters
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config_params=args.deepspeed_config,
    )
    
    # Training loop
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            # Forward pass
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            loss = outputs.loss
            
            # Backward pass (DeepSpeed handles optimization)
            model_engine.backward(loss)
            model_engine.step()
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_with_deepspeed()
```

### Step 3: Run DeepSpeed Training

```bash
deepspeed --num_gpus=4 train_deepspeed.py --deepspeed_config deepspeed_config.json
```

## Part 6: Multi-Node Training

### Step 1: Setup Azure ML Compute

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute

# Create compute cluster
compute_config = AmlCompute(
    name="gpu-cluster",
    size="Standard_NC24ads_A100_v4",  # 1x A100 per node
    min_instances=0,
    max_instances=8,  # 8 nodes max
)

ml_client.compute.begin_create_or_update(compute_config)
```

### Step 2: Multi-Node Training Script

```python
# train_multinode.py
import os
import torch.distributed as dist

def setup_multinode():
    """Setup multi-node distributed training."""
    # Azure ML sets these environment variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Master node address (set by Azure ML)
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = setup_multinode()
    
    print(f"Rank {rank}/{world_size} on node {os.environ.get('NODE_RANK', 'unknown')}")
    
    # Continue with training...
```

### Step 3: Submit Multi-Node Job

```python
from azure.ai.ml import command

job = command(
    code="./src",
    command="python train_multinode.py",
    environment="azureml:pytorch-cuda:1",
    compute="gpu-cluster",
    instance_count=4,  # 4 nodes
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 1,
    },
    display_name="multi-node-training",
)

ml_client.jobs.create_or_update(job)
```

## Part 7: Memory Optimization Techniques

### Gradient Checkpointing

Trade compute for memory:

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or for custom control
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self.transformer_block, x, use_reentrant=False)
```

### Mixed Precision Training

Use FP16 to halve memory usage:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    # Scale loss to prevent underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Memory Monitoring

Track memory usage:

```python
def print_memory_stats():
    """Print GPU memory statistics."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")

# Call during training
print_memory_stats()
```

## Part 8: Troubleshooting Common Issues

### Out of Memory (OOM)

1. **Reduce batch size**:
```python
# Reduce per-device batch size and increase gradient accumulation
per_device_batch_size = 1
gradient_accumulation_steps = 16
```

2. **Enable gradient checkpointing**:
```python
model.gradient_checkpointing_enable()
```

3. **Use CPU offloading**:
```python
# In DeepSpeed config
"zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
}
```

### Slow Training

1. **Check data loading**:
```python
# Increase number of workers
dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)
```

2. **Profile training**:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Training step
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Communication Issues

1. **Check network setup**:
```bash
# Test NCCL
python -c "import torch; torch.distributed.init_process_group('nccl'); print('NCCL working')"
```

2. **Set NCCL debug**:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## Part 9: Best Practices

### 1. Data Pipeline Optimization

```python
# Use streaming datasets for large data
dataset = load_dataset("dataset_name", streaming=True)

# Prefetch data
dataloader = DataLoader(dataset, prefetch_factor=2, persistent_workers=True)
```

### 2. Checkpoint Management

```python
def save_checkpoint(model, optimizer, scheduler, epoch, step):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    torch.save(checkpoint, f"checkpoint_epoch_{epoch}_step_{step}.pt")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["step"]
```

### 3. Monitoring and Logging

```python
import wandb

# Initialize wandb
wandb.init(project="distributed-training")

# Log metrics
wandb.log({
    "loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0],
    "epoch": epoch,
    "step": step,
})
```

## Part 10: Exercise

**Challenge**: Train a 1.3B parameter model using DeepSpeed ZeRO-3

1. Choose a model (e.g., GPT-2 XL or similar)
2. Set up DeepSpeed configuration for ZeRO-3
3. Implement multi-GPU training
4. Monitor memory usage and training speed
5. Compare with regular DDP training

**Success criteria**:
- Model fits in GPU memory
- Training completes without OOM
- Achieve reasonable training speed
- Save and load checkpoints properly

## Summary

You've learned:
- ✅ Different distributed training strategies
- ✅ Setting up DDP, FSDP, and DeepSpeed
- ✅ Multi-node training on Azure ML  
- ✅ Memory optimization techniques
- ✅ Troubleshooting common issues
- ✅ Best practices for production training

## Next Steps

- Try [Tutorial 2: PEFT Techniques](02_peft_techniques.md)
- Explore [Tutorial 3: Model Optimization](03_model_optimization.md)
- Learn about [Tutorial 4: Production Serving](04_production_serving.md)