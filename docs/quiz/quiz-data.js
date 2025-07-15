// Quiz Data - 100 Questions based on Production ML Engineering Handbook
// Categories: Environment Setup (10), Distributed Training (20), PEFT Techniques (20), 
// Model Optimization (15), Architecture (10), Deployment/Serving (15), Monitoring (10)

const quizData = [
    // ========== ENVIRONMENT SETUP (Questions 1-10) ==========
    {
        id: 1,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "What is the minimum CUDA version required for PyTorch 2.0+ with GPU support?",
        code: null,
        options: [
            "CUDA 10.2",
            "CUDA 11.6",
            "CUDA 11.8",
            "CUDA 12.0"
        ],
        correct: 2,
        explanation: "PyTorch 2.0+ requires CUDA 11.8 or higher for optimal performance and feature support. This version includes important improvements for transformer models and distributed training."
    },
    {
        id: 2,
        type: "short-answer",
        category: "Environment Setup",
        question: "What command verifies that PyTorch can detect and use available GPUs?",
        code: null,
        options: null,
        correct: "torch.cuda.is_available()",
        explanation: "torch.cuda.is_available() returns True if PyTorch can detect and use CUDA-capable GPUs. Additional useful commands include torch.cuda.device_count() for GPU count and torch.cuda.get_device_properties() for detailed GPU information."
    },
    {
        id: 3,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "Which environment variable controls which GPUs are visible to PyTorch?",
        code: `# Example usage
export ???=0,1,2,3  # Use GPUs 0-3
python train.py`,
        options: [
            "PYTORCH_CUDA_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDER",
            "NVIDIA_VISIBLE_DEVICES"
        ],
        correct: 1,
        explanation: "CUDA_VISIBLE_DEVICES is the standard environment variable that controls GPU visibility. Setting it to '0,1,2,3' makes only those GPUs visible to CUDA applications."
    },
    {
        id: 4,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "What is the purpose of the NCCL_DEBUG environment variable in distributed training?",
        code: `export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
python -m torch.distributed.launch train.py`,
        options: [
            "Enables debugging output for NCCL communication",
            "Sets the number of communication threads",
            "Configures GPU memory allocation",
            "Controls gradient synchronization frequency"
        ],
        correct: 0,
        explanation: "NCCL_DEBUG=INFO enables detailed logging of NCCL (NVIDIA Collective Communication Library) operations, helping debug communication issues in multi-GPU/multi-node training."
    },
    {
        id: 5,
        type: "short-answer",
        category: "Environment Setup",
        question: "In the docker-compose.yml for ML development, what runtime is required for GPU support?",
        code: `services:
  ml-trainer:
    runtime: ???`,
        options: null,
        correct: "nvidia",
        explanation: "The 'nvidia' runtime is required in Docker Compose to enable GPU access within containers. This requires nvidia-docker2 to be installed on the host system."
    },
    {
        id: 6,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "Which package manager is recommended for creating isolated Python environments in the handbook?",
        code: null,
        options: [
            "pip",
            "virtualenv",
            "conda",
            "poetry"
        ],
        correct: 2,
        explanation: "Conda is recommended because it handles both Python packages and system-level dependencies (like CUDA libraries), making it ideal for ML environments with complex dependencies."
    },
    {
        id: 7,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "What is the primary purpose of including prometheus in the docker-compose stack?",
        code: null,
        options: [
            "Model training acceleration",
            "Metrics collection and monitoring",
            "Distributed computing coordination",
            "Data preprocessing"
        ],
        correct: 1,
        explanation: "Prometheus is included for metrics collection and monitoring. It scrapes metrics from various services and stores time-series data for analysis and alerting."
    },
    {
        id: 8,
        type: "short-answer",
        category: "Environment Setup",
        question: "What Python version is specified in the handbook's environment setup?",
        code: `conda create -n ml-production python=??? -y`,
        options: null,
        correct: "3.9",
        explanation: "Python 3.9 is specified as it provides a good balance of stability, performance, and compatibility with modern ML libraries while avoiding potential issues with newer Python versions."
    },
    {
        id: 9,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "Which Azure package is required for Azure ML integration?",
        code: `pip install ???==1.12.0`,
        options: [
            "azure-ml-client",
            "azureml-sdk",
            "azure-ai-ml",
            "azure-machine-learning"
        ],
        correct: 2,
        explanation: "azure-ai-ml is the modern Azure ML SDK package that provides comprehensive functionality for training, deploying, and managing models on Azure."
    },
    {
        id: 10,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "What is the purpose of setting PYTHONUNBUFFERED=1 in the Dockerfile?",
        code: `ENV PYTHONUNBUFFERED=1`,
        options: [
            "Increases Python execution speed",
            "Forces stdout/stderr to be unbuffered for real-time logging",
            "Enables GPU acceleration",
            "Reduces memory usage"
        ],
        correct: 1,
        explanation: "PYTHONUNBUFFERED=1 forces Python to run in unbuffered mode, ensuring that logs are immediately visible in Docker containers rather than being buffered."
    },

    // ========== DISTRIBUTED TRAINING (Questions 11-30) ==========
    {
        id: 11,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What is the key difference between Data Parallel (DP) and Distributed Data Parallel (DDP)?",
        code: null,
        options: [
            "DP uses multiple GPUs on single node, DDP can span multiple nodes",
            "DP is faster than DDP for all scenarios",
            "DDP only works with transformer models",
            "DP supports gradient accumulation, DDP does not"
        ],
        correct: 0,
        explanation: "Data Parallel (DP) is limited to multiple GPUs on a single node and uses parameter broadcasting which creates bottlenecks. DDP can span multiple nodes and uses more efficient communication patterns."
    },
    {
        id: 12,
        type: "short-answer",
        category: "Distributed Training",
        question: "What is the default backend for DDP on GPU systems?",
        code: `torch.distributed.init_process_group(backend='???')`,
        options: null,
        correct: "nccl",
        explanation: "NCCL (NVIDIA Collective Communication Library) is the default and recommended backend for GPU-based distributed training due to its optimized communication primitives."
    },
    {
        id: 13,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "In FSDP, which sharding strategy provides maximum memory savings?",
        code: `from torch.distributed.fsdp import ShardingStrategy
strategy = ShardingStrategy.???`,
        options: [
            "NO_SHARD",
            "SHARD_GRAD_OP",
            "FULL_SHARD",
            "HYBRID_SHARD"
        ],
        correct: 2,
        explanation: "FULL_SHARD provides maximum memory savings by sharding parameters, gradients, and optimizer states across all devices. Each device only holds 1/N of the model state."
    },
    {
        id: 14,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What is the primary advantage of gradient accumulation in distributed training?",
        code: `for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()`,
        options: [
            "Reduces communication overhead",
            "Simulates larger batch sizes with limited memory",
            "Improves model accuracy",
            "Speeds up forward pass"
        ],
        correct: 1,
        explanation: "Gradient accumulation allows simulation of larger batch sizes by accumulating gradients over multiple forward passes before updating weights, crucial when GPU memory limits batch size."
    },
    {
        id: 15,
        type: "short-answer",
        category: "Distributed Training",
        question: "In DeepSpeed, what is the name of the configuration parameter that enables CPU offloading?",
        code: `{
    "optimizer": {
        "type": "AdamW",
        "???": true
    }
}`,
        options: null,
        correct: "offload_optimizer",
        explanation: "The 'offload_optimizer' parameter in DeepSpeed configuration enables moving optimizer states to CPU memory, significantly reducing GPU memory usage for large models."
    },
    {
        id: 16,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What communication pattern does DDP use for gradient synchronization?",
        code: null,
        options: [
            "Parameter server",
            "All-reduce",
            "Broadcast",
            "Point-to-point"
        ],
        correct: 1,
        explanation: "DDP uses the all-reduce communication pattern where gradients are reduced across all processes and the result is distributed back to all processes, ensuring synchronized updates."
    },
    {
        id: 17,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "Which environment variable sets the master node address for distributed training?",
        code: `export ???=node0
export MASTER_PORT=29500`,
        options: [
            "MASTER_NODE",
            "MASTER_ADDR",
            "DIST_MASTER",
            "HEAD_NODE"
        ],
        correct: 1,
        explanation: "MASTER_ADDR specifies the IP address or hostname of the master node that coordinates the distributed training job."
    },
    {
        id: 18,
        type: "short-answer",
        category: "Distributed Training",
        question: "What is the maximum number of GPUs that can participate in a single NCCL ring?",
        code: null,
        options: null,
        correct: "unlimited",
        explanation: "NCCL doesn't have a hard limit on the number of GPUs in a ring. However, practical limits arise from network topology and bandwidth. Modern NCCL uses tree and double-binary tree algorithms for better scaling."
    },
    {
        id: 19,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "In mixed precision training, what is the purpose of the GradScaler?",
        code: `from torch.cuda.amp import GradScaler
scaler = GradScaler()`,
        options: [
            "Converts gradients to FP16",
            "Prevents gradient underflow in FP16 training",
            "Reduces gradient communication",
            "Accumulates gradients"
        ],
        correct: 1,
        explanation: "GradScaler prevents gradient underflow by scaling losses before backward pass and unscaling gradients before optimizer step, crucial for stable FP16 training."
    },
    {
        id: 20,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What is the recommended way to handle uneven data distribution across distributed workers?",
        code: null,
        options: [
            "Use DistributedSampler with drop_last=False",
            "Manually balance data across workers",
            "Use DistributedSampler with drop_last=True",
            "Ignore the imbalance"
        ],
        correct: 2,
        explanation: "Using DistributedSampler with drop_last=True ensures all workers process the same number of batches, preventing synchronization issues at epoch boundaries."
    },
    {
        id: 21,
        type: "short-answer",
        category: "Distributed Training",
        question: "What PyTorch function synchronizes all processes in distributed training?",
        code: `# Wait for all processes to reach this point
torch.distributed.???()`,
        options: null,
        correct: "barrier",
        explanation: "torch.distributed.barrier() creates a synchronization point where all processes must wait until every process reaches the barrier before continuing."
    },
    {
        id: 22,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "Which DeepSpeed ZeRO stage provides the most memory savings?",
        code: null,
        options: [
            "ZeRO-1 (Optimizer states sharding)",
            "ZeRO-2 (Optimizer + gradients sharding)",
            "ZeRO-3 (Optimizer + gradients + parameters sharding)",
            "ZeRO-Infinity"
        ],
        correct: 2,
        explanation: "ZeRO-3 provides maximum memory savings by sharding optimizer states, gradients, AND model parameters across devices, though ZeRO-Infinity adds CPU/NVMe offloading for even larger models."
    },
    {
        id: 23,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What is the purpose of gradient checkpointing?",
        code: `model.gradient_checkpointing_enable()`,
        options: [
            "Save gradients to disk",
            "Trade compute for memory by recomputing activations",
            "Checkpoint gradients for fault tolerance",
            "Compress gradients"
        ],
        correct: 1,
        explanation: "Gradient checkpointing trades computation for memory by not storing all activations during forward pass, instead recomputing them during backward pass when needed."
    },
    {
        id: 24,
        type: "short-answer",
        category: "Distributed Training",
        question: "In multi-node training, what network interface variable should be set for optimal performance?",
        code: `export NCCL_SOCKET_IFNAME=???`,
        options: null,
        correct: "eth0",
        explanation: "NCCL_SOCKET_IFNAME should be set to the high-speed network interface (often eth0 or ib0 for InfiniBand) to ensure NCCL uses the fastest available network for communication."
    },
    {
        id: 25,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What happens when find_unused_parameters=True in DDP?",
        code: `model = DDP(model, find_unused_parameters=True)`,
        options: [
            "DDP automatically removes unused parameters",
            "DDP tracks and handles parameters not used in forward pass",
            "DDP optimizes memory usage",
            "DDP increases training speed"
        ],
        correct: 1,
        explanation: "With find_unused_parameters=True, DDP tracks which parameters are used in forward pass and handles cases where not all parameters contribute to the loss, at the cost of some overhead."
    },
    {
        id: 26,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What is the primary benefit of using torch.compile() with distributed training?",
        code: `model = torch.compile(model)`,
        options: [
            "Automatic model parallelism",
            "JIT compilation for faster execution",
            "Automatic mixed precision",
            "Distributed checkpointing"
        ],
        correct: 1,
        explanation: "torch.compile() uses TorchDynamo and TorchInductor to JIT compile models, providing significant speedups especially for transformer models in distributed settings."
    },
    {
        id: 27,
        type: "short-answer",
        category: "Distributed Training",
        question: "What is the default timeout (in seconds) for distributed operations in PyTorch?",
        code: null,
        options: null,
        correct: "1800",
        explanation: "The default timeout is 1800 seconds (30 minutes) for distributed operations. This can be adjusted using the timeout parameter in init_process_group()."
    },
    {
        id: 28,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "Which collective operation is most efficient for broadcasting model parameters?",
        code: null,
        options: [
            "all_reduce",
            "broadcast",
            "all_gather",
            "reduce_scatter"
        ],
        correct: 1,
        explanation: "broadcast is the most efficient for sending data from one process to all others, perfect for distributing initial model parameters from rank 0."
    },
    {
        id: 29,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What is pipeline parallelism best suited for?",
        code: null,
        options: [
            "Wide models with many parameters per layer",
            "Deep models with sequential layers",
            "Models with dynamic computation graphs",
            "Small models that fit on single GPU"
        ],
        correct: 1,
        explanation: "Pipeline parallelism is best for deep models with sequential layers, where different layers can be placed on different GPUs and micro-batches pipelined through them."
    },
    {
        id: 30,
        type: "short-answer",
        category: "Distributed Training",
        question: "What FSDP parameter controls the minimum number of parameters before wrapping?",
        code: `auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, 
    min_num_params=???
)`,
        options: null,
        correct: "1e6",
        explanation: "min_num_params (often set to 1e6 or 1 million) determines the minimum number of parameters a layer must have before FSDP wraps it as a separate unit for sharding."
    },

    // ========== PEFT TECHNIQUES (Questions 31-50) ==========
    {
        id: 31,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "In LoRA, if we have a weight matrix W of size 4096×4096 and use rank r=16, how many trainable parameters does the LoRA adapter add?",
        code: `# LoRA decomposition: ΔW = B × A
A = torch.randn(16, 4096)  # Trainable
B = torch.randn(4096, 16)  # Trainable`,
        options: null,
        correct: "131072",
        explanation: "LoRA adds A (16×4096) and B (4096×16). Total = 65,536 + 65,536 = 131,072 parameters, only 0.78% of the original 16.7M parameters."
    },
    {
        id: 32,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "What is the main advantage of QLoRA over standard LoRA?",
        code: null,
        options: [
            "Higher training accuracy",
            "Faster training speed",
            "Significantly reduced memory usage",
            "Better gradient stability"
        ],
        correct: 2,
        explanation: "QLoRA's main advantage is drastically reduced memory usage by using 4-bit quantized base models while keeping LoRA adapters in higher precision."
    },
    {
        id: 33,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "In LoRA, what is the purpose of the alpha parameter?",
        code: `LoraConfig(r=16, lora_alpha=32, ...)`,
        options: [
            "Controls the learning rate",
            "Scaling factor for LoRA updates",
            "Number of attention heads",
            "Dropout probability"
        ],
        correct: 1,
        explanation: "lora_alpha is a scaling factor that controls the magnitude of LoRA updates. The effective scaling is alpha/r, so alpha=32 with r=16 gives a scaling of 2."
    },
    {
        id: 34,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "What is the typical rank (r) range used in LoRA for LLMs?",
        code: null,
        options: null,
        correct: "4-64",
        explanation: "Typical LoRA ranks range from 4 to 64, with 8 or 16 being common choices. Lower ranks save memory but may limit adaptation capacity, while higher ranks increase expressiveness."
    },
    {
        id: 35,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "Which modules are typically targeted for LoRA adaptation in transformer models?",
        code: `target_modules = ["???", "???"]`,
        options: [
            "LayerNorm and embeddings",
            "q_proj and v_proj",
            "FFN layers only",
            "All linear layers"
        ],
        correct: 1,
        explanation: "Query (q_proj) and value (v_proj) projections are the most common targets for LoRA, though k_proj and o_proj can also be included for better performance."
    },
    {
        id: 36,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "What is AdaLoRA's key innovation compared to standard LoRA?",
        code: null,
        options: [
            "Uses higher precision",
            "Adaptive rank allocation across layers",
            "Faster training speed",
            "Better initialization"
        ],
        correct: 1,
        explanation: "AdaLoRA dynamically allocates parameter budgets across different weight matrices based on their importance, optimizing the rank distribution for better performance."
    },
    {
        id: 37,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "In Prefix Tuning, what is the typical prefix length for a model with 12 layers?",
        code: null,
        options: null,
        correct: "20-50",
        explanation: "Prefix lengths typically range from 20-50 tokens per layer. Too short limits adaptation capacity, while too long wastes computation and may hurt performance."
    },
    {
        id: 38,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "What quantization format does QLoRA use for the base model?",
        code: null,
        options: [
            "INT8",
            "FP16",
            "NF4 (4-bit NormalFloat)",
            "INT4"
        ],
        correct: 2,
        explanation: "QLoRA uses NF4 (4-bit NormalFloat) quantization, which is specifically designed for normally distributed weights and provides better accuracy than standard INT4."
    },
    {
        id: 39,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "How does LoRA affect inference speed compared to full model?",
        code: null,
        options: [
            "Significantly slower due to adapter overhead",
            "Same speed after merging adapters",
            "Always faster due to fewer parameters",
            "Depends on batch size only"
        ],
        correct: 1,
        explanation: "After training, LoRA adapters can be merged into base model weights (W' = W + BA), resulting in the same inference speed as the original model."
    },
    {
        id: 40,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "What is the memory saving factor when using QLoRA on a 7B parameter model?",
        code: null,
        options: null,
        correct: "4x",
        explanation: "QLoRA reduces memory by approximately 4x through 4-bit quantization of the base model, enabling 7B model fine-tuning on consumer GPUs with 24GB memory."
    },
    {
        id: 41,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "Which PEFT method modifies input embeddings rather than model weights?",
        code: null,
        options: [
            "LoRA",
            "QLoRA",
            "Prompt Tuning",
            "AdaLoRA"
        ],
        correct: 2,
        explanation: "Prompt Tuning learns continuous prompt embeddings that are prepended to input, modifying model behavior without changing any model weights."
    },
    {
        id: 42,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "What is the recommended dropout rate for LoRA adapters?",
        code: `LoraConfig(r=16, lora_dropout=???)`,
        options: [
            "0.0",
            "0.05-0.1",
            "0.3-0.5",
            "0.7-0.9"
        ],
        correct: 1,
        explanation: "LoRA dropout of 0.05-0.1 (5-10%) is typically recommended. Higher dropout can hurt performance since LoRA already has limited capacity."
    },
    {
        id: 43,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "What initialization method is used for LoRA's A matrix?",
        code: `# A matrix initialization
A = ???`,
        options: null,
        correct: "gaussian",
        explanation: "LoRA's A matrix is initialized with Gaussian/normal distribution (often with small std), while B matrix is initialized to zeros for stable training start."
    },
    {
        id: 44,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "How many LoRA adapters can be loaded simultaneously for multi-task inference?",
        code: null,
        options: [
            "Only one at a time",
            "Two (for comparison)",
            "Unlimited (memory permitting)",
            "Maximum of 5"
        ],
        correct: 2,
        explanation: "Multiple LoRA adapters can be loaded simultaneously, limited only by available memory. This enables efficient multi-task serving with a single base model."
    },
    {
        id: 45,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "What is the computational overhead of LoRA during training?",
        code: null,
        options: [
            "~50% slower than full fine-tuning",
            "~25% slower than full fine-tuning",
            "Negligible overhead (<5%)",
            "2x slower than full fine-tuning"
        ],
        correct: 2,
        explanation: "LoRA adds negligible computational overhead (<5%) during training since the additional matrix multiplications are small compared to the base model computations."
    },
    {
        id: 46,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "In PEFT library, what method merges LoRA weights into the base model?",
        code: `model = model.???()`,
        options: null,
        correct: "merge_and_unload",
        explanation: "merge_and_unload() merges the LoRA adapters into the base model weights and removes the adapter modules, creating a standard model for deployment."
    },
    {
        id: 47,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "Which PEFT technique is most memory-efficient for multi-task learning?",
        code: null,
        options: [
            "Full fine-tuning with task-specific heads",
            "LoRA with task-specific adapters",
            "Prompt tuning with task embeddings",
            "Adapter layers throughout the model"
        ],
        correct: 1,
        explanation: "LoRA with task-specific adapters is most memory-efficient, sharing the base model while using minimal parameters per task (typically <1% of model size)."
    },
    {
        id: 48,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "What happens to gradients of the base model during LoRA training?",
        code: null,
        options: [
            "They are computed but not applied",
            "They are not computed (frozen)",
            "They are computed with reduced precision",
            "They are accumulated separately"
        ],
        correct: 1,
        explanation: "Base model parameters are frozen during LoRA training, so gradients are not computed for them, saving memory and computation."
    },
    {
        id: 49,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "What is the typical learning rate multiplier for LoRA compared to full fine-tuning?",
        code: null,
        options: null,
        correct: "10x",
        explanation: "LoRA typically uses 10x higher learning rates than full fine-tuning (e.g., 1e-4 instead of 1e-5) because the adapters start from random initialization."
    },
    {
        id: 50,
        type: "multiple-choice",
        category: "PEFT Techniques",
        question: "Which quantization method does bitsandbytes use for 8-bit optimization?",
        code: null,
        options: [
            "Symmetric quantization",
            "Asymmetric quantization",
            "LLM.int8() with outlier handling",
            "Standard INT8 quantization"
        ],
        correct: 2,
        explanation: "bitsandbytes uses LLM.int8() which handles outliers in fp16 while quantizing the majority of values to int8, maintaining model quality."
    },

    // ========== MODEL OPTIMIZATION (Questions 51-65) ==========
    {
        id: 51,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "Which quantization technique typically provides the best balance between model size reduction and performance?",
        code: null,
        options: [
            "FP16 (16-bit floating point)",
            "INT8 (8-bit integer)",
            "INT4 (4-bit integer)",
            "Binary quantization (1-bit)"
        ],
        correct: 1,
        explanation: "INT8 quantization offers optimal trade-off, reducing model size by 4x while maintaining 95-99% of original performance with proper calibration."
    },
    {
        id: 52,
        type: "short-answer",
        category: "Model Optimization",
        question: "What is the name of the ONNX Runtime execution provider for NVIDIA GPUs?",
        code: null,
        options: null,
        correct: "CUDAExecutionProvider",
        explanation: "CUDAExecutionProvider enables GPU acceleration in ONNX Runtime, offering optimized kernels for NVIDIA GPUs."
    },
    {
        id: 53,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "What is the primary purpose of knowledge distillation?",
        code: null,
        options: [
            "Compress a large model into a smaller one",
            "Improve model accuracy",
            "Speed up training",
            "Reduce memory usage during training"
        ],
        correct: 0,
        explanation: "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model, maintaining performance while reducing size and inference cost."
    },
    {
        id: 54,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "Which pruning strategy typically preserves model accuracy best?",
        code: null,
        options: [
            "Random unstructured pruning",
            "Magnitude-based structured pruning",
            "Gradual magnitude pruning",
            "One-shot pruning"
        ],
        correct: 2,
        explanation: "Gradual magnitude pruning progressively removes weights while allowing the model to adapt, typically preserving accuracy better than aggressive one-shot methods."
    },
    {
        id: 55,
        type: "short-answer",
        category: "Model Optimization",
        question: "What TensorRT optimization is most effective for transformer models?",
        code: null,
        options: null,
        correct: "kernel fusion",
        explanation: "Kernel fusion in TensorRT combines multiple operations (like attention computations) into single kernels, significantly reducing memory bandwidth and improving speed."
    },
    {
        id: 56,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "What is the main advantage of dynamic quantization over static quantization?",
        code: null,
        options: [
            "Better accuracy",
            "No calibration dataset required",
            "Faster inference",
            "Smaller model size"
        ],
        correct: 1,
        explanation: "Dynamic quantization quantizes weights ahead of time but activations dynamically, requiring no calibration dataset while still providing significant speedups."
    },
    {
        id: 57,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "Which optimization technique is most effective for reducing transformer model latency?",
        code: null,
        options: [
            "Weight pruning",
            "Flash Attention",
            "Quantization only",
            "Model distillation"
        ],
        correct: 1,
        explanation: "Flash Attention dramatically reduces memory bandwidth requirements and improves speed by computing attention in blocks, particularly effective for long sequences."
    },
    {
        id: 58,
        type: "short-answer",
        category: "Model Optimization",
        question: "What is the typical compression ratio achieved by converting FP32 to FP16?",
        code: null,
        options: null,
        correct: "2x",
        explanation: "FP16 uses 16 bits instead of 32 bits per parameter, achieving exactly 2x compression in model size and memory usage."
    },
    {
        id: 59,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "What is the purpose of calibration in INT8 quantization?",
        code: null,
        options: [
            "Train the quantized model",
            "Determine optimal scale factors",
            "Compress the model weights",
            "Validate model accuracy"
        ],
        correct: 1,
        explanation: "Calibration analyzes activation distributions on representative data to determine optimal scale factors for converting FP32 values to INT8 with minimal accuracy loss."
    },
    {
        id: 60,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "Which model format is best for cross-platform deployment?",
        code: null,
        options: [
            "PyTorch .pth",
            "TensorFlow SavedModel",
            "ONNX",
            "TensorRT engine"
        ],
        correct: 2,
        explanation: "ONNX (Open Neural Network Exchange) is designed for cross-platform compatibility, supporting deployment across different frameworks and hardware."
    },
    {
        id: 61,
        type: "short-answer",
        category: "Model Optimization",
        question: "What percentage of weights can typically be pruned from a transformer model without significant accuracy loss?",
        code: null,
        options: null,
        correct: "50-70",
        explanation: "Transformer models can typically handle 50-70% unstructured pruning without significant accuracy loss when using gradual pruning with fine-tuning."
    },
    {
        id: 62,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "What is the main limitation of structured pruning compared to unstructured pruning?",
        code: null,
        options: [
            "Lower compression ratios",
            "Requires special hardware",
            "Slower inference",
            "More complex implementation"
        ],
        correct: 0,
        explanation: "Structured pruning removes entire channels/heads/layers for hardware efficiency but typically achieves lower compression ratios than unstructured pruning."
    },
    {
        id: 63,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "Which optimization provides the best speedup for batch inference?",
        code: null,
        options: [
            "Model quantization",
            "Dynamic batching",
            "Weight pruning",
            "Knowledge distillation"
        ],
        correct: 1,
        explanation: "Dynamic batching groups multiple requests together, amortizing model loading overhead and utilizing GPU parallelism more efficiently."
    },
    {
        id: 64,
        type: "short-answer",
        category: "Model Optimization",
        question: "What is the minimum GPU compute capability required for INT8 Tensor Core operations?",
        code: null,
        options: null,
        correct: "7.5",
        explanation: "GPU compute capability 7.5 (Turing architecture and newer) is required for INT8 Tensor Core operations, providing significant speedups for quantized models."
    },
    {
        id: 65,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "What optimization technique specifically targets reducing KV cache memory in transformers?",
        code: null,
        options: [
            "Weight quantization",
            "Multi-Query Attention (MQA)",
            "Layer pruning",
            "Activation quantization"
        ],
        correct: 1,
        explanation: "Multi-Query Attention (MQA) shares key and value projections across attention heads, dramatically reducing KV cache memory requirements for long sequences."
    },

    // ========== ARCHITECTURE (Questions 66-75) ==========
    {
        id: 66,
        type: "multiple-choice",
        category: "Architecture",
        question: "In the MLOps architecture, which component is responsible for model versioning and experiment tracking?",
        code: null,
        options: [
            "Model Registry",
            "Training Orchestrator",
            "Inference Server",
            "Monitoring Dashboard"
        ],
        correct: 0,
        explanation: "The Model Registry serves as the central hub for model versioning, metadata management, and experiment tracking, maintaining lineage information."
    },
    {
        id: 67,
        type: "short-answer",
        category: "Architecture",
        question: "What pattern is recommended for handling model rollbacks in production?",
        code: null,
        options: null,
        correct: "blue-green",
        explanation: "Blue-green deployment pattern enables instant rollbacks by maintaining two identical production environments and switching traffic between them."
    },
    {
        id: 68,
        type: "multiple-choice",
        category: "Architecture",
        question: "Which component handles request queuing and batching in the serving architecture?",
        code: null,
        options: [
            "Load balancer",
            "Model server",
            "Request scheduler",
            "API gateway"
        ],
        correct: 2,
        explanation: "The request scheduler manages incoming requests, implements dynamic batching, and optimizes throughput by grouping requests efficiently."
    },
    {
        id: 69,
        type: "multiple-choice",
        category: "Architecture",
        question: "What is the recommended storage solution for model artifacts in cloud deployments?",
        code: null,
        options: [
            "Local file system",
            "Relational database",
            "Object storage (S3/Blob)",
            "Network file system"
        ],
        correct: 2,
        explanation: "Object storage like S3 or Azure Blob provides scalable, durable, and cost-effective storage for large model artifacts with versioning support."
    },
    {
        id: 70,
        type: "short-answer",
        category: "Architecture",
        question: "What is the typical target SLA for model inference latency in production?",
        code: null,
        options: null,
        correct: "50-100ms",
        explanation: "Production systems typically target 50-100ms P99 latency for real-time inference, balancing user experience with infrastructure costs."
    },
    {
        id: 71,
        type: "multiple-choice",
        category: "Architecture",
        question: "Which caching layer is most effective for reducing model loading overhead?",
        code: null,
        options: [
            "Redis for model weights",
            "In-memory model cache",
            "CDN for model files",
            "Database query cache"
        ],
        correct: 1,
        explanation: "In-memory model caching on inference servers eliminates repeated loading overhead, crucial for achieving low latency targets."
    },
    {
        id: 72,
        type: "multiple-choice",
        category: "Architecture",
        question: "What is the primary purpose of a feature store in ML architecture?",
        code: null,
        options: [
            "Store trained models",
            "Centralize feature computation and serving",
            "Cache inference results",
            "Store training data"
        ],
        correct: 1,
        explanation: "Feature stores centralize feature computation, storage, and serving, ensuring consistency between training and inference while enabling feature reuse."
    },
    {
        id: 73,
        type: "short-answer",
        category: "Architecture",
        question: "What protocol is commonly used for high-performance model serving?",
        code: null,
        options: null,
        correct: "gRPC",
        explanation: "gRPC provides efficient binary serialization and streaming capabilities, making it ideal for high-performance model serving compared to REST/HTTP."
    },
    {
        id: 74,
        type: "multiple-choice",
        category: "Architecture",
        question: "Which pattern best handles varying inference loads throughout the day?",
        code: null,
        options: [
            "Vertical scaling",
            "Horizontal auto-scaling",
            "Request throttling",
            "Load shedding"
        ],
        correct: 1,
        explanation: "Horizontal auto-scaling automatically adjusts the number of inference servers based on load, efficiently handling daily traffic patterns."
    },
    {
        id: 75,
        type: "multiple-choice",
        category: "Architecture",
        question: "What is the recommended approach for A/B testing model deployments?",
        code: null,
        options: [
            "Deploy to separate clusters",
            "Traffic splitting at load balancer",
            "Client-side routing",
            "Time-based switching"
        ],
        correct: 1,
        explanation: "Traffic splitting at the load balancer level enables controlled A/B testing with configurable traffic percentages and easy rollback capabilities."
    },

    // ========== DEPLOYMENT/SERVING (Questions 76-90) ==========
    {
        id: 76,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "What is the primary advantage of using TorchServe over a custom Flask server?",
        code: null,
        options: [
            "Easier to implement",
            "Built-in model management and scaling",
            "Lower latency",
            "Smaller memory footprint"
        ],
        correct: 1,
        explanation: "TorchServe provides production-ready features like model versioning, automatic batching, metrics, and scaling that would require significant custom development."
    },
    {
        id: 77,
        type: "short-answer",
        category: "Deployment/Serving",
        question: "What is the default port for TorchServe inference API?",
        code: null,
        options: null,
        correct: "8080",
        explanation: "TorchServe uses port 8080 for inference API and port 8081 for management API by default, following common microservice conventions."
    },
    {
        id: 78,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "Which batching strategy is most effective for LLM serving?",
        code: null,
        options: [
            "Fixed-size batching",
            "Dynamic batching with timeout",
            "No batching",
            "Client-side batching"
        ],
        correct: 1,
        explanation: "Dynamic batching with timeout groups requests up to a maximum batch size or timeout, balancing latency and throughput for LLM serving."
    },
    {
        id: 79,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "What is the recommended way to handle model warm-up in production?",
        code: null,
        options: [
            "First real request warms up the model",
            "Pre-load and run dummy inference on startup",
            "Periodic background warm-up",
            "No warm-up needed"
        ],
        correct: 1,
        explanation: "Pre-loading models and running dummy inference on startup ensures consistent latency from the first request, critical for production SLAs."
    },
    {
        id: 80,
        type: "short-answer",
        category: "Deployment/Serving",
        question: "What environment variable controls the number of worker processes in Gunicorn?",
        code: `gunicorn app:app --workers ???`,
        options: null,
        correct: "4",
        explanation: "The --workers flag controls Gunicorn worker processes. A common formula is (2 × CPU cores) + 1, so 4 workers for a 2-core system."
    },
    {
        id: 81,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "Which strategy best handles long-running inference requests?",
        code: null,
        options: [
            "Synchronous processing",
            "Async processing with message queues",
            "Increase timeout values",
            "Reject long requests"
        ],
        correct: 1,
        explanation: "Async processing with message queues (like Celery + Redis) prevents blocking, enables scaling, and provides better reliability for long-running tasks."
    },
    {
        id: 82,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "What is the primary benefit of using NVIDIA Triton Inference Server?",
        code: null,
        options: [
            "Python-only implementation",
            "Multi-framework support with optimizations",
            "Smallest memory footprint",
            "Simplest deployment"
        ],
        correct: 1,
        explanation: "Triton supports multiple frameworks (PyTorch, TensorFlow, ONNX, TensorRT) with built-in optimizations, dynamic batching, and model ensembles."
    },
    {
        id: 83,
        type: "short-answer",
        category: "Deployment/Serving",
        question: "What is the recommended timeout for LLM inference endpoints?",
        code: null,
        options: null,
        correct: "60-120s",
        explanation: "LLM inference typically requires 60-120 second timeouts to handle longer sequences and generation tasks while preventing resource exhaustion."
    },
    {
        id: 84,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "Which deployment pattern provides the best fault isolation?",
        code: null,
        options: [
            "Monolithic deployment",
            "Microservices per model",
            "Serverless functions",
            "Single container multiple models"
        ],
        correct: 1,
        explanation: "Microservices per model provide excellent fault isolation - if one model fails, others continue operating, enabling independent scaling and updates."
    },
    {
        id: 85,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "What is the most important metric to monitor for model serving?",
        code: null,
        options: [
            "CPU usage",
            "Request latency (P99)",
            "Memory usage",
            "Network bandwidth"
        ],
        correct: 1,
        explanation: "P99 request latency directly impacts user experience and SLA compliance. Other metrics are important but latency is the primary user-facing metric."
    },
    {
        id: 86,
        type: "short-answer",
        category: "Deployment/Serving",
        question: "What Kubernetes resource type is best for model serving deployments?",
        code: null,
        options: null,
        correct: "Deployment",
        explanation: "Kubernetes Deployments provide declarative updates, scaling, and rollback capabilities ideal for stateless model serving workloads."
    },
    {
        id: 87,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "How should model artifacts be handled in containerized deployments?",
        code: null,
        options: [
            "Bake into the container image",
            "Download from object storage on startup",
            "Mount from persistent volume",
            "Copy during deployment"
        ],
        correct: 1,
        explanation: "Downloading from object storage on startup keeps images small, enables model updates without rebuilding, and supports multiple model versions."
    },
    {
        id: 88,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "What is the recommended approach for handling GPU memory in multi-model serving?",
        code: null,
        options: [
            "Load all models at startup",
            "Lazy loading with LRU eviction",
            "One model per GPU",
            "Time-based model swapping"
        ],
        correct: 1,
        explanation: "Lazy loading with LRU (Least Recently Used) eviction optimizes GPU memory usage by keeping frequently used models loaded while evicting inactive ones."
    },
    {
        id: 89,
        type: "short-answer",
        category: "Deployment/Serving",
        question: "What HTTP status code should be returned when model inference fails due to invalid input?",
        code: null,
        options: null,
        correct: "400",
        explanation: "HTTP 400 (Bad Request) indicates client error due to invalid input, helping clients distinguish between their errors and server issues."
    },
    {
        id: 90,
        type: "multiple-choice",
        category: "Deployment/Serving",
        question: "Which caching strategy is most effective for LLM responses?",
        code: null,
        options: [
            "Cache all responses indefinitely",
            "Semantic similarity-based caching",
            "No caching due to uniqueness",
            "Time-based expiration only"
        ],
        correct: 1,
        explanation: "Semantic similarity-based caching can identify and reuse responses for similar queries, significantly reducing compute while maintaining response quality."
    },

    // ========== MONITORING (Questions 91-100) ==========
    {
        id: 91,
        type: "multiple-choice",
        category: "Monitoring",
        question: "What is the most critical metric for detecting model drift in production?",
        code: null,
        options: [
            "Inference latency",
            "Prediction distribution shift",
            "Request volume",
            "Model accuracy"
        ],
        correct: 1,
        explanation: "Prediction distribution shift indicates when model outputs deviate from training distribution, an early indicator of model drift before accuracy degrades."
    },
    {
        id: 92,
        type: "short-answer",
        category: "Monitoring",
        question: "What is the standard format for structured logs in production ML systems?",
        code: null,
        options: null,
        correct: "JSON",
        explanation: "JSON format enables structured logging with parseable fields, making it easy to aggregate, search, and analyze logs in systems like ELK stack."
    },
    {
        id: 93,
        type: "multiple-choice",
        category: "Monitoring",
        question: "Which tool is commonly used for distributed tracing in ML systems?",
        code: null,
        options: [
            "Prometheus",
            "Grafana",
            "Jaeger",
            "Tensorboard"
        ],
        correct: 2,
        explanation: "Jaeger provides distributed tracing capabilities, essential for debugging request flow across multiple services in complex ML serving systems."
    },
    {
        id: 94,
        type: "multiple-choice",
        category: "Monitoring",
        question: "What is the recommended sampling rate for high-volume inference logging?",
        code: null,
        options: [
            "Log every request",
            "1-10% sampling",
            "Fixed 1000 requests/hour",
            "No logging needed"
        ],
        correct: 1,
        explanation: "1-10% sampling balances observability with storage costs and performance impact, while still providing statistically significant insights."
    },
    {
        id: 95,
        type: "short-answer",
        category: "Monitoring",
        question: "What prometheus metric type is best for tracking inference latency?",
        code: null,
        options: null,
        correct: "histogram",
        explanation: "Histogram metrics capture latency distribution, enabling calculation of percentiles (P50, P99) crucial for SLA monitoring."
    },
    {
        id: 96,
        type: "multiple-choice",
        category: "Monitoring",
        question: "Which alerting threshold is most appropriate for model serving error rates?",
        code: null,
        options: [
            "Alert on any error",
            "Alert when error rate > 1% for 5 minutes",
            "Alert when error rate > 50%",
            "Daily error summary only"
        ],
        correct: 1,
        explanation: "Alerting on >1% error rate sustained for 5 minutes balances quick detection with avoiding false alarms from transient issues."
    },
    {
        id: 97,
        type: "multiple-choice",
        category: "Monitoring",
        question: "What information should be included in every inference log?",
        code: null,
        options: [
            "Request ID, timestamp, model version, latency",
            "Only errors and exceptions",
            "Input and output data",
            "User personal information"
        ],
        correct: 0,
        explanation: "Request ID (for tracing), timestamp, model version (for debugging), and latency are essential fields for production monitoring and debugging."
    },
    {
        id: 98,
        type: "short-answer",
        category: "Monitoring",
        question: "What is the typical retention period for inference logs in production?",
        code: null,
        options: null,
        correct: "30-90 days",
        explanation: "30-90 days retention balances debugging needs with storage costs. Critical metrics can be aggregated and stored longer term."
    },
    {
        id: 99,
        type: "multiple-choice",
        category: "Monitoring",
        question: "Which monitoring approach best detects gradual performance degradation?",
        code: null,
        options: [
            "Threshold-based alerts",
            "Anomaly detection with baselines",
            "Manual daily reviews",
            "User complaint tracking"
        ],
        correct: 1,
        explanation: "Anomaly detection with baselines can identify gradual degradation that threshold alerts miss, using statistical methods to detect subtle changes."
    },
    {
        id: 100,
        type: "multiple-choice",
        category: "Monitoring",
        question: "What dashboard metric is most important for business stakeholders?",
        code: null,
        options: [
            "GPU utilization percentage",
            "Model inference cost per request",
            "Container restart count",
            "Network packet loss"
        ],
        correct: 1,
        explanation: "Cost per request directly relates to business metrics and ROI, helping stakeholders understand the financial efficiency of ML deployments."
    }
];

// Export for use in quiz application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = quizData;
} else if (typeof window !== 'undefined') {
    window.quizData = quizData;
}