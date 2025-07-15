// Quiz Data - 10 Questions based on Production ML Engineering Handbook
const quizData = [
    {
        id: 1,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "What is the key difference between Data Parallel (DP) and Distributed Data Parallel (DDP) in PyTorch?",
        code: null,
        options: [
            "DP uses multiple GPUs on single node, DDP can span multiple nodes",
            "DP is faster than DDP for all scenarios",
            "DDP only works with transformer models",
            "DP supports gradient accumulation, DDP does not"
        ],
        correct: 0,
        explanation: "Data Parallel (DP) is limited to multiple GPUs on a single node and uses parameter broadcasting which creates bottlenecks. Distributed Data Parallel (DDP) can span multiple nodes and uses more efficient communication patterns with gradient reduction, making it the preferred choice for large-scale training."
    },
    {
        id: 2,
        type: "short-answer",
        category: "PEFT Techniques",
        question: "In LoRA (Low-Rank Adaptation), if we have a weight matrix W of size 4096×4096 and use rank r=16, how many trainable parameters does the LoRA adapter add?",
        code: `# Original weight matrix W
W_original = torch.randn(4096, 4096)  # Frozen
# LoRA decomposition: ΔW = B × A
A = torch.randn(16, 4096)     # Trainable
B = torch.randn(4096, 16)     # Trainable`,
        options: null,
        correct: "131072",
        explanation: "LoRA adds two matrices: A (16×4096) and B (4096×16). Total trainable parameters = 16×4096 + 4096×16 = 65,536 + 65,536 = 131,072 parameters. This is only 0.78% of the original 16.7M parameters in the full matrix!"
    },
    {
        id: 3,
        type: "multiple-choice",
        category: "Model Optimization",
        question: "Which quantization technique typically provides the best balance between model size reduction and performance preservation?",
        code: `# Different quantization approaches
fp32_model = load_model()              # 32-bit floating point
fp16_model = quantize_fp16(model)      # 16-bit floating point  
int8_model = quantize_int8(model)      # 8-bit integer
int4_model = quantize_int4(model)      # 4-bit integer`,
        options: [
            "FP16 (16-bit floating point)",
            "INT8 (8-bit integer)",
            "INT4 (4-bit integer)",
            "Binary quantization (1-bit)"
        ],
        correct: 1,
        explanation: "INT8 quantization typically offers the optimal trade-off, reducing model size by 4x compared to FP32 while maintaining 95-99% of original performance. FP16 provides less compression (2x), while INT4 can cause significant accuracy degradation without careful calibration."
    },
    {
        id: 4,
        type: "multiple-choice",
        category: "Environment Setup",
        question: "What is the purpose of the NCCL_DEBUG environment variable in distributed training?",
        code: `export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
python -m torch.distributed.launch train.py`,
        options: [
            "Enables debugging output for NCCL communication",
            "Sets the number of communication threads",
            "Configures GPU memory allocation",
            "Controls gradient synchronization frequency"
        ],
        correct: 0,
        explanation: "NCCL_DEBUG=INFO enables detailed logging of NCCL (NVIDIA Collective Communication Library) operations, helping debug communication issues in multi-GPU/multi-node training. It shows information about network topology, communication patterns, and potential bottlenecks."
    },
    {
        id: 5,
        type: "short-answer",
        category: "Performance Analysis",
        question: "If a model has 7B parameters and you're using FP16 precision, approximately how much GPU memory is needed just to store the model weights?",
        code: null,
        options: null,
        correct: "14",
        explanation: "With FP16 precision, each parameter takes 2 bytes. For 7B parameters: 7,000,000,000 × 2 bytes = 14,000,000,000 bytes = 14 GB. Note that this is only for model weights - actual training requires additional memory for gradients, optimizer states, and activations."
    },
    {
        id: 6,
        type: "multiple-choice",
        category: "Distributed Training",
        question: "Which sharding strategy in FSDP (Fully Sharded Data Parallel) provides the maximum memory savings?",
        code: `from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# Different sharding strategies
strategies = [
    ShardingStrategy.NO_SHARD,
    ShardingStrategy.SHARD_GRAD_OP, 
    ShardingStrategy.FULL_SHARD,
    ShardingStrategy.HYBRID_SHARD
]`,
        options: [
            "NO_SHARD",
            "SHARD_GRAD_OP", 
            "FULL_SHARD",
            "HYBRID_SHARD"
        ],
        correct: 2,
        explanation: "FULL_SHARD provides maximum memory savings by sharding parameters, gradients, and optimizer states across all devices. Each device only holds 1/N of the model state where N is the number of devices, enabling training of much larger models."
    },
    {
        id: 7,
        type: "multiple-choice", 
        category: "PEFT Techniques",
        question: "What is the main advantage of QLoRA over standard LoRA?",
        code: `# Standard LoRA
model = load_model_fp16()  # 16-bit base model
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])

# QLoRA  
model = load_model_4bit()  # 4-bit quantized base model
qlora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])`,
        options: [
            "Higher training accuracy",
            "Faster training speed",
            "Significantly reduced memory usage",
            "Better gradient stability"
        ],
        correct: 2,
        explanation: "QLoRA's main advantage is drastically reduced memory usage by using 4-bit quantized base models while keeping LoRA adapters in higher precision. This enables fine-tuning of 65B+ parameter models on a single GPU, whereas standard LoRA would require multiple GPUs."
    },
    {
        id: 8,
        type: "short-answer",
        category: "Code Implementation",
        question: "In the provided DistributedTrainer class, what method would you call to initialize the distributed training process?",
        code: `class DistributedTrainer:
    def __init__(self, config):
        self.config = config
        
    def setup_distributed(self):
        # Initialize distributed training
        pass
        
    def train(self):
        # Main training loop
        pass
        
    def cleanup(self):
        # Cleanup distributed resources
        pass`,
        options: null,
        correct: "setup_distributed",
        explanation: "The setup_distributed() method is responsible for initializing the distributed training environment, including setting up process groups, device placement, and communication backends like NCCL."
    },
    {
        id: 9,
        type: "multiple-choice",
        category: "Architecture",
        question: "In the MLOps architecture diagram from Chapter 3, which component is responsible for model versioning and experiment tracking?",
        code: null,
        options: [
            "Model Registry",
            "Training Orchestrator", 
            "Inference Server",
            "Monitoring Dashboard"
        ],
        correct: 0,
        explanation: "The Model Registry serves as the central hub for model versioning, metadata management, and experiment tracking. It maintains lineage information, performance metrics, and enables model promotion through different environments (dev → staging → production)."
    },
    {
        id: 10,
        type: "multiple-choice",
        category: "Optimization Strategy",
        question: "According to the handbook, what is the recommended sequence for applying optimization techniques to a production model?",
        code: `# Optimization pipeline
model = load_pretrained_model()

# Step 1: ?
# Step 2: ?  
# Step 3: ?
# Step 4: Deploy`,
        options: [
            "Quantization → Pruning → Distillation → ONNX Conversion",
            "PEFT Fine-tuning → Quantization → ONNX Conversion → TensorRT",
            "Pruning → Quantization → Distillation → Optimization",
            "Distillation → PEFT → Quantization → Deployment"
        ],
        correct: 1,
        explanation: "The recommended sequence is: 1) PEFT Fine-tuning for task adaptation, 2) Quantization for size/speed optimization, 3) ONNX Conversion for cross-platform compatibility, 4) TensorRT optimization for GPU acceleration. This sequence preserves model quality while maximizing deployment efficiency."
    }
];

// Export for use in quiz application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = quizData;
} else if (typeof window !== 'undefined') {
    window.quizData = quizData;
}