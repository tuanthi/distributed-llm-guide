# Distributed LLM Production Deployment Guide

## Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd distributed-llm-guide

# Install dependencies
make install-dev

# Setup development environment
make setup
```

### 2. Train a Code Model
```bash
# Quick training with LoRA
python examples/train_code_model.py \
    --model_name microsoft/CodeGPT-small-py \
    --use_lora \
    --num_epochs 1 \
    --batch_size 4

# Full training with monitoring
python examples/train_code_model.py \
    --model_name microsoft/phi-2 \
    --use_qlora \
    --num_epochs 3 \
    --enable_monitoring \
    --wandb_project distributed-llm-showcase
```

### 3. Serve the Model
```bash
# Start model server
python src/serving/model_server.py \
    --model-path ./code_model_output \
    --device cuda \
    --max-batch-size 32

# Test inference
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"text": "def fibonacci(n):", "max_new_tokens": 100}'
```

### 4. Run Benchmarks
```bash
# Performance benchmarking
python benchmarks/performance_benchmark.py \
    --model-path ./code_model_output \
    --batch-sizes 1,4,8,16 \
    --sequence-lengths 128,256,512
```

## Production Deployment

### Azure ML Deployment

#### 1. Setup Azure Resources
```bash
# Configure Azure ML workspace
make azure-setup

# Or manually
python scripts/setup.py \
    --azure \
    --subscription-id YOUR_SUBSCRIPTION_ID \
    --resource-group distributed-llm-rg \
    --workspace-name distributed-llm-workspace
```

#### 2. Submit Training Job
```python
from azure.azure_ml_config import AzureMLManager, create_default_config

# Configure Azure ML
config = create_default_config()
azure_manager = AzureMLManager(config)

# Submit distributed training
job = azure_manager.submit_training_job(
    script_path="training/distributed_trainer.py",
    experiment_name="distributed-llm-code-model",
    parameters={
        "model_name": "microsoft/phi-2",
        "dataset_name": "codeparrot/github-code",
        "learning_rate": 5e-5,
        "num_epochs": 3,
    },
    node_count=4  # Multi-node training
)
```

#### 3. Deploy to Endpoint
```python
# Register trained model
model = azure_manager.register_model(
    model_path="./outputs/model",
    model_name="distributed-llm-code-model",
    model_version="1.0"
)

# Deploy to managed endpoint
deployment = azure_manager.deploy_model(
    model_name="distributed-llm-code-model",
    model_version="1.0",
    scoring_script="score.py",
    instance_type="Standard_NC6s_v3"
)
```

### Kubernetes Deployment

#### 1. Build Docker Image
```bash
# Build optimized image
docker build -t distributed-llm-model:latest .

# Push to registry
docker tag distributed-llm-model:latest your-registry/distributed-llm-model:latest
docker push your-registry/distributed-llm-model:latest
```

#### 2. Deploy to Kubernetes
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-llm-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: distributed-llm-model
  template:
    metadata:
      labels:
        app: distributed-llm-model
    spec:
      containers:
      - name: model-server
        image: your-registry/distributed-llm-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
        env:
        - name: MODEL_PATH
          value: "/models/distributed-llm-model"
        - name: DEVICE
          value: "cuda"
```

```bash
# Deploy
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-ingress.yaml
```

### Monitoring Setup

#### 1. Prometheus + Grafana
```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'distributed-llm-model'
    static_configs:
      - targets: ['distributed-llm-model:8000']
    metrics_path: '/metrics'
```

#### 2. Custom Monitoring
```python
from monitoring.model_monitor import ModelMonitor, MonitoringConfig

# Configure monitoring
config = MonitoringConfig(
    model_name="distributed-llm-code-model",
    model_version="1.0",
    environment="production",
    enable_azure_monitor=True,
    enable_prometheus=True,
    latency_threshold_p95=100.0,  # ms
    error_rate_threshold=0.01,    # 1%
)

# Start monitoring
monitor = ModelMonitor(config)
monitor.start()
```

## Performance Optimization

### 1. Model Optimization
```python
from optimization.model_optimizer import ModelOptimizer

optimizer = ModelOptimizer()

# TensorRT optimization
optimized_model = optimizer.optimize_with_tensorrt(
    model_path="./model",
    precision="fp16",
    max_batch_size=32
)

# ONNX optimization
optimizer.convert_to_onnx(
    model_path="./model",
    output_path="./model.onnx",
    opset_version=14
)
```

### 2. Serving Optimization
```python
from serving.model_server import ServingConfig, ModelServer

config = ServingConfig(
    model_path="./optimized_model",
    device="cuda",
    max_batch_size=32,
    optimization_level="O3",
    use_tensorrt=True,
    enable_caching=True,
    cache_size=1000
)

server = ModelServer(config)
```

### 3. Hardware Optimization
```bash
# GPU optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Memory optimization
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
```

## Scaling Strategies

### Horizontal Scaling
```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: distributed-llm-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: distributed-llm-model
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

### Vertical Scaling
```yaml
# VPA configuration
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: distributed-llm-model-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: distributed-llm-model
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: model-server
      minAllowed:
        memory: 4Gi
        nvidia.com/gpu: 1
      maxAllowed:
        memory: 32Gi
        nvidia.com/gpu: 1
```

## A/B Testing & Canary Deployment

### 1. Traffic Splitting
```python
from deployment.ab_testing import ABTestManager

ab_manager = ABTestManager()

# Configure A/B test
test_config = {
    "name": "model-v2-test",
    "control": {
        "model": "distributed-llm-model:v1",
        "traffic": 90
    },
    "treatment": {
        "model": "distributed-llm-model:v2", 
        "traffic": 10
    },
    "metrics": ["accuracy", "latency", "user_satisfaction"],
    "duration": "7d"
}

ab_manager.start_test(test_config)
```

### 2. Canary Deployment
```yaml
# Argo Rollouts canary
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: distributed-llm-model
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 1h}
      - setWeight: 50
      - pause: {duration: 4h}
      - setWeight: 100
      canaryService: distributed-llm-model-canary
      stableService: distributed-llm-model-stable
      trafficRouting:
        nginx:
          stableIngress: distributed-llm-model-ingress
```

## Security & Compliance

### 1. Model Security
```python
# Input validation
def validate_input(text: str) -> bool:
    if len(text) > MAX_INPUT_LENGTH:
        return False
    if contains_malicious_patterns(text):
        return False
    return True

# Output filtering
def filter_output(text: str) -> str:
    # Remove PII
    text = remove_pii(text)
    # Content safety
    text = apply_safety_filters(text)
    return text
```

### 2. Access Control
```yaml
# RBAC configuration
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: distributed-llm-model-operator
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "create", "update", "delete"]
- apiGroups: [""]
  resources: ["services", "configmaps"]
  verbs: ["get", "list", "create", "update"]
```

### 3. Data Privacy
```python
# GDPR compliance
from privacy.data_handler import DataHandler

handler = DataHandler()

# Anonymize training data
anonymized_data = handler.anonymize_dataset(
    dataset,
    fields=["email", "phone", "address"]
)

# Right to be forgotten
handler.delete_user_data(user_id="user123")
```

## Cost Optimization

### 1. Resource Optimization
```python
# Auto-scaling configuration
scaling_config = {
    "min_instances": 1,
    "max_instances": 10,
    "target_cpu_utilization": 70,
    "scale_down_delay": "5m",
    "scale_up_delay": "30s"
}

# Spot instances for training
training_config = {
    "compute_type": "spot",
    "preemption_handler": "graceful_shutdown",
    "checkpoint_interval": "10m"
}
```

### 2. Model Efficiency
```python
# Model compression
from optimization.compression import ModelCompressor

compressor = ModelCompressor()

# Quantization
quantized_model = compressor.quantize(
    model,
    method="int8",
    calibration_dataset=cal_dataset
)

# Pruning
pruned_model = compressor.prune(
    model,
    sparsity=0.5,
    structured=True
)

# Knowledge distillation
student_model = compressor.distill(
    teacher_model=large_model,
    student_architecture="small",
    temperature=3.0
)
```

## Troubleshooting

### Common Issues

#### 1. OOM Errors
```bash
# Reduce batch size
--per_device_train_batch_size 2
--gradient_accumulation_steps 16

# Enable gradient checkpointing
--gradient_checkpointing true

# Use memory-efficient optimizers
--optim adafactor
```

#### 2. Slow Training
```bash
# Enable mixed precision
--fp16 true --bf16 false

# Use faster data loading
--dataloader_num_workers 8
--dataloader_pin_memory true

# Optimize data preprocessing
--preprocessing_num_workers 16
```

#### 3. Model Quality Issues
```python
# Learning rate scheduling
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)

# Regularization
config.weight_decay = 0.01
config.dropout = 0.1
config.attention_dropout = 0.1
```

### Debugging Tools
```bash
# Profile training
python -m torch.profiler examples/train_code_model.py

# Memory profiling
python -m memory_profiler examples/train_code_model.py

# Distributed debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

## Production Checklist

### Pre-deployment
- [ ] Model validation on test set
- [ ] Performance benchmarking
- [ ] Security review
- [ ] Load testing
- [ ] Monitoring setup

### Deployment
- [ ] Blue-green deployment ready
- [ ] Rollback plan defined
- [ ] Health checks configured
- [ ] Alerting rules active
- [ ] Documentation updated

### Post-deployment
- [ ] Monitor key metrics
- [ ] Performance regression check
- [ ] User feedback collection
- [ ] Cost impact analysis
- [ ] Incident response plan

This guide provides comprehensive instructions for deploying distributed LLM models in production environments with proper monitoring, scaling, and optimization strategies.