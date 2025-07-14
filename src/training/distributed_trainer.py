"""
Distributed training system for large-scale models with Azure ML integration.
Supports DeepSpeed, FSDP, and custom parallelism strategies.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_scheduler,
)
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import wandb
from azureml.core import Run
from loguru import logger

# Handle imports gracefully
try:
    from ..data.data_pipeline import DataPipeline
    from ..monitoring.metrics_tracker import MetricsTracker
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from data.data_pipeline import DataPipeline
        from monitoring.metrics_tracker import MetricsTracker
    except ImportError:
        # Mock classes if imports fail
        class DataPipeline:
            def __init__(self, tokenizer, max_length, batch_size):
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.batch_size = batch_size
            
            def create_dataloader(self, dataset, shuffle=True, drop_last=True):
                # Return a simple mock dataloader for demonstration
                return dataset
        
        class MetricsTracker:
            def log_metrics(self, metrics): pass


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""
    
    model_name_or_path: str
    output_dir: str
    dataset_name: str
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Distributed training
    distributed_backend: str = "deepspeed"  # deepspeed, fsdp, ddp
    deepspeed_config: Optional[str] = None
    fsdp_config: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # Azure ML
    use_azure_ml: bool = True
    azure_ml_workspace: Optional[str] = None
    azure_ml_experiment: Optional[str] = None
    
    # Monitoring
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    report_to: List[str] = field(default_factory=lambda: ["wandb", "azureml"])


class DistributedTrainer:
    """Handles distributed training with multiple backend support."""
    
    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        self._setup_logging()
        self._setup_azure_ml()
        self._setup_distributed()
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.metrics_tracker = MetricsTracker()
        
    def _setup_logging(self):
        """Configure logging for distributed training."""
        log_level = logging.INFO if self.rank == 0 else logging.WARNING
        logger.add(
            f"logs/training_rank_{self.rank}.log",
            level=log_level,
            rotation="100 MB"
        )
        
    def _setup_azure_ml(self):
        """Initialize Azure ML run tracking."""
        if self.config.use_azure_ml and self.rank == 0:
            try:
                self.azure_run = Run.get_context()
                logger.info("Azure ML run context initialized")
            except:
                self.azure_run = None
                logger.warning("Azure ML run context not available")
        else:
            self.azure_run = None
            
    def _setup_distributed(self):
        """Initialize distributed training backend."""
        if self.world_size > 1:
            if self.config.distributed_backend == "deepspeed":
                deepspeed.init_distributed()
            else:
                dist.init_process_group(backend="nccl")
                torch.cuda.set_device(self.local_rank)
                
    def _create_deepspeed_config(self) -> Dict[str, Any]:
        """Generate DeepSpeed configuration."""
        if self.config.deepspeed_config:
            with open(self.config.deepspeed_config, 'r') as f:
                return json.load(f)
                
        # Default ZeRO-3 configuration for large models
        return {
            "bf16": {"enabled": self.config.bf16},
            "fp16": {"enabled": self.config.fp16},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 1e9,
                "stage3_prefetch_bucket_size": 1e9,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": 1.0,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": self.config.per_device_train_batch_size,
            "wall_clock_breakdown": False,
            "eigenvalue": {"enabled": True},
        }
        
    def _setup_fsdp(self, model: torch.nn.Module) -> FSDP:
        """Wrap model with FSDP."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        
        # Configure FSDP
        fsdp_config = self.config.fsdp_config
        
        # Auto wrap policy
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={LlamaDecoderLayer},
            recurse=True,
        )
        
        # Mixed precision
        mixed_precision = None
        if self.config.fp16:
            mixed_precision = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif self.config.bf16:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            
        # Wrap model
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=fsdp_config.get("sharding_strategy", "FULL_SHARD"),
            cpu_offload=CPUOffload(offload_params=fsdp_config.get("cpu_offload", False)),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
        )
        
        return model
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configuration."""
        logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
            "use_cache": not self.config.gradient_checkpointing,
            "trust_remote_code": True,
        }
        
        # Load model based on backend
        if self.config.distributed_backend == "deepspeed":
            # DeepSpeed will handle model initialization
            with deepspeed.zero.Init(
                remote_device="cpu",
                config_dict_or_path=self._create_deepspeed_config(),
                enabled=self.world_size > 1,
            ):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name_or_path,
                    **model_kwargs
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs
            )
            
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B parameters")
        
    def setup_training(self, train_dataset, eval_dataset=None):
        """Setup model for distributed training."""
        if self.config.distributed_backend == "deepspeed":
            # DeepSpeed setup
            self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=self.model,
                config=self._create_deepspeed_config(),
                model_parameters=self.model.parameters(),
            )
        elif self.config.distributed_backend == "fsdp":
            # FSDP setup
            self.model = self._setup_fsdp(self.model)
            self._setup_optimizer_and_scheduler()
        else:
            # Standard DDP
            self.model = self.model.to(self.local_rank)
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            self._setup_optimizer_and_scheduler()
            
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Learning rate scheduler
        num_training_steps = (
            self.config.max_steps if self.config.max_steps > 0
            else len(self.train_dataloader) * self.config.num_train_epochs
        )
        
        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop with monitoring and checkpointing."""
        logger.info("Starting distributed training")
        
        # Setup data pipeline
        data_pipeline = DataPipeline(
            tokenizer=self.tokenizer,
            max_length=2048,
            batch_size=self.config.per_device_train_batch_size,
        )
        
        self.train_dataloader = data_pipeline.create_dataloader(
            train_dataset,
            shuffle=True,
            drop_last=True,
        )
        
        if eval_dataset:
            self.eval_dataloader = data_pipeline.create_dataloader(
                eval_dataset,
                shuffle=False,
                drop_last=False,
            )
            
        # Setup training
        self.setup_training(train_dataset, eval_dataset)
        
        # Training loop
        global_step = 0
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            epoch_loss = 0
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                # Forward pass
                if self.config.distributed_backend == "deepspeed":
                    loss = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    ).loss
                    
                    # Backward pass handled by DeepSpeed
                    self.model.backward(loss)
                    self.model.step()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        if self.config.distributed_backend != "fsdp":
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    metrics = {
                        "train/loss": loss.item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                    
                    self._log_metrics(metrics)
                    
                # Evaluation
                if eval_dataset and global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(self.eval_dataloader)
                    self._log_metrics(eval_metrics)
                    
                # Checkpointing
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)
                    
                # Early stopping
                if self.config.max_steps > 0 and global_step >= self.config.max_steps:
                    break
                    
            logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss / len(self.train_dataloader):.4f}")
            
    def evaluate(self, eval_dataloader):
        """Run evaluation on validation set."""
        logger.info("Running evaluation")
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.local_rank) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_tokens += batch["attention_mask"].sum().item()
                
        # Aggregate across all processes
        if self.world_size > 1:
            total_loss = torch.tensor(total_loss).to(self.local_rank)
            total_tokens = torch.tensor(total_tokens).to(self.local_rank)
            
            dist.all_reduce(total_loss)
            dist.all_reduce(total_tokens)
            
            total_loss = total_loss.item()
            total_tokens = total_tokens.item()
            
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }
        
        self.model.train()
        return metrics
        
    def save_checkpoint(self, global_step: int):
        """Save model checkpoint."""
        if self.rank == 0:
            checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving checkpoint to {checkpoint_dir}")
            
            if self.config.distributed_backend == "deepspeed":
                # DeepSpeed checkpoint
                self.model.save_checkpoint(str(checkpoint_dir))
            else:
                # Standard checkpoint
                state_dict = self.model.state_dict()
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
                
                # Save tokenizer
                self.tokenizer.save_pretrained(checkpoint_dir)
                
                # Save training state
                training_state = {
                    "global_step": global_step,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "config": self.config.__dict__,
                }
                torch.save(training_state, checkpoint_dir / "training_state.pt")
                
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to various tracking services."""
        if self.rank == 0:
            # Console logging
            logger.info(f"Metrics: {metrics}")
            
            # WandB
            if "wandb" in self.config.report_to:
                wandb.log(metrics)
                
            # Azure ML
            if self.azure_run and "azureml" in self.config.report_to:
                for key, value in metrics.items():
                    self.azure_run.log(key, value)
                    
            # Custom metrics tracker
            self.metrics_tracker.log_metrics(metrics)