"""
Parameter-Efficient Fine-Tuning (PEFT) implementation with LoRA, QLoRA, and other techniques.
Supports various PEFT methods for efficient large model fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import math
from pathlib import Path
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig,
    PromptTuningConfig,
    PromptTuningInit,
    PrefixTuningConfig,
    IA3Config,
    AdaLoraConfig,
)
import bitsandbytes as bnb
from loguru import logger
import wandb

# Handle imports gracefully
try:
    from ..monitoring.metrics_tracker import MetricsTracker
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from monitoring.metrics_tracker import MetricsTracker
    except ImportError:
        # Mock class if import fails
        class MetricsTracker:
            def log_metrics(self, metrics): pass


@dataclass
class PEFTTrainingConfig:
    """Configuration for PEFT training."""
    
    # Model configuration
    model_name_or_path: str
    output_dir: str
    
    # PEFT method
    peft_method: str = "lora"  # lora, qlora, prompt_tuning, prefix_tuning, ia3, adalora
    
    # LoRA specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_bias: str = "none"
    
    # QLoRA specific
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # AdaLoRA specific
    adalora_init_r: int = 128
    adalora_target_r: int = 8
    adalora_tinit: int = 0
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10
    
    # Prompt tuning specific
    num_virtual_tokens: int = 20
    prompt_tuning_init: str = "random"  # random, text
    prompt_tuning_init_text: Optional[str] = None
    
    # Training configuration
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    
    # Additional features
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    merge_weights: bool = True


class LoRALayer(nn.Module):
    """Custom LoRA layer implementation for detailed control."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return result * self.scaling


class CustomLoRAModel(nn.Module):
    """Custom model with LoRA layers for fine-grained control."""
    
    def __init__(self, base_model: PreTrainedModel, lora_config: Dict[str, Any]):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = nn.ModuleDict()
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Add LoRA layers to specified modules
        self._add_lora_layers(lora_config)
        
    def _add_lora_layers(self, config: Dict[str, Any]):
        """Add LoRA layers to target modules."""
        target_modules = config.get("target_modules", ["q_proj", "v_proj"])
        r = config.get("r", 16)
        alpha = config.get("alpha", 32)
        dropout = config.get("dropout", 0.1)
        
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Add LoRA layer
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        r=r,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    self.lora_layers[name] = lora_layer
                    
                    # Hook to add LoRA output to original output
                    self._register_lora_hook(module, lora_layer)
                    
    def _register_lora_hook(self, module: nn.Module, lora_layer: LoRALayer):
        """Register forward hook to add LoRA output."""
        def lora_forward_hook(module, input, output):
            lora_output = lora_layer(input[0])
            return output + lora_output
            
        module.register_forward_hook(lora_forward_hook)
        
    def merge_lora_weights(self):
        """Merge LoRA weights into base model."""
        for name, module in self.base_model.named_modules():
            if name in self.lora_layers:
                if isinstance(module, nn.Linear):
                    # Get LoRA weights
                    lora_layer = self.lora_layers[name]
                    delta_weight = lora_layer.lora_B.weight @ lora_layer.lora_A.weight
                    delta_weight *= lora_layer.scaling
                    
                    # Merge weights
                    module.weight.data += delta_weight
                    
        # Remove LoRA layers after merging
        self.lora_layers.clear()


class PEFTTrainer:
    """Trainer for various PEFT methods."""
    
    def __init__(self, config: PEFTTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.metrics_tracker = MetricsTracker()
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with appropriate configuration."""
        logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "use_flash_attention_2": self.config.use_flash_attention,
        }
        
        # Load with quantization for QLoRA
        if self.config.use_4bit:
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": getattr(torch, self.config.bnb_4bit_compute_dtype),
                "bnb_4bit_quant_type": self.config.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.config.bnb_4bit_use_double_quant,
            })
            
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs
        )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # Prepare for k-bit training if using quantization
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
    def setup_peft(self):
        """Setup PEFT based on configuration."""
        logger.info(f"Setting up PEFT method: {self.config.peft_method}")
        
        if self.config.peft_method == "lora" or self.config.peft_method == "qlora":
            self._setup_lora()
        elif self.config.peft_method == "adalora":
            self._setup_adalora()
        elif self.config.peft_method == "prompt_tuning":
            self._setup_prompt_tuning()
        elif self.config.peft_method == "prefix_tuning":
            self._setup_prefix_tuning()
        elif self.config.peft_method == "ia3":
            self._setup_ia3()
        else:
            raise ValueError(f"Unknown PEFT method: {self.config.peft_method}")
            
    def _setup_lora(self):
        """Setup LoRA configuration."""
        self.peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
        
    def _setup_adalora(self):
        """Setup AdaLoRA configuration."""
        self.peft_config = AdaLoraConfig(
            init_r=self.config.adalora_init_r,
            target_r=self.config.adalora_target_r,
            tinit=self.config.adalora_tinit,
            tfinal=self.config.adalora_tfinal,
            deltaT=self.config.adalora_delta_t,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        
    def _setup_prompt_tuning(self):
        """Setup prompt tuning configuration."""
        prompt_tuning_init_params = None
        if self.config.prompt_tuning_init == "text" and self.config.prompt_tuning_init_text:
            prompt_tuning_init_params = self.tokenizer(
                self.config.prompt_tuning_init_text,
                return_tensors="pt"
            ).input_ids
            
        self.peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT if self.config.prompt_tuning_init == "text" else PromptTuningInit.RANDOM,
            num_virtual_tokens=self.config.num_virtual_tokens,
            prompt_tuning_init_text=self.config.prompt_tuning_init_text,
            tokenizer_name_or_path=self.config.model_name_or_path,
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        
    def _setup_prefix_tuning(self):
        """Setup prefix tuning configuration."""
        self.peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.config.num_virtual_tokens,
            encoder_hidden_size=self.model.config.hidden_size,
            prefix_projection=True,
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        
    def _setup_ia3(self):
        """Setup IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)."""
        self.peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.lora_target_modules,
            feedforward_modules=["mlp.up_proj", "mlp.down_proj"],
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        
    def train(self, train_dataset, eval_dataset=None):
        """Train model with PEFT method."""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_strategy="steps",
            save_steps=1000,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            fp16=True if not self.config.use_4bit else False,
            bf16=self.config.use_4bit,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to=["wandb"] if wandb.run else [],
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            push_to_hub=False,
        )
        
        # Custom trainer with PEFT-specific features
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )
        
        # Add custom callbacks for AdaLoRA
        if self.config.peft_method == "adalora":
            from transformers.trainer_callback import TrainerCallback
            
            class AdaLoRACallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    # Update AdaLoRA budget allocation
                    if hasattr(kwargs["model"], "update_and_prune"):
                        kwargs["model"].update_and_prune(state.global_step)
                        
            trainer.add_callback(AdaLoRACallback())
            
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        
        # Merge weights if specified
        if self.config.merge_weights and self.config.peft_method in ["lora", "qlora"]:
            self._merge_and_save()
            
    def _merge_and_save(self):
        """Merge PEFT weights with base model and save."""
        logger.info("Merging PEFT weights with base model")
        
        # Merge weights
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_output_dir = Path(self.config.output_dir) / "merged"
        merged_output_dir.mkdir(exist_ok=True)
        
        merged_model.save_pretrained(merged_output_dir)
        self.tokenizer.save_pretrained(merged_output_dir)
        
        logger.info(f"Merged model saved to {merged_output_dir}")
        
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        loss = torch.nn.functional.cross_entropy(
            torch.from_numpy(predictions.reshape(-1, predictions.shape[-1])),
            torch.from_numpy(labels.reshape(-1)),
            ignore_index=-100,
        )
        
        perplexity = torch.exp(loss).item()
        
        return {
            "perplexity": perplexity,
            "eval_loss": loss.item(),
        }
        
    def load_trained_model(self, checkpoint_path: str):
        """Load a trained PEFT model."""
        logger.info(f"Loading trained model from {checkpoint_path}")
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(checkpoint_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
        )
        
        # Load PEFT model
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        return self.model, self.tokenizer


def compare_peft_methods(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Compare different PEFT methods on the same dataset."""
    results = {}
    
    peft_methods = ["lora", "adalora", "prompt_tuning", "prefix_tuning", "ia3"]
    
    for method in peft_methods:
        logger.info(f"Training with {method}")
        
        config = PEFTTrainingConfig(
            model_name_or_path=model_name,
            output_dir=f"{output_dir}/{method}",
            peft_method=method,
            num_train_epochs=1,  # Quick comparison
        )
        
        trainer = PEFTTrainer(config)
        trainer.load_model_and_tokenizer()
        trainer.setup_peft()
        
        # Train and evaluate
        trainer.train(train_dataset, eval_dataset)
        
        # Get final metrics
        eval_results = trainer.model.evaluate()
        
        results[method] = {
            "eval_loss": eval_results["eval_loss"],
            "perplexity": eval_results["eval_perplexity"],
            "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in trainer.model.parameters()),
        }
        
    return results