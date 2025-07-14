#!/usr/bin/env python3
"""
End-to-end example: Training a code-specific model with LoRA fine-tuning.
Demonstrates distributed training, monitoring, and Azure ML integration.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import wandb
from datasets import load_dataset
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Handle imports gracefully for different execution contexts
try:
    from models.code_model_trainer import CodeModelTrainer, CodeTrainingConfig
    from training.peft_trainer import PEFTTrainer, PEFTTrainingConfig
    from monitoring.model_monitor import ModelMonitor, MonitoringConfig
    from azure.azure_ml_config import AzureMLManager, create_default_config
except ImportError:
    # Try relative imports
    try:
        from ..models.code_model_trainer import CodeModelTrainer, CodeTrainingConfig
        from ..training.peft_trainer import PEFTTrainer, PEFTTrainingConfig
        from ..monitoring.model_monitor import ModelMonitor, MonitoringConfig
        from ..azure.azure_ml_config import AzureMLManager, create_default_config
    except ImportError:
        # For demonstration purposes, we'll provide mock implementations
        logger.warning("Could not import all modules. Some features may be limited.")
        
        class CodeModelTrainer:
            def __init__(self, config): pass
            def load_model_and_tokenizer(self): pass
            def prepare_dataset(self): return [], []
            def train(self, train_dataset, eval_dataset): pass
        
        class CodeTrainingConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class PEFTTrainer:
            def __init__(self, config): pass
            def load_model_and_tokenizer(self): pass
            def setup_peft(self): pass
            def train(self, train_dataset, eval_dataset): pass
        
        class PEFTTrainingConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class ModelMonitor:
            def start(self): pass
            def stop(self): pass
        
        class MonitoringConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class AzureMLManager:
            def __init__(self, config): pass
            def register_model(self, **kwargs): pass
            def deploy_model(self, **kwargs): pass
        
        def create_default_config():
            return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train code-specific model")
    
    # Model configuration
    parser.add_argument("--model_name", default="microsoft/CodeGPT-small-py", 
                       help="Base model to fine-tune")
    parser.add_argument("--output_dir", default="./code_model_output", 
                       help="Output directory for trained model")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    
    # PEFT configuration
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                       help="LoRA alpha")
    parser.add_argument("--use_qlora", action="store_true", 
                       help="Use QLoRA (4-bit quantization)")
    
    # Data configuration
    parser.add_argument("--dataset_name", default="codeparrot/github-code-clean",
                       help="Dataset name or path")
    parser.add_argument("--languages", nargs="+", 
                       default=["Python", "JavaScript", "Java", "C++"],
                       help="Programming languages to include")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Azure ML
    parser.add_argument("--use_azure_ml", action="store_true",
                       help="Use Azure ML for training")
    parser.add_argument("--experiment_name", default="code-model-training",
                       help="Azure ML experiment name")
    
    # Monitoring
    parser.add_argument("--enable_monitoring", action="store_true", default=True,
                       help="Enable model monitoring")
    parser.add_argument("--wandb_project", default="distributed-llm-code-models",
                       help="Weights & Biases project name")
    
    return parser.parse_args()


def setup_monitoring(args) -> ModelMonitor:
    """Setup model monitoring."""
    if not args.enable_monitoring:
        return None
        
    config = MonitoringConfig(
        model_name=args.model_name.split("/")[-1],
        model_version="1.0",
        environment="training",
        enable_azure_monitor=args.use_azure_ml,
        enable_wandb=bool(args.wandb_project),
    )
    
    monitor = ModelMonitor(config)
    monitor.start()
    
    return monitor


def prepare_dataset(args):
    """Prepare training dataset."""
    logger.info(f"Loading dataset: {args.dataset_name}")
    
    # Load dataset
    if args.dataset_name.startswith("codeparrot"):
        dataset = load_dataset(
            args.dataset_name,
            split="train",
            streaming=False,
        )
        
        # Filter by languages
        if args.languages:
            language_map = {
                "Python": "python",
                "JavaScript": "javascript", 
                "Java": "java",
                "C++": "cpp",
            }
            
            target_languages = [language_map.get(lang, lang.lower()) 
                              for lang in args.languages]
            
            dataset = dataset.filter(
                lambda x: x.get("language", "").lower() in target_languages
            )
            
        # Take subset for demo
        dataset = dataset.select(range(min(10000, len(dataset))))
        
    else:
        # Load from local path
        dataset = load_dataset("json", data_files=args.dataset_name)["train"]
        
    # Split dataset
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def train_with_peft(args, train_dataset, eval_dataset):
    """Train model using PEFT techniques."""
    logger.info("Training with PEFT (LoRA/QLoRA)")
    
    # PEFT configuration
    peft_config = PEFTTrainingConfig(
        model_name_or_path=args.model_name,
        output_dir=args.output_dir,
        peft_method="qlora" if args.use_qlora else "lora",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_4bit=args.use_qlora,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        merge_weights=True,
    )
    
    # Initialize trainer
    trainer = PEFTTrainer(peft_config)
    trainer.load_model_and_tokenizer()
    trainer.setup_peft()
    
    # Process dataset for training
    def tokenize_function(examples):
        # Combine code with special tokens
        texts = []
        for code in examples["code"]:
            text = f"<code>{code}</code>"
            texts.append(text)
            
        return trainer.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=args.max_length,
            return_overflowing_tokens=False,
        )
        
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    
    # Add labels for language modeling
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
        
    train_dataset = train_dataset.map(add_labels, batched=True)
    eval_dataset = eval_dataset.map(add_labels, batched=True)
    
    # Train model
    trainer.train(train_dataset, eval_dataset)
    
    return trainer


def train_with_code_trainer(args, train_dataset, eval_dataset):
    """Train model using specialized code trainer."""
    logger.info("Training with specialized code trainer")
    
    # Code training configuration
    code_config = CodeTrainingConfig(
        base_model=args.model_name,
        output_dir=args.output_dir,
        languages=[lang.lower() for lang in args.languages],
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        enable_fill_in_middle=True,
        enable_docstring_generation=True,
        enable_code_completion=True,
    )
    
    # Initialize trainer
    trainer = CodeModelTrainer(code_config)
    trainer.load_model_and_tokenizer()
    
    # Prepare dataset with code-specific processing
    processed_train, processed_eval = trainer.prepare_dataset()
    
    # Train model
    trainer.train(processed_train, processed_eval)
    
    return trainer


def deploy_to_azure(args, trainer):
    """Deploy trained model to Azure ML."""
    if not args.use_azure_ml:
        return
        
    logger.info("Deploying model to Azure ML")
    
    # Azure ML configuration
    azure_config = create_default_config()
    azure_manager = AzureMLManager(azure_config)
    
    # Register model
    model = azure_manager.register_model(
        model_path=args.output_dir,
        model_name=f"distributed-llm-code-model",
        model_version="1.0",
        description="Fine-tuned code generation model",
        tags={
            "framework": "pytorch",
            "task": "code_generation",
            "base_model": args.model_name,
            "peft_method": "lora" if args.use_lora else "full",
        }
    )
    
    # Create scoring script
    scoring_script_content = '''
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def init():
    global model, tokenizer
    
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Load PEFT adapter if exists
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
    except:
        model = base_model
        
    model.eval()

def run(raw_data):
    try:
        data = json.loads(raw_data)
        prompt = data["prompt"]
        max_length = data.get("max_length", 128)
        
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": generated_text}
        
    except Exception as e:
        return {"error": str(e)}
'''
    
    # Save scoring script
    scoring_script_path = Path(args.output_dir) / "score.py"
    scoring_script_path.write_text(scoring_script_content)
    
    # Deploy model
    deployment = azure_manager.deploy_model(
        model_name=model.name,
        model_version=model.version,
        scoring_script="score.py",
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    
    logger.info(f"Model deployed to endpoint: {deployment.endpoint_name}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup logging
    logger.add("training.log", rotation="10 MB")
    logger.info("Starting code model training")
    
    # Initialize Weights & Biases
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"code-training-{args.model_name.split('/')[-1]}",
            config=vars(args),
        )
    
    # Setup monitoring
    monitor = setup_monitoring(args)
    
    try:
        # Prepare dataset
        train_dataset, eval_dataset = prepare_dataset(args)
        
        # Train model
        if args.use_lora or args.use_qlora:
            trainer = train_with_peft(args, train_dataset, eval_dataset)
        else:
            trainer = train_with_code_trainer(args, train_dataset, eval_dataset)
            
        # Deploy to Azure ML if enabled
        deploy_to_azure(args, trainer)
        
        # Test the trained model
        logger.info("Testing trained model")
        test_prompts = [
            "def fibonacci(n):",
            "class Calculator:",
            "import numpy as np\n\ndef matrix_multiply(",
        ]
        
        for prompt in test_prompts:
            inputs = trainer.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = trainer.model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=trainer.tokenizer.eos_token_id,
                )
                
            generated = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated}")
            logger.info("-" * 50)
            
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Stop monitoring
        if monitor:
            monitor.stop()
            
        # Finish wandb run
        if args.wandb_project:
            wandb.finish()


if __name__ == "__main__":
    main()