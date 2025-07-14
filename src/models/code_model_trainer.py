"""
Code-specific model fine-tuning system with support for multiple programming languages.
Designed for code completion, generation, and understanding tasks.
"""

import os
import json
import ast
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CodeGenTokenizer,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import tree_sitter
from tree_sitter import Language, Parser
from loguru import logger
import wandb
from tqdm import tqdm

# Handle imports gracefully
try:
    from ..data.code_data_processor import CodeDataProcessor
    from ..training.distributed_trainer import DistributedTrainer, DistributedTrainingConfig
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from data.code_data_processor import CodeDataProcessor
        from training.distributed_trainer import DistributedTrainer, DistributedTrainingConfig
    except ImportError:
        # Mock classes if imports fail
        class CodeDataProcessor:
            def __init__(self, languages, max_length): pass
            def process_dataset(self, dataset): return dataset
            
        class DistributedTrainer:
            def __init__(self, config): pass
            def train(self, train_dataset, eval_dataset): pass
            
        class DistributedTrainingConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)


@dataclass
class CodeTrainingConfig:
    """Configuration for code model training."""
    
    # Model configuration
    base_model: str = "microsoft/CodeGPT-small-py"
    model_type: str = "causal_lm"  # causal_lm, masked_lm, seq2seq
    
    # Data configuration
    dataset_name: Optional[str] = "codeparrot/github-code"
    languages: List[str] = field(default_factory=lambda: ["python", "javascript", "java", "cpp"])
    max_length: int = 2048
    include_docstrings: bool = True
    include_comments: bool = True
    
    # Training configuration
    output_dir: str = "./code_model_output"
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 500
    
    # Fine-tuning strategy
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Code-specific features
    use_ast_features: bool = True
    use_control_flow: bool = True
    use_data_flow: bool = False
    syntax_aware_masking: bool = True
    
    # Specialized tasks
    enable_fill_in_middle: bool = True
    enable_docstring_generation: bool = True
    enable_code_completion: bool = True
    enable_bug_detection: bool = False


class CodeFeatureExtractor:
    """Extracts code-specific features using AST and other analysis."""
    
    def __init__(self, languages: List[str]):
        self.languages = languages
        self.parsers = {}
        self._initialize_parsers()
        
    def _initialize_parsers(self):
        """Initialize Tree-sitter parsers for supported languages."""
        # In production, you would load pre-built language libraries
        # For demo, we'll use simple regex-based extraction
        pass
        
    def extract_ast_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract AST-based features from code."""
        features = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "control_flow": [],
        }
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        features["functions"].append({
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "lineno": node.lineno,
                        })
                    elif isinstance(node, ast.ClassDef):
                        features["classes"].append({
                            "name": node.name,
                            "bases": [base.id for base in node.bases if hasattr(base, 'id')],
                            "lineno": node.lineno,
                        })
                    elif isinstance(node, ast.Import):
                        features["imports"].extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        features["imports"].append(node.module)
                    elif isinstance(node, (ast.For, ast.While, ast.If)):
                        features["control_flow"].append(type(node).__name__)
                        
            except SyntaxError:
                logger.warning("Failed to parse Python code")
                
        return features
        
    def extract_docstrings(self, code: str, language: str) -> List[Dict[str, str]]:
        """Extract docstrings from code."""
        docstrings = []
        
        if language == "python":
            # Simple regex for Python docstrings
            pattern = r'(def\s+\w+[^:]*:)\s*"""([^"]+)"""'
            matches = re.findall(pattern, code, re.MULTILINE | re.DOTALL)
            
            for func_def, docstring in matches:
                docstrings.append({
                    "function": func_def.strip(),
                    "docstring": docstring.strip(),
                })
                
        return docstrings
        
    def extract_code_patterns(self, code: str, language: str) -> Dict[str, Any]:
        """Extract common code patterns and idioms."""
        patterns = {
            "list_comprehensions": 0,
            "lambda_functions": 0,
            "decorators": 0,
            "context_managers": 0,
            "type_hints": 0,
        }
        
        if language == "python":
            patterns["list_comprehensions"] = len(re.findall(r'\[[^\]]+for[^\]]+in[^\]]+\]', code))
            patterns["lambda_functions"] = len(re.findall(r'lambda\s+[^:]+:', code))
            patterns["decorators"] = len(re.findall(r'@\w+', code))
            patterns["context_managers"] = len(re.findall(r'with\s+[^:]+:', code))
            patterns["type_hints"] = len(re.findall(r':\s*[A-Z]\w*(?:\[[^\]]+\])?', code))
            
        return patterns


class CodeModelTrainer:
    """Specialized trainer for code models with code-aware features."""
    
    def __init__(self, config: CodeTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.feature_extractor = CodeFeatureExtractor(config.languages)
        self.code_processor = CodeDataProcessor()
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer."""
        logger.info(f"Loading model: {self.config.base_model}")
        
        # Load tokenizer with special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        
        # Add special tokens for code
        special_tokens = {
            "additional_special_tokens": [
                "<fim_prefix>", "<fim_suffix>", "<fim_middle>",  # Fill-in-middle
                "<code>", "</code>",  # Code boundaries
                "<docstring>", "</docstring>",  # Docstring markers
                "<function>", "</function>",  # Function boundaries
                "<class>", "</class>",  # Class boundaries
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            self._apply_lora()
            
    def _apply_lora(self):
        """Apply LoRA for efficient fine-tuning."""
        # Prepare model for k-bit training if using quantization
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, dataset_path: Optional[str] = None) -> Tuple[Dataset, Dataset]:
        """Prepare code dataset with specialized processing."""
        if dataset_path:
            # Load from local path
            raw_dataset = load_dataset("json", data_files=dataset_path)
        else:
            # Load from HuggingFace
            raw_dataset = load_dataset(
                self.config.dataset_name,
                split="train",
                languages=self.config.languages,
            )
            
        # Process dataset
        processed_examples = []
        
        for example in tqdm(raw_dataset, desc="Processing code examples"):
            code = example.get("code", example.get("content", ""))
            language = example.get("language", "python")
            
            if language not in self.config.languages:
                continue
                
            # Extract features
            features = self.feature_extractor.extract_ast_features(code, language)
            docstrings = self.feature_extractor.extract_docstrings(code, language)
            patterns = self.feature_extractor.extract_code_patterns(code, language)
            
            # Create training examples based on enabled tasks
            if self.config.enable_code_completion:
                # Code completion examples
                completion_examples = self._create_completion_examples(code, language)
                processed_examples.extend(completion_examples)
                
            if self.config.enable_fill_in_middle:
                # Fill-in-middle examples
                fim_examples = self._create_fim_examples(code, language)
                processed_examples.extend(fim_examples)
                
            if self.config.enable_docstring_generation and docstrings:
                # Docstring generation examples
                doc_examples = self._create_docstring_examples(code, docstrings, language)
                processed_examples.extend(doc_examples)
                
        # Create dataset
        dataset = Dataset.from_list(processed_examples)
        
        # Split into train/eval
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        return split_dataset["train"], split_dataset["test"]
        
    def _create_completion_examples(self, code: str, language: str) -> List[Dict[str, str]]:
        """Create code completion training examples."""
        examples = []
        lines = code.split("\n")
        
        # Create examples at different completion points
        for i in range(len(lines) // 4, len(lines), max(1, len(lines) // 10)):
            prefix = "\n".join(lines[:i])
            suffix = "\n".join(lines[i:])
            
            if len(self.tokenizer.encode(prefix)) < self.config.max_length - 100:
                examples.append({
                    "text": f"<code>{prefix}",
                    "completion": suffix + "</code>",
                    "task": "completion",
                    "language": language,
                })
                
        return examples
        
    def _create_fim_examples(self, code: str, language: str) -> List[Dict[str, str]]:
        """Create fill-in-middle training examples."""
        examples = []
        lines = code.split("\n")
        
        # Find function definitions
        for i, line in enumerate(lines):
            if re.match(r'^\s*def\s+\w+', line):
                # Find the end of the function
                end_idx = i + 1
                indent_level = len(line) - len(line.lstrip())
                
                while end_idx < len(lines):
                    next_line = lines[end_idx]
                    if next_line.strip() and (len(next_line) - len(next_line.lstrip())) <= indent_level:
                        break
                    end_idx += 1
                    
                if end_idx - i > 3:  # Function has meaningful body
                    # Create FIM example
                    prefix = "\n".join(lines[:i+1])
                    middle_start = i + 1 + (end_idx - i - 1) // 3
                    middle_end = i + 1 + 2 * (end_idx - i - 1) // 3
                    
                    suffix = "\n".join(lines[middle_end:])
                    middle = "\n".join(lines[middle_start:middle_end])
                    prefix_with_hole = prefix + "\n" + "\n".join(lines[i+1:middle_start])
                    
                    example = {
                        "text": f"<fim_prefix>{prefix_with_hole}<fim_suffix>{suffix}<fim_middle>",
                        "completion": middle,
                        "task": "fill_in_middle",
                        "language": language,
                    }
                    examples.append(example)
                    
        return examples
        
    def _create_docstring_examples(
        self, 
        code: str, 
        docstrings: List[Dict[str, str]], 
        language: str
    ) -> List[Dict[str, str]]:
        """Create docstring generation examples."""
        examples = []
        
        for doc_info in docstrings:
            function_def = doc_info["function"]
            docstring = doc_info["docstring"]
            
            # Find the function in code and extract body
            func_pattern = re.escape(function_def) + r'[^:]*:\s*"""[^"]*"""(.*?)(?=\ndef|\nclass|\Z)'
            match = re.search(func_pattern, code, re.DOTALL)
            
            if match:
                function_body = match.group(1).strip()
                
                # Create example for docstring generation
                example = {
                    "text": f"<function>{function_def}\n{function_body}</function>\nGenerate docstring:",
                    "completion": f'"""{docstring}"""',
                    "task": "docstring_generation",
                    "language": language,
                }
                examples.append(example)
                
        return examples
        
    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Train the code model."""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=["wandb"] if wandb.run else [],
            push_to_hub=False,
        )
        
        # Custom trainer with code-specific metrics
        trainer = CodeTrainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
    def _compute_metrics(self, eval_pred):
        """Compute code-specific metrics."""
        predictions, labels = eval_pred
        
        # Basic metrics
        loss = torch.nn.functional.cross_entropy(
            torch.from_numpy(predictions.reshape(-1, predictions.shape[-1])),
            torch.from_numpy(labels.reshape(-1)),
            ignore_index=-100,
        ).item()
        
        # Code-specific metrics would include:
        # - Syntax validity rate
        # - Exact match accuracy for specific code patterns
        # - BLEU score for generated code
        # - Edit distance for code completion
        
        return {"eval_loss": loss}


class CodeTrainer(Trainer):
    """Custom trainer with code-specific features."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with syntax-aware masking if enabled."""
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply syntax-aware masking if enabled
        if hasattr(self.args, "syntax_aware_masking") and self.args.syntax_aware_masking:
            # Mask certain token types differently
            # This is a simplified version - in practice would use AST info
            mask = labels != -100
            syntax_tokens = self._identify_syntax_tokens(inputs["input_ids"])
            
            # Reduce loss weight for syntax tokens
            loss_weights = torch.ones_like(labels, dtype=torch.float)
            loss_weights[syntax_tokens] = 0.5
            
            # Compute weighted loss
            loss = self._weighted_cross_entropy(logits, labels, loss_weights)
        else:
            # Standard loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
            
        return (loss, outputs) if return_outputs else loss
        
    def _identify_syntax_tokens(self, input_ids):
        """Identify syntax tokens (brackets, operators, etc.)."""
        # This is a placeholder - would use actual token mappings
        syntax_token_ids = set()  # Would contain IDs for {, }, [, ], etc.
        
        return torch.isin(input_ids, torch.tensor(list(syntax_token_ids), device=input_ids.device))
        
    def _weighted_cross_entropy(self, logits, labels, weights):
        """Compute weighted cross-entropy loss."""
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        weighted_loss = loss * weights.view(-1)
        return weighted_loss.mean()


def create_code_model_trainer(config: CodeTrainingConfig) -> CodeModelTrainer:
    """Factory function to create code model trainer."""
    return CodeModelTrainer(config)