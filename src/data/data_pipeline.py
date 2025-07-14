"""
Data pipeline for efficient data loading and preprocessing.
Supports various data formats and streaming for large datasets.
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Iterator, Union, Callable
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PreTrainedTokenizer
import datasets
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
from loguru import logger
from tqdm import tqdm


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    
    # Data source
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    data_files: Optional[Dict[str, str]] = None
    
    # Processing
    max_length: int = 2048
    preprocessing_num_workers: int = 4
    streaming: bool = False
    
    # Tokenization
    padding: str = "max_length"
    truncation: bool = True
    return_overflowing_tokens: bool = False
    
    # DataLoader
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    
    # Filtering
    min_length: int = 10
    max_length_filter: int = 4096
    language_filter: Optional[List[str]] = None
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True


class DataPipeline:
    """Efficient data pipeline for ML training."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def load_dataset(self) -> Union[HFDataset, datasets.IterableDataset]:
        """Load dataset from various sources."""
        logger.info("Loading dataset...")
        
        if self.config.dataset_name:
            # Load from HuggingFace Hub
            dataset = load_dataset(
                self.config.dataset_name,
                streaming=self.config.streaming,
                cache_dir=self.cache_dir,
            )
        elif self.config.dataset_path:
            # Load from local path
            if self.config.dataset_path.endswith('.json') or self.config.dataset_path.endswith('.jsonl'):
                dataset = load_dataset(
                    'json',
                    data_files=self.config.dataset_path,
                    streaming=self.config.streaming,
                    cache_dir=self.cache_dir,
                )
            elif self.config.dataset_path.endswith('.parquet'):
                dataset = load_dataset(
                    'parquet',
                    data_files=self.config.dataset_path,
                    streaming=self.config.streaming,
                    cache_dir=self.cache_dir,
                )
            else:
                raise ValueError(f"Unsupported file format: {self.config.dataset_path}")
        elif self.config.data_files:
            # Load from multiple files
            dataset = load_dataset(
                'json',
                data_files=self.config.data_files,
                streaming=self.config.streaming,
                cache_dir=self.cache_dir,
            )
        else:
            raise ValueError("Must specify either dataset_name, dataset_path, or data_files")
            
        # Get train split
        if isinstance(dataset, dict):
            dataset = dataset.get('train', dataset[list(dataset.keys())[0]])
            
        logger.info(f"Dataset loaded: {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} examples")
        return dataset
        
    def filter_dataset(self, dataset: Union[HFDataset, datasets.IterableDataset]) -> Union[HFDataset, datasets.IterableDataset]:
        """Apply filtering to the dataset."""
        logger.info("Applying dataset filters...")
        
        def filter_function(example):
            text = example.get('text', example.get('content', example.get('code', '')))
            
            # Length filter
            if len(text) < self.config.min_length or len(text) > self.config.max_length_filter:
                return False
                
            # Language filter
            if self.config.language_filter:
                language = example.get('language', example.get('programming_language', 'unknown'))
                if language.lower() not in [lang.lower() for lang in self.config.language_filter]:
                    return False
                    
            return True
            
        if self.config.streaming:
            dataset = dataset.filter(filter_function)
        else:
            original_size = len(dataset)
            dataset = dataset.filter(filter_function, num_proc=self.config.preprocessing_num_workers)
            logger.info(f"Filtered dataset: {len(dataset)}/{original_size} examples remaining")
            
        return dataset
        
    def tokenize_function(self, examples):
        """Tokenize examples."""
        # Extract text from various possible fields
        texts = []
        for i in range(len(examples.get('text', examples.get('content', examples.get('code', ['']))))):
            text = (examples.get('text', examples.get('content', examples.get('code', [''])))[i])
            texts.append(text)
            
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            return_overflowing_tokens=self.config.return_overflowing_tokens,
            return_tensors=None,  # We'll convert later
        )
        
        return tokenized
        
    def preprocess_dataset(self, dataset: Union[HFDataset, datasets.IterableDataset]) -> Union[HFDataset, datasets.IterableDataset]:
        """Preprocess dataset with tokenization."""
        logger.info("Preprocessing dataset...")
        
        # Apply tokenization
        if self.config.streaming:
            dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=1000,
            )
        else:
            dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                num_proc=self.config.preprocessing_num_workers,
                remove_columns=dataset.column_names,
                desc="Tokenizing",
            )
            
        # Add labels for language modeling
        def add_labels(examples):
            examples['labels'] = examples['input_ids'].copy()
            return examples
            
        if self.config.streaming:
            dataset = dataset.map(add_labels, batched=True)
        else:
            dataset = dataset.map(add_labels, batched=True, num_proc=self.config.preprocessing_num_workers)
            
        logger.info("Dataset preprocessing completed")
        return dataset
        
    def create_dataloader(
        self,
        dataset: Union[HFDataset, datasets.IterableDataset],
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = None,
        drop_last: Optional[bool] = None,
    ) -> DataLoader:
        """Create PyTorch DataLoader from dataset."""
        
        shuffle = shuffle if shuffle is not None else self.config.shuffle
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        drop_last = drop_last if drop_last is not None else self.config.drop_last
        
        # Set format for PyTorch
        if hasattr(dataset, 'set_format'):
            dataset.set_format(
                type='torch',
                columns=['input_ids', 'attention_mask', 'labels'],
                output_all_columns=False,
            )
            
        # Create data collator
        def data_collator(features):
            batch = {}
            
            # Get maximum length in batch
            max_len = max(len(f['input_ids']) for f in features)
            
            # Pad sequences
            for key in ['input_ids', 'attention_mask', 'labels']:
                if key in features[0]:
                    sequences = []
                    for f in features:
                        seq = f[key]
                        # Pad to max length
                        if len(seq) < max_len:
                            if key == 'input_ids':
                                pad_value = self.tokenizer.pad_token_id
                            elif key == 'attention_mask':
                                pad_value = 0
                            elif key == 'labels':
                                pad_value = -100
                            else:
                                pad_value = 0
                                
                            seq = torch.cat([
                                seq,
                                torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype)
                            ])
                        sequences.append(seq)
                    batch[key] = torch.stack(sequences)
                    
            return batch
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and not self.config.streaming,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=drop_last,
            collate_fn=data_collator,
        )
        
    def prepare_data(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """Complete data preparation pipeline."""
        # Load dataset
        dataset = self.load_dataset()
        
        # Apply filters
        dataset = self.filter_dataset(dataset)
        
        # Preprocess
        dataset = self.preprocess_dataset(dataset)
        
        # Split if not streaming
        if not self.config.streaming and hasattr(dataset, 'train_test_split'):
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = None
            
        # Create dataloaders
        train_dataloader = self.create_dataloader(train_dataset, shuffle=True)
        eval_dataloader = self.create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None
        
        logger.info(f"Data preparation completed")
        logger.info(f"Train batches: {len(train_dataloader) if hasattr(train_dataloader, '__len__') else 'streaming'}")
        if eval_dataloader:
            logger.info(f"Eval batches: {len(eval_dataloader)}")
            
        return train_dataloader, eval_dataloader


class CodeDataProcessor:
    """Specialized data processor for code datasets."""
    
    def __init__(self):
        self.language_extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h'],
            'c': ['.c', '.h'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
            'ruby': ['.rb'],
            'swift': ['.swift'],
            'kotlin': ['.kt'],
            'scala': ['.scala'],
        }
        
    def extract_functions(self, code: str, language: str) -> List[Dict[str, str]]:
        """Extract function definitions from code."""
        functions = []
        
        if language == 'python':
            import ast
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        start_line = node.lineno - 1
                        # Find end line (simplified)
                        end_line = start_line + 10  # Approximate
                        lines = code.split('\n')
                        func_code = '\n'.join(lines[start_line:min(end_line, len(lines))])
                        
                        functions.append({
                            'name': node.name,
                            'code': func_code,
                            'start_line': start_line,
                            'language': language,
                        })
            except SyntaxError:
                pass
                
        return functions
        
    def create_fill_in_middle_examples(self, code: str, language: str) -> List[Dict[str, str]]:
        """Create fill-in-middle training examples."""
        examples = []
        lines = code.split('\n')
        
        if len(lines) < 10:
            return examples
            
        # Create FIM examples at different split points
        for _ in range(min(3, len(lines) // 10)):
            # Random split points
            split1 = random.randint(2, len(lines) // 3)
            split2 = random.randint(2 * len(lines) // 3, len(lines) - 2)
            
            prefix = '\n'.join(lines[:split1])
            middle = '\n'.join(lines[split1:split2])
            suffix = '\n'.join(lines[split2:])
            
            # Create FIM format
            fim_example = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"
            
            examples.append({
                'text': fim_example,
                'language': language,
                'task': 'fill_in_middle',
            })
            
        return examples
        
    def create_code_completion_examples(self, code: str, language: str) -> List[Dict[str, str]]:
        """Create code completion examples."""
        examples = []
        lines = code.split('\n')
        
        # Create completion examples at function/class boundaries
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in ['def ', 'class ', 'function ', 'public ', 'private ']):
                if i + 5 < len(lines):
                    prefix = '\n'.join(lines[:i+1])
                    completion = '\n'.join(lines[i+1:i+10])
                    
                    examples.append({
                        'text': prefix,
                        'completion': completion,
                        'language': language,
                        'task': 'code_completion',
                    })
                    
        return examples
        
    def augment_code_dataset(self, dataset: HFDataset) -> HFDataset:
        """Augment code dataset with synthetic examples."""
        augmented_examples = []
        
        for example in tqdm(dataset, desc="Augmenting dataset"):
            code = example.get('code', example.get('content', ''))
            language = example.get('language', example.get('programming_language', 'unknown'))
            
            # Original example
            augmented_examples.append(example)
            
            # Add FIM examples
            fim_examples = self.create_fill_in_middle_examples(code, language)
            augmented_examples.extend(fim_examples)
            
            # Add completion examples
            completion_examples = self.create_code_completion_examples(code, language)
            augmented_examples.extend(completion_examples)
            
        return HFDataset.from_list(augmented_examples)


def create_data_pipeline(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    max_length: int = 2048,
    batch_size: int = 8,
    **kwargs
) -> DataPipeline:
    """Factory function to create data pipeline."""
    config = DataConfig(
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        **kwargs
    )
    
    return DataPipeline(tokenizer, config)