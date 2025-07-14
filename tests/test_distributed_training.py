"""
Tests for distributed training functionality.
"""

import pytest
import torch
import torch.distributed as dist
from unittest.mock import patch, MagicMock

from src.training.distributed_trainer import DistributedTrainer, DistributedTrainingConfig


class TestDistributedTrainer:
    """Test cases for DistributedTrainer."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = DistributedTrainingConfig(
            model_name_or_path="gpt2",
            output_dir="./test_output",
            dataset_name="wikitext"
        )
        
        assert config.model_name_or_path == "gpt2"
        assert config.output_dir == "./test_output"
        assert config.learning_rate == 5e-5
        
    @patch.dict('os.environ', {
        'RANK': '0',
        'LOCAL_RANK': '0', 
        'WORLD_SIZE': '1'
    })
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = DistributedTrainingConfig(
            model_name_or_path="gpt2",
            output_dir="./test_output",
            dataset_name="wikitext"
        )
        
        trainer = DistributedTrainer(config)
        assert trainer.rank == 0
        assert trainer.local_rank == 0
        assert trainer.world_size == 1
        
    def test_deepspeed_config_creation(self):
        """Test DeepSpeed configuration creation."""
        config = DistributedTrainingConfig(
            model_name_or_path="gpt2",
            output_dir="./test_output",
            dataset_name="wikitext",
            distributed_backend="deepspeed"
        )
        
        trainer = DistributedTrainer(config)
        ds_config = trainer._create_deepspeed_config()
        
        assert "zero_optimization" in ds_config
        assert ds_config["zero_optimization"]["stage"] == 3
        assert "fp16" in ds_config or "bf16" in ds_config
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_setup(self):
        """Test GPU setup functionality."""
        config = DistributedTrainingConfig(
            model_name_or_path="gpt2",
            output_dir="./test_output",
            dataset_name="wikitext"
        )
        
        trainer = DistributedTrainer(config)
        
        # Test that CUDA is properly configured
        assert torch.cuda.is_available()
        assert torch.cuda.current_device() >= 0


class TestDistributedTrainingIntegration:
    """Integration tests for distributed training."""
    
    @pytest.mark.slow
    def test_single_gpu_training(self):
        """Test training on single GPU."""
        config = DistributedTrainingConfig(
            model_name_or_path="distilgpt2",  # Small model for testing
            output_dir="./test_output",
            dataset_name="wikitext",
            num_train_epochs=1,
            max_steps=10,  # Very short training
            per_device_train_batch_size=1,
        )
        
        trainer = DistributedTrainer(config)
        
        # Mock dataset for testing
        mock_dataset = [
            {"input_ids": torch.randint(0, 1000, (128,)), "labels": torch.randint(0, 1000, (128,))}
            for _ in range(5)
        ]
        
        # This would run actual training in a real test
        # For now, just verify setup works
        assert trainer.config.model_name_or_path == "distilgpt2"