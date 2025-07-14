"""
Multimodal model implementation supporting vision-language tasks.
Inspired by Florence and similar architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPTextModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import BaseModelOutput
import timm
from einops import rearrange, repeat
from loguru import logger

# Handle imports gracefully
try:
    from ..data.multimodal_processor import MultimodalProcessor
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from data.multimodal_processor import MultimodalProcessor
    except ImportError:
        # Mock class if import fails
        class MultimodalProcessor:
            def __init__(self, tokenizer, processor): pass
            def process_batch(self, images, texts): return {"input_ids": [], "pixel_values": []}


@dataclass
class MultimodalConfig(PretrainedConfig):
    """Configuration for multimodal model."""
    
    model_type = "multimodal_vlm"
    
    # Vision encoder
    vision_model_name: str = "openai/clip-vit-large-patch14"
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_num_attention_heads: int = 16
    vision_num_hidden_layers: int = 24
    vision_patch_size: int = 14
    vision_image_size: int = 224
    
    # Language model
    language_model_name: str = "microsoft/phi-2"
    language_hidden_size: int = 2560
    language_intermediate_size: int = 10240
    language_num_attention_heads: int = 32
    language_num_hidden_layers: int = 32
    language_vocab_size: int = 51200
    
    # Cross-modal fusion
    fusion_type: str = "cross_attention"  # cross_attention, mlp, perceiver
    num_fusion_layers: int = 6
    fusion_hidden_size: int = 2048
    fusion_num_attention_heads: int = 16
    fusion_dropout: float = 0.1
    
    # Training
    use_vision_projection: bool = True
    use_language_projection: bool = True
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    
    # Special tokens
    image_token_id: int = 50265
    image_start_token: str = "<image>"
    image_end_token: str = "</image>"


class VisionProjection(nn.Module):
    """Projects vision features to language model dimension."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.vision_hidden_size, config.fusion_hidden_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.fusion_hidden_size, config.language_hidden_size)
        self.norm = nn.LayerNorm(config.language_hidden_size)
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        x = self.linear1(vision_features)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention between vision and language features."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.num_heads = config.fusion_num_attention_heads
        self.hidden_size = config.fusion_hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(config.language_hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(config.vision_hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(config.vision_hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, config.language_hidden_size)
        
        self.dropout = nn.Dropout(config.fusion_dropout)
        self.norm = nn.LayerNorm(config.language_hidden_size)
        
    def forward(
        self,
        language_features: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = language_features.shape
        _, V, _ = vision_features.shape
        
        # Project features
        Q = self.q_proj(language_features).view(B, L, self.num_heads, self.head_dim)
        K = self.k_proj(vision_features).view(B, V, self.num_heads, self.head_dim)
        V = self.v_proj(vision_features).view(B, V, self.num_heads, self.head_dim)
        
        # Rearrange for attention
        Q = rearrange(Q, "b l h d -> b h l d")
        K = rearrange(K, "b v h d -> b h v d")
        V = rearrange(V, "b v h d -> b h v d")
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")
        
        # Output projection
        output = self.o_proj(attn_output)
        output = self.dropout(output)
        
        # Residual connection
        output = self.norm(language_features + output)
        
        return output


class FusionBlock(nn.Module):
    """Single fusion block combining cross-modal attention and FFN."""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.cross_attention = CrossModalAttention(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.language_hidden_size, config.language_intermediate_size),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.language_intermediate_size, config.language_hidden_size),
            nn.Dropout(config.fusion_dropout),
        )
        self.norm = nn.LayerNorm(config.language_hidden_size)
        
    def forward(
        self,
        language_features: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-modal attention
        x = self.cross_attention(language_features, vision_features, attention_mask)
        
        # FFN with residual
        output = x + self.ffn(x)
        output = self.norm(output)
        
        return output


class MultimodalModel(PreTrainedModel):
    """
    Multimodal model combining vision encoder and language model.
    Supports various fusion strategies for vision-language understanding.
    """
    
    config_class = MultimodalConfig
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize vision encoder
        logger.info(f"Loading vision encoder: {config.vision_model_name}")
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_model_name)
        
        if config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
                
        # Initialize language model
        logger.info(f"Loading language model: {config.language_model_name}")
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model_name,
            trust_remote_code=True,
        )
        
        if config.freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
                
        # Vision projection
        if config.use_vision_projection:
            self.vision_projection = VisionProjection(config)
            
        # Cross-modal fusion
        if config.fusion_type == "cross_attention":
            self.fusion_layers = nn.ModuleList([
                FusionBlock(config) for _ in range(config.num_fusion_layers)
            ])
        elif config.fusion_type == "mlp":
            self.fusion_layers = nn.Sequential(
                nn.Linear(
                    config.vision_hidden_size + config.language_hidden_size,
                    config.fusion_hidden_size
                ),
                nn.GELU(),
                nn.Dropout(config.fusion_dropout),
                nn.Linear(config.fusion_hidden_size, config.language_hidden_size),
            )
            
        # Special embeddings for image tokens
        self.image_embedding = nn.Embedding(1, config.language_hidden_size)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        image_positions: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass for multimodal model.
        
        Args:
            input_ids: Text input token IDs
            pixel_values: Image pixel values [B, C, H, W]
            attention_mask: Attention mask for text
            image_positions: Positions where images should be inserted
            labels: Labels for language modeling
        """
        
        batch_size = input_ids.shape[0] if input_ids is not None else pixel_values.shape[0]
        device = input_ids.device if input_ids is not None else pixel_values.device
        
        # Process vision input
        vision_features = None
        if pixel_values is not None:
            vision_outputs = self.vision_encoder(pixel_values)
            vision_features = vision_outputs.last_hidden_state  # [B, num_patches, hidden_size]
            
            if self.config.use_vision_projection:
                vision_features = self.vision_projection(vision_features)
                
        # Get language embeddings
        if input_ids is not None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # Insert vision features at image positions
            if vision_features is not None and image_positions is not None:
                for b in range(batch_size):
                    if image_positions[b].sum() > 0:
                        # Find image token positions
                        img_positions = (image_positions[b] == 1).nonzero(as_tuple=True)[0]
                        
                        if len(img_positions) > 0:
                            # Insert vision features
                            num_patches = vision_features.shape[1]
                            start_pos = img_positions[0]
                            end_pos = min(start_pos + num_patches, inputs_embeds.shape[1])
                            
                            inputs_embeds[b, start_pos:end_pos] = vision_features[b, :end_pos-start_pos]
                            
        else:
            # Vision-only input
            inputs_embeds = vision_features
            
        # Apply cross-modal fusion if using cross-attention
        if self.config.fusion_type == "cross_attention" and vision_features is not None:
            for fusion_layer in self.fusion_layers:
                inputs_embeds = fusion_layer(
                    inputs_embeds,
                    vision_features,
                    attention_mask
                )
                
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )
        
        return outputs
        
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **generate_kwargs
    ) -> torch.LongTensor:
        """Generate text from multimodal input."""
        
        # Process inputs
        if pixel_values is not None:
            # Get vision features
            vision_outputs = self.vision_encoder(pixel_values)
            vision_features = vision_outputs.last_hidden_state
            
            if self.config.use_vision_projection:
                vision_features = self.vision_projection(vision_features)
                
            # Create image tokens
            batch_size = pixel_values.shape[0]
            num_patches = vision_features.shape[1]
            
            if input_ids is not None:
                # Combine with text input
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
                
                # Find image token positions and replace with vision features
                image_token_mask = (input_ids == self.config.image_token_id)
                
                for b in range(batch_size):
                    if image_token_mask[b].any():
                        positions = image_token_mask[b].nonzero(as_tuple=True)[0]
                        if len(positions) > 0:
                            start_pos = positions[0]
                            end_pos = min(start_pos + num_patches, inputs_embeds.shape[1])
                            inputs_embeds[b, start_pos:end_pos] = vision_features[b, :end_pos-start_pos]
            else:
                # Vision-only generation
                inputs_embeds = vision_features
                
            # Generate with embeddings
            return self.language_model.generate(
                inputs_embeds=inputs_embeds,
                **generate_kwargs
            )
        else:
            # Text-only generation
            return self.language_model.generate(
                input_ids=input_ids,
                **generate_kwargs
            )
            
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save model to directory."""
        super().save_pretrained(save_directory, **kwargs)
        
        # Save individual components
        vision_save_path = f"{save_directory}/vision_encoder"
        self.vision_encoder.save_pretrained(vision_save_path)
        
        language_save_path = f"{save_directory}/language_model"
        self.language_model.save_pretrained(language_save_path)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load model from directory."""
        config = MultimodalConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        
        # Load components
        vision_path = f"{pretrained_model_name_or_path}/vision_encoder"
        model.vision_encoder = CLIPVisionModel.from_pretrained(vision_path)
        
        language_path = f"{pretrained_model_name_or_path}/language_model"
        model.language_model = AutoModelForCausalLM.from_pretrained(
            language_path,
            trust_remote_code=True,
        )
        
        # Load fusion layers
        state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin")
        model.load_state_dict(state_dict, strict=False)
        
        return model