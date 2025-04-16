import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput
from transformers.generation import GenerationMixin
from transformers import ProcessorMixin

# Import the Qwen2.5-VL classes
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLConfig
)

@dataclass
class Qwen2_5_VLAnyToAnyLMOutputWithPast(ModelOutput):
    """
    Output type of the any-to-any model.
    """
    loss: Optional[torch.FloatTensor] = None
    text_logits: torch.FloatTensor = None
    vision_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2_5_VLAnyToAnyForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5-VL model that can both consume and generate vision tokens.
    """
    
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        
        # Add a vision head for predicting vision embeddings
        # The vision head projects from the LLM hidden size to the vision encoder's output dimension
        self.vision_head = nn.Linear(config.hidden_size, config.vision_config.out_hidden_size, bias=False)
        
        # Number of image tokens to generate for each vision segment
        self.n_img_tokens = getattr(config, "n_img_tokens", 512)
        
        # Define special tokens for vision control
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)  # <|vision_start|>
        self.vision_end_token_id = getattr(config, "vision_end_token_id", 151653)  # <|vision_end|>
        self.image_pad_token_id = getattr(config, "image_token_id", 151655)  # <|image_pad|>
        
        # Initialize weights for the vision head
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        vision_labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLAnyToAnyLMOutputWithPast]:
        """
        Forward pass of the any-to-any model.
        
        Args:
            vision_labels: Tensor containing vision embeddings to use as targets for the vision generation task
            
        Additional arguments are the same as Qwen2_5_VLForConditionalGeneration.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Run the base model's forward pass
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # We'll handle the loss calculation ourselves
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Always return a dict for internal processing
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )
        
        hidden_states = outputs.hidden_states[-1] if output_hidden_states else outputs.logits
        
        # Get text logits (reuse from base model)
        text_logits = outputs.logits
        
        # Get vision logits by applying the vision head
        vision_logits = self.vision_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Text token prediction loss
            text_loss_fct = CrossEntropyLoss()
            
            # Extract segments that correspond to text prediction
            # Create a mask for positions that should be predicted as text
            is_text = torch.ones_like(labels, dtype=torch.bool)
            
            # Identify the positions where we have vision tokens
            if input_ids is not None:
                # Find sequences of vision tokens (between vision_start and vision_end)
                vision_start_positions = torch.where(input_ids == self.vision_start_token_id)[1]
                vision_end_positions = torch.where(input_ids == self.vision_end_token_id)[1]
                
                # Mark all tokens between vision_start and vision_end as non-text
                for start_pos, end_pos in zip(vision_start_positions, vision_end_positions):
                    is_text[:, start_pos:end_pos+1] = False
            
            # Create shifted labels and logits for text
            shift_labels = labels[..., 1:].contiguous()
            shift_text_logits = text_logits[..., :-1, :].contiguous()
            
            # Apply the mask to get only text token predictions
            text_loss_mask = is_text[..., :-1]
            if text_loss_mask.any():
                text_loss = text_loss_fct(
                    shift_text_logits.view(-1, self.config.vocab_size)[text_loss_mask.view(-1)],
                    shift_labels.view(-1)[text_loss_mask.view(-1)]
                )
            else:
                text_loss = torch.tensor(0.0, device=labels.device)
            
            # Vision token prediction loss (if vision_labels is provided)
            vision_loss = torch.tensor(0.0, device=labels.device)
            if vision_labels is not None:
                # Extract positions where we predict vision tokens
                is_vision = ~is_text
                
                # Apply MSE loss for vision token prediction
                vision_loss_fct = nn.MSELoss()
                
                # Get the actual vision embeddings predicted by our model
                # and compare with the target vision embeddings
                vision_loss_mask = is_vision[..., :-1]
                if vision_loss_mask.any():
                    # Reshape for easier indexing
                    batch_size, seq_len, _ = vision_logits.shape
                    
                    # Extract only the positions where we predict vision tokens
                    predicted_vision_embeddings = vision_logits[..., :-1, :][vision_loss_mask]
                    target_vision_embeddings = vision_labels.view(batch_size, seq_len, -1)[..., :-1, :][vision_loss_mask]
                    
                    vision_loss = vision_loss_fct(predicted_vision_embeddings, target_vision_embeddings)
            
            # Combine losses with optional weighting
            text_weight = getattr(self.config, "text_loss_weight", 1.0)
            vision_weight = getattr(self.config, "vision_loss_weight", 1.0)
            loss = text_weight * text_loss + vision_weight * vision_loss
        
        if not return_dict:
            output = (text_logits, vision_logits) + outputs[1:]
            return (loss,) + output if loss is not None else output
            
        return Qwen2_5_VLAnyToAnyLMOutputWithPast(
            loss=loss,
            text_logits=text_logits,
            vision_logits=vision_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation, delegating to the parent class."""
        return super().prepare_inputs_for_generation(*args, **kwargs)

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False, standardize_cache_format=False
    ):
        """Update model kwargs for the next generation step."""
        return super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, standardize_cache_format
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
        Override the generate method to handle vision token generation.
        """
        # Store original inputs for reference
        original_input_ids = input_ids.clone() if input_ids is not None else None
        
        # Use a custom decode function to track and handle vision token generation
        def custom_logits_processor(input_ids, scores):
            # Check if we're at a vision_start token
            batch_size, seq_len = input_ids.shape
            last_token_ids = input_ids[:, -1]
            
            # Check if the last token is a vision_start token
            is_vision_start = (last_token_ids == self.vision_start_token_id)
            
            if is_vision_start.any():
                # For positions where we need to generate image tokens, force the model
                # to output an image_pad token
                for idx in torch.where(is_vision_start)[0]:
                    scores[idx] = torch.zeros_like(scores[idx])
                    scores[idx, self.image_pad_token_id] = 1.0
            
            return scores
        
        # Custom stopping criteria to handle vision token generation
        class VisionTokenStoppingCriteria:
            def __init__(self, model, max_vision_tokens=512):
                self.model = model
                self.max_vision_tokens = max_vision_tokens
                self.vision_token_counts = {}
                
            def __call__(self, input_ids, scores, **kwargs):
                batch_size = input_ids.shape[0]
                for idx in range(batch_size):
                    # Get the current sequence for this batch item
                    current_seq = input_ids[idx]
                    
                    # Find all vision_start positions
                    vision_start_positions = torch.where(current_seq == self.model.vision_start_token_id)[0]
                    
                    # Find all vision_end positions
                    vision_end_positions = torch.where(current_seq == self.model.vision_end_token_id)[0]
                    
                    # If we have more starts than ends, we're in the middle of generating vision tokens
                    if len(vision_start_positions) > len(vision_end_positions):
                        last_start_pos = vision_start_positions[-1].item()
                        
                        # Calculate how many tokens since the last vision_start
                        tokens_since_start = len(current_seq) - last_start_pos - 1
                        
                        # If we've generated enough vision tokens, force an end token
                        if tokens_since_start >= self.max_vision_tokens:
                            scores[idx] = torch.zeros_like(scores[idx])
                            scores[idx, self.model.vision_end_token_id] = 1.0
                
                # Never stop generation based on this criteria alone
                return False
        
        # Add our custom logits processor
        logits_processor = kwargs.get('logits_processor', [])
        logits_processor.append(custom_logits_processor)
        kwargs['logits_processor'] = logits_processor
        
        # Add our custom stopping criteria
        stopping_criteria = kwargs.get('stopping_criteria', [])
        stopping_criteria.append(VisionTokenStoppingCriteria(self, max_vision_tokens=self.n_img_tokens))
        kwargs['stopping_criteria'] = stopping_criteria
        
        # Call the parent's generate method with our modified kwargs
        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)