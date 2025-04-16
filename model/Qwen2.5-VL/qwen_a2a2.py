import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

class AnyToAnyQwen2_5_VL(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # Number of vision tokens to generate after <|vision_start|>
        # Special token IDs
        self.vision_start_token_id = 151652  # <|vision_start|>
        self.vision_end_token_id = 151653    # <|vision_end|>
        # Setup vision head
        self.setup_vision_head()
        # Initialize weights
        self.post_init()

    def setup_vision_head(self):
        """Initialize the vision head that predicts vision embeddings."""
        self.vision_head = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def generate_labels(self, input_ids, vision_positions=None):
        """
        Generate labels for language modeling (same as input_ids for next-token prediction).
        Masks out vision positions with -100 so they don't contribute to text loss.
        """
        labels = input_ids.clone()
        
        # Mask out vision positions
        if vision_positions is not None:
            labels = labels.masked_fill(vision_positions, -100)
        
        return labels
        
    def identify_vision_positions(self, input_ids):
        """
        Create a boolean mask identifying positions where vision tokens should be predicted.
        These are positions that follow a vision_start_token.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            
        Returns:
            is_vision_position: Boolean mask of shape (batch_size, seq_length)
        """
        batch_size, seq_length = input_ids.shape
        is_vision_position = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Find positions where we need to predict vision tokens
        for b in range(batch_size):
            # Find start tokens in the sequence
            starts = (input_ids[b] == self.vision_start_token_id).nonzero(as_tuple=True)[0]
            ends = (input_ids[b] == self.vision_end_token_id).nonzero(as_tuple=True)[0]
            
            # For each start token, mark the next n_img_tokens positions for vision prediction
            for i in range(len(starts)):
                start_pos = starts[i]
                # Find the corresponding end position or use sequence length if not found
                end_pos = seq_length
                for end_idx in ends:
                    if end_idx > start_pos:
                        end_pos = end_idx
                        break
                
                # Mark the positions between start and end as vision positions
                # (not including the start token itself, but including positions up to end token)
                if start_pos + 1 < seq_length:
                    is_vision_position[b, start_pos+1:min(end_pos+1, seq_length)] = True
        
        return is_vision_position

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for the any-to-any model that can generate both text and vision tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Store original input_embeds for vision loss calculation
        original_inputs_embeds = None
        
        # Process inputs and prepare embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # Process image inputs if available
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                )
            
            # Process video inputs if available
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    mask, video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                )
                
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
            
            # Save the original inputs_embeds for use in loss calculation
            # Using detach() to prevent gradient flow through the target values
            original_inputs_embeds = inputs_embeds.clone().detach()

        # Handle position_ids and rope_deltas as in the original model
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Forward pass through the LLM
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        
        # Identify positions where we need to predict vision tokens
        vision_positions = None
        if input_ids is not None:
            vision_positions = self.identify_vision_positions(input_ids)
        
        # Get text logits for the entire sequence
        text_logits = self.lm_head(hidden_states)
        
        # Get vision predictions for the entire sequence
        vision_predictions = self.vision_head(hidden_states)
        
        # Calculate loss (if in training mode)
        loss = None
        
        # If labels are not provided but we have input_ids, generate them
        if labels is None and input_ids is not None:
            labels = self.generate_labels(input_ids, vision_positions)
            
        if labels is not None:
            # Initialize loss
            loss = 0.0
            
            # 1. Calculate text loss (cross-entropy)
            text_logits_float = text_logits.float()
            
            # Shift logits and labels for next-token prediction
            shift_logits = text_logits_float[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Mask out vision positions in labels to avoid computing text loss for them
            if vision_positions is not None:
                # Shift vision positions to align with shifted labels
                shift_vision_positions = vision_positions[..., 1:].contiguous()
                shift_labels = shift_labels.masked_fill(shift_vision_positions, -100)
            
            # Calculate cross-entropy loss for text positions
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = loss_fct(shift_logits, shift_labels)
            loss += text_loss
            
            # 2. Calculate vision loss (MSE or cosine similarity)
            if vision_positions is not None and vision_positions.any() and original_inputs_embeds is not None:
                # Get the shifted vision positions for next-token prediction
                shift_vision_positions = vision_positions[..., :-1].contiguous()
                
                if shift_vision_positions.any():
                    # Get predictions and targets for vision positions
                    # Predictions: what our vision head predicts for the next position
                    # Targets: the actual embeddings from the input at the next position
                    
                    # Get predicted vision embeddings (from current positions)
                    pred_vision_embeds = vision_predictions[..., :-1, :][shift_vision_positions]
                    
                    # Get target embeddings (from next positions in original_inputs_embeds)
                    target_vision_embeds = original_inputs_embeds[..., 1:, :][shift_vision_positions]
                    
                    # Calculate MSE loss between predicted and target vision embeddings
                    vision_loss = F.mse_loss(
                        pred_vision_embeds, 
                        target_vision_embeds.to(pred_vision_embeds.dtype)
                    )
                    
                    # Alternative: Cosine similarity loss
                    # vision_loss = 1 - F.cosine_similarity(
                    #     pred_vision_embeds, 
                    #     target_vision_embeds.to(pred_vision_embeds.dtype), 
                    #     dim=-1
                    # ).mean()
                    
                    loss += vision_loss

        # Return the appropriate output format
        if not return_dict:
            output = (text_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=text_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )