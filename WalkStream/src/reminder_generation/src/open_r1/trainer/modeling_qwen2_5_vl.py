# -*- coding: utf-8 -*-
"""
Patched Qwen2.5-VL modeling with cache_embeds support.

Save as:
    src/open_r1/trainer/modeling_qwen2_5_vl.py

Import:
    from src.open_r1.trainer.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from transformers.utils import is_torchdynamo_compiling, logging
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import LossKwargs

# Import official implementations to subclass/patch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel as _HF_Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration as _HF_Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLCausalLMOutputWithPast,
)

logger = logging.get_logger(__name__)


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    """Just a typed mixin matching upstream signatures."""
    pass


class Qwen2_5_VLModel(_HF_Qwen2_5_VLModel):
    """
    Patch on top of HF Qwen2_5_VLModel:
    - Support `cache_embeds` injection:
        * If provided (shape [K, H] or [1, K, H]), only encode the real image row from `image_grid_thw`,
          then concatenate `cache_embeds` after real image features, and scatter back to <image> token slots.
    - Assumes batch=1 for training/generation loop that feeds per-sample cache (as in your trainer).
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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
        # --------- NEW: cache embeddings ----------
        cache_embeds: Optional[torch.Tensor] = None,
        # ------------------------------------------
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:

        # keep upstream behavior
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ===== vision/text embed path with our patch =====
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # ----- IMAGE PATH (patched for cache_embeds) -----
            if pixel_values is not None:
                if cache_embeds is not None:
                    # Expect image_grid_thw to have at least one row for the real image.
                    if image_grid_thw is None or image_grid_thw.dim() != 2 or image_grid_thw.size(0) < 1:
                        raise ValueError(
                            "When `cache_embeds` is provided, `image_grid_thw` must have at least one row for the real image."
                        )

                    # 1) Encode only the real image (first row).
                    grid_real = image_grid_thw[0:1, :]  # [1, 3]
                    image_embeds_real_list = self.get_image_features(pixel_values, grid_real)
                    image_embeds_real = torch.cat(image_embeds_real_list, dim=0)  # [Nr, H]

                    # 2) Normalize/cache shape
                    if cache_embeds.dim() == 3:  # [B, K, H] -> assume B==1
                        cache_cat = cache_embeds[0]
                    elif cache_embeds.dim() == 2:  # [K, H]
                        cache_cat = cache_embeds
                    else:
                        raise ValueError(f"`cache_embeds` must be [K,H] or [1,K,H], got {tuple(cache_embeds.shape)}")

                    cache_cat = cache_cat.to(image_embeds_real.device, image_embeds_real.dtype)  # [K, H]

                    # 3) Concat real image features + cache features
                    image_embeds_all = torch.cat([image_embeds_real, cache_cat], dim=0)  # [Nr+K, H]
                else:
                    # Upstream path: encode all images by image_grid_thw
                    image_embeds_list = self.get_image_features(pixel_values, image_grid_thw)
                    image_embeds_all = torch.cat(image_embeds_list, dim=0)  # [N_all, H]

                # Scatter back into <image> token positions
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = int(image_embeds_all.shape[0])
                if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds_all = image_embeds_all.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_all)

            # ----- VIDEO PATH (unchanged) -----
            if pixel_values_videos is not None:
                video_embeds_list = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds_list, dim=0)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = int(video_embeds.shape[0])
                if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ===== position ids (keep upstream logic) =====
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = (not is_torchdynamo_compiling()) and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
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

        # forward into text LM (unchanged)
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class Qwen2_5_VLForConditionalGeneration(_HF_Qwen2_5_VLForConditionalGeneration):
    """
    Patch on top of HF CausalLM wrapper:
    - forward(..., cache_embeds=...) passes through to model.
    - prepare_inputs_for_generation cleans `cache_embeds` after the first decoding step
      (mirrors pixel_values handling).
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # --------- NEW: pass-through cache embeddings ----------
        cache_embeds: Optional[torch.Tensor] = None,
        # -------------------------------------------------------
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            # pass-through cache_embeds
            cache_embeds=cache_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(  # type: ignore[override]
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        # --------- NEW: receive cache_embeds at generate() call ----------
        cache_embeds=None,
        # ---------------------------------------------------------------
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2.5-VL computes position_ids inside forward using rope_deltas
        model_inputs["position_ids"] = None

        cp = model_inputs.get("cache_position", None)
        # After the first decoding step, drop heavy inputs, including our cache_embeds
        if cp is not None and cp[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs.pop("cache_embeds", None)
        else:
            if cache_embeds is not None:
                model_inputs["cache_embeds"] = cache_embeds

        return model_inputs


__all__ = [
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLModelOutputWithPast",
    "Qwen2_5_VLCausalLMOutputWithPast",
]
