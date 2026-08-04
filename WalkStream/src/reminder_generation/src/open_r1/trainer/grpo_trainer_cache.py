# -*- coding: utf-8 -*-
# Qwen2VLGRPOTrainer (per-sample cache version)

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from contextlib import contextmanager

import copy
import json
import torch
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModel,
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from src.open_r1.trainer.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

@contextmanager
def generation_cache_mode(model):
    """
    Temporarily:
      - switch to eval()
      - disable gradient checkpointing
      - force config.use_cache=True
    Then restore everything.
    """
    # 记录当前状态
    prev_training = model.training
    prev_use_cache = getattr(model.config, "use_cache", True)

    # 有的实现把标志挂在 language_model / visual 子模块上，一起关掉更稳妥
    def _disable_gc(m):
        if hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable()
        # 兼容手动 flag
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = False

    def _enable_gc(m):
        if hasattr(m, "gradient_checkpointing_enable"):
            m.gradient_checkpointing_enable()
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = True

    # 检测之前是否启用了 GC，等会只在启用过的情况下恢复
    def _gc_enabled(m):
        return bool(getattr(m, "gradient_checkpointing", False))

    was_gc_enabled_model = _gc_enabled(model)
    was_gc_enabled_lm = _gc_enabled(getattr(model, "language_model", model))
    was_gc_enabled_vis = _gc_enabled(getattr(model, "visual", model))

    try:
        model.eval()
        _disable_gc(model)
        if hasattr(model, "language_model"):
            _disable_gc(model.language_model)
        if hasattr(model, "visual"):
            _disable_gc(model.visual)
        # 强制用 cache
        if hasattr(model, "config"):
            model.config.use_cache = True
        yield
    finally:
        # 恢复 use_cache 与 train/eval
        if hasattr(model, "config"):
            model.config.use_cache = prev_use_cache
        if was_gc_enabled_model:
            _enable_gc(model)
        if was_gc_enabled_lm and hasattr(model, "language_model"):
            _enable_gc(model.language_model)
        if was_gc_enabled_vis and hasattr(model, "visual"):
            _enable_gc(model.visual)
        model.train(prev_training)


class Qwen2VLGRPOTrainer(Trainer):
    """
    GRPO Trainer customized for Qwen2-VL to inject per-sample cache visual tokens.

    Differences from vanilla:
      - In compute_loss(): read `cache_embeds` from sample (inputs[0]["cache_embeds"]).
      - Insert K extra <image> tokens, append [K, s, s] to image_grid_thw.
      - Pass `cache_embeds` to model/generate so that modeling_qwen2_vl.py can拼接视觉特征.
      - Do per-generation scoring sequentially with batch=1 (to keep visual features aligned).
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, str) and torch_dtype not in ("auto", None):
                model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache", None)
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "InternVL2" in model_id:
                model_init_kwargs.pop("use_cache", None)
                model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError("You passed `model_init_kwargs` in GRPOConfig but model is already instantiated.")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "InternVL2" in model_id:
                self.ref_model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, padding_side="left")
                processing_class.tokenizer.padding_side = "left"
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen2-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            elif "InternVL2" in model_id:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True, padding_side="left")
                processing_class.tokenizer.padding_side = "left"
                pad_token_id = processing_class.pad_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                processing_class.tokenizer.padding_side = "left"
                pad_token_id = processing_class.pad_token_id
        self.processing_class = processing_class
        self.processing_class.tokenizer.padding_side = "left"

        # Reward funcs
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(reward_func, num_labels=1, **model_init_kwargs)
        self.reward_funcs = reward_funcs

        # Reward tokenizers
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("reward_processing_classes length mismatch.")
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Collator
        def data_collator(features):
            return features

        # Training args
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=1,  # 循环生成 G 次
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # Init trainer
        model.warnings_issued["estimate_tokens"] = True
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
        )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # useful attrs
        self._metrics = defaultdict(list)
        umodel = self.accelerator.unwrap_model(self.model)
        self.image_token_id = umodel.config.image_token_id if hasattr(umodel.config, "image_token_id") else None
        self.spatial_merge_size = umodel.visual.spatial_merge_size if hasattr(umodel, "visual") else 2
        self.hidden_size = umodel.config.text_config.hidden_size
        self.emb_dtype = umodel.get_input_embeddings().weight.dtype

    def _prepare_inputs(self, inputs):
        return inputs

    @staticmethod
    def _insert_after_image_block(input_ids_1d: torch.Tensor, attn_mask_1d: torch.Tensor, image_token_id: int, K: int):
        dev, dtype = input_ids_1d.device, input_ids_1d.dtype
        pos = (input_ids_1d == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        if pos.numel() == 0:
            raise ValueError("No <image> tokens found in input_ids.")
        end = pos[-1].item() + 1
        insert_ids = torch.full((K,), image_token_id, device=dev, dtype=dtype)
        insert_mask = torch.ones(K, device=dev, dtype=attn_mask_1d.dtype)
        new_ids = torch.cat([input_ids_1d[:end], insert_ids, input_ids_1d[end:]], dim=0)
        new_mask = torch.cat([attn_mask_1d[:end], insert_mask, attn_mask_1d[end:]], dim=0)
        return new_ids, new_mask

    def _append_cache_grid(self, image_grid_thw: torch.Tensor, K: int):
        dev = image_grid_thw.device
        s = int(self.spatial_merge_size)
        add = torch.tensor([K, s, s], device=dev, dtype=image_grid_thw.dtype).unsqueeze(0)
        return torch.cat([image_grid_thw, add], dim=0)

    # @torch.no_grad()
    def _per_token_logps_single(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        cache_embeds: torch.Tensor,
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_embeds=cache_embeds,
            use_cache=False,
        )
        logits = out.logits  # (1, L, V)
        logits = logits[:, :-1, :]
        tgt = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logp = torch.gather(log_probs, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
        return token_logp

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        # === 每个样本自带 cache_embeds ===
        sample = inputs[0]  # batch=1
        cache_embeds = sample["cache_embeds"]                  # torch.Tensor [K, hidden]
        if not isinstance(cache_embeds, torch.Tensor):
            cache_embeds = torch.tensor(cache_embeds)
        K = cache_embeds.size(0)
        if cache_embeds.size(1) != self.hidden_size:
            raise ValueError(f"cache_embeds dim mismatch: expect {self.hidden_size}, got {cache_embeds.size(1)}")
        cache_embeds = cache_embeds.to(device=device, dtype=self.emb_dtype)

        prompts = [sample["prompt"]]
        images = [sample["image"]]
        keywords = [sample["prompt"][-1]["content"][1].get("keywords", [])]

        prompts_text = ['<|vision_start|><|image_pad|><|vision_end|>' + sample['prompt'][-1]['content'][-1]['text']]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        for k, v in list(prompt_inputs.items()):
            if isinstance(v, torch.Tensor):
                prompt_inputs[k] = v.to(device)

        # 插入 K 个 <image> token，追加 [K,s,s]
        if self.image_token_id is None:
            raise ValueError("image_token_id not found in model config.")
        input_ids = prompt_inputs["input_ids"]
        attn_mask = prompt_inputs["attention_mask"]
        new_ids, new_mask = self._insert_after_image_block(input_ids[0], attn_mask[0], self.image_token_id, K)
        prompt_inputs["input_ids"] = new_ids.unsqueeze(0)
        prompt_inputs["attention_mask"] = new_mask.unsqueeze(0)
        prompt_inputs["image_grid_thw"] = self._append_cache_grid(prompt_inputs["image_grid_thw"], K)
        prompt_inputs["cache_embeds"] = cache_embeds

        # 逐次生成 G 次
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # （强烈建议）文本 tokenizer 左填充，避免 FA2 对右填充的限制
            try:
                tok = getattr(self, "tokenizer", None) or getattr(self, "processor", None)
                if tok is not None and hasattr(tok, "padding_side"):
                    tok.padding_side = "left"
            except Exception:
                pass

            gens = []
            # 仅在 generate 的这段时间里：关闭GC + 强制 use_cache=True
            with generation_cache_mode(unwrapped_model):
                # 如果你用的是 FA2，确保半精度（也可以在 from_pretrained 时传 torch_dtype）
                use_amp = getattr(unwrapped_model.config, "_attn_implementation", "") == "flash_attention_2"
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                ctx = torch.autocast(device_type=device_type) if use_amp else contextmanager(lambda: (yield))()

                with ctx:
                    for _ in range(self.num_generations):
                        gen_ids = unwrapped_model.generate(
                            **prompt_inputs,
                            generation_config=self.generation_config,
                            use_cache=True,                    # << 关键：确保用KV-cache
                            return_dict_in_generate=False,     # 需要scores再改True
                            output_scores=False,
                        )
                        gens.append(gen_ids)

            max_len = max(t.size(1) for t in gens)
            padded = []
            for t in gens:
                if t.size(1) < max_len:
                    pad = torch.full((t.size(0), max_len - t.size(1)),
                                     self.processing_class.tokenizer.pad_token_id,
                                     dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            prompt_completion_ids = torch.cat(padded, dim=0)  # (G, Lmax)

        prompt_len = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_len:]  # (G, Cmax)

        # 逐条算 logps（保持 batch=1，与视觉特征对齐）
        eos_id = self.processing_class.tokenizer.eos_token_id
        token_logps_list, ref_token_logps_list, completion_masks = [], [], []
        for gi in range(self.num_generations):
            seq_ids = prompt_completion_ids[gi:gi+1, :]
            pmask = prompt_inputs["attention_mask"]
            comp = completion_ids[gi:gi+1, :]
            if eos_id is None:
                cmask = torch.ones_like(comp)
            else:
                is_eos = (comp == eos_id)
                if is_eos.any():
                    first_eos = is_eos.int().argmax(dim=1)
                    rng = torch.arange(comp.size(1), device=comp.device).unsqueeze(0)
                    cmask = (rng <= first_eos.unsqueeze(1)).int()
                else:
                    cmask = torch.ones_like(comp)
            attn_mask_seq = torch.cat([pmask, cmask], dim=1)

            # pixel / grid
            pixel_values = prompt_inputs["pixel_values"]
            image_grid_thw = prompt_inputs["image_grid_thw"]

            logp = self._per_token_logps_single(
                model,
                input_ids=seq_ids,
                attention_mask=attn_mask_seq,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                cache_embeds=cache_embeds,
            )
            token_logps_list.append(logp)

            if self.ref_model is not None:
                with torch.no_grad():
                    ref_logp = self._per_token_logps_single(
                        self.ref_model,
                        input_ids=seq_ids,
                        attention_mask=attn_mask_seq,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        cache_embeds=cache_embeds,
                    )
                ref_logp = ref_logp.detach()
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_logp = self._per_token_logps_single(
                        model,
                        input_ids=seq_ids,
                        attention_mask=attn_mask_seq,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        cache_embeds=cache_embeds,
                    )
            ref_token_logps_list.append(ref_logp)
            completion_masks.append(cmask)

        per_token_logps = torch.cat(token_logps_list, dim=0)[:, prompt_len - 1 :]
        ref_per_token_logps = torch.cat(ref_token_logps_list, dim=0)[:, prompt_len - 1 :]
        completion_mask = torch.cat(completion_masks, dim=0)

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(sample):
            completions = [[{"role": "assistant", "content": c}] for c in completions]

        # 奖励
        prompts_rep = [prompts[0] for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(self.num_generations, len(self.reward_funcs), device=device)
        for i, (reward_func, reward_tok) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(sample):
                    messages = [{"messages": p + c} for p, c in zip(prompts_rep, completions)]
                    texts = [apply_chat_template(x, reward_tok)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts_rep, completions)]
                reward_inputs = reward_tok(texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
                for k, v in list(reward_inputs.items()):
                    if isinstance(v, torch.Tensor):
                        reward_inputs[k] = v.to(device)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in sample.keys() if key not in ["prompt", "completion", "keywords", "image", "cache_embeds", "image_path"]}
                for key in reward_kwargs:
                    reward_kwargs[key].extend([sample[key]] * self.num_generations)
                kws = keywords[0] if len(keywords) > 0 else []
                reward_kwargs["keywords"] = [kws] * self.num_generations
                out = reward_func(prompts=prompts_rep, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)
        mean_grouped = rewards.mean()
        std_grouped = rewards.std()
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.view(-1, 1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()

        # metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            name = reward_func.config._name_or_path.split("/")[-1] if isinstance(reward_func, PreTrainedModel) else reward_func.__name__
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(std_grouped.item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # 可选：轨迹
        try:
            save_dir = f"trajectories/trajectories_{self.args.run_name}/step{self.state.global_step}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"rank{self.accelerator.process_index}.jsonl")
            with open(save_path, "w") as f:
                json.dump({
                    "trajectories": [
                        {"messages": {"prompt": prompts[0][0], "response": completions[i]}, "reward": float(rewards[i].item())}
                        for i in range(self.num_generations)
                    ]
                }, f, indent=2)
        except Exception:
            pass

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {k: sum(v) / len(v) for k, v in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(self, model_name: Optional[str] = None, dataset_name: Optional[str] = None, tags: Union[str, list[str], None] = None):
        if not self.is_world_process_zero():
            return
        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None
        tags = [tags] if isinstance(tags, str) else (tags or [])
        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")
        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )
        mc = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )
        mc.save(os.path.join(self.args.output_dir, "README.md"))
