# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""
import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from transformers import AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLConfig, Qwen2VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
AutoModelForCausalLM.register(config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration)
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)

from torch.utils.data import Dataset

from PIL import Image
missing_image_count = 0

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# oracle answer 

def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if script_args.dataset_name == "LLaVA-OneVision-Data":
        def make_conversation(example, source="LLaVA-OneVision-Data"):
            if source == "LLaVA-OneVision-Data":
                return [{"role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": example["image"],
                                    },
                                    {
                                        "type": "text",
                                        "text": example["conversations"][0]["value"],
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": example["conversations"][1]["value"]}],
                            },
                        ]
            elif source == "multimodal-open-r1-8k-verified":
                return [{"role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": example["image"],
                                    },
                                    {
                                        "type": "text",
                                        "text": example["problem"],
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": example["solution"]}],
                            },
                        ]
            data_list = ['CLEVR-Math(MathV360K)', 'GEOS(MathV360K)', 'Geometry3K(MathV360K)', 'UniGeo(MathV360K)', 'TabMWP(MathV360K)', 'FigureQA(MathV360K)']
        
            dataset_list = []
            data_len = []

            for data_name in data_list:
                data = load_dataset("lmms-lab/LLaVA-OneVision-Data", data_name, split="train")
                dataset_list.append(data)
                data_len.append(len(data))

            dataset = []
            # random sample dataset with ratio 0.2
            for i in range(len(dataset_list)):
                dataset_list[i] = dataset_list[i].shuffle(seed=42).select(range(int(data_len[i]*0.1)))
                dataset += [make_conversation(example) for example in dataset_list[i]]

            dataset_list = load_dataset("lmms-lab/multimodal-open-r1-8k-verified", split="train")
            dataset = [make_conversation(example, source="multimodal-open-r1-8k-verified") for example in dataset_list]
    elif script_args.dataset_name == "SAT":
        def make_conversation_sat(example):
            if model_args.model_name_or_path.split("/")[-1] == "Qwen2-VL-2B":
                image = Image.open(dataset_prefix + example["images"][0])
                question = example["messages"][0]["content"]
                question = question.replace("<image>", "")
                prompt = f'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \nUser: {question} \nAssistant:'
                return [{"image": image,
                    "prompt": [
                                {"type": "image"},
                                {"type": "text", "text": "<image>" + prompt}],
                    "solution":  example["messages"][1]["content"], 
                }]
            else:
                if 'alter' not in example:
                    return None
                image_path = dataset_prefix + jpg_path + example["frame_path"] + "/8.jpg"
                if not os.path.exists(image_path):
                    # print(f"Image not found: {image_path}")
                    # missing_image_count += 1
                    return None
                user_message = "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. The prompt should:1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles).2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock).3. Provide specific details about objects (e.g., shape, size, material, distance).4. Include action suggestions (e.g., avoid, move forward, turn left).5. Prioritize safety and clarity. Output Prompt:"
                assistant_message = f"Alter: {example['alter']}."
                return [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "image": image_path,
                                        },
                                        {
                                            "type": "text",
                                            "text": user_message,
                                        },
                                    ],
                                },
                                {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": assistant_message}],
                                },
                            ]

        dataset_prefix = ""
        jpg_path = "src_data/"
        dataset_path = "test_alter_withkey.json"
        dataset_path_val = "test_alter.json"
        
        import json
        # load json file 
        with open(dataset_prefix + dataset_path, 'r') as f:
            # sat_dataset = json.load(f)
            sat_dataset = [json.loads(line.strip()) for line in f]
        # import pdb; pdb.set_trace()
        # dataset = [make_conversation_sat(sample) for sample in sat_dataset]
        dataset = []
        for sample in sat_dataset:
            conversation = make_conversation_sat(sample)
            if conversation is not None:
                dataset.append(conversation)

        #构建val dataset
        with open(dataset_prefix + dataset_path_val, 'r') as f:
            # sat_dataset = json.load(f)
            sat_dataset_val = [json.loads(line.strip()) for line in f]
        # import pdb; pdb.set_trace()
        # dataset = [make_conversation_sat(sample) for sample in sat_dataset]
        dataset_val = []
        for sample in sat_dataset_val:
            conversation = make_conversation_sat(sample)
            if conversation is not None:
                dataset_val.append(conversation)


        # dataset = {'train': dataset}
        num_samples = 20000
        dataset = dataset[:num_samples]
        print("Dataset is ready")
        # print(f"Total missing images: {missing_image_count}")

    dataset = CustomDataset(dataset)
    dataset_val = CustomDataset(dataset_val)

    # import pdb; pdb.set_trace()

    ################
    # Define processor
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing

        if model_args.model_name_or_path.split("/")[-1] == "Qwen2-VL-2B":
            image_inputs = [example[0]['image'] for example in examples]  # Process the images to extract inputs
        else:
            image_inputs = [process_vision_info(example)[0] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels
            
        batch["labels"] = labels  # Add labels to the batch

        return batch
    
    def collate_fn_progressive(examples, progress_ratio=0):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing

        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        for i, example in enumerate(examples):
            # Extract the answer tokens
            answer = example['answer']
            answer_tokens = processor.tokenizer(answer)['input_ids']

            # Determine the split point based on progress_ratio
            split_idx = int(len(answer_tokens) * progress_ratio)
            
            # Mask tokens beyond the split point
            start_idx = (labels[i] != -100).nonzero(as_tuple=True)[0][-len(answer_tokens):][0]
            labels[i, start_idx + split_idx : start_idx + len(answer_tokens)] = -100
            
        # import pdb; pdb.set_trace()
        batch["labels"] = labels  # Add labels to the batch

        return batch

    ################
    # Training
    ################
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                    torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2",
    )

    min_pixels = 256*28*28
    max_pixels = 156800
    model.visual.requires_grad_ = False
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, max_pixels=max_pixels, padding_side='right')
    processor.tokenizer.padding_side = 'left'

    training_args.model_init_kwargs = None
    training_args.dataset_text_field = ""
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    progressive_training = False
    # import pdb; pdb.set_trace()
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_val,
        # peft_config=get_peft_config(model_args),
        # tokenizer=processor.tokenizer,
        data_collator=collate_fn if not progressive_training else collate_fn_progressive,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # print(training_args)
    main(script_args, training_args, model_args)
