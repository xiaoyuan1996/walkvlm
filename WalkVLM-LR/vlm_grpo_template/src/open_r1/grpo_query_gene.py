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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
import math

from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import base64
import requests
from openai import OpenAI
from collections import Counter

from math_verify import parse, verify
from src.open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    reward_funcs: list[str] = field(
        default_factory=lambda: ['fluency', 'simplicity', 'sematic','keywords'],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the LLM parameters during training"},
    )
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the vision model parameters during training"},
    )
    dataset_prefix: str = field(
        default="/mnt/default",
        metadata={"help": "Base directory path for the dataset."}
    )
    dataset_path: str = field(
        default="/mnt/default.json",
        metadata={"help": "Full path to the dataset file."}
    )        

def extract_letters(text): # for RAVEN
    pattern = r'(^|\s|\[|\()([A-H])(\s|\]|\)|$)'
    matches = re.findall(pattern, text)
    return [match[1] for match in matches]

#reward of mine#
def compute_ngram_diversity(text, n=2):
    """
    Compute the n-gram diversity of the text. This calculates the ratio of unique n-grams
    to the total number of n-grams in the text. Higher values indicate higher diversity.
    """
    tokens = text.split()
    
    n_grams = list(ngrams(tokens, n))
    
    n_gram_counts = Counter(n_grams)
    
    diversity = len(n_gram_counts) / len(n_grams) if len(n_grams) > 0 else 0
    
    return diversity

def fluency_reward_ngram(completions, **kwargs):
    """
    Reward function that evaluates the fluency of the completions using N-gram diversity.
    Higher diversity means better fluency, and therefore higher reward.
    """
    n1_weight=0.7
    n2_weight=0.3
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]

    rewards = []
    
    for content in contents:
        # Compute the N-gram diversity (higher is better)
        ngram_diversity_1 = compute_ngram_diversity(content, n=1)
        
        # Compute the 2-gram diversity (higher is better)
        ngram_diversity_2 = compute_ngram_diversity(content, n=2)
        
        # Compute weighted average of 1-gram and 2-gram diversities
        weighted_diversity = (n1_weight * ngram_diversity_1) + (n2_weight * ngram_diversity_2)
        
        # Ensure the reward is within the range [0.1, 1.0]
        reward = max(0.0, min(weighted_diversity, 1.0))

        rewards.append(reward)
    return rewards


def mean_token_accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that evaluates the mean token accuracy of the completions against the solution.
    Higher accuracy means better reward.
    """
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]

    rewards = []

    for content, reference in zip(contents, solution):
        # Convert both the generated sentence and solution sentence into token lists
        generated_tokens = content.lower().split()  # Convert to lowercase and split into tokens
        reference_tokens = reference.lower().split()  # Convert to lowercase and split into tokens

        # Ensure both sentences have the same length
        min_length = min(len(generated_tokens), len(reference_tokens))

        # Calculate mean token accuracy (number of correct tokens divided by total tokens)
        correct_tokens = sum(1 for i in range(min_length) if generated_tokens[i] == reference_tokens[i])
        mean_accuracy = correct_tokens / min_length if min_length > 0 else 0

        # Ensure the reward is within the range [0.0, 1.0]
        reward = max(0.0, min(mean_accuracy, 1.0))

        rewards.append(reward)

    return rewards


def load_gpt2_model():
    model_path = ''
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def compute_perplexity(text, model, tokenizer):
    if not text.strip():  # If the text is empty or just whitespace
        return 1000000  # Return a high perplexity for empty text
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    if inputs["input_ids"].shape[1] == 0:
        return 1000000  # If tokenized text is empty, return high perplexity
    
    # Compute the model's logits
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
    
    # Compute the loss (Cross-Entropy Loss)
    shift_logits = logits[..., :-1, :].contiguous()  # Remove the last token
    shift_labels = inputs["input_ids"][..., 1:].contiguous()  # Remove the first token
    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Perplexity is the exponential of the loss
    perplexity = torch.exp(loss)
    
    return perplexity.item()

def fluency_reward(completions, **kwargs):
    n1_weight=0.7
    n2_weight=0.3
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]

    rewards = []
    
    # Load GPT-2 model and tokenizer
    model, tokenizer = load_gpt2_model()
    
    for content in contents:
        # Compute the perplexity (lower is better)
        perplexity = compute_perplexity(content, model, tokenizer)
        
        scaled_perplexity = perplexity * 0.0001  # Scale perplexity for better sigmoid behavior
        
        # Ensure scaled_perplexity is a tensor
        scaled_perplexity_tensor = torch.tensor(scaled_perplexity)  # Convert to tensor
        
        reward = 1 / (1 + torch.exp(scaled_perplexity_tensor))  # Sigmoid function
        
        # Adjust the reward range to be between 0.1 and 1 using scaling and shifting
        gpt_reward = 0.9 * reward + 0.1  # This shifts the range to [0.1, 1]

        # Compute the N-gram diversity (higher is better)
        ngram_diversity_1 = compute_ngram_diversity(content, n=1)
        
        # Compute the 2-gram diversity (higher is better)
        ngram_diversity_2 = compute_ngram_diversity(content, n=2)
        
        # Compute weighted average of 1-gram and 2-gram diversities
        weighted_diversity = (n1_weight * ngram_diversity_1) + (n2_weight * ngram_diversity_2)
        
        lambda_value = 0.5
        reward = weighted_diversity / (weighted_diversity + lambda_value * gpt_reward)
        # Optional: Apply a lower bound to the reward to prevent very small values
        reward = max(0.0, min(reward, 1.0))  # Ensure reward is within [0.1, 1]
        
        rewards.append(reward)

    return rewards


def load_clip_model():
    """
    Load the pre-trained CLIP model and processor for encoding text.
    """
    model_path = ""
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    return model, processor

clip_model, clip_processor = load_clip_model()

def encode_text_with_clip(text, model, processor):
    """
    Encode the input text using the CLIP model and return its embedding.
    """
    # Preprocess the text (tokenize and pad) using CLIP processor
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # Get the text features (embeddings) from the CLIP model
        outputs = model.get_text_features(**inputs)
    return outputs

def semantic_similarity_reward(completions, solution, **kwargs):
    """
    Reward function that evaluates the semantic similarity between completion and solution using CLIP.
    """
    # If completions is a list of strings, use them directly, otherwise extract content
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]

    rewards = []
    
    # Load CLIP model and processor
    # model, processor = load_clip_model()
    
    for content, sol in zip(contents, solution):
        # semantic_similarity
        # Encode both the content and the solution using CLIP
        content_embedding = encode_text_with_clip(content, clip_model, clip_processor)
        solution_embedding = encode_text_with_clip(sol, clip_model, clip_processor)
        
        # Compute cosine similarity between content and solution
        similarity = cosine_similarity(content_embedding.detach().numpy(), solution_embedding.detach().numpy())[0][0]
        # semantic_similarity

        generated_tokens = content.lower().split()  # Convert to lowercase and split into tokens
        reference_tokens = sol.lower().split()  # Convert to lowercase and split into tokens

        # Ensure both sentences have the same length
        min_length = min(len(generated_tokens), len(reference_tokens))

        # Calculate mean token accuracy (number of correct tokens divided by total tokens)
        correct_tokens = sum(1 for i in range(min_length) if generated_tokens[i] == reference_tokens[i])
        mean_accuracy = correct_tokens / min_length if min_length > 0 else 0

        
        # Reward based on similarity (1.0 for exact match, 0.0 for no similarity)
        a = 0.75
        reward = a*mean_accuracy + (1-a)*similarity
        reward = min(1.0, reward)  # Ensure reward is between 0 and 1
        rewards.append(reward)
    
    return rewards

def generate_ngrams(text, n=2):
    """
    Generate n-grams from the input text.
    """
    words = text.split()  
    ngrams_list = list(ngrams(words, n))  
    ngram_strings = [' '.join(ngram) for ngram in ngrams_list]  
    return ngram_strings

def calculate_keyword_density(text, keywords, model, processor):
    """
    Calculate keyword density based on semantic similarity between each word and the keywords using CLIP.
    """
    threshold=0.9
    text = re.sub(r'[^\w\s]', '', text.lower())  
    word_tokens = text.split()  
    unique_word_tokens = set(word_tokens)  
    total_unique_words = len(unique_word_tokens)  
    
    if total_unique_words == 0:
        return 0.0  
    
    ngrams_list = generate_ngrams(text, 2)
    
    all_tokens = unique_word_tokens.union(set(ngrams_list))
    total_tokens = len(all_tokens)
    
    keyword_count = 0
    
    for token in all_tokens:
        token_embedding = encode_text_with_clip(token, model, processor)
        
        for keyword in keywords:
            keyword_embedding = encode_text_with_clip(keyword, model, processor)
            
            similarity = cosine_similarity(token_embedding.detach().numpy(), keyword_embedding.detach().numpy())[0][0]
            
            if similarity >= threshold:
                keyword_count += 1
                break  
    keyword_density = keyword_count / total_tokens
    return keyword_density


def calculate_redundancy_penalty(text, keywords):
    """
    Calculate penalty for redundancy based on repeated keywords.
    """
    word_tokens = text.split()
    keyword_count = 0
    
    for keyword in keywords:
        if isinstance(keyword, str):  
            keyword = keyword.strip().lower()  
            keyword_count += word_tokens.count(keyword)
    
    redundancy_penalty = 0
    if keyword_count > len(keywords):
        redundancy_penalty = (keyword_count - len(keywords)) * 0.1  
    
    return redundancy_penalty


def keywords_reward(completions, solution, **kwargs):
    """
    Reward function that evaluates the conciseness of the completion based on keyword density.
    Now, the reward is calculated based on keyword density in the sentence after deduplication.
    """
    keywords = kwargs.get("keywords", [])  
    
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    
    for completion in contents:
        keyword_density = calculate_keyword_density(completion, keywords, clip_model, clip_processor)
        reward = max(0.0, keyword_density)  
        rewards.append(reward)
    
    return rewards


def include_tags_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion include solution tags"""
    if isinstance(completions[0],str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        tags = [item for item in sol.replace('<answer>', '').replace('</answer>', '').split('#') if item != '']
        len_sol = len(tags)
        
        include_cnt = 0
        for tag in tags:
            if '<answer>' in content: 
                reward_answer = 0.05
                post_content = content.split('<answer>')[-1]
                
                if tag not in ['', ' '] and tag in post_content:
                    include_cnt += 1
            
            else:
                reward_answer = 0.0
                break
                
        reward = min(1.0, include_cnt/len_sol + reward_answer)
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} include_tags_reward reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
                
    return rewards
    

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    if isinstance(completions[0],str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                if student_answer == ground_truth:
                    reward = 1.0

                if extract_letters(student_answer)[-1] == ground_truth:
                    reward = 1.0
                # Compare the extracted answers

            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def length_reward(completions, **kwargs):
    
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    rewards = []
    for completion in completions:
        if isinstance(completions[0],str):
            rewards.append(len(processor.tokenizer(completion)['input_ids']) * 0.001)
        else:
            rewards.append(len(processor.tokenizer(completion[0]["content"])['input_ids']) * 0.001)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    if isinstance(completions[0],str):
        completion_contents = ["<think>" + completion for completion in completions]
    else:
        completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def simplicity_reward(completions, **kwargs):
    if isinstance(completions[0],str):
        contents = [completion for completion in completions]
    else:
        contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    L_max = 20  
    max_reward = 1  
    for content in contents:
        print(content)
        cleaned_content = re.sub(r'[^\w\s]', '', content)
        words = cleaned_content.split()
        content_length = len(words)
        if content_length == 0:
            reward = 0
        else:
            reward = max_reward - ((abs(content_length - L_max) / L_max) ** 2)
            reward = max(reward, 0)
        rewards.append(reward)

        
    return rewards

    



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward,
    'include_tags': include_tags_reward,
    'simplicity': simplicity_reward,
    'sematic': semantic_similarity_reward,
    'keywords': keywords_reward,
    'fluency': fluency_reward
}

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    
    def make_conversation_sat(example):
        if 'alter' not in example:
            return None
        image_path = dataset_prefix + jpg_path + example["frame_path"] + "/8.jpg"
        if not os.path.exists(image_path):
    
            return None
        assistant_message = f"Alter: {example['alter']}."
        keywords = example['keywords']
        with Image.open(image_path).convert('RGB') as image:
            image = Image.open(image_path)

        user_message = "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. The prompt should:1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles).2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock).3. Provide specific details about objects (e.g., shape, size, material, distance).4. Include action suggestions (e.g., avoid, move forward, turn left).5. Prioritize safety and clarity. Output Prompt: "
        
        return {"image": image,
                "image_path": image_path,
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "keywords": keywords},
                            {"type": "text", "text": user_message},
                        ],
                    },
                ],
                "solution":  "<answer>" + example['alter'] + "</answer>", 
            }        
    
    dataset_prefix = "walkvlm/wad_dataset/"
    jpg_path = "src_data/"
    dataset_path = "train.json"
    dataset_path_val = "test_alter.json"

    import json

    with open(dataset_prefix + dataset_path, 'r') as f:
            sat_dataset = [json.loads(line.strip()) for line in f]
    dataset = []
    for sample in sat_dataset:
        conversation = make_conversation_sat(sample)
        if conversation is not None:
            dataset.append(conversation)

        #构建val dataset
    with open(dataset_prefix + dataset_path_val, 'r') as f:
        sat_dataset_val = [json.loads(line.strip()) for line in f]
    dataset_val = []
    for sample in sat_dataset_val:
        conversation = make_conversation_sat(sample)
        if conversation is not None:
            dataset_val.append(conversation)

    dataset = {'train': dataset, 'test':dataset_val}

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if script_args.freeze_vision:
        trainer.model.visual.requires_grad_ = False
    elif script_args.freeze_llm:
        trainer.model.model.requires_grad_ = False
    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    print(training_args)
    print(model_args)
    
    main(script_args, training_args, model_args)
    