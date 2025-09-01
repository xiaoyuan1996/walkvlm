import json
import os
import torch
import torch.distributed as dist
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GPT2LMHeadModel, GPT2Tokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from rouge_score import rouge_scorer
import numpy as np
import math
from collections import Counter
from tqdm import tqdm  
from qwen_vl_utils import process_vision_info
import requests
import base64
import time
import hmac
import hashlib
import io
from transformers import AutoConfig
from GPTScore import evaluate_image  
from accelerate import Accelerator  
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import re
from nltk.util import ngrams


accelerator = Accelerator()
appid = ""
appkey = ""
source = ""

def load_clip_model():
    """
    Load the pre-trained CLIP model and processor for encoding text.
    """
    model_path = ""
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    return model, processor

def encode_text_with_clip(text, model, processor):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs

def generate_ngrams(text, n=2):
    words = text.split()  
    ngrams_list = list(ngrams(words, n))  
    ngram_strings = [' '.join(ngram) for ngram in ngrams_list]  
    return ngram_strings

def calculate_keyword_density(text, keywords, model, processor, threshold=0.9):
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

model_clip, processor_clip = load_clip_model()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "", 
    torch_dtype="auto",
)

processor = AutoProcessor.from_pretrained(
    ""
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = accelerator.prepare(model)

dataset_prefix = ""  
dataset_path_val = ""

with open(dataset_prefix + dataset_path_val, 'r') as f:
    sat_dataset_val = [json.loads(line.strip()) for line in f]

user_message = "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. The prompt should:1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles).2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock).3. Provide specific details about objects (e.g., shape, size, material, distance).4. Include action suggestions (e.g., avoid, move forward, turn left).5. Prioritize safety and clarity. Output Prompt: "

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def calculate_rouge(reference, generated):
    scores = scorer.score(reference, generated)
    return scores


def generate_text_from_image(image_path, user_message, keywords):
    if not os.path.exists(image_path):
        print(image_path)
        return None  
    
    with Image.open(image_path).convert('RGB') as image:
        image_data = np.array(image)  
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_message}
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(accelerator.device)  

    generated_ids = model.generate(**inputs, max_new_tokens=1280)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

total_rouge1, total_rouge2, total_rougel = 0, 0, 0
total_keyword_density, total_perplexity, total_gpt_score = 0, 0, 0
num_samples = len(sat_dataset_val)

inference_results = []

for example in tqdm(sat_dataset_val, desc="Processing samples", ncols=100):
    if 'alter' not in example:
        print("no alter")
        continue 
    
    image_path = dataset_prefix + "wad_dataset/src_data/" + example["frame_path"] + "/8.jpg"
    reference = example["alter"]  
    keywords = example["keywords"]
    qwen_text = example["qwen72b_output"]
    
    generated_text = generate_text_from_image(image_path, user_message, keywords)
    print(generated_text)
    
    if generated_text is None:
        continue
    
    print("re:", reference, "ge:", generated_text)
    rouge_scores = calculate_rouge(reference, generated_text)
    total_rouge1 += rouge_scores['rouge1'].fmeasure
    total_rouge2 += rouge_scores['rouge2'].fmeasure
    total_rougel += rouge_scores['rougeL'].fmeasure
    
    keyword_density = calculate_keyword_density(generated_text, keywords, model_clip, processor_clip)
    total_keyword_density += keyword_density
    
    gpt_score = evaluate_image(appid, appkey, source, qwen_text, generated_text)
    total_gpt_score += gpt_score
    
    inference_results.append({
        "reference": reference,
        "generated_text": generated_text,
        "rouge_scores": rouge_scores,
        "keyword_density": keyword_density,
        "gpt_score": gpt_score  
    })

output_file = "inference_results_walkvlm.json"
with open(output_file, 'w') as f:
    json.dump(inference_results, f, ensure_ascii=False, indent=4)

print(f"Average ROUGE-1: {total_rouge1 / num_samples}")
print(f"Average ROUGE-2: {total_rouge2 / num_samples}")
print(f"Average ROUGE-L: {total_rougel / num_samples}")
print(f"Average Keyword Density: {total_keyword_density / num_samples}")
print(f"Average GPT Score: {total_gpt_score / num_samples}")  
