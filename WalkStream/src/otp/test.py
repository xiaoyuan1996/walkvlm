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
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
from qwen_vl_utils import process_vision_info
import requests
import base64
import time
import hmac
import hashlib
import io
from transformers import AutoConfig
from GPTScore import evaluate_image  # 引入 GPT-4 打分函数
from accelerate import Accelerator  # 导入accelerate
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import re
from nltk.util import ngrams


accelerator = Accelerator()
appid = ""
appkey = ""
source = ""

# 加载 CLIP 模型和处理器
def load_clip_model():
    """
    Load the pre-trained CLIP model and processor for encoding text.
    """
    model_path = ""
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    return model, processor

# 对输入文本进行编码并返回其嵌入
def encode_text_with_clip(text, model, processor):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs

# 生成 2-gram
def generate_ngrams(text, n=2):
    words = text.split()  # 拆分成单词
    ngrams_list = list(ngrams(words, n))  # 生成2-gram
    ngram_strings = [' '.join(ngram) for ngram in ngrams_list]  # 转为字符串形式
    return ngram_strings

# 计算关键词密度（包括2-gram）
def calculate_keyword_density(text, keywords, model, processor, threshold=0.9):
    # 预处理文本，去除标点符号并转换为小写
    text = re.sub(r'[^\w\s]', '', text.lower())  # 转小写并去除标点
    word_tokens = text.split()  # 将文本拆分为单词
    unique_word_tokens = set(word_tokens)  # 获取唯一单词
    total_unique_words = len(unique_word_tokens)  # 计算总唯一单词数
    
    if total_unique_words == 0:
        return 0.0  # 如果文本没有单词，返回0
    
    ngrams_list = generate_ngrams(text, 2)
    
    all_tokens = unique_word_tokens.union(set(ngrams_list))
    total_tokens = len(all_tokens)
    
    # 计算与每个关键词的相似度
    keyword_count = 0
    
    for token in all_tokens:
        token_embedding = encode_text_with_clip(token, model, processor)
        
        # 对每个关键词计算与当前token的相似度
        for keyword in keywords:
            # 对关键词进行编码
            keyword_embedding = encode_text_with_clip(keyword, model, processor)
            
            # 计算余弦相似度
            similarity = cosine_similarity(token_embedding.detach().numpy(), keyword_embedding.detach().numpy())[0][0]
            
            # 如果相似度超过阈值，认为该token是关键词
            if similarity >= threshold:
                keyword_count += 1
                break  # 找到一个相似的关键词，跳出循环
    
    # 计算关键词密度：匹配的关键词数量 / 总的唯一单词数
    keyword_density = keyword_count / total_tokens
    return keyword_density

# 加载模型和处理器
model_clip, processor_clip = load_clip_model()

# 加载视觉语言模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "", 
    torch_dtype="auto",
)

processor = AutoProcessor.from_pretrained(
    ""
)


# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 将模型转移到GPU（如果可用）
model = accelerator.prepare(model)

# 读取数据
dataset_prefix = ""  # 修改为实际路径
dataset_path_val = ""

with open(dataset_prefix + dataset_path_val, 'r') as f:
    sat_dataset_val = [json.loads(line.strip()) for line in f]

# 用户消息模板
user_message = "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. The prompt should:1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles).2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock).3. Provide specific details about objects (e.g., shape, size, material, distance).4. Include action suggestions (e.g., avoid, move forward, turn left).5. Prioritize safety and clarity. Output Prompt: "

# ROUGE评分计算器
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def calculate_rouge(reference, generated):
    scores = scorer.score(reference, generated)
    return scores

# gpt score

def generate_text_from_image(image_path, user_message, keywords):
    # 检查图像路径是否存在
    if not os.path.exists(image_path):
        print(image_path)
        return None  # 跳过该样本
    
    # 加载图像
    with Image.open(image_path).convert('RGB') as image:
        image_data = np.array(image)  # 将图像转换为数组，假设模型接收此输入格式
    
    # 创建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_message}
            ],
        }
    ]
    
    # 处理图像输入
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
    inputs = inputs.to(accelerator.device)  # 将输入数据转移到GPU

    # 推理生成输出文本
    generated_ids = model.generate(**inputs, max_new_tokens=1280)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

# 计算所有样本的平均指标
total_rouge1, total_rouge2, total_rougel = 0, 0, 0
total_keyword_density, total_perplexity, total_gpt_score = 0, 0, 0
num_samples = len(sat_dataset_val)

inference_results = []

# 使用 tqdm 显示进度条
for example in tqdm(sat_dataset_val, desc="Processing samples", ncols=100):
    # 检查是否存在alter字段
    if 'alter' not in example:
        print("no alter")
        continue  # 如果没有alter字段，跳过该样本
    
    image_path = dataset_prefix + "wad_dataset/src_data/" + example["frame_path"] + "/8.jpg"
    reference = example["alter"]  # 标注文本
    keywords = example["keywords"]
    qwen_text = example["qwen72b_output"]
    
    # 生成模型的输出文本
    generated_text = generate_text_from_image(image_path, user_message, keywords)
    print(generated_text)
    
    # 如果没有生成文本（图片不存在或其他问题），跳过该样本
    if generated_text is None:
        continue
    
    # 计算ROUGE评分
    print("re:", reference, "ge:", generated_text)
    rouge_scores = calculate_rouge(reference, generated_text)
    total_rouge1 += rouge_scores['rouge1'].fmeasure
    total_rouge2 += rouge_scores['rouge2'].fmeasure
    total_rougel += rouge_scores['rougeL'].fmeasure
    
    # 计算关键词密度
    keyword_density = calculate_keyword_density(generated_text, keywords, model_clip, processor_clip)
    total_keyword_density += keyword_density
    
    # 调用 GPT-4 进行打分 (GPT Score)
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

# 输出平均指标
print(f"Average ROUGE-1: {total_rouge1 / num_samples}")
print(f"Average ROUGE-2: {total_rouge2 / num_samples}")
print(f"Average ROUGE-L: {total_rougel / num_samples}")
print(f"Average Keyword Density: {total_keyword_density / num_samples}")
print(f"Average GPT Score: {total_gpt_score / num_samples}")  
