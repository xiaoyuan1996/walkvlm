import base64
import io
import openai
import requests
from PIL import Image
from time import time
import hashlib
import hmac
import json


# 计算授权签名（用于GPT-4调用）
def calc_authorization(source, appkey):
    timestamp = str(int(time()))
    sign_str = f"x-timestamp: {timestamp}\nx-source: {source}"
    sign = hmac.new(appkey.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha256).digest()
    return sign.hex(), timestamp


# 调用Qwen72B API处理图像
def call_vlm(ip_port: str, model_name: str, system_prompt, user_prompt, user_img_bin):
    try:
        # 将图像转换为Base64
        b64_img = base64.b64encode(user_img_bin).decode("utf8")
        b64_img = f"data:image/image;base64,{b64_img}"
        base_url = f"http://{ip_port}/v1/"
        client = openai.OpenAI(api_key="test", base_url=base_url)
        
        # 调用Qwen72B模型
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": b64_img}},
                    {"type": "text", "text": user_prompt},
                ]}
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        raise e


# 调用GPT-4进行打分
def call_gpt4(input_text, appid, appkey, source):
    gpt4_prompt = f"""
    You are an evaluation model. Please rate the following sentence on a scale from 0 to 10, where 0 is very poor and 10 is excellent.
    Evaluation criteria: accuracy, fluency, clarity of expression.

    Sentence: "{input_text}"

    Please return only a numeric score, between 0 and 10.
    """
    
    auth, timestamp = calc_authorization(source, appkey)

    headers = {
        "X-AppID": appid,
        "X-Source": source,
        "X-Timestamp": timestamp,
        "X-Authorization": auth,
    }

    content = json.dumps({
        "model": "gpt-4-0613",
        "stream": False,
        "messages": [{"role": "user", "content": [{"type": "text", "text": gpt4_prompt}]}],
        "max_tokens": 50
    })

    response = requests.post("xxx.url", data=content, headers=headers)
    result = json.loads(response.text)
    return float(result['response'])


# 主函数：处理图像和参考文本，计算得分比值
def evaluate_image(appid, appkey, source, qwen_text, reference_text):
    try:
        # 调用GPT-4分别打分
        score_reference = call_gpt4(reference_text, appid, appkey, source)
        score_output = call_gpt4(qwen_text, appid, appkey, source)

        # print(f"参考文本得分: {score_reference}")
        # print(f"Qwen72B输出文本得分: {score_output}")

        # 计算得分比值
        if score_output != 0:
            score_ratio = score_reference / score_output
            return score_ratio  # 返回得分比值
        else:
            return None
    
    except Exception as e:
        return None


# 示例调用
if __name__ == "__main__":
    ip_port = ""  # Qwen72B API地址
    model_name = ""    # Qwen72B模型名称
    appid = ""
    appkey = "T"
    source = ""
    
    # 输入图像路径和参考文本
    img_path = ""
    reference_text = "This is a picture describing a beach with some people surfing in the water."

    evaluate_image(appid, appkey, source, reference_text, reference_text)
