from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import gradio as gr
import os

# 全局加载模型和处理器（只加载一次）
model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        model_dir = "/mnt/nanjing3cephfs/wx-mm-spr-xxxx/yuanzhiqiang/code/open_r1_all/VisualThinker-R1-Zero/src/open-r1-multimodal/outputs/Qwen2-VL-2B-GRPO-Base-QUERY_GENE_think/checkpoint-1000"
        
        # 配置动态分辨率
        # min_pixels = 256 * 28 * 28
        # max_pixels = 1280 * 28 * 28
        
        # 加载模型
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16, 
            use_flash_attention_2=True
        ).eval()
        
        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            model_dir,
            # min_pixels=min_pixels,
            # max_pixels=max_pixels
        )

# 预加载模型
load_model()

def analyze_image(image):
    """处理上传的图片并返回分析结果"""
    try:
        # 保存临时图片文件
        temp_path = "/tmp/gradio_temp_image.png"
        image.save(temp_path)
        
        # 固定提示词模板
        prompt = """给这张图片打出中文标签。 
        使用 <think>  这样的格式输出思考过程，并使用 <answer> </answer> 来输出最终结果，
        输出示例： <think> 可能包含「keyword1」和「keyword2」的标签，原因是...   
        <answer> #keyword1#keyword2 </answer>。 
        这张图像的思考过程和标签为: <think> """
        
        # 执行推理
        result = qwen2_vl_inference(temp_path, prompt)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return result
        
    except Exception as e:
        return f"处理出错：{str(e)}"

def qwen2_vl_inference(image_path, prompt):
    print(image_path, prompt)
    """包装原有推理逻辑"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = '<|vision_start|><|image_pad|><|vision_end|>' + prompt
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=False,
        temperature=0.8,
        num_beams=1,
        top_p=0.9
    )
    
    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return response

# 创建Gradio界面
demo = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil", label="上传图片"),
    outputs=gr.Textbox(label="分析结果", lines=10),
    title="Qwen-VL图像分析系统",
    description="上传图片获取视觉语言模型的分析结果",
    examples=[
        os.path.join(os.path.dirname(__file__), "-753023143.png")
    ]
)

# 配置服务器设置
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=True
    )