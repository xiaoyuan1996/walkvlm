from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import json
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import os
from EAD import VisionDangerClassification  

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    else:
        return obj  

def load_images_from_folder(folder, time_points):
    images = {}
    for time_point in time_points:
        image_files = [
            os.path.join(folder, f"{time_point:.2f}_{i+1}.jpg") for i in range(3)
        ]
        images[time_point] = [Image.open(image_file) for image_file in image_files if os.path.exists(image_file)]
    return images

def save_results_to_json(results, output_file=""):
    results = convert_ndarray_to_list(results)  
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def get_time_points_from_folder(folder):
    time_points = set()
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            time_point = filename.split('_')[0]
            time_points.add(float(time_point))
    
    return sorted(list(time_points))

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  
        new_state_dict[name] = v
    
    return new_state_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "checkpoint_path", 
    torch_dtype="auto", 
).to(device)
processor = AutoProcessor.from_pretrained(
    "checkpoint_path"
)

regression_head_model = VisionDangerClassification(input_dim=1536, num_classes=3)
checkpoint_path = "checkpoint.pth"  
checkpoint = torch.load(checkpoint_path)
checkpoint['model_state_dict'] = remove_module_prefix(checkpoint['model_state_dict'])

regression_head_model.load_state_dict(checkpoint['model_state_dict'])
regression_head_model = regression_head_model.to(device)  

regression_head_model = nn.DataParallel(regression_head_model, device_ids=list(range(torch.cuda.device_count())))
model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))  

def generate_text_from_image(image_path, user_message, model, processor):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None  
    
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
    inputs = inputs.to(device) 

    generated_ids = model.module.generate(**inputs, max_new_tokens=1280)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    target_size = (544, 960) 
    
    if image.size[0] > image.size[1]:  
        image = image.rotate(90, expand=True)
    else:
        image = image.resize(target_size)

    return image

def infer_batch(image_paths):
    images = [preprocess_image(image_path) for image_path in image_paths]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
        for image in images
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

 
    inputs = {key: value.to(device) for key, value in inputs.items()}

    image_grid_thw = inputs['image_grid_thw']
    grid_thw = inputs.get('image_grid_thw', None)
    if grid_thw is None:
        raise ValueError("grid_thw parameter is required for the visual model.")

    visual_model = model.module.visual  
    visual_features_list = []
    with torch.no_grad():
        for i in range(len(images)):
            a=int(inputs['pixel_values'].shape[0]/3)
            image_feature = visual_model(hidden_states=inputs['pixel_values'][i*a:i*a+a], grid_thw=grid_thw[i:i+1])
            visual_features_list.append(image_feature.unsqueeze(0))  
    visual_features = torch.cat(visual_features_list, dim=0)

    visual_features = visual_features.to(torch.float32)

    logits = regression_head_model(visual_features)  
    predicted_classes = torch.argmax(logits, dim=1)  

    label_encoder = LabelEncoder()
    label_encoder.fit(["A", "B", "C"])
    predicted_labels = label_encoder.inverse_transform(predicted_classes.tolist())

    return predicted_labels

def infer_with_check(image_paths, predictions, model, processor):
    if 'C' in predictions:
        image_path = image_paths[-1]  
        user_message = "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. The prompt should:1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles).2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock).3. Provide specific details about objects (e.g., shape, size, material, distance).4. Include action suggestions (e.g., avoid, move forward, turn left).5. Prioritize safety and clarity. "
        generated_text = generate_text_from_image(image_path, user_message, model, processor)
        return generated_text
    else:
        return "No reminder needed"  
    
def infer_with_check_other_model(image_paths,model, processor):
    image_path = image_paths[-1]  
    user_message = "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. The prompt should:1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles).2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock).3. Provide specific details about objects (e.g., shape, size, material, distance).4. Include action suggestions (e.g., avoid, move forward, turn left).5. Prioritize safety and clarity."  
    generated_text = generate_text_from_image(image_path, user_message, model, processor)
    return generated_text

image_paths = [
    "",
    "",
    ""
]  
predictions = infer_batch(image_paths)
result = infer_with_check(image_paths, predictions, model, processor)
print(result) 
