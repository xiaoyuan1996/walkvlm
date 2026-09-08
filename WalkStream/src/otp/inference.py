import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from otp_model import VisionDangerClassification


LABELS = ["A", "B", "C"]
DEFAULT_USER_MESSAGE = (
    "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. "
    "The prompt should: 1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles). "
    "2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock). "
    "3. Provide specific details about objects (e.g., shape, size, material, distance). "
    "4. Include action suggestions (e.g., avoid, move forward, turn left). "
    "5. Prioritize safety and clarity."
)


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    return obj


def remove_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    target_size = (544, 960)
    if image.size[0] > image.size[1]:
        image = image.rotate(90, expand=True)
    return image.resize(target_size)


def load_generator(model_name_or_path, device):
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype="auto")
    model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return model, processor


def load_classifier(classifier_checkpoint, device):
    if not classifier_checkpoint:
        return None

    checkpoint_path = Path(classifier_checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = remove_module_prefix(state_dict)

    model = VisionDangerClassification(input_dim=1536, num_classes=3)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def generate_text_from_image(image_path, user_message, model, processor, device, max_new_tokens):
    image_path = str(Path(image_path).expanduser().resolve())
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_message},
            ],
        }
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
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def extract_visual_feature(image_path, model, processor, device):
    image = preprocess_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
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
    inputs = inputs.to(device)

    with torch.no_grad():
        feature = model.visual(hidden_states=inputs["pixel_values"], grid_thw=inputs.get("image_grid_thw"))

    if feature.dim() == 3:
        feature = feature.squeeze(0)
    return feature.detach().to(torch.float32)


def predict_trigger_labels(image_paths, classifier_model, generator_model, processor, device):
    features = [extract_visual_feature(image_path, generator_model, processor, device) for image_path in image_paths]
    max_len = max(feature.size(0) for feature in features)
    hidden_size = features[0].size(1)
    padded = torch.zeros(len(features), max_len, hidden_size, dtype=torch.float32, device=device)
    for idx, feature in enumerate(features):
        padded[idx, : feature.size(0)] = feature.to(device)

    with torch.no_grad():
        logits = classifier_model(padded)
        predicted_classes = torch.argmax(logits, dim=1).tolist()
    return [LABELS[pred] for pred in predicted_classes]


def run_inference(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    image_paths = [str(Path(path).expanduser().resolve()) for path in args.image_paths]
    for image_path in image_paths:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    generator_model, processor = load_generator(args.model_name_or_path, device)
    classifier_model = load_classifier(args.classifier_checkpoint, device)

    predicted_labels = []
    should_generate = args.always_generate or classifier_model is None
    if classifier_model is not None:
        predicted_labels = predict_trigger_labels(image_paths, classifier_model, generator_model, processor, device)
        should_generate = should_generate or ("C" in predicted_labels)

    reminder = "No reminder needed"
    if should_generate:
        reminder = generate_text_from_image(
            image_paths[-1],
            args.user_message,
            generator_model,
            processor,
            device,
            args.max_new_tokens,
        )

    result = {
        "image_paths": image_paths,
        "predicted_labels": predicted_labels,
        "should_generate": should_generate,
        "reminder": reminder,
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_ndarray_to_list(result), f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run WalkStream OTP inference on 1-3 images.")
    parser.add_argument("--model_name_or_path", required=True, help="Path or Hugging Face ID of the VLM checkpoint.")
    parser.add_argument("--image_paths", nargs="+", required=True, help="One to three image paths ordered by time.")
    parser.add_argument("--classifier_checkpoint", default=None, help="Optional OTP classifier checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--always_generate", action="store_true", help="Generate reminder text even without OTP gating.")
    parser.add_argument("--user_message", default=DEFAULT_USER_MESSAGE)
    parser.add_argument("--output_json", default=None, help="Optional path to save the inference result JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
