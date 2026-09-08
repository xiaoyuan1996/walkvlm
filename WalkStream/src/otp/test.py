import argparse
import json
import re
import sys
from pathlib import Path

import torch

try:
    from nltk.util import ngrams
except ModuleNotFoundError:  # pragma: no cover - allows `--help` without full runtime deps
    ngrams = None

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - allows `--help` without full runtime deps
    Image = None

try:
    from qwen_vl_utils import process_vision_info
except ModuleNotFoundError:  # pragma: no cover - allows `--help` without full runtime deps
    process_vision_info = None

try:
    from rouge_score import rouge_scorer
except ModuleNotFoundError:  # pragma: no cover - allows `--help` without full runtime deps
    rouge_scorer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:  # pragma: no cover - allows `--help` without full runtime deps
    cosine_similarity = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - allows `--help` without full runtime deps
    tqdm = None

try:
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, Qwen2VLForConditionalGeneration
except ModuleNotFoundError:  # pragma: no cover - allows `--help` without full runtime deps
    AutoProcessor = None
    CLIPModel = None
    CLIPProcessor = None
    Qwen2VLForConditionalGeneration = None

CURRENT_DIR = Path(__file__).resolve().parent
INFER_DIR = CURRENT_DIR.parent / "infer"
if str(INFER_DIR) not in sys.path:
    sys.path.append(str(INFER_DIR))

try:
    from GPTScore import evaluate_image  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - optional GPT scoring
    evaluate_image = None


DEFAULT_USER_MESSAGE = (
    "Given the input image, generate a clear and concise navigation prompt for visually impaired individuals. "
    "The prompt should: 1. Identify key elements in the environment (e.g., obstacles, landmarks, pedestrians, vehicles). "
    "2. Use clear directional terms (e.g., left, right, front, or clock directions like 1 o'clock). "
    "3. Provide specific details about objects (e.g., shape, size, material, distance). "
    "4. Include action suggestions (e.g., avoid, move forward, turn left). "
    "5. Prioritize safety and clarity. Output Prompt:"
)
EN_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "the", "to", "with", "this", "that", "there", "here", "near", "ahead",
}


def require_runtime_dependency(module_name, imported_symbol, install_name=None):
    if imported_symbol is None:
        package_name = install_name or module_name
        raise ModuleNotFoundError(
            f"Missing optional dependency '{package_name}'. Install it before running evaluation."
        )
    return imported_symbol


def load_annotation_records(annotation_path: Path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "annotations", "samples", "test", "eval"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported annotation format: {annotation_path}")


def unique_items(items):
    results = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            results.append(text)
    return results


def normalize_keywords(value):
    if value is None:
        return []
    if isinstance(value, list):
        merged = []
        for item in value:
            merged.extend(normalize_keywords(item))
        return unique_items(merged)

    text = str(value).strip()
    if not text:
        return []

    split_tokens = unique_items(re.split(r"[#;,，；、/|\n\t]+", text))
    if len(split_tokens) > 1:
        return split_tokens[:12]

    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    en_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text)
        if len(token) > 2 and token.lower() not in EN_STOPWORDS
    ]
    tokens = unique_items(zh_tokens + en_tokens)
    return tokens[:12] if tokens else [text]


def extract_reference(example):
    for key in ("reminder", "alter", "answer", "caption"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_keywords(example):
    for key in ("keywords", "keyword", "tags", "tag"):
        keywords = normalize_keywords(example.get(key))
        if keywords:
            return keywords
    reference = extract_reference(example)
    return normalize_keywords(reference)


def resolve_image_path(frame_path: str, dataset_root: Path, image_root: Path | None):
    frame_rel = Path(str(frame_path).strip("/"))
    candidates = []
    if image_root:
        candidates.append(image_root)
    candidates.extend(
        [
            dataset_root,
            dataset_root / "src_data",
            dataset_root / "src_data4",
            dataset_root / "test_data",
            dataset_root / "test_data4",
            dataset_root / "keyframes",
            dataset_root.parent / "wad_dataset" / "src_data",
            dataset_root.parent / "wad_dataset" / "src_data4",
        ]
    )

    for base in unique_items(candidates):
        base = Path(base)
        for ext in ("jpg", "jpeg", "png"):
            candidate = base / frame_rel / f"8.{ext}"
            if candidate.exists():
                return candidate
    return None


def load_generation_model(model_name_or_path, device):
    generator_cls = require_runtime_dependency(
        "transformers", Qwen2VLForConditionalGeneration, "transformers"
    )
    processor_cls = require_runtime_dependency("transformers", AutoProcessor, "transformers")
    model = generator_cls.from_pretrained(model_name_or_path, torch_dtype="auto")
    model = model.to(device)
    model.eval()
    processor = processor_cls.from_pretrained(model_name_or_path)
    return model, processor


def load_clip_model(model_name_or_path, device):
    clip_processor_cls = require_runtime_dependency("transformers", CLIPProcessor, "transformers")
    clip_model_cls = require_runtime_dependency("transformers", CLIPModel, "transformers")
    processor = clip_processor_cls.from_pretrained(model_name_or_path)
    model = clip_model_cls.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return model, processor


def encode_text_with_clip(text, model, processor, device):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        return model.get_text_features(**inputs)


def generate_ngrams(text, n=2):
    ngram_builder = require_runtime_dependency("nltk", ngrams, "nltk")
    words = text.split()
    return [" ".join(gram) for gram in ngram_builder(words, n)]


def calculate_keyword_density(text, keywords, model, processor, device, threshold=0.9):
    similarity_fn = require_runtime_dependency("scikit-learn", cosine_similarity, "scikit-learn")
    if not keywords:
        return 0.0
    text = re.sub(r"[^\w\s]", "", text.lower())
    word_tokens = text.split()
    unique_word_tokens = set(word_tokens)
    if not unique_word_tokens:
        return 0.0

    all_tokens = unique_word_tokens.union(set(generate_ngrams(text, 2)))
    total_tokens = len(all_tokens)
    keyword_count = 0

    keyword_embeddings = [encode_text_with_clip(keyword, model, processor, device) for keyword in keywords]
    for token in all_tokens:
        token_embedding = encode_text_with_clip(token, model, processor, device)
        for keyword_embedding in keyword_embeddings:
            similarity = similarity_fn(token_embedding.cpu().numpy(), keyword_embedding.cpu().numpy())[0][0]
            if similarity >= threshold:
                keyword_count += 1
                break
    return keyword_count / total_tokens


def generate_text_from_image(image_path, user_message, model, processor, device, max_new_tokens):
    process_vision = require_runtime_dependency("qwen_vl_utils", process_vision_info, "qwen-vl-utils")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_message},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision(messages)
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


def maybe_compute_gpt_score(args, qwen_text, generated_text):
    if not (args.gpt_appid and args.gpt_appkey and args.gpt_source and qwen_text):
        return None
    score_fn = require_runtime_dependency("GPTScore", evaluate_image, "openai")
    return score_fn(args.gpt_appid, args.gpt_appkey, args.gpt_source, qwen_text, generated_text)


def run_evaluation(args):
    annotation_path = Path(args.dataset_path).expanduser().resolve()
    if not annotation_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {annotation_path}")

    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else annotation_path.parent
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    progress = require_runtime_dependency("tqdm", tqdm, "tqdm")
    scorer_factory = require_runtime_dependency("rouge-score", rouge_scorer, "rouge-score")
    generation_model, processor = load_generation_model(args.model_name_or_path, device)
    clip_model, clip_processor = load_clip_model(args.clip_model_name_or_path, device)
    records = load_annotation_records(annotation_path)
    if args.max_samples:
        records = records[: args.max_samples]

    scorer = scorer_factory.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    processed = 0
    total_rouge1 = total_rouge2 = total_rougel = 0.0
    total_keyword_density = total_gpt_score = 0.0
    gpt_score_count = 0
    results = []

    for example in progress(records, desc="Evaluating", ncols=100):
        reference = extract_reference(example)
        frame_path = example.get("frame_path")
        if reference is None or frame_path is None:
            continue

        image_path = resolve_image_path(frame_path, dataset_root, image_root)
        if image_path is None:
            continue

        generated_text = generate_text_from_image(
            image_path,
            args.user_message,
            generation_model,
            processor,
            device,
            args.max_new_tokens,
        )
        rouge_scores = scorer.score(reference, generated_text)
        keywords = extract_keywords(example)
        keyword_density = calculate_keyword_density(
            generated_text,
            keywords,
            clip_model,
            clip_processor,
            device,
        )
        gpt_score = maybe_compute_gpt_score(args, example.get("qwen72b_output"), generated_text)

        processed += 1
        total_rouge1 += rouge_scores["rouge1"].fmeasure
        total_rouge2 += rouge_scores["rouge2"].fmeasure
        total_rougel += rouge_scores["rougeL"].fmeasure
        total_keyword_density += keyword_density
        if gpt_score is not None:
            total_gpt_score += gpt_score
            gpt_score_count += 1

        results.append(
            {
                "frame_path": frame_path,
                "reference": reference,
                "generated_text": generated_text,
                "keywords": keywords,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
                "keyword_density": keyword_density,
                "gpt_score": gpt_score,
            }
        )

    if processed == 0:
        raise RuntimeError("No valid evaluation samples were found. Please check dataset_path/image_root.")

    summary = {
        "num_samples": processed,
        "average_rouge1": total_rouge1 / processed,
        "average_rouge2": total_rouge2 / processed,
        "average_rougeL": total_rougel / processed,
        "average_keyword_density": total_keyword_density / processed,
        "average_gpt_score": (total_gpt_score / gpt_score_count) if gpt_score_count else None,
        "results": results,
    }

    output_path = Path(args.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WalkStream reminder generation on a JSON/JSONL dataset.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True, help="Path to a test/eval JSON or JSONL annotation file.")
    parser.add_argument("--dataset_root", default=None, help="Optional dataset root for resolving frame_path.")
    parser.add_argument("--image_root", default=None, help="Optional explicit image root that contains frame folders.")
    parser.add_argument("--clip_model_name_or_path", default="openai/clip-vit-base-patch16")
    parser.add_argument("--output_file", default="inference_results_walkstream.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--user_message", default=DEFAULT_USER_MESSAGE)
    parser.add_argument("--gpt_appid", default=None)
    parser.add_argument("--gpt_appkey", default=None)
    parser.add_argument("--gpt_source", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
