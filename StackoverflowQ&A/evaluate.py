"""
Simple evaluation script for the LoRA-tuned GPT-Neo model.
- Loads the tokenizer and base model
- Loads any LoRA adapter from `lora-gptneo` if present
- Runs generation for a few samples from `commandline_qa.json` and prints outputs

Usage:
    python evaluate.py --num 5
"""

import argparse
import json
import os
import math
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    import evaluate as hf_evaluate
except Exception:
    hf_evaluate = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
import nltk

MODEL_NAME = "EleutherAI/gpt-neo-125M"
LORA_DIR = "./lora-gptneo"


def load_model_and_tokenizer(model_name: str = MODEL_NAME, adapter_dir: str | None = None):
    tokenizer_source = adapter_dir if (adapter_dir and os.path.exists(os.path.join(adapter_dir, "tokenizer.json"))) else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = base_model
    if adapter_dir and os.path.exists(adapter_dir) and any(f.startswith("adapter") or f.startswith("checkpoint") for f in os.listdir(adapter_dir)):
        try:
            model = PeftModel.from_pretrained(base_model, adapter_dir)
        except Exception:
            model = base_model

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def generate_answer(tokenizer, model, question, max_new_tokens=128):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=3, pad_token_id=tokenizer.eos_token_id
        )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ans = ans.replace(prompt, "", 1).strip()
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=0, help="Number of examples to run; 0 means all")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter folder")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Base model to load for generation")
    args = parser.parse_args()

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)
 
    using_adapter = bool(args.adapter)
    if using_adapter:
        if not os.path.exists(args.adapter):
            print(f"Adapter {args.adapter} not found. Falling back to base model: {args.model}")
            args.adapter = None
            using_adapter = False
        else:
            print(f"Evaluating using base model: {args.model} + adapter: {args.adapter}")
    else:
        print(f"Evaluating using base model: {args.model} (no adapter)")

    tokenizer, model = load_model_and_tokenizer(model_name=args.model, adapter_dir=args.adapter or None)

    with open("commandline_qa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    num_examples = args.num if args.num > 0 else len(data)

    references = []
    hypotheses = []

    for i, item in enumerate(data[:num_examples]):
        question = item.get("question")
        answer = item.get("answer")
        gen = generate_answer(tokenizer, model, question)
        references.append([answer.split()])
        hypotheses.append(gen.split())

    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        smoothie = SmoothingFunction().method1
        bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    except Exception as e:
        print("Could not compute BLEU (nltk not installed or error):", e)

    def normalize_text(s):
        return " ".join(s.lower().strip().split())

    try:
        if schre := rouge_scorer:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = [scorer.score(' '.join(ref[0]), ' '.join(hyp))['rougeL'].fmeasure for ref, hyp in zip(references, hypotheses)]
            rouge_l = sum(scores) / len(scores) if len(scores) > 0 else 0.0
            print(f"BLEU: {bleu_score:.4f}")
            print(f"ROUGE-L: {rouge_l:.4f}")
        elif hf_evaluate is not None:
            rouge = hf_evaluate.load('rouge')
            refs = [' '.join(ref[0]) for ref in references]
            hyps = [' '.join(hyp) for hyp in hypotheses]
            rouge_result = rouge.compute(predictions=hyps, references=refs)
            r_l = rouge_result.get('rougeL', None)
            if isinstance(r_l, dict):
                print(f"BLEU: {bleu_score:.4f}")
                if 'rougeL' in rouge_result:
                    print(f"ROUGE-L: {rouge_result['rougeL']}")
            else:
                print(f"BLEU: {bleu_score:.4f}")
                print(f"ROUGE-L: {rouge_result}")
        else:
            print(f"BLEU: {bleu_score:.4f}")
            print("ROUGE-L: 0.0000")
    except Exception as e:
        print('Error computing ROUGE:', e)


if __name__ == "__main__":
    main()
