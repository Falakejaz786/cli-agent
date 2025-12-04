"""
Agent utility:
- If the asked question exactly matches an entry in `commandline_qa.json`, return the saved reference answer.
- If not, fall back to the model (base or LoRA adapter) and generate an answer.

Usage:
    python agent.py "How do I check if a variable is set in Bash?"
"""

import sys
import json
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "EleutherAI/gpt-neo-125M"
LORA_DIR = "./lora-gptneo"
DATA_FILE = "commandline_qa.json"


def load_reference_answer(question: str):
    """Try to find the reference answer in `commandline_qa.json`.
    First an exact match, then a case-insensitive substring match.
    """
    if not os.path.exists(DATA_FILE):
        return None
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    def normalize(q: str) -> str:
        import re
        if not q:
            return ""
        q = q.lower()
        q = re.sub(r"[^a-z0-9\s]", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    q_norm = normalize(question)
    for item in data:
        if normalize(item.get("question")) == q_norm:
            return item.get("answer")
    for item in data:
        if q_norm and q_norm in normalize(item.get("question")):
            return item.get("answer")
    try:
        import difflib
        all_questions = [normalize(item.get("question")) for item in data if item.get("question")]
        matches = difflib.get_close_matches(q_norm, all_questions, n=1, cutoff=0.6)
        if matches:
            for item in data:
                if normalize(item.get("question")) == matches[0]:
                    return item.get("answer")
    except Exception:
        pass
    return None


def load_model(adapter_path: str | None = None):
    tokenizer_source = adapter_path if (adapter_path and os.path.exists(os.path.join(adapter_path, "tokenizer.json"))) else (LORA_DIR if os.path.exists(os.path.join(LORA_DIR, "tokenizer.json")) else MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    model = base_model
    target_adapter = adapter_path or LORA_DIR
    if target_adapter and os.path.exists(target_adapter) and any(f.startswith("adapter") or f.startswith("checkpoint") for f in os.listdir(target_adapter)):
        try:
            model = PeftModel.from_pretrained(base_model, target_adapter)
        except Exception:
            model = base_model

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


def generate_answer(tokenizer, model, question: str):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
            num_beams=1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "", 1).strip()
    return ans


def modernize_git_commands(text: str) -> str:
        """Replace legacy git commands with modern equivalents for readability.
        Examples:
            - git branch <name> + git checkout <name> -> git switch -c <name>
            - git checkout -b <name> -> git switch -c <name>
        """
        import re
        if not text:
                return text
        lines = text.splitlines()
        def modernize_line(line: str) -> str:
            if "(alternative)" in line:
                return line
            l = re.sub(r"git\s+checkout\s+-b\s+<([^>]+)>", r"git switch -c <\1>", line)
            l = re.sub(r"git\s+branch\s+<([^>]+)>\s*(?:\\n|\s)*\d?\.\s*git\s+checkout\s+<\1>", r"git switch -c <\1>", l)
            l = re.sub(r"git\s+checkout\s+-b\s+([A-Za-z0-9/_-]+)", r"git switch -c \1", l)
            l = re.sub(r"git\s+branch\s+([A-Za-z0-9/_-]+)\s*(?:\\n|\s)*\d?\.\s*git\s+checkout\s+\1", r"git switch -c \1", l)
            return l
        lines = [modernize_line(ln) for ln in lines]
        text = "\n".join(lines)
        return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--use-reference", action="store_true", help="Return dataset reference answer if available instead of model output")
    parser.add_argument("--force-model", action="store_true", help="Ignore reference answers and force model generation")
    parser.add_argument("--no-modernize", action="store_true", help="Do not rewrite legacy git commands to modern equivalents")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter folder (overrides default)")
    args = parser.parse_args()
    question = args.question

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.ERROR)

    if not args.force_model:
        ref = load_reference_answer(question)
        if ref:
            ref_text = ref
            if not getattr(args, "no_modernize", False):
                ref_text = modernize_git_commands(ref_text)
            print(ref_text)
            sys.exit(0)
    if args.use_reference and not args.force_model:
        ref = load_reference_answer(question)
        if ref:
            print(ref)
            sys.exit(0)

    tokenizer, model = load_model(adapter_path=args.adapter)
    gen = generate_answer(tokenizer, model, question)
    if not args.no_modernize:
        gen = modernize_git_commands(gen)
    print(gen)
