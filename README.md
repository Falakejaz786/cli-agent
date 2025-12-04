# 🧠 StackOverflow Command-Line Support Agent

A hybrid AI agent that answers command-line related questions using **dataset lookup first** and **LLM generation as fallback**, trained using **LoRA-fine-tuning on GPT-Neo**.

This tool behaves like a real IT support assistant:

✔ If the answer exists → return verified response
✔ Otherwise → generate answer using fine-tuned model

---

## ⭐ Project Highlights

🚀 Hybrid approach (Search-then-Generate)
🧠 Fine-tuned GPT-Neo using LoRA (lightweight training)
🔍 Fast semantic lookup from dataset
📊 Evaluation performed using BLEU & ROUGE-L
🖥 CLI interface for real workflow usage
⚡ Runs locally on CPU

This makes it ideal for **training chatbots, internal knowledge systems, developer support bots, and intelligent assistants**.

---

# 📌 Real-World Problem Solved

Organizations repeatedly face the same technical queries:

> “How to create a branch?”
> “How to compress a folder?”
> “How to install curl on Ubuntu?”

Typical workflow today:

🧑‍💻 Engineer Googles / searches past tickets
⌛ Wastes time
❌ Inconsistent answers

This project replaces that process with automation:

### 💡 Known → Accurate

### 💬 Unknown → Generated

### 📌 Missing Data → Can be added

This is how real intelligent systems evolve.

---

# 📁 Project Structure

```
├── stackoverflow_lora.ipynb     # Fine-tuning notebook
├── agent.py                     # CLI answering agent
├── evaluate.py                  # Model performance evaluator
├── commandline_qa.json          # Dataset used for lookup
├── lora-gptneo/                 # Fine-tuned adapter weights
└── README.md
```

---

# 🛠️ Installation & Setup

### 1️⃣ Clone Repo

```bash
git clone <repo-link>
cd StackoverflowQ&A
```

---

### 2️⃣ Create Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Make Sure `lora-gptneo/` Exists

It must contain:

✔ adapter_model.bin
✔ adapter_config.json
✔ tokenizer files

---

# ▶️ Running the Agent

### Ask a question:

```bash
python agent.py --question "How to create and switch to a new git branch?"
```

Example output:

```
1. git switch -c <branch_name>
2. (alternative) git checkout -b <branch_name>
```

### Another example:

```bash
python agent.py --question "How do I check if a variable is set in Bash?"
```

Output:

```
if [ -z ${var+x} ]; then echo "var is unset"; else echo "var is set"; fi
```

---

# 🧠 How It Works Internally

### STEP 1: Normalize user query

→ lowercase
→ remove punctuation

### STEP 2: Search dataset

```
commandline_qa.json
```

If exact or fuzzy match found → return verified answer.

### STEP 3: If not found → Model inference

* Loads `EleutherAI/gpt-neo-125M`
* Merges LoRA adapter weights
* Generates answer

This ensures:

🟢 Correct responses when already known
🤖 AI-generated fallback when unknown

---

# 📊 Evaluation and Metrics

To evaluate model performance:

```bash
python evaluate.py --adapter lora-gptneo --num 3
```

And for comparison baseline:

```bash
python evaluate.py --num 3

---

# 🌍 Applications

### 🏢 1. Internal Developer Support Bot

Automates repeated DevOps queries.

### 🧑‍🎓 2. Learning Assistant

Beginner asks:

> "How to remove a directory?"

Agent replies:

```
rm -rf <folder>
```

### 🚀 3. Onboarding Tool For New Engineers

Interns do not need documentation.

### 🏗 4. Knowledge Base Builder

Unknown query = new dataset entry

Knowledge grows over time.

### 🤖 5. AI-ready Extensions

* FastAPI APIs
* Slack integration
* Browser plugins
* VS Code extension

---

# 🔮 Future Enhancements

✨ Vector-based semantic search
✨ Auto-update dataset from unknown responses
✨ Web UI using Streamlit/React
✨ Full evaluation dashboard
✨ Logging and versioning

---

# 🙌 Contributions Welcome

Steps to contribute:

```bash
git checkout -b new-feature
git commit -m "Improvement"
git push origin new-feature
```

---

# 🏁 Final Notes

This project demonstrates:

✔ Data-driven answer retrieval
✔ Lightweight LoRA fine-tuning
✔ Real-time inference pipeline
✔ Proper evaluation metrics
✔ Fully usable command-line interface

This is a complete real-world ML system—**from training → inference → evaluation → utility**.

---
