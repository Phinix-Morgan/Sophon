# Sophon

**A 40M-parameter decoder-only language model trained from scratch on everyday conversation.**

Sophon is a compact GPT-style transformer trained end-to-end on ~4.5M tokens of
dialogue from DailyDialog and PersonaChat. It learns *how people talk* — greetings,
small talk, turn-taking, casual phrasing — at a scale that fits on a single 6 GB
consumer GPU or a free Colab T4.

> **Scope, up front:** Sophon is a research/educational chit-chat model, not an
> assistant. It has no world knowledge, cannot follow instructions, and should not
> be deployed as a user-facing product. See [Limitations](#limitations).

| | |
|---|---|
| **Parameters** | 39,980,032 (≈40M), weight-tied embeddings |
| **Architecture** | Decoder-only transformer, pre-LN, GELU, learned positions |
| **Context** | 512 tokens |
| **Vocabulary** | 16,002 (16k ByteLevel BPE + 2 chat role tokens) |
| **Training data** | 100,000 per-turn samples from DailyDialog + PersonaChat |
| **Best validation loss** | **3.5926** (epoch 8 of 12) → perplexity **36.33** |
| **Hardware** | Single Colab T4 / RTX 3050 6 GB |
| **Weights** | Not in this repository — see [Checkpoints](#checkpoints) |

---

## Contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Chat format & tokenizer](#chat-format--tokenizer)
- [Checkpoints](#checkpoints)
- [HTTP API](#http-api)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Architecture

A minimal, from-scratch GPT implementation (no `transformers` model classes, no
pretrained initialization):

```
input ids ──► token embedding ──┐
                                ├─► + ──► dropout ──► 10 × TransformerBlock ──► LayerNorm ──► lm_head ──► logits
positions ──► position embedding┘
```

| Component | Configuration |
|---|---|
| Layers | 10 |
| Hidden size (`d_model`) | 512 |
| Attention heads | 8 (head dim 64) |
| Feed-forward width (`d_ff`) | 2048 |
| Dropout | 0.1 |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Positional encoding | Learned, max 512 |
| Output head | Linear, **tied** to token embedding |

**Parameter breakdown**

| Component | Parameters |
|---|---|
| Token embedding (tied with `lm_head`) | 8.19M |
| Position embedding | 0.26M |
| 10 transformer blocks | 31.5M |
| **Total** | **39.98M** |

Implementation lives in two places that share an identical state-dict layout:

- `Sophon/Sophon.ipynb` — training + evaluation code
- `Sophon/Sophon_chat_ui.py` — self-contained inference server (no training imports)

---

## Dataset

Two open dialogue corpora, normalized, deduplicated, and split by conversation.

| Source | Train conversations | Val conversations | Train text tokens |
|---|---:|---:|---:|
| DailyDialog | 10,455 | 994 | 1,194,405 |
| PersonaChat | 17,878 | 1,000 | 3,329,266 |
| **Total** | **28,333** | **1,994** | **4,523,671** |

- Train: 341,410 turns · validation: 23,258 turns
- Unique corpus: **4.84M text tokens** (5.94M including BOS/EOS and role markers)
- Per-turn sampling: keep **all** DailyDialog turns, fill the remaining quota with
  shuffled PersonaChat → **100,000 train** and **8,000 val** samples
- Packed into 512-token rows: **24,633 train rows** (85.2% fill) and **2,052 val rows**
- Supervised signal: only bot turns + final EOS (~1.5M graded tokens in the 100k set)

Each training example is one bot reply plus its preceding conversation context.
Files are newline-delimited JSON (`{"source": ..., "turns": [user, bot, ...]}`).

**Data licensing:** DailyDialog is distributed under **CC BY-NC-SA 4.0**
(non-commercial); PersonaChat follows its ParlAI/MIT terms. Review both before
redistributing or using commercially.

---

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = 0.9, 0.95) |
| Learning rate | 3e-4, 5% linear warmup + cosine decay |
| Weight decay | 0.01 (matrix weights only) |
| Batch size | 2 sequences × 4 accumulation steps = **8 effective** |
| Sequence length | 512 |
| Epochs | 12 |
| Optimizer steps / epoch | 3,080 |
| Total optimizer steps | 36,960 |
| Gradient clipping | 1.0 |
| Precision | fp16 autocast + GradScaler |
| Checkpoint cadence | every 100 steps, plus best-so-far and end-of-epoch |

The notebook is **crash-resumable**: it stores model/optimizer/scheduler/scaler
state, RNG state, partial-epoch progress, and best validation loss. On Colab,
checkpoints are written atomically to Google Drive every 100 optimizer steps, so a
disconnected session resumes exactly where it stopped (with a deterministic
per-epoch shuffle seed so batch order is replayed, not restarted).

A `SMOKE_TEST` flag runs the entire pipeline on 400 samples in a few minutes.

---

## Results

Training curve (cross-entropy per token):

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 5.0283 | 4.5299 |
| 2 | 4.2610 | 4.1461 |
| 3 | 3.9612 | 3.9504 |
| 4 | 3.7638 | 3.8184 |
| 5 | 3.6032 | 3.7214 |
| 6 | 3.4484 | 3.6583 |
| 7 | 3.2922 | 3.6093 |
| **8** | **3.1336** | **3.5926** ← best |
| 9 | 2.9804 | 3.5931 |
| 10 | 2.8485 | 3.6050 |
| 11 | 2.7560 | 3.6156 |
| 12 | 2.7103 | 3.6196 |

- **Best validation loss:** 3.5926 at epoch 8 → **validation perplexity 36.33**
- **Final training loss:** 2.7103 at epoch 12
- Final validation perplexity: 37.32

Validation bottoms out at epoch 8 and then drifts upward while training loss keeps
falling — the expected overfitting regime for a 40M model on ~4.5M tokens. The
`chat_100k_best.pt` checkpoint captures the epoch-8 weights and is the recommended
checkpoint for inference.

---

## Repository layout

```
.
├── .gitignore
├── README.md
└── Sophon/
    ├── Sophon.ipynb                 # training, evaluation, export, chat demo
    ├── Sophon_chat_ui.py            # FastAPI + HTML chat UI (inference only)
    ├── tokenizer_chat.json          # 16,002-token BPE + chat role tokens
    ├── data/
    │   ├── chat_train.jsonl         # 28,333 conversations
    │   ├── chat_validation.jsonl    # 1,994 conversations
    │   └── tokenizer_chat.json
    └── checkpoints/                 # git-ignored; weights are local only
        ├── chat_100k_best.pt        # epoch 8, val 3.5926  (recommended)
        ├── chat_100k_final.pt       # exported best checkpoint (inference schema)
        └── chat_100k_last.pt        # full resumable state (~480 MB)
```

> The GitHub repository root contains the `Sophon/` project directory. Clone, then
> `cd Sophon/Sophon` before running the commands below.

---

## Quickstart

### Requirements

- Python 3.10+
- PyTorch 2.x (CUDA, Apple MPS, or CPU)
- `tokenizers`, `numpy`, `matplotlib`, `tqdm`
- For the web UI: `fastapi`, `uvicorn`, `pydantic`

```bash
git clone git@github.com:Phinix-Morgan/Sophon.git
cd Sophon/Sophon

python -m venv .venv
source .venv/bin/activate
pip install torch tokenizers numpy matplotlib tqdm fastapi uvicorn pydantic
```

### Run the chat UI

Place a checkpoint at `checkpoints/chat_100k_best.pt`, then:

```bash
python Sophon_chat_ui.py --checkpoint checkpoints/chat_100k_best.pt --port 8000
# open http://127.0.0.1:8000
```

Options: `--device cuda|mps|cpu`, `--host`, `--port`, `--max-new-tokens`.

### Train from scratch (Colab)

1. Open `Sophon.ipynb` in Google Colab and select a **T4 GPU** runtime.
2. Run Section 2 to mount Drive; the notebook looks for data in
   `/content/drive/MyDrive/chat_max_100k/data` and falls back to local `./data`.
3. Run all cells. Training auto-resumes from `chat_100k_last.pt` if present.
4. Section 7 exports `chat_100k_final.pt`; the final cells run a scripted demo and
   an interactive `chat(model)` loop.

To train locally, run the notebook from the `Sophon/` directory (non-Colab paths
resolve to `./data` and `./checkpoints`). A CUDA GPU is required by the notebook's
`COLAB_GPU_REQUIRED` guard; disable that flag for CPU-only experimentation.

---

## Chat format & tokenizer

Every conversation is a single sequence with role markers:

```
[BOS] <|user|> hi, how are you? <|bot|> i'm fine, thanks. and you? <|user|> ... [EOS]
```

Loss is masked to bot turns only, so the model is trained to predict replies, never
user text:

```
ids:    [BOS] <|user|> u1 <|bot|> b1 <|user|> u2 <|bot|> b2 [EOS]
labels: -100  -100 ....  -100  -100  b1  -100 ....  -100  b2  [EOS]
```

| Token | ID | Role |
|---|---:|---|
| `[PAD]` | 0 | padding (masked, banned at generation) |
| `[UNK]` | 1 | unknown (banned) |
| `[BOS]` | 2 | sequence start (banned) |
| `[EOS]` | 3 | sequence end / stop token |
| `<|user|>` | 16000 | user turn marker (banned) |
| `<|bot|>` | 16001 | bot turn marker (banned) |

Generation uses temperature + top-k sampling, bans all structural tokens, stops at
`[EOS]`, and keeps a sliding 512-token history window.

---

## Checkpoints

Weights are **not** committed to git (the `.pt` files total ~785 MB). Checkpoints
use this schema:

```python
{
    "model_state_dict": ...,
    "epoch": 8,
    "val_loss": 3.5925737954569463,
    "history": {"train_loss": [...], "val_loss": [...]},
    "notebook_version": "1.1",
    "config": {
        "vocab_size": 16002,
        "max_seq_len": 512,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 10,
        "d_ff": 2048,
        "dropout": 0.1,
        "tokenizer_path": ".../tokenizer_chat.json",
        "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3, "user": 16000, "bot": 16001}
    }
}
```

- `chat_100k_best.pt` — best validation weights (recommended for inference)
- `chat_100k_final.pt` — exported best checkpoint with the full inference config
- `chat_100k_last.pt` — resumable training state (model + optimizer + RNG + progress)

Regenerate them by running the notebook. To share weights, upload the `.pt` files to
Hugging Face Hub or a GitHub release (they exceed GitHub's 100 MB file limit) and
link them here.

---

## HTTP API

`Sophon_chat_ui.py` serves a single-page UI plus a small JSON API.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI (HTML) |
| `GET` | `/api/health` | Model status, device, checkpoint, epoch, val loss, params |
| `POST` | `/api/chat` | Generate one reply (JSON) |
| `POST` | `/api/chat/stream` | Generate a reply as Server-Sent Events |

**Request**

```json
{
  "message": "hi, how are you?",
  "history": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hey!"}],
  "temperature": 0.8,
  "top_k": 50,
  "max_new_tokens": 64
}
```

**Examples**

```bash
curl -s http://127.0.0.1:8000/api/health

curl -s http://127.0.0.1:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hi, how are you?","history":[],"temperature":0.8,"top_k":50}'
```

Inputs are clamped server-side (message ≤ 2,000 chars, ≤ 60 history items, ≤ 256 new
tokens). The health endpoint reports `loaded: false` with an error if the checkpoint
is missing, rather than crashing the server.

---

## Evaluation

The notebook reports token-level cross-entropy on a held-out validation split
(8,000 per-turn samples from 1,994 conversations disjoint from training). Because
loss is computed only on bot tokens, train and validation numbers are directly
comparable.

This repository reports **loss/perplexity only**. Reference-based metrics
(unigram/bigram F1, distinct-n) and retrieval-augmented variants are planned; for a
40M from-scratch model, human evaluation of short chat replies is more informative
than n-gram overlap.

---

## Limitations

Sophon is intentionally small and narrowly trained. Do not expect:

- **Knowledge.** It cannot answer factual questions ("what is photosynthesis?").
  There is no encyclopedic data in the training set and no retrieval.
- **Instruction following.** It was trained on casual dialogue, not tasks.
- **Long context.** The window is 512 tokens; older turns are dropped.
- **Reliability.** Expect repetition, generic replies, and occasional incoherence,
  especially under greedy decoding.
- **Safety.** No RLHF, filtering, or safety tuning. Training corpora are
  crowd-sourced and may contain biased or offensive text.

**Out of scope:** production chatbots, customer support, medical/legal/financial
advice, or any deployment where incorrect output causes harm. The model is suitable
for research, education, and experimentation with small-scale language modeling.

---

## Roadmap

- [x] 40M from-scratch baseline with resumable Colab training
- [x] Best-checkpoint tracking and production-shaped inference server
- [ ] Retrieval + LM reranker (retrieve candidate replies, rank with Sophon)
- [ ] Reference-based evaluation harness (F1, distinct-n, latency)
- [ ] Pretrained small-model + retrieval path for factual question answering
- [ ] Publish weights on Hugging Face Hub

---

## Acknowledgements

- **DailyDialog** — Li et al., *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset* (IJCNLP 2017)
- **PersonaChat** — Zhang et al., *Personalizing Dialogue Agents* (ACL 2018), via ParlAI
- Architecture and training conventions follow the GPT-2 / nanoGPT lineage
- Built with PyTorch, Hugging Face `tokenizers`, FastAPI, and Uvicorn

---

## License

Code in this repository is released under the **MIT License**.

Model weights and training data retain their upstream terms: DailyDialog is
**CC BY-NC-SA 4.0** (non-commercial) and PersonaChat follows ParlAI's licensing.
You are responsible for complying with those terms if you redistribute the data or
derived artifacts.

---

## Citation

```bibtex
@misc{sophon2026,
  title        = {Sophon: A 40M-Parameter From-Scratch Conversational Language Model},
  author       = {Morgan, Phinix},
  year         = {2026},
  howpublished = {\url{https://github.com/Phinix-Morgan/Sophon}},
  note         = {Decoder-only transformer trained on DailyDialog and PersonaChat}
}
```
