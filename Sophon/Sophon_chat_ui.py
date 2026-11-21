"""Chat Max 100K — self-contained FastAPI chat UI.

Run:
    .venv/bin/python chat_max_100k/chat_ui.py
    .venv/bin/python chat_max_100k/chat_ui.py --port 8000 --device cpu

Defaults to checkpoints/chat_100k_best.pt beside this file.
"""

import argparse
import json
import logging
import math
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

import uvicorn

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "chat_100k_best.pt"
DEFAULT_MAX_NEW_TOKENS = 64

MAX_MESSAGE_CHARS = 2000
MAX_NEW_TOKEN_LIMIT = 256
MAX_HISTORY_ITEMS = 60

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("chat_max")


# ---------------------------------------------------------------------------
# Model (identical state-dict layout to chatbot_core.GPTLanguageModel)
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        return self.embedding(positions)


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids) + self.position_embedding(input_ids)
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~key_mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        output = attention_weights @ V
        output = output.transpose(1, 2).contiguous().reshape(B, T, C)
        output = self.output_projection(output)
        output = self.residual_dropout(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, attention_mask=None):
        for block in self.blocks:
            x = block(x, attention_mask)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.embedding = GPTEmbedding(vocab_size, d_model, max_seq_len, dropout)
        self.transformer = TransformerStack(n_layers, d_model, n_heads, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds maximum {self.max_seq_len}")

        x = self.embedding(input_ids)
        x = self.transformer(x, attention_mask)
        x = self.final_norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_device(preference=None):
    if preference:
        requested = str(preference).lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU.")
        elif requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            log.warning("MPS requested but unavailable; falling back to CPU.")
        else:
            return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def resolve_tokenizer(checkpoint_path, config):
    raw = config.get("tokenizer_path") or "tokenizer_chat.json"
    name = Path(raw).name

    candidates = [
        checkpoint_path.parent.parent / name,
        SCRIPT_DIR / name,
        SCRIPT_DIR / "data" / name,
        checkpoint_path.parent / name,
        SCRIPT_DIR.parent / name,
    ]

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return Tokenizer.from_file(str(candidate))

    searched = "\n  ".join(str(c) for c in dict.fromkeys(candidates))
    raise FileNotFoundError(
        f"Could not find tokenizer {name!r}. Looked in:\n  {searched}"
    )


def load_inference_model(checkpoint=DEFAULT_CHECKPOINT, device=None):
    checkpoint_path = resolve_checkpoint(checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        ckpt = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except Exception:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no config block; cannot rebuild the model.")

    special_ids = config.get("special_ids", {})
    required_ids = ("pad", "unk", "bos", "eos", "user", "bot")
    missing = [key for key in required_ids if key not in special_ids]
    if missing:
        raise ValueError(f"Checkpoint config is missing special ids: {missing}")

    model = GPTLanguageModel(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(resolve_device(device))
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tokenizer = resolve_tokenizer(checkpoint_path, config)

    info = {
        "checkpoint": checkpoint_path.name,
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss", ckpt.get("best_val_loss")),
        "notebook_version": ckpt.get("notebook_version"),
        "device": str(next(model.parameters()).device),
        "params": sum(p.numel() for p in model.parameters()),
        "max_seq_len": model.max_seq_len,
    }

    return model, tokenizer, special_ids, config, info


# ---------------------------------------------------------------------------
# Prompting / generation
# ---------------------------------------------------------------------------

def build_prompt_ids(tokenizer, special_ids, turns):
    input_ids = [special_ids["bos"]]

    for index, text in enumerate(turns):
        input_ids.append(special_ids["user"] if index % 2 == 0 else special_ids["bot"])
        input_ids.extend(tokenizer.encode(text).ids)

    input_ids.append(special_ids["bot"])
    return input_ids


def trim_history(tokenizer, turns, max_seq_len, margin=80):
    while (
        len(turns) > 2
        and sum(len(tokenizer.encode(turn).ids) for turn in turns) > max_seq_len - margin
    ):
        turns = turns[2:]
    return turns


def message_content_to_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif hasattr(part, "text") and isinstance(getattr(part, "text"), str):
                parts.append(getattr(part, "text"))
        return " ".join(part for part in parts if part)

    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""

    if content is not None and isinstance(getattr(content, "text", None), str):
        return content.text

    return ""


def history_to_turns(history):
    turns = []

    for item in history or []:
        role = None
        content = None

        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        elif hasattr(item, "role") and hasattr(item, "content"):
            role = getattr(item, "role")
            content = getattr(item, "content")
        elif isinstance(item, (list, tuple)):
            turns.extend(part for part in item if isinstance(part, str) and part)
            continue
        elif isinstance(item, str):
            content = item

        if role == "system" or content is None:
            continue

        text = message_content_to_text(content)
        if text.strip():
            turns.append(text)

    return turns


class GenerationState:
    """One incremental decoding session.

    The lock is held only for each individual forward pass, never while a token
    is being yielded. A disconnected client therefore cannot leave the model
    lock stuck.
    """

    def __init__(self, model, tokenizer, special_ids, turns, max_new_tokens, temperature, top_k):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = model.max_seq_len
        self.max_new_tokens = max(0, int(max_new_tokens))
        self.temperature = float(temperature)
        self.top_k = int(top_k or 0)

        input_ids = build_prompt_ids(tokenizer, special_ids, turns)
        self.generated_ids = torch.tensor(
            [input_ids[-self.max_seq_len:]],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )

        self.banned_ids = [special_ids[key] for key in ("pad", "unk", "bos", "user", "bot")]
        self.eos_id = special_ids["eos"]
        self.reply_ids = []
        self.previous_text = ""
        self.steps = 0
        self.finished = False

    def _next_token(self):
        context_ids = self.generated_ids[:, -self.max_seq_len:]
        logits = self.model(context_ids, torch.ones_like(context_ids))
        next_token_logits = logits[:, -1, :].clone()

        for banned_id in self.banned_ids:
            next_token_logits[:, banned_id] = float("-inf")

        if self.temperature <= 0:
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token_logits = next_token_logits / self.temperature

        if self.top_k:
            k = min(self.top_k, next_token_logits.shape[-1])
            top_values, _ = torch.topk(next_token_logits, k, dim=-1)
            next_token_logits[next_token_logits < top_values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(next_token_logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    @torch.no_grad()
    def step(self):
        if self.finished or self.steps >= self.max_new_tokens:
            self.finished = True
            return "", True

        next_token = self._next_token()
        self.steps += 1

        if next_token.item() == self.eos_id:
            self.finished = True
            return "", True

        self.generated_ids = torch.cat([self.generated_ids, next_token], dim=1)
        self.reply_ids.append(next_token.item())

        current_text = self.tokenizer.decode(self.reply_ids, skip_special_tokens=True)

        if current_text.startswith(self.previous_text):
            delta = current_text[len(self.previous_text):]
        else:
            split = 0
            limit = min(len(self.previous_text), len(current_text))
            while split < limit and self.previous_text[split] == current_text[split]:
                split += 1
            delta = current_text[split:]

        self.previous_text = current_text

        if self.steps >= self.max_new_tokens:
            self.finished = True

        return delta, self.finished


class InferenceEngine:
    def __init__(self, model, tokenizer, special_ids, info):
        self.model = model
        self.tokenizer = tokenizer
        self.special_ids = special_ids
        self.info = info
        self.lock = threading.Lock()

    def prepare_turns(self, message, history):
        turns = history_to_turns((history or [])[-MAX_HISTORY_ITEMS:])
        turns.append(message)
        return trim_history(self.tokenizer, turns, self.model.max_seq_len)

    def stream(self, message, history, temperature, top_k, max_new_tokens):
        turns = self.prepare_turns(message, history)
        state = GenerationState(
            self.model,
            self.tokenizer,
            self.special_ids,
            turns,
            max_new_tokens,
            temperature,
            top_k,
        )

        while not state.finished:
            with self.lock:
                delta, finished = state.step()

            if delta:
                yield delta

            if finished:
                break

    def reply(self, message, history, temperature, top_k, max_new_tokens):
        return "".join(
            self.stream(message, history, temperature, top_k, max_new_tokens)
        ).strip()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS * 2)
    history: list = Field(default_factory=list)
    temperature: float | None = 0.8
    top_k: int | None = 50
    max_new_tokens: int | None = None


def clamp(value, low, high, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number) or math.isinf(number):
        return fallback
    return max(low, min(high, number))


def create_app(checkpoint=DEFAULT_CHECKPOINT, device=None, default_max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    checkpoint_path = resolve_checkpoint(checkpoint)

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        app.state.load_error = None
        app.state.default_max_new_tokens = int(default_max_new_tokens)

        try:
            model, tokenizer, special_ids, config, info = load_inference_model(
                checkpoint_path, device
            )
            app.state.engine = InferenceEngine(model, tokenizer, special_ids, info)
            log.info(
                "Loaded %s on %s | %s parameters | val %.4f",
                info["checkpoint"],
                info["device"],
                f"{info['params']:,}",
                info["val_loss"] if info["val_loss"] is not None else float("nan"),
            )
        except Exception as exc:
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            log.error("Could not load model: %s", app.state.load_error)

        yield

    app = FastAPI(title="Chat Max", docs_url=None, redoc_url=None, lifespan=lifespan)

    def get_engine():
        engine = getattr(app.state, "engine", None)
        if engine is None:
            error = getattr(app.state, "load_error", None) or "Model is not loaded."
            raise HTTPException(status_code=503, detail=error)
        return engine

    def generation_params(payload):
        temperature = clamp(payload.temperature, 0.0, 2.0, 0.8)
        top_k = int(clamp(payload.top_k, 0, 4096, 50))
        max_new_tokens = int(
            clamp(
                payload.max_new_tokens,
                1,
                MAX_NEW_TOKEN_LIMIT,
                getattr(app.state, "default_max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            )
        )
        return temperature, top_k, max_new_tokens

    def prepared_message(payload):
        message = (payload.message or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        if len(message) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Message is too long (limit {MAX_MESSAGE_CHARS} characters).",
            )
        return message

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/health")
    def health():
        engine = getattr(app.state, "engine", None)

        if engine is None:
            return {
                "status": "error",
                "loaded": False,
                "error": getattr(app.state, "load_error", None) or "Model is not loaded.",
                "checkpoint": checkpoint_path.name,
            }

        info = engine.info
        return {
            "status": "ok",
            "loaded": True,
            "device": info["device"],
            "checkpoint": info["checkpoint"],
            "epoch": info["epoch"],
            "val_loss": info["val_loss"],
            "params": info["params"],
            "max_seq_len": info["max_seq_len"],
            "default_max_new_tokens": getattr(
                app.state, "default_max_new_tokens", DEFAULT_MAX_NEW_TOKENS
            ),
        }

    @app.post("/api/chat")
    def chat(payload: ChatRequest):
        engine = get_engine()
        message = prepared_message(payload)
        temperature, top_k, max_new_tokens = generation_params(payload)

        reply = engine.reply(
            message,
            payload.history,
            temperature,
            top_k,
            max_new_tokens,
        )
        return {"reply": reply}

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest):
        engine = get_engine()
        message = prepared_message(payload)
        temperature, top_k, max_new_tokens = generation_params(payload)

        def events():
            collected = []
            error = None

            try:
                for delta in engine.stream(
                    message,
                    payload.history,
                    temperature,
                    top_k,
                    max_new_tokens,
                ):
                    collected.append(delta)
                    yield f"data: {json.dumps({'token': delta})}\n\n"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                log.exception("Generation failed")
                yield f"data: {json.dumps({'error': error})}\n\n"

            yield f"data: {json.dumps({'done': True, 'reply': ''.join(collected).strip(), 'error': error})}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chat Max</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;550;600;650;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
  :root {
    --canvas: #f6f8f4;
    --surface: #ffffff;
    --surface-soft: #fbfcfa;
    --ink: #18231c;
    --ink-soft: #394a3e;
    --muted: #66776b;
    --faint: #93a196;
    --line: #e2e8e0;
    --line-strong: #d2dbd3;
    --green-900: #0b3d2c;
    --green-800: #0d5038;
    --green-700: #0f6b4f;
    --green-600: #187a5a;
    --green-100: #e6f1e8;
    --green-50: #eff6f0;
    --rose: #a33a3a;
    --rose-50: #fbefef;
    --amber: #9a6a18;
    --radius: 18px;
    --shadow-card: 0 1px 2px rgba(24, 35, 28, .04), 0 14px 36px rgba(24, 35, 28, .045);
    --shadow-soft: 0 1px 2px rgba(24, 35, 28, .05);
    --ease: cubic-bezier(.2, .7, .3, 1);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background: var(--canvas);
    color: var(--ink);
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    font-size: 15px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }
  button, textarea, input { font: inherit; color: inherit; }
  button { cursor: pointer; }
  :focus-visible { outline: 2px solid var(--green-700); outline-offset: 2px; }

  .shell {
    width: min(1180px, 100%);
    height: 100dvh;
    margin: 0 auto;
    padding: 22px 22px 26px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  /* ---------- top bar ---------- */
  .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-shrink: 0; }
  .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
  .mark {
    width: 38px; height: 38px; border-radius: 11px; flex-shrink: 0;
    background: var(--green-900); color: #fff;
    display: grid; place-items: center;
  }
  .mark svg { width: 20px; height: 20px; }
  .brand-text { min-width: 0; }
  .brand-text h1 { font-size: 17px; font-weight: 650; letter-spacing: -.02em; line-height: 1.15; }
  .brand-text p { font-size: 12px; color: var(--muted); margin-top: 2px; letter-spacing: .005em; white-space: nowrap; }

  .top-actions { display: flex; align-items: center; gap: 9px; }
  .status {
    display: inline-flex; align-items: center; gap: 7px;
    height: 34px; padding: 0 12px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--surface);
    font-size: 12.5px; font-weight: 550; color: var(--muted);
    white-space: nowrap;
  }
  .status .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--faint); }
  .status[data-state="ready"] { color: var(--green-800); border-color: #cfe0d3; background: var(--green-50); }
  .status[data-state="ready"] .dot { background: var(--green-600); }
  .status[data-state="error"] { color: var(--rose); border-color: #ecd4d4; background: var(--rose-50); }
  .status[data-state="error"] .dot { background: var(--rose); }
  .status[data-state="offline"] { color: var(--amber); border-color: #ecdfc8; background: #fcf7ee; }
  .status[data-state="offline"] .dot { background: var(--amber); }

  .ghost-btn, .soft-btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 7px;
    height: 36px; padding: 0 14px;
    border-radius: 10px; border: 1px solid var(--line-strong);
    background: var(--surface); color: var(--ink-soft);
    font-size: 13px; font-weight: 550;
    box-shadow: var(--shadow-soft);
    transition: border-color .15s var(--ease), color .15s var(--ease), background .15s var(--ease);
  }
  .ghost-btn:hover, .soft-btn:hover { border-color: #b9c9bd; color: var(--green-800); }
  .ghost-btn:active, .soft-btn:active { background: var(--surface-soft); }
  .settings-toggle { display: none; }

  /* ---------- layout ---------- */
  .workspace {
    flex: 1; min-height: 0;
    display: grid; grid-template-columns: minmax(0, 1fr) 296px;
    gap: 16px;
  }

  .chat-card {
    display: flex; flex-direction: column; min-height: 0;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow-card);
    overflow: hidden;
  }

  /* ---------- messages ---------- */
  .messages {
    flex: 1; min-height: 0;
    overflow-y: auto;
    padding: 26px 26px 10px;
    display: flex; flex-direction: column; gap: 20px;
    scroll-behavior: smooth;
    overscroll-behavior: contain;
  }
  .messages::-webkit-scrollbar { width: 10px; }
  .messages::-webkit-scrollbar-thumb { background: #dfe6de; border-radius: 8px; border: 3px solid var(--surface); }
  .messages::-webkit-scrollbar-track { background: transparent; }

  .empty {
    margin: auto; max-width: 560px; padding: 30px 10px 26px;
    text-align: center;
  }
  .empty .seal {
    width: 52px; height: 52px; margin: 0 auto 18px; border-radius: 50%;
    border: 1px solid #cfe0d3; background: var(--green-50);
    display: grid; place-items: center; color: var(--green-800);
  }
  .empty .seal svg { width: 24px; height: 24px; }
  .empty h2 {
    font-family: 'Instrument Serif', Georgia, serif;
    font-weight: 400; font-size: 31px; line-height: 1.15;
    letter-spacing: -.015em; color: var(--green-900);
  }
  .empty h2 em { font-style: italic; color: var(--green-700); }
  .empty > p { margin: 9px auto 0; max-width: 410px; color: var(--muted); font-size: 13.5px; }
  .chips { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 22px; }
  .chip {
    border: 1px solid var(--line-strong); background: var(--surface);
    color: var(--ink-soft); border-radius: 999px; padding: 8px 14px;
    font-size: 12.5px; font-weight: 500;
    transition: border-color .15s var(--ease), background .15s var(--ease), color .15s var(--ease);
  }
  .chip:hover { border-color: #b9c9bd; background: var(--green-50); color: var(--green-800); }

  .message { display: flex; gap: 11px; align-items: flex-start; max-width: 82%; }
  .message.user { align-self: flex-end; justify-content: flex-end; }
  .message.bot { align-self: flex-start; }
  .avatar {
    width: 30px; height: 30px; border-radius: 9px; flex-shrink: 0; margin-top: 21px;
    display: grid; place-items: center;
    font-size: 11px; font-weight: 700; letter-spacing: .01em;
    background: var(--green-100); color: var(--green-900);
    border: 1px solid #d8e7db;
  }
  .message.user .avatar { display: none; }

  .message-body { min-width: 0; display: flex; flex-direction: column; }
  .message.user .message-body { align-items: flex-end; }
  .meta {
    display: flex; align-items: center; gap: 8px;
    font-size: 11.5px; color: var(--faint);
    margin-bottom: 5px; padding: 0 2px; min-height: 18px;
  }
  .meta .who { color: var(--ink-soft); font-weight: 600; }
  .copy-btn {
    border: 0; background: transparent; color: var(--faint);
    font-size: 11px; font-weight: 550; padding: 1px 4px; border-radius: 5px;
    opacity: 0; transition: opacity .15s var(--ease), color .15s var(--ease);
  }
  .message.bot:hover .copy-btn, .copy-btn:focus-visible { opacity: 1; }
  .copy-btn:hover { color: var(--green-700); }

  .bubble {
    position: relative;
    padding: 11px 15px;
    border-radius: 14px;
    font-size: 14.5px; line-height: 1.58;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .message.bot .bubble {
    background: var(--surface);
    border: 1px solid var(--line);
    border-top-left-radius: 5px;
    box-shadow: var(--shadow-soft);
  }
  .message.user .bubble {
    background: var(--green-100);
    border: 1px solid #d5e6d9;
    border-top-right-radius: 5px;
    color: #173c29;
  }
  .bubble.error { background: var(--rose-50); border-color: #eccfcf; color: var(--rose); }

  .caret {
    display: none;
    width: 6px; height: 15px; margin-left: 2px; border-radius: 1px;
    background: var(--green-600); vertical-align: -2px;
  }
  .message.streaming .caret { display: inline-block; animation: blink 1.05s steps(2, start) infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  .typing { display: inline-flex; align-items: center; gap: 5px; height: 20px; }
  .typing i {
    width: 6px; height: 6px; border-radius: 50%; background: #a9bbae;
    animation: dot 1.2s infinite var(--ease);
  }
  .typing i:nth-child(2) { animation-delay: .14s; }
  .typing i:nth-child(3) { animation-delay: .28s; }
  @keyframes dot { 0%, 70%, 100% { opacity: .35; transform: translateY(0); } 35% { opacity: 1; transform: translateY(-3px); } }

  /* ---------- composer ---------- */
  .composer-wrap { padding: 12px 18px 14px; border-top: 1px solid var(--line); background: var(--surface); }
  .composer {
    display: flex; align-items: flex-end; gap: 10px;
    padding: 7px 7px 7px 15px;
    border: 1px solid var(--line-strong);
    border-radius: 15px;
    background: var(--surface);
    transition: border-color .15s var(--ease), box-shadow .15s var(--ease);
  }
  .composer:focus-within { border-color: #9dbaa5; box-shadow: 0 0 0 3px rgba(15, 107, 79, .08); }
  textarea {
    flex: 1; resize: none; border: 0; outline: 0; background: transparent;
    min-height: 40px; max-height: 150px; padding: 9px 0 7px;
    font-size: 14.5px; line-height: 1.5;
  }
  textarea::placeholder { color: var(--faint); }
  textarea:focus, textarea:focus-visible { outline: none; }
  .send {
    width: 40px; height: 40px; border-radius: 11px; border: 0; flex-shrink: 0;
    background: var(--green-800); color: #fff;
    display: grid; place-items: center;
    transition: background .15s var(--ease), opacity .15s var(--ease), transform .1s var(--ease);
  }
  .send:hover:not(:disabled) { background: var(--green-700); }
  .send:active:not(:disabled) { transform: scale(.97); }
  .send:disabled { opacity: .42; cursor: default; }
  .send svg { width: 18px; height: 18px; }
  .send .spinner {
    width: 15px; height: 15px; border-radius: 50%;
    border: 2px solid rgba(255, 255, 255, .35); border-top-color: #fff;
    animation: spin .8s linear infinite; display: none;
  }
  .send.busy svg { display: none; }
  .send.busy .spinner { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .composer-foot {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 8px 3px 0; color: var(--faint); font-size: 11.5px;
  }
  .composer-foot kbd {
    font-family: inherit; font-size: 11px; color: var(--muted);
    border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; background: var(--surface-soft);
  }

  /* ---------- sidebar ---------- */
  .sidebar { display: flex; flex-direction: column; gap: 12px; min-height: 0; overflow-y: auto; padding-right: 1px; }
  .panel {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 15px; padding: 16px;
    box-shadow: var(--shadow-soft);
  }
  .panel h2 {
    font-size: 11.5px; font-weight: 650; letter-spacing: .075em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 14px;
  }
  .panel-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .panel-head h2 { margin-bottom: 0; }
  .link-btn {
    border: 0; background: transparent; padding: 0;
    font-size: 11.5px; font-weight: 550; color: var(--green-700);
  }
  .link-btn:hover { color: var(--green-900); text-decoration: underline; text-underline-offset: 2px; }

  .control { padding: 9px 0 11px; border-top: 1px solid #eef2ec; }
  .control:first-of-type { border-top: 0; padding-top: 0; }
  .control:last-child { padding-bottom: 0; }
  .control label {
    display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
    font-size: 12.5px; font-weight: 550; color: var(--ink-soft); margin-bottom: 7px;
  }
  .control output { font-variant-numeric: tabular-nums; color: var(--green-800); font-weight: 600; font-size: 12px; }
  .control p { margin-top: 5px; font-size: 11px; color: var(--faint); line-height: 1.45; }

  input[type="range"] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 4px; border-radius: 4px;
    background: #e3eae2; outline-offset: 5px;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 15px; height: 15px; border-radius: 50%;
    background: var(--green-700); border: 3px solid #fff;
    box-shadow: 0 0 0 1px #b9cdbe;
    cursor: pointer;
  }
  input[type="range"]::-moz-range-thumb {
    width: 15px; height: 15px; border-radius: 50%;
    background: var(--green-700); border: 3px solid #fff;
    box-shadow: 0 0 0 1px #b9cdbe;
    cursor: pointer;
  }

  .stats { display: flex; flex-direction: column; }
  .stats > div {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 7px 0; border-top: 1px solid #eef2ec; font-size: 12px;
  }
  .stats > div:first-child { border-top: 0; padding-top: 0; }
  .stats > div:last-child { padding-bottom: 0; }
  .stats dt { color: var(--faint); }
  .stats dd { color: var(--ink-soft); font-weight: 550; text-align: right; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }

  .note p { font-size: 12px; color: var(--muted); line-height: 1.6; }
  .note strong { color: var(--ink-soft); font-weight: 600; }

  .scrim { display: none; }

  /* ---------- responsive ---------- */
  @media (max-width: 1023px) {
    .shell { padding: 16px; }
    .workspace { grid-template-columns: minmax(0, 1fr); }
    .settings-toggle { display: inline-flex; }
    .sidebar {
      position: fixed; z-index: 30; top: 0; right: 0; bottom: 0;
      width: min(330px, 88vw);
      padding: 18px; gap: 12px;
      background: var(--canvas);
      border-left: 1px solid var(--line);
      transform: translateX(102%);
      transition: transform .22s var(--ease);
      overflow-y: auto;
    }
    .sidebar.open { transform: translateX(0); }
    .scrim {
      display: block; position: fixed; inset: 0; z-index: 20;
      background: rgba(24, 35, 28, .28);
      opacity: 0; pointer-events: none;
      transition: opacity .22s var(--ease);
    }
    .scrim.show { opacity: 1; pointer-events: auto; }
  }

  @media (max-width: 640px) {
    .shell { padding: 10px; gap: 12px; }
    .topbar { gap: 8px; }
    .brand-text p { display: none; }
    .status { height: 32px; padding: 0 10px; }
    .status #statusText { display: none; }
    .status { width: 32px; justify-content: center; }
    .ghost-btn, .soft-btn { height: 34px; padding: 0 11px; font-size: 12.5px; }
    .messages { padding: 18px 16px 8px; gap: 17px; }
    .message { max-width: 92%; }
    .empty h2 { font-size: 27px; }
    .composer-wrap { padding: 10px 12px 12px; }
    .composer-foot span:first-child { display: none; }
    .composer-foot { justify-content: flex-end; }
    .avatar { display: none; }
  }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; transition-duration: .001ms !important; scroll-behavior: auto !important; }
  }
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 18V7.5A2.5 2.5 0 0 1 6.5 5h11A2.5 2.5 0 0 1 20 7.5V18"/>
          <path d="M8 10.5h8M8 14h5"/>
        </svg>
      </div>
      <div class="brand-text">
        <h1>Chat Max</h1>
        <p>100K conversational model</p>
      </div>
    </div>
    <div class="top-actions">
      <div class="status" id="status" data-state="loading" role="status" aria-live="polite" title="Model status">
        <span class="dot"></span><span id="statusText">Connecting</span>
      </div>
      <button class="soft-btn settings-toggle" id="settingsToggle" type="button" aria-controls="sidebar" aria-expanded="false">
        Settings
      </button>
      <button class="ghost-btn" id="newChat" type="button">New chat</button>
    </div>
  </header>

  <div class="workspace">
    <main class="chat-card">
      <div class="messages" id="messages" aria-live="polite" aria-label="Conversation">
        <section class="empty" id="emptyState">
          <div class="seal" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 3.5c3.8 2.2 5.7 5.6 5.7 9.4 0 3.5-2.6 6.1-5.7 7.6-3.1-1.5-5.7-4.1-5.7-7.6 0-3.8 1.9-7.2 5.7-9.4Z"/>
              <path d="M12 7v9"/>
            </svg>
          </div>
          <h2>A quiet place to <em>talk.</em></h2>
          <p>Chat Max is a small language model trained on everyday conversation. Start with something simple.</p>
          <div class="chips">
            <button class="chip" type="button" data-prompt="Hey! How's your day going?">How's your day going?</button>
            <button class="chip" type="button" data-prompt="Tell me something interesting.">Tell me something interesting</button>
            <button class="chip" type="button" data-prompt="What do you like to talk about?">What do you like to talk about?</button>
          </div>
        </section>
      </div>

      <div class="composer-wrap">
        <form class="composer" id="composer">
          <textarea id="input" rows="1" maxlength="2000" placeholder="Write a message..." aria-label="Message" autocomplete="off"></textarea>
          <button class="send" id="send" type="submit" aria-label="Send message" disabled>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M5 12h13M13 6l6 6-6 6"/>
            </svg>
            <span class="spinner" aria-hidden="true"></span>
          </button>
        </form>
        <div class="composer-foot">
          <span><kbd>Enter</kbd> to send &nbsp;·&nbsp; <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line</span>
          <span id="charCount">0 / 2000</span>
        </div>
      </div>
    </main>

    <aside class="sidebar" id="sidebar" aria-label="Settings">
      <section class="panel">
        <div class="panel-head">
          <h2>Generation</h2>
          <button class="link-btn" id="resetSettings" type="button">Reset</button>
        </div>
        <div class="control">
          <label for="temperature">Temperature <output id="temperatureValue" for="temperature">0.80</output></label>
          <input type="range" id="temperature" min="0" max="1.5" step="0.05" value="0.8">
          <p>Lower is focused and predictable; higher is more playful.</p>
        </div>
        <div class="control">
          <label for="topK">Top-k <output id="topKValue" for="topK">50</output></label>
          <input type="range" id="topK" min="0" max="200" step="10" value="50">
          <p>Only sample from the k most likely tokens. 0 disables the limit.</p>
        </div>
        <div class="control">
          <label for="maxTokens">Max tokens <output id="maxTokensValue" for="maxTokens">64</output></label>
          <input type="range" id="maxTokens" min="16" max="256" step="8" value="64">
          <p>Upper bound on the length of each reply.</p>
        </div>
      </section>

      <section class="panel">
        <h2>Model</h2>
        <dl class="stats">
          <div><dt>Checkpoint</dt><dd id="statCheckpoint">—</dd></div>
          <div><dt>Device</dt><dd id="statDevice">—</dd></div>
          <div><dt>Parameters</dt><dd id="statParams">—</dd></div>
          <div><dt>Val loss</dt><dd id="statLoss">—</dd></div>
        </dl>
      </section>

      <section class="panel note">
        <p><strong>A small research model.</strong> Chat Max imitates the rhythm of everyday conversation, not facts. Replies may drift, loop, or miss the point.</p>
      </section>
    </aside>
  </div>
</div>
<div class="scrim" id="scrim"></div>

<script>
(() => {
  "use strict";

  const messagesEl = document.getElementById("messages");
  const emptyTemplate = document.getElementById("emptyState").outerHTML;
  const composer = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const newChatBtn = document.getElementById("newChat");
  const charCount = document.getElementById("charCount");
  const statusEl = document.getElementById("status");
  const statusText = document.getElementById("statusText");
  const sidebar = document.getElementById("sidebar");
  const settingsToggle = document.getElementById("settingsToggle");
  const scrim = document.getElementById("scrim");

  const controls = {
    temperature: { el: document.getElementById("temperature"), out: document.getElementById("temperatureValue") },
    topK: { el: document.getElementById("topK"), out: document.getElementById("topKValue") },
    maxTokens: { el: document.getElementById("maxTokens"), out: document.getElementById("maxTokensValue") }
  };

  const DEFAULTS = { temperature: 0.8, topK: 50, maxTokens: 64 };
  const STORAGE_KEY = "chat-max-settings-v1";

  const state = {
    history: [],
    busy: false,
    controller: null,
    seq: 0,
    settings: { ...DEFAULTS },
    userTouchedSettings: false
  };

  /* ---------------- settings ---------------- */
  function readSettings() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!saved || typeof saved !== "object") return;
      for (const key of Object.keys(DEFAULTS)) {
        const value = Number(saved[key]);
        if (Number.isFinite(value)) state.settings[key] = value;
      }
      state.userTouchedSettings = true;
    } catch (_) { /* ignore */ }
  }

  function saveSettings() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.settings)); } catch (_) { /* ignore */ }
  }

  function paintSettings() {
    const { temperature, topK, maxTokens } = state.settings;
    controls.temperature.el.value = String(temperature);
    controls.topK.el.value = String(topK);
    controls.maxTokens.el.value = String(maxTokens);
    controls.temperature.out.textContent = Number(temperature).toFixed(2);
    controls.topK.out.textContent = topK === 0 ? "Off" : String(topK);
    controls.maxTokens.out.textContent = String(maxTokens);
  }

  function bindSettings() {
    controls.temperature.el.addEventListener("input", () => {
      state.settings.temperature = Number(controls.temperature.el.value);
      state.userTouchedSettings = true;
      paintSettings(); saveSettings();
    });
    controls.topK.el.addEventListener("input", () => {
      state.settings.topK = Number(controls.topK.el.value);
      state.userTouchedSettings = true;
      paintSettings(); saveSettings();
    });
    controls.maxTokens.el.addEventListener("input", () => {
      state.settings.maxTokens = Number(controls.maxTokens.el.value);
      state.userTouchedSettings = true;
      paintSettings(); saveSettings();
    });

    document.getElementById("resetSettings").addEventListener("click", () => {
      state.settings = { ...DEFAULTS };
      state.userTouchedSettings = false;
      paintSettings(); saveSettings();
    });
  }

  /* ---------------- composer ---------------- */
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 150) + "px";
  }

  function updateComposer() {
    const length = input.value.length;
    charCount.textContent = length + " / 2000";
    sendBtn.disabled = state.busy || input.value.trim().length === 0;
  }

  function setBusy(value) {
    state.busy = value;
    sendBtn.classList.toggle("busy", value);
    sendBtn.setAttribute("aria-label", value ? "Generating reply" : "Send message");
    updateComposer();
  }

  /* ---------------- messages ---------------- */
  function timeLabel() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function atBottom() {
    return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 140;
  }

  function scrollToBottom(force) {
    if (force || atBottom()) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeEmpty() {
    const empty = document.getElementById("emptyState");
    if (empty) empty.remove();
  }

  function restoreEmpty() {
    const wrap = document.createElement("div");
    wrap.innerHTML = emptyTemplate;
    const node = wrap.firstElementChild;
    if (node) messagesEl.appendChild(node);
  }

  function addUserMessage(text) {
    removeEmpty();
    const row = document.createElement("div");
    row.className = "message user";

    const body = document.createElement("div");
    body.className = "message-body";

    const meta = document.createElement("div");
    meta.className = "meta";
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = "You";
    const time = document.createElement("span");
    time.textContent = timeLabel();
    meta.append(who, time);

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    body.append(meta, bubble);
    row.appendChild(body);
    messagesEl.appendChild(row);
    scrollToBottom(true);
  }

  function createBotMessage() {
    removeEmpty();
    const row = document.createElement("div");
    row.className = "message bot streaming";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "M";

    const body = document.createElement("div");
    body.className = "message-body";

    const meta = document.createElement("div");
    meta.className = "meta";
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = "Chat Max";
    const time = document.createElement("span");
    time.textContent = timeLabel();
    const copy = document.createElement("button");
    copy.className = "copy-btn";
    copy.type = "button";
    copy.textContent = "Copy";
    copy.setAttribute("aria-label", "Copy reply");
    meta.append(who, time, copy);

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const textEl = document.createElement("span");
    textEl.className = "text";
    const typing = document.createElement("span");
    typing.className = "typing";
    typing.setAttribute("aria-label", "Typing");
    typing.innerHTML = "<i></i><i></i><i></i>";
    textEl.appendChild(typing);

    const caret = document.createElement("span");
    caret.className = "caret";
    caret.setAttribute("aria-hidden", "true");

    bubble.append(textEl, caret);
    body.append(meta, bubble);
    row.append(avatar, body);
    messagesEl.appendChild(row);
    scrollToBottom(true);

    let started = false;

    return {
      row,
      textEl,
      copy,
      start() {
        if (started) return;
        started = true;
        textEl.textContent = "";
      },
      setText(text, streaming) {
        if (!started) this.start();
        textEl.textContent = text;
        row.classList.toggle("streaming", Boolean(streaming));
        scrollToBottom(false);
      },
      finish() {
        row.classList.remove("streaming");
        if (!textEl.textContent.trim()) {
          textEl.textContent = "I didn't manage to form a reply. Try again?";
          bubble.classList.add("error");
        }
      },
      fail(message) {
        this.start();
        row.classList.remove("streaming");
        bubble.classList.add("error");
        textEl.textContent = message;
      }
    };
  }

  function copyReply(button) {
    const row = button.closest(".message");
    const text = row ? row.querySelector(".text") : null;
    if (!text) return;
    const value = text.textContent || "";
    const done = () => {
      const old = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => { button.textContent = old; }, 1300);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(value).then(done).catch(() => {});
    }
  }

  /* ---------------- API ---------------- */
  function payloadFor(message) {
    return {
      message,
      history: state.history.slice(),
      temperature: Number(state.settings.temperature),
      top_k: Number(state.settings.topK),
      max_new_tokens: Number(state.settings.maxTokens)
    };
  }

  function friendlyError(error) {
    const raw = error && error.message ? error.message : String(error);
    if (raw.includes("503") || raw.toLowerCase().includes("not loaded")) {
      return "The model is not available. Check the server terminal for details.";
    }
    if (raw.includes("Failed to fetch") || raw.includes("NetworkError")) {
      return "Lost connection to the server. Is it still running?";
    }
    return "Something went wrong while generating a reply. Please try again.";
  }

  async function fallbackReply(payload) {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      let detail = "HTTP " + response.status;
      try {
        const body = await response.json();
        if (body && body.detail) detail = body.detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    return response.json();
  }

  function commitReply(message, reply) {
    state.history.push({ role: "user", content: message });
    state.history.push({ role: "assistant", content: reply });
    if (state.history.length > 60) state.history = state.history.slice(-60);
  }

  async function streamReply(message, bot, seq) {
    const controller = new AbortController();
    state.controller = controller;

    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFor(message)),
      signal: controller.signal
    });

    if (!response.ok || !response.body) {
      throw new Error("HTTP " + response.status);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let full = "";
    let started = false;
    let finished = false;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (seq !== state.seq) return;

      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const line = chunk.split("\n").find((entry) => entry.startsWith("data:"));
        if (!line) continue;

        let data;
        try {
          data = JSON.parse(line.slice(5).trim());
        } catch (_) {
          continue;
        }

        if (data.error) throw new Error(data.error);

        if (typeof data.token === "string") {
          if (!started) { started = true; bot.start(); }
          full += data.token;
          bot.setText(full, true);
        }

        if (data.done) {
          const reply = (data.reply || full || "").trim();
          if (!started) { started = true; bot.start(); }
          bot.setText(reply, false);
          bot.finish();
          commitReply(message, reply);
          finished = true;
          return;
        }
      }
    }

    if (seq !== state.seq) return;

    if (!finished && full) {
      bot.setText(full, false);
      bot.finish();
      commitReply(message, full.trim());
    } else if (!finished) {
      throw new Error("Empty response");
    }
  }

  async function sendMessage(forcedText) {
    if (state.busy) return;

    const message = (forcedText !== undefined ? forcedText : input.value).trim();
    if (!message) return;

    if (message.length > 2000) {
      input.focus();
      return;
    }

    input.value = "";
    autosize();
    updateComposer();

    addUserMessage(message);
    const bot = createBotMessage();

    setBusy(true);
    const seq = ++state.seq;

    try {
      await streamReply(message, bot, seq);
    } catch (error) {
      if (error && error.name === "AbortError") return;
      if (seq !== state.seq) return;

      try {
        const data = await fallbackReply(payloadFor(message));
        if (seq !== state.seq) return;
        const reply = (data.reply || "").trim();
        bot.setText(reply, false);
        bot.finish();
        commitReply(message, reply);
      } catch (fallbackError) {
        if (seq !== state.seq) return;
        bot.fail(friendlyError(fallbackError));
      }
    } finally {
      if (seq === state.seq) {
        state.controller = null;
        setBusy(false);
        input.focus();
      }
    }
  }

  /* ---------------- status ---------------- */
  function setStatus(stateName, text) {
    statusEl.dataset.state = stateName;
    statusText.textContent = text;
    statusEl.title = text;
  }

  function formatParams(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return (number / 1e6).toFixed(1) + "M";
  }

  function setStat(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
  }

  async function loadHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      const data = await response.json();

      if (!data.loaded) {
        setStatus("error", "Model unavailable");
        return;
      }

      const device = String(data.device || "cpu").toUpperCase();
      setStatus("ready", "Ready · " + device);
      setStat("statCheckpoint", data.checkpoint || "—");
      setStat("statDevice", device);
      setStat("statParams", formatParams(data.params));
      setStat("statLoss", Number.isFinite(Number(data.val_loss)) ? Number(data.val_loss).toFixed(4) : "—");

      if (!state.userTouchedSettings && data.default_max_new_tokens) {
        state.settings.maxTokens = Number(data.default_max_new_tokens);
        paintSettings();
      }
    } catch (_) {
      setStatus("offline", "Server offline");
    }
  }

  /* ---------------- events ---------------- */
  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });

  input.addEventListener("input", () => {
    autosize();
    updateComposer();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });

  messagesEl.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-prompt]");
    if (chip) {
      sendMessage(chip.dataset.prompt || "");
      return;
    }
    const copy = event.target.closest(".copy-btn");
    if (copy) copyReply(copy);
  });

  newChatBtn.addEventListener("click", () => {
    state.seq += 1;
    if (state.controller) state.controller.abort();
    state.controller = null;
    state.history = [];
    messagesEl.innerHTML = "";
    restoreEmpty();
    setBusy(false);
    input.value = "";
    autosize();
    updateComposer();
    input.focus();
  });

  function openSidebar(open) {
    sidebar.classList.toggle("open", open);
    scrim.classList.toggle("show", open);
    settingsToggle.setAttribute("aria-expanded", String(open));
  }

  settingsToggle.addEventListener("click", () => {
    openSidebar(!sidebar.classList.contains("open"));
  });

  scrim.addEventListener("click", () => openSidebar(false));

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && sidebar.classList.contains("open")) {
      openSidebar(false);
      settingsToggle.focus();
    }
  });

  /* ---------------- init ---------------- */
  readSettings();
  bindSettings();
  paintSettings();
  autosize();
  updateComposer();
  loadHealth();
  window.setInterval(loadHealth, 30000);
  input.focus();
})();
</script>
</body>
</html>
"""


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat Max 100K web UI")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="path to a .pt checkpoint")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu (default: auto)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = parser.parse_args()

    application = create_app(
        checkpoint=args.checkpoint,
        device=args.device,
        default_max_new_tokens=args.max_new_tokens,
    )

    uvicorn.run(application, host=args.host, port=args.port, log_level="info")
