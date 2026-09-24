"""JSSA basis calibration with HuggingFace transformers.

For each calibration sequence, one prefill forward records the post-RoPE queries
and keys of every attention layer and accumulates, per (layer, KV head),

    G_Qm = sum_t q_bar_t q_bar_t^T     q_bar = mean of the G query heads of the KV head
    G_K  = sum_t k_t k_t^T

The basis is the closed-form solution of the marginal objective,

    M = TopEig_r( G_Qm / tr(G_Qm) + G_K / tr(G_K) )     (rows = eigenvectors)

saved as ``{'M': {layer_idx: Tensor[H_kv, r, d]}}``, the format read by the vLLM
JSSA backend (``attention_config.jssa_basis_path``).

MLA (GLM-4.7-Flash): the query is the absorbed query q_latent = W_UK^T q_nope and
the key is the normed latent c_KV (one latent KV head; all attention heads form
one group). The decoupled RoPE part is not projected. Output M: [1, r, kv_lora_rank].

Gemma 4: only the global-attention layers are calibrated (sliding-window layers
are served densely), so the basis contains only those layers.

Default recipe (paper): 8 sequences of 131,072 tokens packed from LongAlign-10k
(see prepare_longalign.py), r = 32 for d = 128 models, r = 64 for the Gemma 4
global layers and for MLA.

Examples:
  python jssa_calibrate.py --model meta-llama/Llama-3.1-8B-Instruct --rank 32 \\
      --output bases/llama_3_1_8b_instruct.pt
  python jssa_calibrate.py --model Qwen/Qwen3-14B --yarn --rank 32 --output bases/qwen3_14b.pt
  JSSA_MOE_CHUNK=16384 python jssa_calibrate.py --model google/gemma-4-26b-a4b-it \\
      --rank 64 --attn_impl chunked_math --output bases/gemma_4_26b_a4b_it.pt
  python jssa_calibrate.py --model zai-org/GLM-4.7-Flash --rank 64 --output bases/glm_4_7_flash.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Calibration data: LongAlign-10k documents packed into long sequences.
# ---------------------------------------------------------------------------

def load_longalign_packed(data_path: str, sample_idx: int, target_tokens: int) -> str:
    """Concatenate consecutive LongAlign-10k user turns (instruction + long
    document) until their token count reaches ``target_tokens``. Single
    LongAlign documents are at most ~64k tokens, so packing is needed to fill a
    128k context. ``sample_idx`` selects a distinct window of documents."""
    rows = [json.loads(line) for line in open(data_path)]
    n = len(rows)
    off = (sample_idx * 5) % n
    parts, tot, k = [], 0, 0
    while tot < target_tokens and k < n:
        r = rows[(off + k) % n]
        parts.append(r["user"])
        tot += int(r.get("length", 0))
        k += 1
    return "\n\n".join(parts)


def tokenize_truncate(tokenizer, prompt: str, max_len: int) -> torch.Tensor:
    """Apply the chat template (thinking disabled where supported) and tokenize;
    if too long, keep the first and last max_len // 2 tokens."""
    messages = [{"role": "user", "content": prompt}]
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        chat_kwargs["enable_thinking"] = False
    p = tokenizer.apply_chat_template(messages, **chat_kwargs)
    ids = tokenizer(p, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] > max_len:
        half = max_len // 2
        p = (tokenizer.decode(ids[0, :half], skip_special_tokens=True)
             + tokenizer.decode(ids[0, -half:], skip_special_tokens=True))
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=True).input_ids
    return ids


# ---------------------------------------------------------------------------
# Second-moment accumulation.
# ---------------------------------------------------------------------------

class GramAccumulator:
    """Accumulates G_Qm (group-mean query) and G_K per (layer, KV head), float64."""

    CHUNK = 16384  # tokens per einsum, bounds peak memory at 128k prefill

    def __init__(self):
        self.G_Qm: dict[int, torch.Tensor] = {}
        self.G_K: dict[int, torch.Tensor] = {}

    def __call__(self, layer_idx: int, Q: torch.Tensor, K: torch.Tensor, G: int) -> None:
        # Q: [1, H_q, L, d] post-RoPE, K: [1, H_kv, L, d] post-RoPE;
        # query head h belongs to KV head h // G.
        H_kv, L, d = K.shape[1], K.shape[2], K.shape[-1]
        GQm = torch.zeros(H_kv, d, d, dtype=torch.float64)
        GK = torch.zeros(H_kv, d, d, dtype=torch.float64)
        for s0 in range(0, L, self.CHUNK):
            s1 = min(s0 + self.CHUNK, L)
            Qh = Q[0, :, s0:s1, :].float().reshape(H_kv, G, s1 - s0, d)
            Kh = K[0, :, s0:s1, :].float()
            GK += torch.einsum("hnd,hne->hde", Kh, Kh).double().cpu()
            Qkv = Qh.mean(dim=1)  # group-mean query [H_kv, n, d]
            GQm += torch.einsum("hnd,hne->hde", Qkv, Qkv).double().cpu()
        if layer_idx in self.G_K:
            self.G_Qm[layer_idx] += GQm
            self.G_K[layer_idx] += GK
        else:
            self.G_Qm[layer_idx] = GQm
            self.G_K[layer_idx] = GK


def compute_basis(G_Qm: dict[int, torch.Tensor], G_K: dict[int, torch.Tensor],
                  rank: int) -> dict[int, torch.Tensor]:
    """M = TopEig_r(G_Qm / tr(G_Qm) + G_K / tr(G_K)) per (layer, KV head).
    Returns {layer: [H_kv, r, d]} float32 (rows = basis vectors)."""
    out: dict[int, torch.Tensor] = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for layer, GQm in G_Qm.items():
        A = GQm.double().to(dev)
        B = G_K[layer].double().to(dev)
        tra = A.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
        trb = B.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
        S = A / tra.view(-1, 1, 1) + B / trb.view(-1, 1, 1)
        S = 0.5 * (S + S.transpose(-1, -2))
        _evals, evecs = torch.linalg.eigh(S)  # ascending
        V = evecs[..., -rank:].flip(-1)  # [H_kv, d, r]
        out[layer] = V.transpose(-1, -2).contiguous().float().cpu()
    return out


# ---------------------------------------------------------------------------
# Recording attention forwards: same math as the HF implementations, plus a
# hook that sees post-RoPE prefill Q/K.
# ---------------------------------------------------------------------------

_hook: Optional[Callable[[int, torch.Tensor, torch.Tensor, int], None]] = None


def model_family(name: str) -> str:
    n = name.lower()
    if "gemma-4" in n or "gemma4" in n:
        return "gemma4_text"
    if "qwen3" in n:
        return "qwen3"
    if "ministral-3" in n or "ministral3" in n or "mistral3" in n:
        return "ministral3"
    if "glm" in n:
        return "glm4_moe_lite"
    return "llama"


def _install_generic(family: str) -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if family == "llama":
        from transformers.models.llama.modeling_llama import (
            LlamaAttention as cls, apply_rotary_pos_emb, eager_attention_forward)
        has_qk_norm = False
    elif family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention as cls, apply_rotary_pos_emb, eager_attention_forward)
        has_qk_norm = True
    elif family == "ministral3":
        from transformers.models.ministral3.modeling_ministral3 import (
            Ministral3Attention as cls, apply_rotary_pos_emb, eager_attention_forward)
        has_qk_norm = False
    else:
        raise ValueError(family)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        if has_qk_norm:
            q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        else:
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        if _hook is not None and q.shape[-2] > 1:
            _hook(int(self.layer_idx), q, k, int(self.num_key_value_groups))
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        extra = {}
        if family in ("qwen3", "ministral3"):
            extra["sliding_window"] = getattr(self, "sliding_window", None)
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **extra, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    cls.forward = forward


def _install_gemma4() -> None:
    """Gemma 4: record the global-attention (non-sliding) layers only."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextAttention, apply_rotary_pos_emb, eager_attention_forward)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                shared_kv_states=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = (self.v_proj(hidden_states).view(hidden_shape)
                            if self.v_proj is not None else key_states)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key_states, value_states

        if _hook is not None and query_states.shape[-2] > 1 and not self.is_sliding:
            _hook(int(self.layer_idx), query_states, key_states,
                  int(self.num_key_value_groups))

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    Gemma4TextAttention.forward = forward


def _install_glm4_moe_lite() -> None:
    """GLM-4.7-Flash (MLA): record the absorbed query and the normed latent."""
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteAttention)

    orig_forward = Glm4MoeLiteAttention.forward
    if getattr(orig_forward, "_jssa_recording", False):
        return  # already wrapped (install_recording runs before and after loading)

    def forward(self, hidden_states, *args, **kwargs):
        if _hook is not None and hidden_states.shape[1] > 1:
            b, s = hidden_states.shape[:2]
            qk_nope = self.qk_nope_head_dim
            kvlr = self.kv_lora_rank
            if getattr(self, "q_lora_rank", None) is None:
                q_states = self.q_proj(hidden_states)
            else:
                q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
            H = q_states.shape[-1] // self.qk_head_dim
            q_nope = q_states.view(b, s, H, self.qk_head_dim)[..., :qk_nope]
            W_UK = self.kv_b_proj.weight.view(H, qk_nope + self.v_head_dim, kvlr)[:, :qk_nope, :]
            q_latent = torch.einsum("bshn,hnl->bshl", q_nope.float(), W_UK.float())
            compressed = self.kv_a_proj_with_mqa(hidden_states)
            c_kv = self.kv_a_layernorm(compressed[..., :kvlr])
            _hook(int(self.layer_idx),
                  q_latent.transpose(1, 2).contiguous(),  # [1, H, s, kv_lora]
                  c_kv.unsqueeze(1),  # [1, 1, s, kv_lora]
                  int(H))
        return orig_forward(self, hidden_states, *args, **kwargs)

    forward._jssa_recording = True
    Glm4MoeLiteAttention.forward = forward


def install_recording(family: str) -> None:
    if family == "gemma4_text":
        _install_gemma4()
    elif family == "glm4_moe_lite":
        _install_glm4_moe_lite()
    else:
        _install_generic(family)


# ---------------------------------------------------------------------------
# Model loading.
# ---------------------------------------------------------------------------

def load_model(name: str, family: str, attn_impl: str, device: str, yarn: bool):
    hf_config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    if yarn and family == "qwen3":
        # Qwen3-14B is served with YaRN (factor 4, 32k -> 128k); calibrate with the
        # same RoPE as deployment.
        rs = getattr(hf_config, "rope_scaling", None) or {}
        if rs.get("rope_type", "default") == "default":
            theta = rs.get("rope_theta", getattr(hf_config, "rope_theta", None) or 1000000)
            hf_config.rope_theta = theta
            hf_config.rope_scaling = {"rope_type": "yarn", "factor": 4.0,
                                      "original_max_position_embeddings": 32768}
    if attn_impl == "chunked_math":
        import chunked_attn  # noqa: F401  (registers the attention implementation)
    import chunked_moe
    chunked_moe.install()

    qc = None
    qcfg = getattr(hf_config, "quantization_config", None)
    qm = (qcfg.get("quant_method") if isinstance(qcfg, dict)
          else getattr(qcfg, "quant_method", None)) if qcfg else None
    if qm == "fp8":
        # FP8 checkpoints (Ministral-3-8B-Instruct-2512) are dequantized to bf16.
        from transformers import FineGrainedFP8Config
        import transformers.integrations.finegrained_fp8 as fp8mod
        qc = FineGrainedFP8Config(dequantize=True)
        DQ = fp8mod.Fp8Dequantize
        if not getattr(DQ, "_per_tensor_patch", False):
            orig = DQ._dequantize_one

            def _dq_one(self, quantized, scales, _o=orig):
                if scales.numel() == 1:  # per-tensor scale
                    scales = scales.reshape(1, 1)
                return _o(self, quantized, scales)

            DQ._dequantize_one = _dq_one
            DQ._per_tensor_patch = True

    kw = dict(config=hf_config, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
              device_map={"": device}, quantization_config=qc, trust_remote_code=True)
    if family == "ministral3":
        # Ministral-3-2512 ships as a vision-language model; the text backbone is used.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    return model.eval()


def main() -> None:
    global _hook
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True, help="basis .pt to write")
    ap.add_argument("--rank", type=int, default=32, help="subspace rank r")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=131072, help="tokens per sequence")
    ap.add_argument("--data", default=os.path.join(_HERE, "data", "longalign_calib.jsonl"),
                    help="LongAlign cache written by prepare_longalign.py")
    ap.add_argument("--yarn", action="store_true",
                    help="Qwen3: calibrate with YaRN factor 4 (use when serving with YaRN)")
    ap.add_argument("--attn_impl", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager", "chunked_math"],
                    help="chunked_math for head_dim > 256 (Gemma 4 global layers)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_grams", default=None, help="optionally also save G_Qm / G_K")
    args = ap.parse_args()

    family = model_family(args.model)
    print(f"[calib] model={args.model} family={family} rank={args.rank}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    install_recording(family)  # before loading (accelerate snapshots forwards)
    model = load_model(args.model, family, args.attn_impl, args.device, args.yarn)
    install_recording(family)

    acc = GramAccumulator()
    _hook = acc
    t0 = time.time()
    for idx in range(args.num_samples):
        prompt = load_longalign_packed(args.data, idx, int(args.max_len * 1.15))
        ids = tokenize_truncate(tokenizer, prompt, args.max_len).to(args.device)
        print(f"[calib] sample {idx}: {ids.shape[1]} tokens")
        with torch.no_grad():
            model.model(input_ids=ids, use_cache=False)  # prefill only, no LM head
    _hook = None
    print(f"[calib] {args.num_samples} prefills in {time.time() - t0:.1f}s; "
          f"{len(acc.G_K)} layers recorded")
    if not acc.G_K:
        raise SystemExit("no attention layer was recorded (unsupported model family?)")

    if args.save_grams:
        torch.save({"G_Qm": acc.G_Qm, "G_K": acc.G_K, "model": args.model}, args.save_grams)
    M = compute_basis(acc.G_Qm, acc.G_K, args.rank)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"M": M, "rank": args.rank, "model": args.model,
                "objective": "TopEig(G_Qm/tr + G_K/tr)"}, args.output)
    k0 = sorted(M)[0]
    print(f"[calib] wrote {args.output}: {len(M)} layers, M[{k0}] {tuple(M[k0].shape)}")


if __name__ == "__main__":
    main()
