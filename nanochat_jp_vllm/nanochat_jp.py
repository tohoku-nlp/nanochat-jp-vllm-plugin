# Copyright 2026 The nanochat-jp authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native vLLM implementation of the NanoChatJP architecture.

Ground truth is the HF reference implementation supplied with nanochat-jp
checkpoints. This module implements
the same numerics on top of vLLM's continuous-batching primitives (paged KV
cache, `vllm.model_executor.layers.attention.Attention`, tensor-parallel
linear layers), which requires reworking a few mechanisms that the HF
reference expresses in terms of a padded `(batch, seq)` tensor:

- RoPE uses the nanochat sign convention (not vLLM's neox `rotate_half`), so
  it is implemented by hand instead of via `get_rope()`.
- Sliding-window per layer is passed to `Attention` as
  `layer_window_sizes[i] + 1` to account for vLLM/FlashAttention's exclusive
  `window_size = (W - 1, 0)` convention vs. nanochat's inclusive
  `q_pos - k_pos <= window` convention.
- The "smear" mechanism (mixing in the previous token's pre-smear embedding)
  needs the previous token id even when it is not part of the current
  flattened batch (e.g. the first token of a decode step, or of a
  chunked-prefill continuation). This is implemented with a persistent,
  slot-indexed token-id buffer -- see `NanoChatJPModel.compute_prev_ids`.

Execution is split into an eager python preamble and a compiled core:
`NanoChatJPForCausalLM.forward` (plain python, runs outside dynamo every
step) computes `prev_input_ids` from the forward context's attention
metadata, then calls the `@support_torch_compile`-decorated
`NanoChatJPModel` with pure tensor arguments only. Everything inside the
compiled forward is dynamo-traceable with static control flow; everything
metadata-dependent lives in the preamble. The preamble itself is CUDA-graph
capture-safe (no host/device syncs, no data-dependent shapes, no
allocations after the slot buffer is created), which matters because
vLLM's FULL cudagraph mode wraps and captures the *entire* top-level model
-- preamble included (`gpu_model_runner.py`, `self.model =
CUDAGraphWrapper(self.model, ...)`).

The HF config class is duck-typed (attribute access only, no isinstance
checks): at runtime vLLM may hand this model a dynamically-loaded
`trust_remote_code` config class rather than this package's own
`NanoChatJPConfig`.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .configuration_nanochat_jp import compute_layer_window_sizes


def norm(x: torch.Tensor) -> torch.Tensor:
    """Parameterless RMS norm (matches nanochat.gpt.norm): torch default eps,
    computed in the activation dtype, normalizing over the last dim only."""
    return F.rms_norm(x, (x.size(-1),))


def has_value_embedding(layer_idx: int, num_hidden_layers: int) -> bool:
    """matches nanochat.gpt.has_ve: alternating layers, last layer always included."""
    return layer_idx % 2 == (num_hidden_layers - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """nanochat RoPE sign convention (NOT vLLM's neox rotate_half).

    x: [num_tokens, num_heads, head_dim]. cos/sin: [num_tokens, 1, head_dim / 2].
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def _get_layer_window_sizes(config) -> list[int]:
    sizes = getattr(config, "layer_window_sizes", None)
    if sizes is not None:
        return list(sizes)
    window_pattern = getattr(config, "window_pattern", None)
    if window_pattern is None:
        raise ValueError(
            "NanoChatJP config has neither `layer_window_sizes` nor "
            "`window_pattern`; cannot derive per-layer attention windows."
        )
    return compute_layer_window_sizes(
        window_pattern, config.num_hidden_layers, config.max_position_embeddings
    )


class NanoChatJPMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        # Attribute names must be fc1/fc2 so checkpoint tensors
        # `model.layers.{i}.mlp.{fc1,fc2}.weight` map directly.
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = F.relu(x).square()  # relu^2 activation
        x, _ = self.fc2(x)
        return x


class NanoChatJPAttention(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        num_hidden_layers = config.num_hidden_layers

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.qk_scalar = getattr(config, "qk_scalar", 1.2)
        self.ve_gate_channels = getattr(config, "ve_gate_channels", 12)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Value-embedding gate (ResFormer): reads the first ve_gate_channels
        # channels of the (normed) attention input, produces one gate per kv
        # head. Small and not TP-sharded (see the TP>1 guard in
        # NanoChatJPModel.__init__), kept as a plain nn.Linear so its
        # checkpoint name (`self_attn.ve_gate.weight`) loads via
        # AutoWeightsLoader's default_weight_loader.
        self.has_ve = has_value_embedding(layer_idx, num_hidden_layers)
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, num_kv_heads, bias=False) if self.has_ve else None
        )

        window_sizes = _get_layer_window_sizes(config)
        window = window_sizes[layer_idx]
        max_position_embeddings = config.max_position_embeddings
        # FlashAttention's window_size=(W-1, 0) means "q_pos - k_pos <= W-1",
        # one less than nanochat's inclusive "q_pos - k_pos <= window"
        # convention (see vllm/v1/attention/backends/flash_attn.py, DECODER
        # branch of FlashAttentionImpl.__init__). Passing window + 1 as
        # per_layer_sliding_window makes the two match. Long (full-context)
        # layers -- window >= max_position_embeddings, which always includes
        # the last layer -- get no sliding window at all.
        per_layer_sliding_window = (
            None if window >= max_position_embeddings else window + 1
        )

        self.attn = Attention(
            num_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_embedding: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if value_embedding is not None:
            # gate in (0, 3), one per (token, kv head)
            gate = 3.0 * torch.sigmoid(
                self.ve_gate(hidden_states[..., : self.ve_gate_channels])
            )
            v = v + gate.unsqueeze(-1) * value_embedding

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        # QK norm after RoPE, then the "sharper attention" scalar.
        q = norm(q) * self.qk_scalar
        k = norm(k) * self.qk_scalar

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NanoChatJPDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.self_attn = NanoChatJPAttention(
            config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
        )
        intermediate_size = getattr(config, "intermediate_size", None) or (
            4 * config.hidden_size
        )
        self.mlp = NanoChatJPMLP(
            config.hidden_size, intermediate_size, quant_config, prefix=f"{prefix}.mlp"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_embedding: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            norm(hidden_states), value_embedding, cos, sin
        )
        hidden_states = hidden_states + self.mlp(norm(hidden_states))
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "prev_input_ids": {0: "b"},
    },
)
class NanoChatJPModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        window_sizes = _get_layer_window_sizes(config)
        if not window_sizes or len(window_sizes) != config.num_hidden_layers:
            raise ValueError(
                "NanoChatJP requires one attention window per decoder layer "
                "and at least one decoder layer."
            )
        if window_sizes[-1] < config.max_position_embeddings:
            raise ValueError(
                "NanoChatJP requires full attention in the final decoder layer "
                "for the smear token-id cache: layer_window_sizes[-1] must be "
                f">= max_position_embeddings ({config.max_position_embeddings}), "
                f"got {window_sizes[-1]}."
            )

        if get_tensor_model_parallel_world_size() > 1:
            raise NotImplementedError(
                "NanoChatJP does not support tensor parallelism: ve_gate, "
                "value_embeds and the smear/backout parameters are small, "
                "replicated (non-TP-sharded) modules. Use --tensor-parallel-size 1."
            )
        if get_pp_group().world_size > 1:
            raise NotImplementedError(
                "NanoChatJP does not support pipeline parallelism: the "
                "resid/x0 lambdas and the mid-layer backout residual span "
                "the full depth of the model. Use --pipeline-parallel-size 1."
            )

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads
        num_hidden_layers = config.num_hidden_layers
        vocab_size = config.vocab_size

        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.rope_theta = config.rope_theta
        self.smear_channels = getattr(config, "smear_channels", 24)
        self.backout_layer_idx = num_hidden_layers // 2
        self.block_size = vllm_config.cache_config.block_size

        self.embed_tokens = VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # NOTE: make_layers invokes the factory BY KEYWORD --
        # `layer_fn(prefix=f"{prefix}.{idx}")` (see
        # vllm/model_executor/models/utils.py) -- so the lambda's
        # parameter must be named exactly `prefix`.
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_hidden_layers,
            lambda prefix: NanoChatJPDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        kv_dim = num_kv_heads * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(vocab_size, kv_dim)
                for i in range(num_hidden_layers)
                if has_value_embedding(i, num_hidden_layers)
            }
        )
        self.resid_lambdas = nn.Parameter(torch.ones(num_hidden_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(num_hidden_layers))
        self.smear_gate = nn.Linear(self.smear_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.full((1,), 0.2))

        # The last decoder layer is always full-attention (see the
        # per_layer_sliding_window derivation above: layer_window_sizes[-1]
        # is always >= max_position_embeddings), so its block table stays
        # valid for a request's entire lifetime -- unlike short-window
        # layers, whose KV cache (and therefore block table) may only cover
        # a recent suffix of positions. The smear slot buffer below is keyed
        # off this layer's slot_mapping/block_table for that reason.
        last_attn: NanoChatJPAttention = self.layers[-1].self_attn
        self._last_attn_layer_name = last_attn.attn.layer_name

        # Slot-indexed token-id buffer used by the smear mechanism to recover
        # the previous token id across forward-pass boundaries (decode steps,
        # chunked-prefill continuations). Allocated exactly once, in
        # compute_prev_ids, on the first forward after the KV cache is bound
        # (that forward is always an eager warmup run, never a CUDA graph
        # capture; see _maybe_allocate_slot_buffer). Sized to the number of
        # physical KV slots plus one trailing scratch slot that absorbs
        # writes for padded rows (slot_mapping == -1) without any boolean-
        # mask indexing (which would force a device sync and is illegal
        # during graph capture). Not a model parameter/buffer: pure runtime
        # state that must not appear in the checkpoint.
        self.slot_token_ids: torch.Tensor | None = None
        self._scratch_slot: int = -1
        # PIECEWISE graphs capture the compiled core's input addresses, while
        # this model's Python preamble runs again on every step. Keep its
        # output in one persistent buffer so graph replay sees fresh ids.
        # Allocate lazily during profiling, using the runner's input dtype
        # and device; model construction may happen on the meta device.
        self._prev_input_ids: torch.Tensor | None = None
        self._max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _rotary_cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Recomputed on the fly every forward (no persistent buffer), exactly
        # like the HF reference's NanoChatJPRotaryEmbedding.
        device = positions.device
        channel_range = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.rope_theta ** (channel_range / self.head_dim))
        freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = freqs.cos().to(dtype).unsqueeze(1)  # [num_tokens, 1, head_dim / 2]
        sin = freqs.sin().to(dtype).unsqueeze(1)
        return cos, sin

    def _get_last_layer_metadata(self):
        """Return this forward pass's AttentionMetadata for the last decoder
        layer, or None if unavailable (e.g. a memory-profiling dummy run)."""
        ctx = get_forward_context()
        raw = ctx.attn_metadata
        if raw is None:
            return None
        if isinstance(raw, list):
            # DBO / speculative decoding: [0] is the base-model metadata dict.
            raw = raw[0]
        if isinstance(raw, dict):
            return raw.get(self._last_attn_layer_name)
        return raw

    def _maybe_allocate_slot_buffer(self, device: torch.device) -> None:
        """Allocate the slot buffer once the KV cache is bound. Deterministic,
        exact-size, one-time allocation -- never grows, never drops contents.

        Safe-by-ordering: vLLM's engine flow is memory profiling (KV cache
        unbound, attn_metadata None) -> KV cache bind -> eager compile/warmup
        dummy runs -> CUDA graph capture, and `cudagraph_num_of_warmups` is
        forced to 1 whenever cudagraphs are enabled, so the first forward
        that reaches this code with the KV cache bound is always an eager
        run. The stream-capture check below turns any violation of that
        ordering into a loud error instead of a corrupted graph.
        """
        if self.slot_token_ids is not None:
            return
        kv_cache = self.layers[-1].self_attn.attn.kv_cache
        if kv_cache is None or kv_cache.numel() == 0:
            return  # pre-bind (memory profiling); nothing to allocate yet
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "NanoChatJP smear slot buffer allocation was reached during "
                "CUDA graph capture. This should be impossible (the first "
                "post-KV-cache-bind forward is an eager warmup run); please "
                "report this, and serve with --enforce-eager as a workaround."
            )
        # kv_cache layout for FA is (num_blocks, 2, block_size, ...), so
        # shape[0] * block_size is the exact number of physical slots. One
        # extra trailing scratch slot absorbs padded-row writes.
        num_slots = kv_cache.shape[0] * self.block_size
        self._scratch_slot = num_slots
        self.slot_token_ids = torch.zeros(
            num_slots + 1, dtype=torch.int64, device=device
        )

    def compute_prev_ids(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Eager (non-compiled) preamble: previous-token ids for the smear.

        Runs outside dynamo every step, so it may read the forward context.
        Every tensor op in here must stay CUDA-graph capture-safe: static
        shapes only, no host/device syncs (no `.item()`, no boolean-mask
        advanced indexing, no `.tolist()`), and no allocations that depend
        on batch content -- under FULL cudagraph mode this code is captured
        along with the rest of the model and replayed without re-running
        the python.
        """
        num_tokens = input_ids.shape[0]
        if num_tokens > self._max_num_tokens:
            raise ValueError(
                f"NanoChatJP received {num_tokens} tokens, exceeding the "
                f"previous-token buffer capacity ({self._max_num_tokens})."
            )
        if self._prev_input_ids is None:
            if input_ids.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "NanoChatJP's previous-token buffer must be initialized "
                    "during profiling or warmup, before CUDA graph capture."
                )
            self._prev_input_ids = input_ids.new_empty(self._max_num_tokens)
        if (
            self._prev_input_ids.dtype != input_ids.dtype
            or self._prev_input_ids.device != input_ids.device
        ):
            raise ValueError(
                "NanoChatJP input_ids dtype and device must remain unchanged "
                "after the previous-token buffer is initialized."
            )
        prev_ids = self._prev_input_ids[:num_tokens]
        # Default: within-batch shift. Correct for every token except the
        # first token of each request's scheduled chunk (fixed up below);
        # index 0 is always such a chunk start too, so this never reads
        # input_ids[-1]. In-place writes preserve the address captured by
        # PIECEWISE graphs; under FULL graphs these writes are captured too.
        prev_ids[:1].zero_()
        prev_ids[1:].copy_(input_ids[:-1])

        self._maybe_allocate_slot_buffer(input_ids.device)

        metadata = self._get_last_layer_metadata()
        if metadata is None:
            # Memory-profiling dummy run: no real slot mapping/block table to
            # read. Skip the buffer entirely but keep the surrounding tensor
            # ops (smear gate, embedding lookups) unchanged so profiling
            # still sees representative memory usage.
            return prev_ids

        slot_mapping = getattr(metadata, "slot_mapping", None)
        query_start_loc = getattr(metadata, "query_start_loc", None)
        block_table = getattr(metadata, "block_table", None)
        missing = [
            name
            for name, value in (
                ("slot_mapping", slot_mapping),
                ("query_start_loc", query_start_loc),
                ("block_table", block_table),
            )
            if value is None
        ]
        if missing:
            raise NotImplementedError(
                f"NanoChatJP's smear mechanism requires attention metadata "
                f"field(s) {missing}, which are missing on backend "
                f"{type(metadata).__name__}."
            )
        if self.slot_token_ids is None:
            raise RuntimeError(
                "NanoChatJP: attention metadata is present but the KV cache "
                "of the last decoder layer is not bound, so the smear slot "
                "buffer could not be sized. This breaks the assumed engine "
                "ordering (profile -> KV bind -> warmup -> capture)."
            )

        device = input_ids.device
        slot_mapping = slot_mapping[:num_tokens]

        # Write this forward pass's tokens into the slot buffer. Padded rows
        # (slot_mapping == -1, e.g. cudagraph tail padding) are redirected to
        # the trailing scratch slot, which is never read: the read path below
        # only uses block-table-derived slots, and position-0 reads are
        # discarded by the has_prev mask in the compiled forward. masked_fill
        # keeps the write static-shaped (no nonzero()/device sync), unlike
        # boolean-mask indexing.
        safe_slots = slot_mapping.masked_fill(slot_mapping < 0, self._scratch_slot)
        # The v1 runner hands the model int32 input_ids; index_put requires
        # source/destination dtypes to match, so cast explicitly.
        self.slot_token_ids[safe_slots] = input_ids.to(self.slot_token_ids.dtype)

        # Fix up chunk-start rows. All ops below are static-shaped gathers/
        # scatters sized by num_reqs (padded and fixed per cudagraph batch
        # descriptor). Padded request rows are garbage-in-discarded-out:
        # their query_start_loc entries repeat the last real cumsum (a padded
        # token row), their positions are 0, and padded block-table rows are
        # the reserved null block -- every value read or written for them is
        # masked off by has_prev = positions > 0 in the compiled forward.
        num_reqs = query_start_loc.numel() - 1
        if num_reqs > 0:
            starts = query_start_loc[:num_reqs].to(torch.int64)
            p = positions[starts]
            # Clamped to 0 for true sequence starts (p == 0); the gathered
            # value is discarded there by the has_prev mask.
            prev_pos = (p - 1).clamp(min=0)
            block_idx = (prev_pos // self.block_size).to(torch.int64)
            block_offset = (prev_pos % self.block_size).to(torch.int64)
            req_idx = torch.arange(num_reqs, device=device)
            phys_block = block_table[req_idx, block_idx].to(torch.int64)
            prev_slot = phys_block * self.block_size + block_offset
            prev_ids[starts] = self.slot_token_ids[prev_slot].to(prev_ids.dtype)

        return prev_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        prev_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Compiled (dynamo, fullgraph) core: pure tensor args, static
        # control flow only. All forward-context/metadata access happens in
        # the eager preamble (NanoChatJPForCausalLM.forward), which passes
        # the result in as prev_input_ids.
        x_hat = norm(self.embed_tokens(input_ids))
        prev_x_hat = norm(self.embed_tokens(prev_input_ids))
        has_prev = (positions > 0).unsqueeze(-1)
        gate = self.smear_lambda.to(x_hat.dtype) * torch.sigmoid(
            self.smear_gate(x_hat[:, : self.smear_channels])
        )
        x = x_hat + has_prev * gate * prev_x_hat

        x0 = x
        cos, sin = self._rotary_cos_sin(positions, x.dtype)

        x_backout = None
        for i, layer in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            key = str(i)
            value_embedding = None
            if key in self.value_embeds:
                value_embedding = self.value_embeds[key](input_ids).to(x.dtype)
                value_embedding = value_embedding.view(-1, self.num_kv_heads, self.head_dim)
            x = layer(x, value_embedding, cos, sin)
            if i == self.backout_layer_idx:
                x_backout = x

        assert x_backout is not None
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class NanoChatJPForCausalLM(nn.Module):
    hf_to_vllm_mapper = NanoChatJPModel.hf_to_vllm_mapper
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = NanoChatJPModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Eager python preamble (this method is NOT compiled): validate
        # unsupported inputs, derive prev_input_ids from the forward
        # context's attention metadata, then hand pure tensor args to the
        # @support_torch_compile-decorated model.
        if inputs_embeds is not None:
            raise NotImplementedError(
                "NanoChatJP requires input_ids: value embeddings and the "
                "smear mechanism are token-id lookups, so inputs_embeds "
                "cannot be supported."
            )
        if intermediate_tensors is not None:
            raise NotImplementedError(
                "NanoChatJP does not support pipeline parallelism."
            )
        positions = positions[: input_ids.shape[0]]
        prev_input_ids = self.model.compute_prev_ids(input_ids, positions)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            prev_input_ids=prev_input_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


__all__ = ["NanoChatJPForCausalLM", "NanoChatJPModel"]
