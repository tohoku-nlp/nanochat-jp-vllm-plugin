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
"""vLLM out-of-tree plugin registering the NanoChatJP architecture.

Discovered via the `vllm.general_plugins` entry point (see pyproject.toml).
vLLM calls `register()` once per process (worker, engine core, and the API
server front-end); it may be called more than once within a process across
different code paths, so it must be idempotent.
"""

_ARCH_NAME = "NanoChatJPForCausalLM"
_MODEL_CLS_PATH = "nanochat_jp_vllm.nanochat_jp:NanoChatJPForCausalLM"


def register() -> None:
    """Entry point invoked by vLLM's plugin loader (`load_general_plugins`)."""
    from vllm import ModelRegistry

    if _ARCH_NAME not in ModelRegistry.get_supported_archs():
        # Register with the lazy (string) form so importing this module never
        # imports torch/vLLM model code eagerly -- that avoids
        # "Cannot re-initialize CUDA in forked subprocess" when the plugin is
        # loaded in a process that later forks workers.
        ModelRegistry.register_model(_ARCH_NAME, _MODEL_CLS_PATH)

    # Best-effort convenience: also register the config class with
    # transformers.AutoConfig under model_type "nanochat_jp", so that
    # `AutoConfig.from_pretrained(...)` (without trust_remote_code) resolves
    # to this package's config. This is not required for correctness: the
    # checkpoint's config.json carries an `auto_map` entry, so
    # `trust_remote_code=True` works regardless of whether this succeeds.
    try:
        from transformers import AutoConfig

        from .configuration_nanochat_jp import NanoChatJPConfig

        AutoConfig.register("nanochat_jp", NanoChatJPConfig)
    except Exception:
        pass


__all__ = ["register"]
