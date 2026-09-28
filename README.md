# nanochat-jp-vllm

[日本語](#日本語) | [English](#english)

## 日本語

nanochat-jp の `NanoChatJPForCausalLM` を vLLM で動かすためのプラグインです。vLLM の paged attention と連続バッチ処理を利用できます。


### インストール

**Docker イメージ [`vllm/vllm-openai:v0.25.1`](https://hub.docker.com/layers/vllm/vllm-openai/v0.25.1/images/sha256-f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268) を変換した Singularity イメージで動作検証済みです。`v0.25.1` はイメージのタグを指します。他のイメージタグでは未検証です。**

Python 3.10 以上と、CUDA 対応 GPU、`torch`、`vllm`、`transformers` が必要です。上記の検証済みイメージに含まれる環境を使用してください。

リポジトリのルートで以下のコマンドを実行してください。

```sh
pip install -e .
```

依存パッケージは自動インストールされません。上記のパッケージを用意した環境にインストールしてください。

### 起動

プラグインはインストール後に自動で読み込まれます。以下は GPU 1 台での起動例です。チェックポイントのパスを置き換えて実行してください。

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve /path/to/nanochat-jp-checkpoint \
    --served-model-name tohokunlp/nanochat-jp \
    --trust-remote-code \
    --enforce-eager \
    --chat-template-content-format string \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --data-parallel-size 1 \
    --seed 42 \
    --port 5000
```

この例では GPU 0 を使用し、BF16・eager 実行で OpenAI 互換 API をポート `5000` に起動します。`--enforce-eager` によりコンパイルと CUDA グラフを無効にします。API に指定するモデル名は `tohokunlp/nanochat-jp` です。

`--max-model-len 32768` は入出力を合わせた最大トークン数、`--gpu-memory-utilization 0.9` は GPU メモリ使用率の設定です。モデルと GPU 容量に合わせて調整してください。`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` はモデル設定を超える長さの指定を許可しますが、その長さでのモデル品質を保証するものではありません。

**本プラグインのコンパイル・CUDA グラフを使用する実行は experimental（実験的機能）です。** `--enforce-eager` を省略すると有効になります。基本的な検証は通過していますが、検証範囲は限定的です。通常は上記の eager 実行を推奨します。

### 主な制限

- テンソル並列・パイプライン並列は、いずれも並列数 1 のみ対応しています。
- `inputs_embeds` による入力には対応していません。
- 投機的デコーディングとプレフィックスキャッシュは未検証です。

### 検証

以下は起動例と同じ eager 実行での検証です。
HF transformers の参照実装と、生成トークンおよび logprob を比較できます。HF でも読み込めるチェックポイントを用意し、CUDA 対応 GPU のある環境で実行してください。

```sh
python tools/check_parity.py \
    --model_path /path/to/nanochat-jp-checkpoint \
    --enforce_eager
```

---

## English

A vLLM plugin for nanochat-jp's `NanoChatJPForCausalLM`, with support for vLLM's paged attention and continuous batching.

### Installation

**Tested with a Singularity image converted from the Docker image [`vllm/vllm-openai:v0.25.1`](https://hub.docker.com/layers/vllm/vllm-openai/v0.25.1/images/sha256-f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268). `v0.25.1` refers to the image tag. Other image tags have not been tested.**

Requires Python 3.10 or later, a CUDA-capable GPU, and `torch`, `vllm`, and `transformers`. Use the environment provided by the tested image above.

Run the following command from the repository root:

```sh
pip install -e .
```

Dependencies are not installed automatically. Install this plugin into an environment that already provides the packages listed above.

### Serving

The plugin loads automatically after installation. This example uses one GPU. Replace the checkpoint path before running:

```sh
CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve /path/to/nanochat-jp-checkpoint \
    --served-model-name tohokunlp/nanochat-jp \
    --trust-remote-code \
    --enforce-eager \
    --chat-template-content-format string \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --data-parallel-size 1 \
    --seed 42 \
    --port 5000
```

This starts an OpenAI-compatible API on port `5000`, using GPU 0 with BF16 and eager execution. `--enforce-eager` disables compilation and CUDA graphs. Use `tohokunlp/nanochat-jp` as the model name in API requests.

`--max-model-len 32768` sets the maximum combined input and output length, and `--gpu-memory-utilization 0.9` sets GPU memory utilization. Adjust these for your model and GPU capacity. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` permits a length beyond the model configuration; it does not guarantee model quality at that length.

**Execution with compilation and CUDA graphs is experimental in this plugin.** Enable it by omitting `--enforce-eager` when serving. Basic verification has passed, but coverage remains limited. Eager execution as shown above is recommended for normal use.

### Limitations

- Tensor parallelism and pipeline parallelism are supported only with a parallel size of 1.
- `inputs_embeds` is not supported.
- Speculative decoding and prefix caching are untested.

### Verification

The following checks eager execution, matching the serving example.
Compare generated tokens and logprobs against the HF transformers reference implementation. Use a checkpoint that can also be loaded by HF, and run in an environment with a CUDA-capable GPU.

```sh
python tools/check_parity.py \
    --model_path /path/to/nanochat-jp-checkpoint \
    --enforce_eager
```
