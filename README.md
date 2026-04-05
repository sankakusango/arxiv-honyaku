# arxiv-honyaku

arXiv の TeX ソースを取得し、チャンク単位で翻訳して PDF を再コンパイルするための新規実装です。

## 設計方針

- `core`: 純粋な業務ロジック
- `adapters`: arXiv 通信、LLM 通信、latexmk 実行、チェックポイント保存
- `interfaces`: CLI / Web

ロジック層は UI や具体的な通信手段を知りません。  
CLI、Web、ライブラリは同じ `core` を呼び出すだけです。

## 使い方 (CLI)

```bash
pip install -e .
arxiv-honyaku 2401.01234
arxiv-honyaku --force 2401.01234
```

CLI は, 既にコンパイル成功済みの成果物が `./runs/<arxiv-id>/` にあれば自動で全処理をスキップします。  
最初からやり直したいときだけ `--force` を使います。

設定は `arxiv-honyaku.toml` で管理します。

```toml
[llm]
provider = "deepseek"   # "openai" | "deepseek" | "vllm" | "ollama"
model = "deepseek-chat"
base_url = "https://api.deepseek.com"
# api_key = "sk-..."  # 直接書く場合のみ

[run]
workspace_root = "./runs"
max_chunk_chars = 2200
concurrency = 10
max_retries = 3
translate_section_titles = false
japanese_layout_mode = "adaptive" # "safe" | "adaptive" | "preserve"
japanese_font_mode = "compat"     # "compat" | "paper-like"
texlive_versions = [2025, 2023]   # 例: 2015 / [2015, 2013] / [2015, 2015]

[prompts]
catalog_path = "./prompts/translation-prompts.toml"
profile = "ja_default"
```

`openai` は `OPENAI_API_KEY` を自動で参照します。  
`deepseek` は `OPENAI_API_KEY` を優先して参照し, 必要なら `DEEPSEEK_API_KEY` も利用できます。  
さらに, 設定ファイルと同じディレクトリにある `.env` も自動で読み込みます。追加の独自環境変数は不要です。

`translate_section_titles = false` にすると, `\section`, `\subsection`, `\subsubsection`, `\chapter` のタイトルは原文のまま残します。

`japanese_layout_mode` は日本語化後のレイアウト補正方針です。`safe` は安全優先, `adaptive` は元の雰囲気を残しつつ回り込みを崩れにくくし, `preserve` は原文のレイアウト構造をそのまま残します。

`japanese_font_mode = "paper-like"` を使うと, 現在の `pdflatex + CJKutf8` ベースの動作は保ったまま, 日本語フォントだけ IPAex 明朝寄りへ切り替えます。従来どおりの挙動を使いたい場合は `compat` のままにしてください。

`texlive_versions` は PDF コンパイル時に試す TeX Live バージョン順です。  
例えば `[2025, 2023]` なら 2025 で失敗したとき 2023 を試します。`2015` のような単体指定や `[2015, 2015]` のような同一バージョン再試行も可能です。未指定時は `/opt/texlive` 配下から年ディレクトリを自動検出して降順で試します。

`ollama` / `vllm` も OpenAI 互換 API として同じクライアント経由で実行します。
ただし `TranslateGemma` は推奨プロンプト形式が通常のchatモデルと異なるため, `profile = "translategemma_ja"` を使ってください。

DeepSeek を non-thinking mode で使う場合は, `model = "deepseek-chat"` を指定してください。

プロンプト本文は `prompts/translation-prompts.toml` に一元化しています。  
プロンプトを変更・追加したい場合はこのファイルだけ編集してください。

```toml
[profiles.default]
output_open_tag = "<translated>"
output_close_tag = "</translated>"
default_system_prompt = """..."""
user_prompt_template = """...{{CHUNK_TEXT}}..."""

[[profiles.default.system_rules]]
pattern = "ollama:*"
system_prompt = """..."""
```

`arxiv-honyaku.toml` 側では `profile` を切り替えるだけで運用できます。
（例: `default`, `literal_guard_v2`, `ja_default`）

## Web UI

```bash
pip install -e ".[web]"
uvicorn arxiv_honyaku.interfaces.web:create_app --factory --reload
```

## Docker (DeepSeek / OpenAI)

`.env` の `OPENAI_API_KEY` または環境変数を使って Web UI を起動します。

```bash
export OPENAI_API_KEY="sk-..."
docker compose up --build
```

ブラウザで `http://localhost:8000` を開きます。

## Docker (Ollama 同居)

Web UI と Ollama サーバを同時に起動します。

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.ollama.yml exec ollama ollama pull translategemma:4b
```

ブラウザで `http://localhost:8000` を開きます。  
この構成では `arxiv-honyaku.ollama.toml` が使われます。

## ゼロから実行, Ollama + Small Translate Gemma で論文を和訳

GPUメモリ量を見て, `TranslateGemma` の 4B, 12B, 27B から適切なサイズを選ぶスクリプトを用意しています.
GPUで実行する場合は, NVIDIA Container Toolkit を有効化した Docker 環境が必要です.
Docker からGPUが使えない場合は, スクリプトが自動で CPU-only 実行へフォールバックします.

```bash
bash scripts/run_ollama_gemma_translate.sh
bash scripts/run_ollama_gemma_translate.sh --force
```

サンプルの arXiv 論文IDを指定する場合:

```bash
bash scripts/run_ollama_gemma_translate.sh 1706.03762
bash scripts/run_ollama_gemma_translate.sh --force 1706.03762
```

このスクリプトは以下を順番に実行します.

1. `nvidia-smi` でGPU VRAMを取得し, `translategemma` のサイズを選択.
2. `docker compose` で `web` と `ollama` を起動.
3. 選択モデルを `ollama pull`.
4. 選択モデル名を環境変数で渡して `arxiv-honyaku` CLIで和訳を実行.

すでに `./runs/<arxiv-id>/build/*.pdf` がある場合は, スクリプト側でも最初に検出して全体をスキップします.  
完全にやり直したい場合は `--force` を付けてください。

モデル選択の基準:

1. VRAM 24GB以上: `translategemma:27b`
2. VRAM 12GB以上: `translategemma:12b`
3. それ未満または未検出: `translategemma:4b`

出力は `./runs/<arxiv-id>/` 以下に保存されます.

## 注意

- Docker イメージは, arXiv 論文の再コンパイル成功率を優先して `texlive-full` を含みます. そのぶん初回ビルドはかなり重くなります.
- `latexmk` が必要です。
- `vllm` / `ollama` を使う場合は `base_url` を `toml` 側で設定してください。
