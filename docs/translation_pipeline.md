# 翻訳前処理パイプライン

ユーザ向け手順書. v04 の反省を踏まえ, 「文字列の置換と復元」を軸に再設計したもの.

---

## 設計方針

1. **非可逆な操作は最小限にし, パイプラインの先頭で1度だけ行う**
   - コメント削除, 安全な空白整理, TeX 表記の小さな正規化がこれに当たる.
2. **可逆な操作はすべて placeholder への文字列置換として行い, 置換プールを保存する**
   - 図, 式, 連続コマンドなど LLM に触らせたくないものを `XQK0001` のような placeholder に退避.
   - 元本文中の `XQK` は固定で `XQK0000` に退避し, 生成 placeholder との衝突を避ける.
   - チャンクと置換プールを JSON で保存しておけば, 翻訳前 (= 非可逆操作だけ済んだ状態) にいつでも戻せる.
3. **チャンクと置換プールは独立**
   - チャンク同士を素朴に連結すれば置換済み本文に戻る. 置換は別関数で逆置換するだけ.
4. **賢い置換と正確な復元を分離**
   - 置換ロジックがどれほど雑でも, 置換時に取った original 文字列をそのまま戻せば常に復元できる.

---

## パイプライン全体

入力: `source_dir` (= ソースツリーのルート), 出力: 各 `.tex` に対する JSON ファイル.

```
[各 .tex ファイル]
        │
        ├─ ① clean_tex          (非可逆: コメント除去 + 安全な空白整理, verbatim 内は保護)
        │
        ├─ ② pad_unitless_si    (非可逆: 単位なし \SI{...} に空単位 {} を補う)
        │
        ├─ ③ substitute_blocks  (可逆: 数式環境, figure/table の caption 以外を placeholder へ)
        │
        ├─ ④ substitute_commands (可逆: 単発の \\cmd{...} を placeholder へ. section/caption 等は除外)
        │
        ├─ ⑤ filter             (コマンドのみのファイルは translatable=False で記録)
        │
        ├─ ⑥ chunk              (本文/caption 中身だけを翻訳対象 chunk へ. 見出しは構造 chunk)
        │
        └─ ⑦ save_json          ({ chunks, pool, irreversible_ops, ... } を JSON へ)
```

---

## 各ステップの詳細

### ① `clean_tex` (非可逆)

行頭の `%` で始まる純コメント行と, 行内の未エスケープ `%` 以降を削除する.
`verbatim` / `lstlisting` / `minted` 環境内の `%` と, 本文中の `\%` は保護.
さらに, 同じく verbatim 系環境の外だけで以下を整理する.

- 行末の半角スペース/タブを削除.
- 空白だけの行を空行にする.
- 連続する空行を 1 つの空行に畳む.

なぜ非可逆: コメントを placeholder で残すと数が膨大になり, チャンクサイズに悪影響. 翻訳に不要なら捨てる方が素直.
空白整理も TeX のコンパイル結果に影響しにくい範囲だけに限定する.

### ② `pad_unitless_si` (非可逆)

単位引数を欠いた `\SI{8e-5}` のような siunitx 呼び出しを `\SI{8e-5}{}` に正規化する.
arXiv ソースではこの省略形が混入することがあり, 英文のままでは通っても,
日本語化後に直後の和文文字列との組み合わせで siunitx が失敗する場合がある.

この処理は `substitute_commands` の前に実行する. そのため `\SI{...}{}` 全体が
placeholder として退避され, 翻訳対象 chunk には露出しない.

### ③ `substitute_blocks` (可逆, 構造単位)

以下を順番に検出して placeholder へ退避:

1. 数式環境: `\begin{equation}...\end{equation}`, `\begin{align}...\end{align}` 等
2. インライン数式: `$...$`, `\(...\)`, `\[...\]`
3. 図表環境: `\begin{figure}...\end{figure}`, `\begin{table}...\end{table}` (含 `*` 付き, `wrap*`)
   - `\caption{...}` / `\caption[...]{...}` は本文側へ残し, 画像・表本体・label など caption 以外を placeholder に退避.
4. bibliography 環境: `thebibliography` は block 全体を placeholder に退避.
5. 表本体環境: `tabular`, `NiceTabular`, `longtable`, `tblr` など (列区切りの `&` を含むため翻訳対象外)
6. リスト/コード環境: `verbatim`, `lstlisting`, `minted` など (本文翻訳の対象外)

各 placeholder は 1 つの `Substitution` レコードを生成し, kind と元文字列を保持.

### ④ `substitute_commands` (可逆, コマンド単位)

連続するインライン LaTeX コマンドをまとめて 1 つの placeholder に. 例: `\cite{a}~\ref{b}` → `XQK0042`.

- 対象: 基本的にすべての LaTeX コマンド.
- 引数 `{...}` を含む形式と, `\textbackslash` のような単独コマンドの両方をカバー.
- `\section`, `\subsection`, `\caption`, `\footnote`, `\textbf`, `\emph` などの **構造/装飾/本文保持コマンドは置換対象外**.
- `\title`, `\author` などの metadata command は丸ごと placeholder に退避し, 翻訳対象にしない.
- `\textbf{xxx} yyy` のように後ろにテキストが続くケースは, コマンド部分 (引数まで) のみを退避. 続く ` yyy` は本文として残す.
- 連続するコマンド (空白だけを挟むもの) は 1 つの placeholder にまとめる.

### ⑤ `filter`

ファイル内の置換後本文が空白と placeholder のみの場合, ファイル全体を `translatable=False` で記録.
チャンクレベルでも同様の判定を ⑥ で行う.

### ⑥ `chunk`

- 左から TeX を走査し, `\section{...}` / `\subsection{...}` / `\paragraph{...}` などの見出し command は `translatable=False, skip_reason="structure"` として保持する. 見出し中身は翻訳対象にしない.
- section の内外を前提にせず, 見出しの外へそのまま書かれた本文も通常の body chunk として扱う.
- `\caption{...}` / `\caption[...]{...}` と `\abstract{...}` は command wrapper と中身に分け, 中身だけを独立した `translatable=True` chunk にする.
- `text` には末尾改行を入れず, chunk 後続の改行は `join_after` に保存する. 翻訳で `text` 末尾の改行が揺れても, 復元時は `text + join_after` で構造側の改行を戻す.
- placeholder と空白だけの chunk: `translatable=False, skip_reason="command_only"`.

### ⑦ `save_json`

JSON スキーマ:

```jsonc
{
  "version": 1,
  "source_path": "sections/intro.tex",        // source_dir からの相対パス
  "is_main": false,                           // メイン .tex か (復元時の CJK 注入対象か)
  "irreversible_ops": [                       // 適用した非可逆操作の記録
    "clean_tex",
    "pad_unitless_si"
  ],
  "substitutions": [                          // 置換プール
    {"placeholder": "XQK0001", "kind": "math_env", "original": "\\begin{equation}...\\end{equation}"},
    {"placeholder": "XQK0002", "kind": "math_inline", "original": "$x^2$"},
    {"placeholder": "XQK0003", "kind": "figure_env", "original": "\\begin{figure}...\\end{figure}"},
    {"placeholder": "XQK0004", "kind": "command_block", "original": "\\cite{a}~\\ref{b}"}
  ],
  "chunks": [
    {"index": 0, "section_path": "preamble",      "translatable": false, "skip_reason": "structure", "text": "\\section{Intro}", "join_after": "\n"},
    {"index": 1, "section_path": "Intro",         "translatable": true,  "skip_reason": null,        "text": "Body text.",       "join_after": "\n"},
    {"index": 2, "section_path": "Intro",         "translatable": false, "skip_reason": "structure", "text": "\\caption{",      "join_after": ""},
    {"index": 3, "section_path": "Intro/caption", "translatable": true,  "skip_reason": null,        "text": "Caption text.",    "join_after": ""},
    {"index": 4, "section_path": "Intro",         "translatable": false, "skip_reason": "structure", "text": "}",               "join_after": "\n"}
  ]
}
```

---

## 復元 (round-trip) 手順

```
JSON を読む
  ├─ 全 chunks の text + join_after を index 順に連結 → 置換後の本文を得る
  └─ substitutions を後ろから順に 1 回だけ処理し, 本文中の placeholder を original に置換
       (original に別 placeholder が含まれる階層置換も復元でき, ループしない)
```

**不変条件**: この復元結果は ①② 適用直後の本文と完全に一致する. ③以降は文字列置換のみで構造変更を伴わないため.

(①② は非可逆なので, 元の `.tex` までは戻せない. 戻したければ source_tree から元ファイルをコピーしておく.)

`tests/test_reconstruction.py` はこの round-trip 確認も含めて source tree 全体を検証する.

---

## source_tree.json

source tree は source root と, その配下の entry 一覧を保存する.

```jsonc
{
  "source_root": "source",
  "entries": [
    {"path": "ms.tex", "is_dir": false, "is_tex": true},
    {"path": "Figures", "is_dir": true, "is_tex": false}
  ]
}
```

`prepare_from_source_tree(source_tree_path, prep_dir, ...)` はこの JSON を読み,
`is_tex=true` の entry だけを `prep_dir` に `.tex.json` として出力する.

`reconstruct_from_source_tree(source_tree_path, prep_dir, output_dir)` は TeX を
prep JSON から復元し, 非 TeX ファイルは元 source tree からコピーする.

---

## 翻訳実行

`translate_prep_dir(prep_dir, output_jsonl, ...)` は `prep_dir` 配下の
`translatable=true` chunk だけを LLM に投げ, 結果を JSONL に書く.

- 同時 request 数は `config.toml` の `llm.max_concurrency`.
- prompt/再試行/検証は `llm.translation_logic` で切り替える.
- JSONL 書き込みは完了した chunk を受け取る側で直列に行い, 同時書き込みしない.
- `general_chat` は初回の翻訳が不正なら, その応答とエラー内容を渡してもう 1 回だけ翻訳する.
- LLM 応答は `<translated>...</translated>` から本文だけを取り出す. tag の前後に任意の説明文が付くことは許すが, tag 欠落, open/close 数不一致, 複数 pair はエラー.
- 翻訳前後で `XQK0001` のような placeholder の種類, 個数, 順序が変わった場合はエラー.
- 翻訳で壊れやすい LaTeX 特殊文字は, 平文中の bare `&`, `%`, `$`, `#`, `_`, `^` を escape し, `XQK0001は...` のように placeholder 後ろの空白が落ちた場合は必要な空白を戻す.
- 和文句読点は論文 TeX として扱いやすいように `、` → `, `, `。` → `. ` などへ機械的に正規化する.

```toml
[llm]
max_concurrency = 2
translation_logic = "general_chat"
```

JSONL は chunk ごとに 1 行:

```jsonc
{
  "source_path": "introduction.tex",
  "chunk_index": 12,
  "section_path": "Introduction",
  "source_text": "The model ...",
  "translated_text": "このモデルは...",
  "status": "ok",
  "logic": "general_chat",
  "attempts": [{"attempt": 1, "ok": true, "error": null}],
  "error": null
}
```

`reconstruct_translated_from_source_tree(source_tree_path, prep_dir, translated_dir, translations_jsonl=...)`
は JSONL から `status="ok"` の翻訳だけを使って translated source tree を作る.
翻訳失敗や JSONL に存在しない chunk は原文 chunk のまま復元する.
非 TeX ファイルは元 source tree からコピーする.
復元した TeX には, mode ごとに `japanese_setup` を適用する. メイン `.tex` には CJK
パッケージと本文ラッパを注入し, 全 `.tex` に layout 補正をかける.

prep 生成, 翻訳, translated source 再構成, build までまとめて行う場合は:

```python
from arxiv_honyaku import translate_source_tree_to_pdf

translate_source_tree_to_pdf(
    source_tree_path,
    prep_dir,
    translations_jsonl,
    translated_dir,
    build_root,
)
```

実行用の最小スクリプトとして `tests/test_translate.py` を置いている.

---

## ファイル単位 vs ツリー単位

- 1 つの .tex に対し 1 つの JSON.
- JSON は `source_dir` からの相対パスを保ったまま `rel.with_suffix(".tex.json")` で作る.
  例: `sections/intro.tex` → `prep/sections/intro.tex.json`.
  別階層に同名 `.tex` があっても JSON パスは衝突しない.
- ツリーレベルの操作は wrapper で並列実行. ファイル間で状態は共有しない (placeholder index も file-local).

---

## 既知の TODO / 将来拡張

- 翻訳結果との合流処理 (= 置換プールの一部 placeholder を翻訳済みテキストで置き換える形).
- `translation_preparation` JSON 同士の差分マージ (再翻訳サポート).
