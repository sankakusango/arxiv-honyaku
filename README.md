# arxiv-honyaku

arXiv の TeX ソースを取得し, チャンク単位で翻訳して PDF を再コンパイルするための新規実装です. `arxiv-honyaku-v04` の書き直しです.

## 開発者向けドキュメント

- [docs/install_texlive.md](docs/install_texlive.md) — Docker イメージに TeX Live を入れるスクリプトの仕様.

## Web UI

内輪向けの小さな Web UI を同梱しています. ログインは作らず, 管理者がユーザーごとの
リンクを発行して共有する想定です.

```bash
arxiv-honyaku-web --host 0.0.0.0 --port 8000
```

初回起動時に管理者リンクが表示されます. その画面からユーザーを作成するか,
サーバを起動せずに次のコマンドでリンクを発行できます.

```bash
arxiv-honyaku-web --create-user Alice --base-url http://localhost:8000
```

Web UI では arXiv の URL または ID を入力し, `config.toml` の
`run.japanese_layout_modes` から複数の layout mode を選んで翻訳を実行できます.
入力に `v2` のようなバージョンが含まれていればその版を使い, 無印なら arXiv API から
最新バージョンを解決してから実行します.
PDF 候補は論文・バージョン単位で共有され, スターと個人メモだけがユーザー別に保存されます.
生成された候補が合わない場合は TeX 編集対象を選んでビルドし直せます.

複数人が同時に翻訳を押した場合, Web 側のジョブ同時実行数は
`ARXIV_HONYAKU_WEB_CONCURRENCY` で制御します. docker compose のデフォルトは `1` なので,
翻訳ジョブは 1 本ずつ実行されます. 1 ジョブ内で推論 API へ同時接続する chunk 数は
`config.toml` の `llm.max_concurrency` です.
