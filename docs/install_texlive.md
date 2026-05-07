# scripts/install_texlive.sh

指定した年の TeX Live を `${TEXLIVE_INSTALL_ROOT:-/opt/texlive}/<year>` にインストールするスクリプト. Docker イメージで複数年の TeX Live を共存させる用途を想定.

```bash
bash scripts/install_texlive.sh <year> <repository_url>
# 例
bash scripts/install_texlive.sh 2025 https://example.com/tlnet/2025
```

デフォルトはスキーム `scheme-full`, バイナリ `x86_64-linux` のみ. 変えたい場合はスクリプト冒頭の `SELECTED_SCHEME` / `BINARY_PLATFORM` を編集する. 利用可能なスキーム名やバイナリ識別子は [install-tl 公式ドキュメント](https://www.tug.org/texlive/doc/install-tl.html) を参照.
