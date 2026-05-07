"""Static assets for the small built-in web UI."""
from __future__ import annotations

import html
import json
from typing import Any


def render_app_html(initial_state: dict[str, Any]) -> str:
    """Render the user-facing single page app."""
    data = html.escape(json.dumps(initial_state, ensure_ascii=False), quote=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>arxiv-honyaku</title>
  <style>{APP_CSS}</style>
</head>
<body>
  <script id="app-data" type="application/json">{data}</script>
  <div class="app-shell">
    <header class="topbar">
      <div>
        <div class="brand">arxiv-honyaku</div>
        <div class="subtle" id="user-name"></div>
      </div>
    </header>

    <main class="workspace">
      <aside class="sidebar">
        <form class="translate-form" id="translate-form">
          <label class="field-label" for="arxiv-input">arXiv</label>
          <input id="arxiv-input" name="arxiv" type="text" placeholder="1234.56789v2 / URL" autocomplete="off" required>

          <div>
            <div class="field-label">Layout</div>
            <div class="check-grid" id="layout-options"></div>
          </div>

          <div class="button-row">
            <button class="primary-button" type="submit">翻訳</button>
            <label class="force-toggle">
              <input id="force-input" type="checkbox">
              <span>再実行</span>
            </label>
          </div>
        </form>

        <div class="list-header">
          <span>論文</span>
          <span class="count-label" id="paper-count">0</span>
        </div>
        <div class="paper-list" id="paper-list"></div>
      </aside>

      <section class="content">
        <section class="job-panel" id="job-panel" hidden>
          <div class="job-heading">
            <div>
              <div class="job-title">翻訳キュー</div>
              <div class="subtle" id="job-count"></div>
            </div>
          </div>
          <div class="job-list" id="job-list"></div>
        </section>

        <section class="paper-detail" id="paper-detail" hidden>
          <div class="paper-toolbar">
            <div class="paper-title-wrap">
              <button class="star-button" id="star-button" type="button" aria-label="star"></button>
              <div>
                <h1 id="paper-title"></h1>
                <a class="paper-link" id="arxiv-link" target="_blank" rel="noreferrer">arXiv</a>
              </div>
            </div>
            <div class="toolbar-controls">
              <select id="paper-version-select"></select>
              <select id="candidate-select"></select>
            </div>
          </div>

          <div class="tabs" role="tablist">
            <button class="tab-button active" type="button" data-tab="pdf">PDF</button>
            <button class="tab-button" type="button" data-tab="tex">TeX</button>
            <button class="tab-button" type="button" data-tab="board">メモ</button>
            <button class="tab-button" type="button" data-tab="build-logs">ログ</button>
          </div>

          <section class="tab-panel active" id="tab-pdf">
            <div class="pdf-actions">
              <a id="open-pdf-link" target="_blank" rel="noreferrer">PDFを開く</a>
            </div>
            <iframe class="pdf-frame" id="pdf-frame" title="translated PDF"></iframe>
          </section>

          <section class="tab-panel tex-panel" id="tab-tex">
            <div class="tex-toolbar">
              <select id="tex-candidate-select"></select>
              <button class="ghost-button" id="workspace-button" type="button">選択</button>
              <select id="tex-file-select"></select>
            </div>
            <textarea id="tex-editor" spellcheck="false"></textarea>
            <div class="button-row">
              <button class="primary-button" id="save-tex-button" type="button">保存</button>
              <button class="secondary-button" id="build-tex-button" type="button">ビルド</button>
              <span class="subtle" id="tex-status"></span>
            </div>
          </section>

          <section class="tab-panel board-panel" id="tab-board">
            <div class="note-editor">
              <label class="field-label" for="paper-note">非公開メモ</label>
              <textarea id="paper-note"></textarea>
              <div class="button-row">
                <button class="primary-button" id="save-note-button" type="button">保存</button>
                <span class="subtle" id="note-status"></span>
              </div>
            </div>

            <div class="board-compose">
              <label class="field-label" for="post-body">公開メモ</label>
              <textarea id="post-body"></textarea>
              <button class="secondary-button" id="post-button" type="button">投稿</button>
            </div>
            <div class="posts" id="posts"></div>
          </section>

          <section class="tab-panel build-logs-panel" id="tab-build-logs">
            <div class="build-log-toolbar">
              <button class="ghost-button" id="reload-build-logs-button" type="button">更新</button>
              <span class="subtle" id="build-log-status"></span>
            </div>
            <div class="build-log-list" id="build-log-list"></div>
          </section>
        </section>

        <section class="empty-state" id="empty-state">
          <h1>arXiv翻訳</h1>
          <p>論文を選ぶか、新しく翻訳を開始してください。</p>
        </section>
      </section>
    </main>
  </div>
  <script>{APP_JS}</script>
</body>
</html>"""


def render_admin_html(initial_state: dict[str, Any]) -> str:
    """Render the admin user-link page."""
    data = html.escape(json.dumps(initial_state, ensure_ascii=False), quote=False)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>arxiv-honyaku admin</title>
  <style>{APP_CSS}</style>
</head>
<body>
  <script id="admin-data" type="application/json">{data}</script>
  <main class="admin-page">
    <header class="topbar">
      <div>
        <div class="brand">arxiv-honyaku admin</div>
        <div class="subtle">リンク発行</div>
      </div>
    </header>
    <section class="admin-panel">
      <form class="admin-form" id="admin-form">
        <label class="field-label" for="display-name">Name</label>
        <div class="inline-submit">
          <input id="display-name" type="text" required autocomplete="off">
          <button class="primary-button" type="submit">作成</button>
        </div>
      </form>
      <div class="paper-list" id="user-list"></div>
    </section>
  </main>
  <script>{ADMIN_JS}</script>
</body>
</html>"""


APP_CSS = r"""
:root {
  --bg: #f6f6f3;
  --panel: #ffffff;
  --text: #242424;
  --muted: #6b706d;
  --line: #d9ddd7;
  --accent: #2f6f73;
  --accent-strong: #20565a;
  --amber: #b7791f;
  --danger: #a33a32;
  --ok: #2f7551;
  --shadow: 0 8px 24px rgba(39, 45, 42, 0.08);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}

button,
input,
select,
textarea {
  font: inherit;
}

button {
  cursor: pointer;
}

.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.topbar {
  min-height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 0 24px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 5;
}

.brand {
  font-weight: 700;
  font-size: 17px;
}

.subtle {
  color: var(--muted);
  font-size: 12px;
}

.workspace {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
  min-height: 0;
}

.sidebar {
  border-right: 1px solid var(--line);
  padding: 18px;
  min-height: calc(100vh - 64px);
  background: #fbfbf9;
}

.translate-form {
  display: grid;
  gap: 12px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--line);
}

.field-label {
  display: block;
  margin-bottom: 5px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}

input,
select,
textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
  padding: 9px 10px;
  outline: none;
}

input:focus,
select:focus,
textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(47, 111, 115, 0.12);
}

.check-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
}

.check-grid label,
.force-toggle {
  min-height: 34px;
  display: flex;
  align-items: center;
  gap: 7px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 9px;
  background: var(--panel);
  color: var(--text);
  font-size: 13px;
}

.check-grid input,
.force-toggle input {
  width: auto;
}

.button-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.primary-button,
.secondary-button,
.ghost-button {
  min-height: 34px;
  border-radius: 6px;
  border: 1px solid transparent;
  padding: 7px 12px;
  font-weight: 650;
}

.primary-button {
  background: var(--accent);
  color: white;
}

.primary-button:hover {
  background: var(--accent-strong);
}

.secondary-button {
  background: #edf4f1;
  color: var(--accent-strong);
  border-color: #bdd2cd;
}

.ghost-button {
  background: transparent;
  color: var(--text);
  border-color: var(--line);
}

.ghost-button:hover,
.secondary-button:hover {
  border-color: var(--accent);
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 18px 0 10px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

.count-label {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 2px 8px;
}

.paper-list {
  display: grid;
  gap: 8px;
}

.paper-item,
.post-item,
.user-item {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  box-shadow: 0 1px 0 rgba(39, 45, 42, 0.02);
}

.paper-item {
  cursor: pointer;
}

.paper-item.active {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(47, 111, 115, 0.1);
}

.paper-item-top,
.post-top,
.user-item {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
}

.paper-id {
  font-weight: 700;
  word-break: break-word;
}

.paper-title-small {
  margin-top: 5px;
  color: var(--text);
  font-size: 13px;
  line-height: 1.35;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.paper-meta,
.post-meta {
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}

.content {
  min-width: 0;
  padding: 20px;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 16px;
}

.job-panel,
.paper-detail,
.empty-state,
.admin-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.job-panel {
  padding: 14px;
}

.job-heading,
.paper-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.job-title {
  font-weight: 700;
}

.job-list {
  display: grid;
  gap: 10px;
  margin-top: 12px;
}

.job-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfbf9;
}

.job-card-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.job-card-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.cancel-button {
  min-height: 28px;
  border: 1px solid rgba(163, 58, 50, 0.35);
  border-radius: 6px;
  background: rgba(163, 58, 50, 0.06);
  color: var(--danger);
  padding: 4px 8px;
  font-weight: 700;
  font-size: 12px;
}

.job-logs summary {
  margin-top: 8px;
  color: var(--accent-strong);
  cursor: pointer;
  font-weight: 700;
  font-size: 12px;
}

.status-pill {
  min-width: 72px;
  text-align: center;
  border-radius: 999px;
  padding: 4px 10px;
  color: var(--muted);
  border: 1px solid var(--line);
  font-size: 12px;
  font-weight: 700;
}

.status-pill.done,
.status-pill.success {
  color: var(--ok);
  border-color: rgba(47, 117, 81, 0.35);
  background: rgba(47, 117, 81, 0.08);
}

.status-pill.failed {
  color: var(--danger);
  border-color: rgba(163, 58, 50, 0.35);
  background: rgba(163, 58, 50, 0.08);
}

.status-pill.cancelled,
.status-pill.canceling {
  color: var(--danger);
  border-color: rgba(163, 58, 50, 0.35);
  background: rgba(163, 58, 50, 0.08);
}

.meter-row {
  display: grid;
  grid-template-columns: 68px minmax(160px, 1fr) 72px;
  gap: 10px;
  align-items: center;
  margin-top: 10px;
  font-size: 12px;
  color: var(--muted);
}

progress {
  width: 100%;
  height: 9px;
  accent-color: var(--accent);
}

.log-view {
  max-height: 170px;
  overflow: auto;
  margin: 12px 0 0;
  padding: 10px;
  background: #171a19;
  color: #d8e7de;
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  white-space: pre-wrap;
}

.paper-detail {
  min-height: 0;
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
}

.paper-toolbar {
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
}

.paper-title-wrap {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.paper-title-wrap h1 {
  margin: 0;
  font-size: 18px;
  line-height: 1.25;
  word-break: break-word;
}

.paper-link {
  color: var(--accent-strong);
  font-size: 12px;
}

.star-button {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--line);
  border-radius: 50%;
  background: var(--panel);
  color: var(--muted);
  font-size: 19px;
}

.star-button.active {
  color: var(--amber);
  border-color: rgba(183, 121, 31, 0.45);
  background: rgba(183, 121, 31, 0.08);
}

.toolbar-controls,
.tex-toolbar {
  display: flex;
  gap: 8px;
  align-items: center;
}

.toolbar-controls select,
.tex-toolbar select {
  min-width: 150px;
}

.tabs {
  display: flex;
  gap: 2px;
  padding: 0 12px;
  border-bottom: 1px solid var(--line);
}

.tab-button {
  border: 0;
  background: transparent;
  min-height: 42px;
  padding: 0 12px;
  color: var(--muted);
  font-weight: 700;
  border-bottom: 2px solid transparent;
}

.tab-button.active {
  color: var(--accent-strong);
  border-color: var(--accent);
}

.tab-panel {
  min-height: 0;
  display: none;
  padding: 14px;
}

.tab-panel.active {
  display: block;
}

.pdf-actions {
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
}

.pdf-actions a {
  color: var(--accent-strong);
  font-weight: 650;
  font-size: 13px;
}

.pdf-frame {
  display: block;
  width: 100%;
  height: calc(100vh - 292px);
  min-height: 440px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #ededeb;
}

.tex-panel.active,
.board-panel.active,
.build-logs-panel.active {
  display: grid;
  gap: 10px;
  align-content: start;
}

.tex-toolbar {
  flex-wrap: wrap;
}

#tex-editor {
  min-height: 430px;
  resize: vertical;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13px;
  line-height: 1.55;
}

#paper-note,
#post-body {
  min-height: 92px;
  resize: vertical;
}

.note-editor {
  display: grid;
  gap: 8px;
}

.note-editor .field-label,
.board-compose .field-label {
  margin-bottom: 0;
}

.board-compose {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px 10px;
  align-items: start;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}

.board-compose .field-label {
  grid-column: 1 / -1;
}

.posts {
  display: grid;
  gap: 8px;
}

.build-log-toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
}

.build-log-list {
  display: grid;
  gap: 10px;
}

.build-log-attempt {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfbf9;
}

.build-log-attempt.failed {
  border-color: rgba(163, 58, 50, 0.35);
}

.build-log-attempt.success {
  border-color: rgba(47, 117, 81, 0.28);
}

.build-log-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.build-log-title {
  font-weight: 700;
}

.build-log-files {
  display: grid;
  gap: 8px;
  margin-top: 8px;
}

.build-log-files summary {
  color: var(--accent-strong);
  cursor: pointer;
  font-weight: 700;
  font-size: 12px;
}

.post-body {
  margin-top: 8px;
  white-space: pre-wrap;
  line-height: 1.5;
}

.delete-post {
  border: 0;
  background: transparent;
  color: var(--danger);
  font-weight: 700;
  padding: 4px 0;
}

.empty-state {
  padding: 36px;
  align-self: start;
}

.empty-state h1 {
  margin: 0 0 8px;
  font-size: 24px;
}

.empty-state p {
  margin: 0;
  color: var(--muted);
}

.admin-page {
  max-width: 760px;
  margin: 0 auto;
  padding: 20px;
}

.admin-panel {
  margin-top: 20px;
  padding: 16px;
}

.admin-form {
  display: grid;
  gap: 8px;
  margin-bottom: 18px;
}

.inline-submit {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
}

.user-link {
  color: var(--accent-strong);
  word-break: break-all;
  font-size: 13px;
}

@media (max-width: 900px) {
  .workspace {
    grid-template-columns: 1fr;
  }

  .sidebar {
    min-height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .paper-toolbar,
  .job-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .toolbar-controls,
  .tex-toolbar,
  .board-compose {
    width: 100%;
    grid-template-columns: 1fr;
  }

  .toolbar-controls {
    flex-direction: column;
  }

  .toolbar-controls select,
  .tex-toolbar select {
    width: 100%;
  }

  .pdf-frame {
    height: 64vh;
    min-height: 340px;
  }
}
"""


APP_JS = r"""
(() => {
  const boot = JSON.parse(document.getElementById("app-data").textContent);
  const token = boot.token;
  const layoutModes = boot.layout_modes || [];
  const user = boot.user || {};

  let papers = [];
  let selectedPaperId = null;
  let detail = null;
  let selectedVersion = null;
  let selectedCandidateId = null;
  let jobPollTimer = null;
  let jobRenderSeq = 0;
  let stateLoadInFlight = false;
  let stateLoadQueued = false;
  const jobLogStickToBottom = new Map();
  let buildLogRenderSeq = 0;
  let activeJobs = [];
  let currentWorkspace = null;
  let currentTexPath = null;

  const $ = (id) => document.getElementById(id);

  const api = async (path, options = {}) => {
    const response = await fetch(`/api/u/${encodeURIComponent(token)}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    const text = await response.text();
    let payload = {};
    if (text) {
      payload = JSON.parse(text);
    }
    if (!response.ok) {
      throw new Error(payload.error || response.statusText);
    }
    return payload;
  };

  const setText = (id, value) => {
    $(id).textContent = value == null ? "" : String(value);
  };

  const option = (value, label, selected = false) => {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    node.selected = selected;
    return node;
  };

  const formatTime = (value) => {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) return value;
    return date.toLocaleString();
  };

  const percent = (current, total) => {
    if (!total || total <= 0) return 0;
    return Math.max(0, Math.min(100, Math.round((current / total) * 100)));
  };

  const isNearBottom = (node) => {
    return node.scrollTop + node.clientHeight >= node.scrollHeight - 24;
  };

  const scrollLogToBottom = (node) => {
    node.scrollTop = node.scrollHeight;
  };

  const phaseName = (phase) => ({
    queued: "待機",
    download: "取得",
    source_tree: "解析",
    prepare: "準備",
    translate: "翻訳",
    reconstruct: "再構成",
    build: "ビルド",
    done: "完了",
    failed: "失敗",
    canceling: "取消中",
    cancelled: "取消済み",
  }[phase] || phase || "Phase");

  const statusName = (status) => ({
    queued: "待機",
    running: "実行中",
    canceling: "取消中",
    cancelled: "取消済み",
    done: "完了",
    failed: "失敗",
  }[status] || status || "");

  const renderLayouts = () => {
    const wrap = $("layout-options");
    wrap.replaceChildren();
    for (const mode of layoutModes) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = mode;
      input.checked = true;
      label.append(input, document.createTextNode(mode));
      wrap.append(label);
    }
  };

  const renderPapers = () => {
    const list = $("paper-list");
    list.replaceChildren();
    $("paper-count").textContent = String(papers.length);
    for (const paper of papers) {
      const item = document.createElement("div");
      item.className = `paper-item${paper.paper_id === selectedPaperId ? " active" : ""}`;
      item.tabIndex = 0;
      item.addEventListener("click", () => loadPaper(paper.paper_id));
      item.addEventListener("keydown", (event) => {
        if (event.key === "Enter") loadPaper(paper.paper_id);
      });

      const top = document.createElement("div");
      top.className = "paper-item-top";
      const id = document.createElement("div");
      id.className = "paper-id";
      id.textContent = paper.paper_id;
      const star = document.createElement("div");
      star.textContent = paper.starred ? "★" : "☆";
      star.style.color = paper.starred ? "var(--amber)" : "var(--muted)";
      top.append(id, star);

      const meta = document.createElement("div");
      meta.className = "paper-meta";
      const status = paper.latest_status ? ` · ${paper.latest_status}` : "";
      meta.textContent = `${paper.candidate_count || 0} PDFs${status}`;
      item.append(top);
      if (paper.title) {
        const title = document.createElement("div");
        title.className = "paper-title-small";
        title.textContent = paper.title;
        item.append(title);
      }
      item.append(meta);
      list.append(item);
    }
  };

  const loadState = async () => {
    if (stateLoadInFlight) {
      stateLoadQueued = true;
      return;
    }
    stateLoadInFlight = true;
    try {
      const state = await api("/state");
      papers = state.papers || [];
      activeJobs = state.jobs || [];
      renderPapers();
      await renderJobQueue(activeJobs);
      if (!selectedPaperId && papers.length) {
        await loadPaper(papers[0].paper_id);
      }
      scheduleJobPolling();
    } finally {
      stateLoadInFlight = false;
      if (stateLoadQueued) {
        stateLoadQueued = false;
        setTimeout(() => {
          loadState().catch((error) => console.warn(error));
        }, 0);
      }
    }
  };

  const loadPaper = async (paperId) => {
    selectedPaperId = paperId;
    detail = await api(`/papers/${encodeURIComponent(paperId)}`);
    selectedVersion = selectedVersion || detail.default_version || (detail.versions[0] && detail.versions[0].version_label);
    if (!detail.versions.some((v) => v.version_label === selectedVersion)) {
      selectedVersion = detail.default_version || (detail.versions[0] && detail.versions[0].version_label);
    }
    selectedCandidateId = null;
    currentWorkspace = null;
    currentTexPath = null;
    $("empty-state").hidden = true;
    $("paper-detail").hidden = false;
    renderPapers();
    renderDetail();
  };

  const renderDetail = () => {
    if (!detail) return;
    setText("paper-title", detail.paper.title || detail.paper.paper_id);
    $("arxiv-link").href = `https://arxiv.org/abs/${detail.paper.paper_id}`;
    $("arxiv-link").textContent = detail.paper.paper_id;
    const star = $("star-button");
    star.textContent = detail.meta.starred ? "★" : "☆";
    star.classList.toggle("active", Boolean(detail.meta.starred));
    $("paper-note").value = detail.meta.note || "";

    const versionSelect = $("paper-version-select");
    versionSelect.replaceChildren();
    for (const version of detail.versions) {
      versionSelect.append(option(version.version_label, version.version_label, version.version_label === selectedVersion));
    }

    const candidates = candidatesForVersion();
    const primary = candidates.find((c) => c.is_primary) || candidates[0] || null;
    if (!selectedCandidateId || !candidates.some((c) => c.candidate_id === selectedCandidateId)) {
      selectedCandidateId = primary ? primary.candidate_id : "";
    }
    renderCandidateSelects(candidates);
    renderPdf(candidates);
    renderPosts();
    loadBuildLogs().catch((error) => {
      setText("build-log-status", error.message);
    });
  };

  const candidatesForVersion = () => {
    if (!detail) return [];
    return (detail.candidates || []).filter((candidate) => candidate.version_label === selectedVersion);
  };

  const pdfFilename = (candidate) => {
    const raw = `${candidate.paper_id}${candidate.version_label}.pdf`;
    return raw.replace(/[^A-Za-z0-9._-]+/g, "_");
  };

  const renderCandidateSelects = (candidates) => {
    for (const id of ["candidate-select", "tex-candidate-select"]) {
      const select = $(id);
      select.replaceChildren();
      if (!candidates.length) {
        select.append(option("", "none", true));
        select.disabled = true;
      } else {
        select.disabled = false;
        for (const candidate of candidates) {
          const baseLabel = candidate.label || `${candidate.font_mode || ""} ${candidate.layout_mode || ""}`.trim();
          const label = candidate.pdf_path ? baseLabel : `[失敗] ${baseLabel}`;
          select.append(option(candidate.candidate_id, label, candidate.candidate_id === selectedCandidateId));
        }
      }
    }
  };

  const renderPdf = (candidates) => {
    const candidate = candidates.find((c) => c.candidate_id === selectedCandidateId);
    const frame = $("pdf-frame");
    const link = $("open-pdf-link");
    if (!candidate || !candidate.pdf_path) {
      frame.removeAttribute("src");
      link.removeAttribute("href");
      link.textContent = candidate ? "ビルド失敗 (TeXタブで編集できます)" : "none";
      return;
    }
    const url = `/pdf/${candidate.candidate_id}/${encodeURIComponent(pdfFilename(candidate))}`;
    frame.src = url;
    link.href = url;
    link.textContent = "PDFを開く";
  };

  const renderPosts = () => {
    const posts = $("posts");
    posts.replaceChildren();
    for (const post of detail.posts || []) {
      const item = document.createElement("div");
      item.className = "post-item";
      const top = document.createElement("div");
      top.className = "post-top";
      const meta = document.createElement("div");
      meta.className = "post-meta";
      meta.textContent = `${post.display_name} · ${formatTime(post.created_at)}`;
      top.append(meta);
      if (post.can_delete) {
        const del = document.createElement("button");
        del.className = "delete-post";
        del.type = "button";
        del.textContent = "削除";
        del.addEventListener("click", async () => {
          await api(`/posts/${post.post_id}`, { method: "DELETE" });
          await loadPaper(selectedPaperId);
        });
        top.append(del);
      }
      const body = document.createElement("div");
      body.className = "post-body";
      body.textContent = post.body;
      item.append(top, body);
      posts.append(item);
    }
  };

  const loadBuildLogs = async () => {
    if (!selectedPaperId || !selectedVersion) return;
    const seq = ++buildLogRenderSeq;
    setText("build-log-status", "読み込み中");
    const payload = await api(
      `/papers/${encodeURIComponent(selectedPaperId)}/build-logs?version=${encodeURIComponent(selectedVersion)}`
    );
    if (seq !== buildLogRenderSeq) return;
    renderBuildLogs(payload.attempts || []);
    setText("build-log-status", `${(payload.attempts || []).length}件`);
  };

  const renderBuildLogs = (attempts) => {
    const list = $("build-log-list");
    list.replaceChildren();
    if (!attempts.length) {
      const empty = document.createElement("div");
      empty.className = "subtle";
      empty.textContent = "none";
      list.append(empty);
      return;
    }
    for (const attempt of attempts) {
      list.append(renderBuildLogAttempt(attempt));
    }
  };

  const renderBuildLogAttempt = (attempt) => {
    const item = document.createElement("div");
    item.className = `build-log-attempt ${attempt.status || ""}`;

    const head = document.createElement("div");
    head.className = "build-log-head";
    const left = document.createElement("div");
    const title = document.createElement("div");
    title.className = "build-log-title";
    title.textContent = `${attempt.label} · ${attempt.attempt}`;
    const meta = document.createElement("div");
    meta.className = "subtle";
    meta.textContent = `${attempt.source} · ${attempt.status} · ${formatTime(attempt.created_at)}`;
    left.append(title, meta);

    const status = document.createElement("span");
    status.className = `status-pill ${attempt.status}`;
    status.textContent = attempt.status || "";
    head.append(left, status);

    const files = document.createElement("div");
    files.className = "build-log-files";
    for (const file of attempt.files || []) {
      const details = document.createElement("details");
      if ((attempt.status || "") === "failed" && file.name.includes("stderr")) {
        details.open = true;
      }
      const summary = document.createElement("summary");
      summary.textContent = `${file.name} · ${file.size} bytes`;
      const pre = document.createElement("pre");
      pre.className = "log-view";
      pre.textContent = file.text || "";
      details.append(summary, pre);
      files.append(details);
    }
    item.append(head, files);
    return item;
  };

  const renderJobQueue = async (jobs) => {
    const panel = $("job-panel");
    const list = $("job-list");
    const seq = ++jobRenderSeq;
    setText("job-count", jobs.length ? `${jobs.length}件` : "");
    if (!jobs.length) {
      list.replaceChildren();
      panel.hidden = true;
      jobLogStickToBottom.clear();
      return;
    }
    panel.hidden = false;

    const payloads = await Promise.all(jobs.map(async (job) => {
      try {
        return await api(`/jobs/${job.job_id}`);
      } catch (error) {
        return { job, logs: [{ level: "ERROR", message: error.message, created_at: new Date().toISOString() }] };
      }
    }));
    if (seq !== jobRenderSeq) return;
    const openLogs = new Set(
      Array.from(list.querySelectorAll(".job-logs[open]")).map((node) => node.dataset.jobId)
    );
    for (const pre of list.querySelectorAll(".log-view")) {
      if (pre.dataset.jobId) {
        jobLogStickToBottom.set(pre.dataset.jobId, isNearBottom(pre));
      }
    }
    const fragment = document.createDocumentFragment();
    const activeJobIds = new Set();
    for (const payload of payloads) {
      activeJobIds.add(payload.job.job_id);
      fragment.append(renderJobCard(payload, openLogs.has(payload.job.job_id)));
    }
    for (const jobId of Array.from(jobLogStickToBottom.keys())) {
      if (!activeJobIds.has(jobId)) jobLogStickToBottom.delete(jobId);
    }
    list.replaceChildren(fragment);
    requestAnimationFrame(() => {
      for (const pre of list.querySelectorAll(".log-view")) {
        const jobId = pre.dataset.jobId;
        if (!jobId || jobLogStickToBottom.get(jobId) !== false) {
          scrollLogToBottom(pre);
        }
      }
    });
  };

  const renderJobCard = (payload, logsOpen = false) => {
    const job = payload.job;
    const card = document.createElement("div");
    card.className = "job-card";

    const top = document.createElement("div");
    top.className = "job-card-top";
    const titleWrap = document.createElement("div");
    const title = document.createElement("div");
    title.className = "job-title";
    title.textContent = `${job.paper_id || ""} ${job.version_label || ""}`.trim() || job.job_type;
    const message = document.createElement("div");
    message.className = "subtle";
    message.textContent = `${phaseName(job.phase)} · ${job.message || ""}`;
    titleWrap.append(title, message);

    const actions = document.createElement("div");
    actions.className = "job-card-actions";
    const status = document.createElement("span");
    status.className = `status-pill ${job.status}`;
    status.textContent = statusName(job.status);
    actions.append(status);
    if (!["done", "failed", "cancelled", "canceling"].includes(job.status)) {
      const cancel = document.createElement("button");
      cancel.className = "cancel-button";
      cancel.type = "button";
      cancel.textContent = "キャンセル";
      cancel.addEventListener("click", async () => {
        cancel.disabled = true;
        await api(`/jobs/${job.job_id}/cancel`, { method: "POST" });
        await loadState();
      });
      actions.append(cancel);
    }
    top.append(titleWrap, actions);

    const overall = document.createElement("div");
    overall.className = "meter-row";
    const overallLabel = document.createElement("label");
    overallLabel.textContent = "全体";
    const overallProgress = document.createElement("progress");
    overallProgress.max = 100;
    overallProgress.value = percent(job.overall_current, job.overall_total);
    const overallText = document.createElement("span");
    overallText.textContent = `${Math.round(job.overall_current || 0)}/${Math.round(job.overall_total || 0)}`;
    overall.append(overallLabel, overallProgress, overallText);

    const phase = document.createElement("div");
    phase.className = "meter-row";
    const phaseLabel = document.createElement("label");
    phaseLabel.textContent = phaseName(job.phase);
    const phaseProgress = document.createElement("progress");
    phaseProgress.max = 100;
    phaseProgress.value = percent(job.phase_current, job.phase_total);
    const phaseText = document.createElement("span");
    phaseText.textContent = `${Math.round(job.phase_current || 0)}/${Math.round(job.phase_total || 0)}`;
    phase.append(phaseLabel, phaseProgress, phaseText);

    const logs = document.createElement("details");
    logs.className = "job-logs";
    logs.dataset.jobId = job.job_id;
    logs.open = logsOpen;
    const summary = document.createElement("summary");
    summary.textContent = "ログ";
    const pre = document.createElement("pre");
    pre.className = "log-view";
    pre.dataset.jobId = job.job_id;
    pre.textContent = (payload.logs || []).map((line) => {
      return `${formatTime(line.created_at)} [${line.level}] ${line.message}`;
    }).join("\n");
    pre.addEventListener("scroll", () => {
      jobLogStickToBottom.set(job.job_id, isNearBottom(pre));
    });
    logs.addEventListener("toggle", () => {
      if (logs.open && jobLogStickToBottom.get(job.job_id) !== false) {
        requestAnimationFrame(() => scrollLogToBottom(pre));
      }
    });
    logs.append(summary, pre);
    card.append(top, overall, phase, logs);
    return card;
  };

  const scheduleJobPolling = () => {
    if (activeJobs.length) {
      if (!jobPollTimer) {
        jobPollTimer = setInterval(() => {
          loadState().catch((error) => console.warn(error));
        }, 1200);
      }
    } else if (jobPollTimer) {
      clearInterval(jobPollTimer);
      jobPollTimer = null;
    }
  };

  const selectedLayouts = () => {
    return Array.from(document.querySelectorAll("#layout-options input:checked")).map((input) => input.value);
  };

  $("translate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await api("/translate", {
        method: "POST",
        body: JSON.stringify({
          input: $("arxiv-input").value.trim(),
          layouts: selectedLayouts(),
          force: $("force-input").checked,
        }),
      });
      await loadState();
    } catch (error) {
      alert(error.message);
    }
  });

  $("star-button").addEventListener("click", async () => {
    if (!selectedPaperId) return;
    const payload = await api(`/papers/${encodeURIComponent(selectedPaperId)}/star`, {
      method: "POST",
      body: JSON.stringify({ starred: !detail.meta.starred }),
    });
    detail.meta.starred = payload.starred;
    await loadState();
    renderDetail();
  });

  $("save-note-button").addEventListener("click", async () => {
    if (!selectedPaperId) return;
    await api(`/papers/${encodeURIComponent(selectedPaperId)}/note`, {
      method: "POST",
      body: JSON.stringify({ note: $("paper-note").value }),
    });
    setText("note-status", "保存済み");
    setTimeout(() => setText("note-status", ""), 1400);
    await loadState();
  });

  $("post-button").addEventListener("click", async () => {
    const body = $("post-body").value.trim();
    if (!selectedPaperId || !body) return;
    await api(`/papers/${encodeURIComponent(selectedPaperId)}/posts`, {
      method: "POST",
      body: JSON.stringify({ body }),
    });
    $("post-body").value = "";
    await loadPaper(selectedPaperId);
  });

  $("paper-version-select").addEventListener("change", () => {
    selectedVersion = $("paper-version-select").value;
    selectedCandidateId = null;
    currentWorkspace = null;
    currentTexPath = null;
    renderDetail();
  });

  $("reload-build-logs-button").addEventListener("click", () => {
    loadBuildLogs().catch((error) => setText("build-log-status", error.message));
  });

  for (const id of ["candidate-select", "tex-candidate-select"]) {
    $(id).addEventListener("change", () => {
      selectedCandidateId = $(id).value;
      currentWorkspace = null;
      currentTexPath = null;
      renderDetail();
    });
  }

  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((node) => node.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      $(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });

  const loadWorkspaceFile = async (path) => {
    if (!currentWorkspace || !path) return;
    const payload = await api(`/workspaces/${currentWorkspace.workspace_id}/files?path=${encodeURIComponent(path)}`);
    $("tex-editor").value = payload.text || "";
    currentTexPath = path;
  };

  const renderTexFiles = async () => {
    const select = $("tex-file-select");
    select.replaceChildren();
    $("tex-editor").value = "";
    if (!currentWorkspace) {
      select.disabled = true;
      return;
    }
    select.disabled = false;
    for (const file of currentWorkspace.files || []) {
      select.append(option(file, file, file === currentTexPath));
    }
    if (currentWorkspace.files && currentWorkspace.files.length) {
      await loadWorkspaceFile(currentWorkspace.files[0]);
    }
  };

  $("workspace-button").addEventListener("click", async () => {
    if (!selectedCandidateId) return;
    const payload = await api(`/candidates/${selectedCandidateId}/workspace`, { method: "POST" });
    currentWorkspace = payload.workspace;
    setText("tex-status", "編集対象を選択しました");
    await renderTexFiles();
  });

  $("tex-file-select").addEventListener("change", () => loadWorkspaceFile($("tex-file-select").value));

  $("save-tex-button").addEventListener("click", async () => {
    if (!currentWorkspace || !currentTexPath) return;
    await api(`/workspaces/${currentWorkspace.workspace_id}/files?path=${encodeURIComponent(currentTexPath)}`, {
      method: "PUT",
      body: JSON.stringify({ text: $("tex-editor").value }),
    });
    setText("tex-status", "保存済み");
  });

  $("build-tex-button").addEventListener("click", async () => {
    if (!currentWorkspace) return;
    await api(`/workspaces/${currentWorkspace.workspace_id}/build`, { method: "POST" });
    await loadState();
  });

  renderLayouts();
  setText("user-name", user.display_name || "");
  loadState().catch((error) => alert(error.message));
  setInterval(() => {
    loadState().catch((error) => console.warn(error));
  }, 10000);
})();
"""


ADMIN_JS = r"""
(() => {
  const boot = JSON.parse(document.getElementById("admin-data").textContent);
  const token = boot.admin_token;
  let users = boot.users || [];
  const $ = (id) => document.getElementById(id);

  const api = async (path, options = {}) => {
    const response = await fetch(`/api/admin/${encodeURIComponent(token)}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    if (!response.ok) throw new Error(payload.error || response.statusText);
    return payload;
  };

  const render = () => {
    const list = $("user-list");
    list.replaceChildren();
    for (const user of users) {
      const item = document.createElement("div");
      item.className = "user-item";
      const left = document.createElement("div");
      const name = document.createElement("div");
      name.className = "paper-id";
      name.textContent = user.display_name;
      const link = document.createElement("a");
      link.className = "user-link";
      link.href = user.url;
      link.textContent = user.url;
      left.append(name, link);
      item.append(left);
      list.append(item);
    }
  };

  $("admin-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const displayName = $("display-name").value.trim();
    if (!displayName) return;
    const payload = await api("/users", {
      method: "POST",
      body: JSON.stringify({ display_name: displayName }),
    });
    users = payload.users || [];
    $("display-name").value = "";
    render();
  });

  render();
})();
"""
