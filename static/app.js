const state = {
  folders: [],
  folder: '',
  expanded: new Set(),
  moveDoc: null,
  tab: 'ask',
  docs: [],
  pending: [],
  jobs: [],
  chats: [],
  threadId: '',
  messages: [],
  sending: false,
  uploadFiles: [],
  inspectDocId: null,
  inspectInfo: null,
  inspectToc: [],
  inspectPage: null,
  inspectSearchResults: [],
  inspectPanel: 'page',
  inspecting: false,
};

const VALID_TABS = new Set(['ask', 'check', 'documents', 'ingest', 'jobs', 'inspect']);
const INSPECT_PANELS = new Set(['metadata', 'toc', 'page', 'search']);
const $ = (id) => document.getElementById(id);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: opts.body instanceof FormData ? opts.headers : {
      'Content-Type': 'application/json',
      ...(opts.headers || {}),
    },
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      message = data.detail || data.error || message;
    } catch (_) {}
    throw new Error(message);
  }
  return res.json();
}

function enc(value) {
  return encodeURIComponent(value);
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function showToast(message, type = '') {
  const toast = $('toast');
  toast.textContent = message;
  toast.className = `toast ${type}`.trim();
  setTimeout(() => toast.classList.add('hidden'), 3500);
}

// Mark the row owning a clicked button as busy: dim it, disable its buttons,
// and relabel the clicked button. Cleared on the next render (refreshActiveData).
function setActionBusy(el, label) {
  if (!el) return;
  const row = el.closest('.list-row');
  if (row) {
    row.classList.add('busy');
    row.querySelectorAll('button').forEach((btn) => { btn.disabled = true; });
  } else {
    el.disabled = true;
  }
  if (label) el.textContent = label;
}

function setHash() {
  if (!state.folder) return;
  const params = new URLSearchParams();
  params.set('folder', state.folder);
  params.set('tab', state.tab);
  if (state.tab === 'inspect' && state.inspectDocId) {
    params.set('doc', String(state.inspectDocId));
    params.set('panel', state.inspectPanel);
  }
  const hash = params.toString();
  if (location.hash.slice(1) !== hash) {
    history.replaceState(null, '', `${location.pathname}#${hash}`);
  }
}

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  state.folder = params.get('folder') || state.folder;
  const tab = params.get('tab') || state.tab || 'ask';
  state.tab = VALID_TABS.has(tab) ? tab : 'ask';
  const panel = params.get('panel') || state.inspectPanel || 'page';
  state.inspectPanel = INSPECT_PANELS.has(panel) ? panel : 'page';
  const doc = Number(params.get('doc'));
  state.inspectDocId = Number.isFinite(doc) && doc > 0 ? doc : state.inspectDocId;
  if (state.tab === 'inspect' && !state.inspectDocId) state.tab = 'documents';
}

function renderMarkdown(text) {
  if (!text) return '';
  if (window.marked && window.DOMPurify) {
    return DOMPurify.sanitize(marked.parse(text));
  }
  return `<p>${escapeHtml(text)}</p>`;
}

function renderMath(root) {
  if (!window.renderMathInElement) return;
  renderMathInElement(root, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true },
    ],
    throwOnError: false,
  });
}

async function loadFolders() {
  state.folders = await api('/api/folders');
  if (!state.folder && state.folders.length) state.folder = state.folders[0].folder;
  if (state.folder && !state.folders.some((f) => f.folder === state.folder)) {
    state.folder = state.folders[0]?.folder || '';
  }
  renderShell();
}

async function selectFolder(folder) {
  state.folder = folder;
  expandAncestors(folder);
  saveExpanded();
  state.threadId = '';
  state.messages = [];
  state.docs = [];
  state.pending = [];
  clearInspector();
  if (state.tab === 'inspect') state.tab = 'documents';
  setHash();
  renderShell();
  await refreshActiveData();
}

async function refreshActiveData() {
  if (!state.folder) return;
  // loadFolders refreshes the sidebar document/pending counts; without it the
  // counts stay stale after deletes/ingests until a manual page reload.
  await Promise.all([loadFolders(), loadDocuments(), loadChats(), loadJobs()]);
  if (state.tab === 'inspect' && state.inspectDocId && !state.inspectInfo) {
    await loadInspection(state.inspectDocId);
  }
  renderActiveView();
}

async function loadDocuments() {
  const data = await api(`/api/folders/${enc(state.folder)}/documents`);
  state.docs = data.documents || [];
  state.pending = data.pending || [];
  renderShell();
}

async function loadChats() {
  state.chats = await api(`/api/folders/${enc(state.folder)}/chats`);
  if (!state.threadId && state.chats.length) state.threadId = state.chats[0].id;
  if (state.threadId) await loadMessages();
}

async function loadMessages() {
  if (!state.threadId) {
    state.messages = [];
    return;
  }
  const data = await api(`/api/chats/${enc(state.threadId)}/messages`);
  state.messages = data.messages || [];
}

async function loadJobs() {
  state.jobs = await api('/api/ingestion/jobs');
}

const EXPANDED_STORAGE_KEY = 'ocr-rag-expanded-folders';

function loadExpanded() {
  try {
    const raw = localStorage.getItem(EXPANDED_STORAGE_KEY);
    if (!raw) return new Set();
    return new Set(JSON.parse(raw));
  } catch (_) {
    return new Set();
  }
}

function saveExpanded() {
  try {
    localStorage.setItem(EXPANDED_STORAGE_KEY, JSON.stringify([...state.expanded]));
  } catch (_) {}
}

function expandAncestors(path) {
  if (!path) return;
  const parts = path.split('/');
  for (let i = 1; i < parts.length; i++) {
    state.expanded.add(parts.slice(0, i).join('/'));
  }
}

function buildFolderTree(folders) {
  const byPath = new Map(folders.map((f) => [f.folder, { ...f, children: [] }]));
  const roots = [];
  for (const node of byPath.values()) {
    if (node.parent && byPath.has(node.parent)) {
      byPath.get(node.parent).children.push(node);
    } else {
      roots.push(node);
    }
  }
  const sortRec = (list) => {
    list.sort((a, b) => (a.display_name || a.folder).localeCompare(b.display_name || b.folder));
    list.forEach((n) => sortRec(n.children));
  };
  sortRec(roots);
  return roots;
}

function renderFolderList() {
  const search = ($('folder-search')?.value || '').toLowerCase().trim();
  if (search) {
    const matches = state.folders.filter((f) => f.folder.toLowerCase().includes(search));
    return matches.length
      ? matches.map((f) => renderFolderRow(f, { depth: 0, hasChildren: false, flat: true })).join('')
      : '<div class="empty-state">No folders</div>';
  }
  const tree = buildFolderTree(state.folders);
  if (!tree.length) return '<div class="empty-state">No folders</div>';
  return tree.map((node) => renderFolderNode(node, 0)).join('');
}

function renderFolderNode(node, depth) {
  const hasChildren = node.children.length > 0;
  const isExpanded = state.expanded.has(node.folder);
  const row = renderFolderRow(node, { depth, hasChildren, isExpanded, flat: false });
  if (hasChildren && isExpanded) {
    return row + node.children.map((child) => renderFolderNode(child, depth + 1)).join('');
  }
  return row;
}

function renderFolderRow(folder, { depth, hasChildren, isExpanded = false, flat = false }) {
  const isActive = folder.folder === state.folder;
  const chevron = hasChildren
    ? `<button class="folder-chevron ${isExpanded ? 'open' : ''}" type="button" data-action="toggle-folder" data-folder="${escapeHtml(folder.folder)}" aria-label="${isExpanded ? 'Collapse' : 'Expand'}"></button>`
    : `<span class="folder-chevron empty"></span>`;
  const title = flat ? escapeHtml(folder.folder) : escapeHtml(folder.display_name || folder.folder);
  return `
    <div class="folder-row-wrap ${isActive ? 'active' : ''}" style="--depth: ${depth};">
      ${chevron}
      <button class="folder-row" type="button" data-action="select-folder" data-folder="${escapeHtml(folder.folder)}">
        <span class="folder-row-text">
          <span class="folder-title">${title}</span>
        </span>
        <span class="folder-count">${folder.docs || 0}</span>
      </button>
      <button class="folder-add" type="button" data-action="new-subfolder" data-parent="${escapeHtml(folder.folder)}" title="New sub-folder">+</button>
    </div>
  `;
}

function renderShell() {
  $('folder-list').innerHTML = renderFolderList();

  const active = state.folders.find((f) => f.folder === state.folder);
  $('active-folder').textContent = active ? active.folder : 'Select a folder';
  if (active) {
    $('folder-stats').innerHTML = `<b>${active.docs || 0}</b> document${(active.docs || 0) === 1 ? '' : 's'}`
      + ` &nbsp;·&nbsp; <b>${active.pages || 0}</b> pages`
      + ` &nbsp;·&nbsp; <b>${active.pending || 0}</b> pending`;
  } else {
    $('folder-stats').textContent = '';
  }

  for (const tab of VALID_TABS) {
    const btn = $(`${tab}-tab`);
    const view = $(`${tab}-view`);
    if (btn) btn.classList.toggle('active', state.tab === tab);
    if (view) view.classList.toggle('hidden', state.tab !== tab);
  }
  $('inspect-tab').classList.toggle('hidden', !state.inspectDocId);
  for (const t of ['documents', 'ingest', 'jobs']) {
    const b = $(`${t}-tab`);
    if (b) b.classList.toggle('hidden', !state.isAdmin);
  }
}

function setTab(tab) {
  if (!VALID_TABS.has(tab)) return;
  if (tab === 'inspect' && !state.inspectDocId) return;
  if (['documents', 'ingest', 'jobs'].includes(tab) && !state.isAdmin) return;
  state.tab = tab;
  setHash();
  renderShell();
  renderActiveView();
}

function renderActiveView() {
  renderShell();
  if (state.tab === 'ask') renderAsk();
  if (state.tab === 'check') renderCheck();
  if (state.tab === 'documents') renderDocuments();
  if (state.tab === 'ingest') renderIngest();
  if (state.tab === 'jobs') renderJobs();
  if (state.tab === 'inspect') renderInspect();
}

function pageHeader(title, subtitle = '', actions = '') {
  return `
    <div class="page-head">
      <div>
        <h2>${escapeHtml(title)}</h2>
        ${subtitle ? `<p>${escapeHtml(subtitle)}</p>` : ''}
      </div>
      ${actions ? `<div class="page-actions">${actions}</div>` : ''}
    </div>
  `;
}

const ICON = {
  plus: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>`,
  refresh: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36M21 4v5h-5"/></svg>`,
  send: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>`,
  trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>`,
};

function renderAsk(opts = {}) {
  const view = $('ask-view');
  if (!state.folder) {
    view.innerHTML = `<div class="empty-state">Select a folder to ask questions.</div>`;
    return;
  }

  const hasMessages = state.messages.length > 0;
  view.innerHTML = `
    <div class="ask-toolbar">
      <div class="ask-toolbar-title">${renderThreadControl()}</div>
      <div class="ask-toolbar-actions">
        <button class="chip-btn" type="button" data-action="new-chat">${ICON.plus}New chat</button>
        ${state.threadId ? `<button class="chip-btn danger" type="button" data-action="delete-chat" title="Delete this chat">${ICON.trash}Delete</button>` : ''}
        <button class="chip-btn" type="button" data-action="refresh">${ICON.refresh}Refresh</button>
      </div>
    </div>
    <div class="messages" id="messages">
      ${hasMessages ? `<div class="thread-col">${state.messages.map(renderMessage).join('')}</div>` : renderAskEmpty()}
    </div>
    <form class="chat-input" id="chat-form">
      <div class="composer-wrap">
        <textarea id="message-input" placeholder="Ask about this folder…" ${state.sending ? 'disabled' : ''}></textarea>
        <button class="send-btn" type="submit" ${state.sending ? 'disabled' : ''} aria-label="Send">${ICON.send}</button>
      </div>
      <div class="composer-hint">DocLens answers only from documents in this folder, with citations.</div>
    </form>
  `;
  const messages = $('messages');
  scrollMessages(messages, opts.scroll || 'bottom');
  renderMath(view);
}

// 'answer' aligns the top of the latest assistant message to the top of the
// scroll viewport (so the reader starts where the response begins); 'bottom'
// jumps to the end (used for loading/switching threads).
function scrollMessages(container, mode) {
  if (!container) return;
  if (mode === 'answer') {
    const items = container.querySelectorAll('.message.assistant');
    const last = items[items.length - 1];
    if (last) {
      const offset = last.getBoundingClientRect().top - container.getBoundingClientRect().top;
      container.scrollTop += offset - 12;
      return;
    }
  }
  container.scrollTop = container.scrollHeight;
}

function renderThreadControl() {
  if (!state.chats.length) return `<div class="chat-status">No chat yet</div>`;
  return `
    <select class="field thread-select" id="thread-select" data-action="switch-thread">
      ${state.chats.map((chat) => `
        <option value="${escapeHtml(chat.id)}" ${chat.id === state.threadId ? 'selected' : ''}>
          ${escapeHtml(chat.title || 'New chat')}
        </option>
      `).join('')}
    </select>
  `;
}

function renderAskEmpty() {
  const active = state.folders.find((f) => f.folder === state.folder);
  const pages = (active && active.pages) || 0;
  const grounding = pages
    ? `Every answer is grounded in the ${pages} pages of this folder, with citations back to the exact clause and page.`
    : `Every answer is grounded in this folder's documents, with citations back to the exact clause and page.`;
  return `
    <div class="ask-empty">
      <div class="ask-empty-card">
        <div class="ask-empty-kicker">Ask the folder</div>
        <h1>What would you like to know about ${escapeHtml(state.folder)}?</h1>
        <p class="ask-empty-lede">${escapeHtml(grounding)}</p>
      </div>
    </div>
  `;
}

// Turn inline [1], [2,3] style citation markers into superscripts keyed to source cards.
function markCitations(text) {
  return (text || '').replace(/\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, '<sup class="cite">$1</sup>');
}

// Live "thinking" trace shown in the assistant bubble while the model works,
// before the final answer replaces it. Steps are thinking summaries and tool
// calls, in the order the model produced them.
function renderLiveTrace(live) {
  const head = '<div class="live-head"><span class="live-dot"></span>DocLens working…</div>';
  if (!live) return head;
  const steps = (live.steps || []).map((s) => (
    s.kind === 'tool'
      ? `<div class="live-step tool">${escapeHtml(s.text)}</div>`
      : `<div class="live-step think">${escapeHtml(s.text)}</div>`
  )).join('');
  return `${head}${steps ? `<div class="live-steps">${steps}</div>` : ''}`;
}

// Patch just the streaming assistant bubble in place (no full re-render) so the
// composer and scroll position survive each event.
function updateLiveBubble(live) {
  const container = $('messages');
  if (!container) return;
  const bodies = container.querySelectorAll('.message.assistant .message-body');
  const last = bodies[bodies.length - 1];
  if (last) last.innerHTML = `<div class="live-thinking">${renderLiveTrace(live.live)}</div>`;
  container.scrollTop = container.scrollHeight;
}

function renderMessage(message) {
  const sources = Array.isArray(message.sources) ? message.sources : [];
  const isUser = message.role === 'user';
  const streaming = !isUser && message.live && !message.content;
  const body = isUser
    ? renderMarkdown(message.content || '')
    : streaming
      ? `<div class="live-thinking">${renderLiveTrace(message.live)}</div>`
      : renderMarkdown(markCitations(message.content || ''));
  const meta = !isUser && sources.length ? `<span class="msg-meta">· ${sources.length} source${sources.length > 1 ? 's' : ''}</span>` : '';
  const hasTrace = !isUser && message.trace && Array.isArray(message.trace.events) && message.trace.events.length;
  const traceBlock = hasTrace ? `
      <div class="search-trace" data-msg="${escapeHtml(String(message.id || ''))}">
        <button class="chip-btn trace-toggle" type="button" data-action="toggle-trace">${ICON.search || ''}Search trace</button>
        <div class="trace-panel hidden"></div>
      </div>` : '';
  return `
    <article class="message ${isUser ? 'user' : 'assistant'}">
      <div class="msg-who">
        <span class="msg-avatar ${isUser ? 'you' : 'ai'}">${isUser ? 'You' : 'E'}</span>
        <span class="msg-name">${isUser ? 'You' : 'DocLens'}</span>
        ${meta}
      </div>
      <div class="message-body">${body}</div>
      ${sources.length ? `<div class="sources"><div class="sources-label">Sources</div>${sources.map(renderSource).join('')}</div>` : ''}
      ${traceBlock}
    </article>
  `;
}

// Group the flat trace event stream into rounds for display + diagramming.
function traceRounds(trace) {
  const rounds = new Map();
  const get = (n) => { if (!rounds.has(n)) rounds.set(n, { n, thinking: '', calls: [], stale: false, usage: null }); return rounds.get(n); };
  for (const ev of (trace.events || [])) {
    const r = get(ev.round || 1);
    if (ev.type === 'thinking') r.thinking += (r.thinking ? '\n' : '') + (ev.text || '');
    else if (ev.type === 'tool_result') r.calls.push(ev);
    else if (ev.type === 'round_end') { r.stale = !!ev.stale; r.usage = ev.usage || null; r.final = !!ev.final; }
  }
  return [...rounds.values()].sort((a, b) => a.n - b.n);
}

// Build a mermaid flowchart of the search: question → rounds → tool calls → answer,
// with bottlenecks flagged (slow >2s, dead-end = no new sources, repeated calls).
function traceMermaid(rounds) {
  const L = ['flowchart TD', '  Q([Question])'];
  const cls = [];
  let prev = 'Q';
  rounds.forEach((r) => {
    const rid = `R${r.n}`;
    L.push(`  ${rid}["Round ${r.n}${r.stale ? ' · stale' : ''}"]`);
    L.push(`  ${prev} --> ${rid}`);
    if (r.stale) cls.push(`class ${rid} stale;`);
    r.calls.forEach((c, i) => {
      const tid = `${rid}c${i}`;
      const secs = (c.ms / 1000).toFixed(1);
      const out = c.new_sources > 0 ? `${c.new_sources} new src` : 'no new';
      const label = `${c.name}<br/>${(c.query || '').slice(0, 38)}<br/>${secs}s · ${out}`;
      L.push(`  ${tid}["${label.replace(/"/g, "'")}"]`);
      L.push(`  ${rid} --> ${tid}`);
      if (c.error) cls.push(`class ${tid} err;`);
      else if (c.new_sources <= 0) cls.push(`class ${tid} dead;`);
      else if (c.ms > 2000) cls.push(`class ${tid} slow;`);
      else cls.push(`class ${tid} good;`);
      if (c.repeated) cls.push(`class ${tid} repeat;`);
    });
    prev = rid;
  });
  L.push('  A([Answer])');
  L.push(`  ${prev} --> A`);
  L.push('  classDef good fill:#e7f3ea,stroke:#4c9b6a,color:#1d3d2a;');
  L.push('  classDef dead fill:#fdeaea,stroke:#d7263d,color:#7a1420;');
  L.push('  classDef slow fill:#fdf2e0,stroke:#e8a33d,color:#6b4a10;');
  L.push('  classDef err fill:#fdeaea,stroke:#d7263d,color:#7a1420,stroke-width:2px;');
  L.push('  classDef stale fill:#f0ede6,stroke:#b9b2a3,color:#5b554a;');
  L.push('  classDef repeat stroke-dasharray:5 4;');
  return L.join('\n') + '\n  ' + cls.join('\n  ');
}

function traceBreakdown(trace, rounds) {
  const totalCalls = rounds.reduce((a, r) => a + r.calls.length, 0);
  const dead = rounds.reduce((a, r) => a + r.calls.filter((c) => c.new_sources <= 0).length, 0);
  const repeats = rounds.reduce((a, r) => a + r.calls.filter((c) => c.repeated).length, 0);
  const slow = rounds.reduce((a, r) => a + r.calls.filter((c) => c.ms > 2000).length, 0);
  const summary = `<div class="trace-summary">
    <span><b>${rounds.length}</b> rounds</span>
    <span><b>${totalCalls}</b> searches</span>
    <span><b>${(trace.ms_total / 1000).toFixed(1)}s</b> total</span>
    ${dead ? `<span class="warn"><b>${dead}</b> dead-end</span>` : ''}
    ${repeats ? `<span class="warn"><b>${repeats}</b> repeated</span>` : ''}
    ${slow ? `<span class="warn"><b>${slow}</b> slow</span>` : ''}
  </div>`;
  const rows = rounds.map((r) => {
    const calls = r.calls.map((c) => {
      const tag = c.error ? 'err' : c.new_sources <= 0 ? 'dead' : c.ms > 2000 ? 'slow' : 'good';
      const bits = [`${(c.ms / 1000).toFixed(1)}s`, `${c.result_chars || 0} chars`,
        c.new_sources > 0 ? `${c.new_sources} new sources` : 'no new sources'];
      if (c.repeated) bits.push('repeated');
      return `<li class="tcall ${tag}"><span class="tcall-name">${escapeHtml(c.name)}</span>
        <span class="tcall-q">${escapeHtml(c.query || '')}</span>
        <span class="tcall-meta">${bits.join(' · ')}</span></li>`;
    }).join('');
    return `<div class="tround">
      <div class="tround-head">Round ${r.n}${r.stale ? ' <span class="warn">· stale (no progress)</span>' : ''}</div>
      ${r.thinking ? `<div class="tround-think">${escapeHtml(r.thinking)}</div>` : ''}
      ${calls ? `<ul class="tcalls">${calls}</ul>` : '<div class="muted">no searches this round</div>'}
    </div>`;
  }).join('');
  return summary + `<div class="mermaid-wrap"></div>` + rows;
}

async function renderSearchTrace(panel, trace) {
  const rounds = traceRounds(trace);
  panel.innerHTML = traceBreakdown(trace, rounds);
  const wrap = panel.querySelector('.mermaid-wrap');
  try {
    const m = await loadMermaid();
    const { svg } = await m.render('tg' + Math.random().toString(36).slice(2), traceMermaid(rounds));
    if (wrap) wrap.innerHTML = svg;
  } catch (e) {
    if (wrap) wrap.innerHTML = '<div class="muted">Diagram unavailable.</div>';
  }
}

let _mermaidPromise = null;
function loadMermaid() {
  if (window.mermaid) return Promise.resolve(window.mermaid);
  if (_mermaidPromise) return _mermaidPromise;
  _mermaidPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
    s.onload = () => { window.mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'strict' }); resolve(window.mermaid); };
    s.onerror = reject;
    document.head.appendChild(s);
  });
  return _mermaidPromise;
}

// Pull a leading clause/section number (e.g. "6.2.1", "B31.3") out of a breadcrumb.
function clauseFromBreadcrumb(breadcrumb) {
  if (!breadcrumb) return '';
  const last = breadcrumb.split(/\s*[›>|/]\s*/).pop().trim();
  const m = last.match(/^([A-Z]?\d+(?:\.\d+)*[A-Za-z]?)\b/);
  return m ? m[1] : '';
}

function renderSource(source) {
  const title = source.doc_title || source.filename || 'Source';
  const page = source.page_num ? `Page ${source.page_num}` : '';
  const breadcrumb = source.breadcrumb || '';
  const clause = clauseFromBreadcrumb(breadcrumb);
  const metaText = [title, page].filter(Boolean).join(' · ');
  return `
    <div class="source-card">
      <div class="source-num">${escapeHtml(String(source.id || ''))}</div>
      <div class="source-main">
        <div class="source-meta">
          ${clause ? `<span class="clause-pill">§ ${escapeHtml(clause)}</span>` : ''}
          <span>${escapeHtml(metaText)}</span>
        </div>
        ${source.snippet ? `<div class="source-snippet">${escapeHtml(source.snippet)}</div>` : ''}
      </div>
    </div>
  `;
}

function renderDocuments() {
  const view = $('documents-view');
  if (!state.folder) {
    view.innerHTML = `<div class="empty-state">Select a folder to view documents.</div>`;
    return;
  }
  view.innerHTML = `
    ${pageHeader('Documents', 'Indexed documents available for search and manual inspection.', `
      <button class="ghost" type="button" data-action="refresh">Refresh</button>
      <button class="primary" type="button" data-action="go-ingest">Add documents</button>
    `)}
    <div class="page-body single-column">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Indexed documents</div>
          <div class="muted">${state.docs.length}</div>
        </div>
        <div class="panel-body">
          <div class="list document-list">
            ${state.docs.length ? state.docs.map(renderDoc).join('') : '<div class="muted">No indexed documents.</div>'}
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderDoc(doc) {
  const docFolder = doc.folder || doc.project || '';
  const inSubfolder = docFolder && docFolder !== state.folder;
  return `
    <div class="list-row document-row">
      <span class="row-main">
        <span class="row-title">${escapeHtml(doc.title || doc.filename)}</span>
        <span class="doc-meta">${doc.total_pages || 0} pages | ${doc.sections || 0} sections${doc.document_type ? ` | ${escapeHtml(doc.document_type)}` : ''}${inSubfolder ? ` | ${escapeHtml(docFolder)}` : ''}</span>
      </span>
      <span class="row-actions">
        <button class="ghost small" type="button" data-action="move-doc" data-doc-id="${doc.id}">Move</button>
        <button class="ghost small" type="button" data-action="inspect-doc" data-doc-id="${doc.id}">Inspect</button>
        <button class="ghost small danger" type="button" data-action="delete-doc" data-doc-id="${doc.id}" data-doc-title="${escapeHtml(doc.title || doc.filename || '')}">Delete</button>
      </span>
    </div>
  `;
}

function renderIngest() {
  const view = $('ingest-view');
  if (!state.folder) {
    view.innerHTML = `<div class="empty-state">Select a folder to ingest documents.</div>`;
    return;
  }
  const ingestingCount = state.pending.filter((f) => activeJobForPending(f)).length;
  const idlePendingCount = state.pending.length - ingestingCount;
  view.innerHTML = `
    ${pageHeader('Ingest', 'Upload new files and start ingestion for pending documents.', `
      <button class="ghost" type="button" data-action="refresh">Refresh</button>
      <button class="ghost" type="button" data-action="go-documents">View documents</button>
    `)}
    <div class="page-body single-column">
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Upload documents</div>
            <div class="muted">${escapeHtml(state.folder)}</div>
          </div>
        </div>
        <div class="panel-body">
          <div class="upload-box">
            <label class="drop-zone" id="drop-zone">
              <input id="file-input" type="file" multiple hidden data-action="stage-files">
              <span>
                <strong>Choose files or drop them here</strong>
                <div class="muted">PDF, DOCX, XLSX, images, archives</div>
              </span>
            </label>
            <div class="file-actions">
              <button class="primary" type="button" data-action="upload-staged" ${state.uploadFiles.length ? '' : 'disabled'}>
                Upload ${state.uploadFiles.length ? `(${state.uploadFiles.length})` : ''}
              </button>
              <button class="ghost" type="button" data-action="clear-staged" ${state.uploadFiles.length ? '' : 'disabled'}>Clear</button>
            </div>
            <div id="staged-files" class="list">${renderStagedFiles()}</div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Pending ingestion</div>
            <div class="muted">${state.pending.length} pending${ingestingCount ? ` · ${ingestingCount} ingesting` : ''}</div>
          </div>
          <button class="primary small" type="button" data-action="ingest-all" ${idlePendingCount ? '' : 'disabled'}>Ingest pending</button>
        </div>
        <div class="panel-body">
          <div class="list">
            ${state.pending.length ? state.pending.map(renderPending).join('') : '<div class="muted">No pending files.</div>'}
          </div>
        </div>
      </div>
    </div>
  `;
  bindDropZone();
}

function renderStagedFiles() {
  return state.uploadFiles.length ? state.uploadFiles.map((file) => `
    <div class="list-row">
      <span class="row-title">${escapeHtml(file.name)}</span>
      <span class="muted">${formatBytes(file.size)}</span>
    </div>
  `).join('') : '';
}

// Find a queued/running ingestion job for a pending file so the row can show
// progress and hide its action buttons (preventing duplicate ingestion).
function activeJobForPending(file) {
  const folder = file.folder || state.folder;
  const src = `${folder}/${file.filename}`;
  return state.jobs.find((job) =>
    (job.status === 'queued' || job.status === 'running') &&
    (job.source_path === src ||
      (job.project === folder && (job.filename === file.filename || job.filename === file.relative_path)))
  );
}

function renderPending(file) {
  const folder = file.folder || state.folder;
  const job = activeJobForPending(file);
  const actions = job
    ? `<span class="status-pill ${escapeHtml(job.status)}" title="${escapeHtml(job.stage || '')}">${escapeHtml(job.stage || 'Ingesting…')}</span>`
    : `
        <button class="secondary small" type="button" data-action="ingest-single" data-folder="${escapeHtml(folder)}" data-filename="${escapeHtml(file.filename)}">Ingest</button>
        <button class="secondary small" type="button" data-action="ingest-single-nosplit" data-folder="${escapeHtml(folder)}" data-filename="${escapeHtml(file.filename)}">Ingest (no split)</button>
        <button class="ghost small danger" type="button" data-action="delete-pending" data-folder="${escapeHtml(folder)}" data-filename="${escapeHtml(file.filename)}">Delete</button>
      `;
  return `
    <div class="list-row${job ? ' ingesting' : ''}">
      <span class="row-main">
        <span class="row-title">${escapeHtml(file.relative_path || file.filename)}</span>
        <span class="doc-meta">${escapeHtml(folder)}</span>
      </span>
      <span class="row-actions">${actions}</span>
    </div>
  `;
}

function renderJobs() {
  const view = $('jobs-view');
  if (!state.folder) {
    view.innerHTML = `<div class="empty-state">Select a folder to view jobs.</div>`;
    return;
  }
  const scopedJobs = state.jobs.filter((job) => job.project === state.folder || job.project?.startsWith(`${state.folder}/`));
  view.innerHTML = `
    ${pageHeader('Jobs', 'Recent ingestion jobs for the selected folder.', `
      <button class="ghost" type="button" data-action="refresh-jobs">Refresh</button>
    `)}
    <div class="page-body single-column">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Recent jobs</div>
          <div class="muted">${scopedJobs.length}</div>
        </div>
        <div class="panel-body">
          <div class="list">${scopedJobs.length ? scopedJobs.map(renderJob).join('') : '<div class="muted">No recent jobs.</div>'}</div>
        </div>
      </div>
    </div>
  `;
}

function renderJob(job) {
  const canRetry = job.status === 'failed' || job.status === 'cancelled';
  const retryBtn = canRetry
    ? `<button class="secondary small" type="button" data-action="reingest-job" data-job-id="${escapeHtml(job.id)}">Re-ingest</button>`
    : '';
  return `
    <div class="list-row">
      <span class="row-main">
        <span class="row-title">${escapeHtml(job.filename || job.id)}</span>
        <span class="job-meta">${escapeHtml(job.stage || '')}${job.error ? ` | ${escapeHtml(job.error)}` : ''}</span>
      </span>
      <span class="status-pill ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>
      ${retryBtn}
    </div>
  `;
}

function renderInspect() {
  const view = $('inspect-view');
  if (!state.inspectDocId) {
    view.innerHTML = `<div class="empty-state">Select a document from the Documents page.</div>`;
    return;
  }
  if (state.inspecting || !state.inspectInfo) {
    view.innerHTML = `
      ${pageHeader('Inspect document', 'Loading document details...', '<button class="ghost" type="button" data-action="go-documents">Back to documents</button>')}
      <div class="page-body single-column"><div class="panel"><div class="panel-body muted">Loading document inspection...</div></div></div>
    `;
    return;
  }

  const info = state.inspectInfo;
  view.innerHTML = `
    ${pageHeader(info.title || info.filename || 'Inspect document', info.filename || '', `
      <button class="ghost" type="button" data-action="go-documents">Back to documents</button>
    `)}
    <div class="inspect-tabs" aria-label="Document inspection views">
      ${renderInspectTab('metadata', 'Metadata')}
      ${renderInspectTab('toc', 'TOC')}
      ${renderInspectTab('page', 'Page text')}
      ${renderInspectTab('search', 'Search')}
    </div>
    <div class="page-body single-column">
      <div class="panel inspect-page-panel">
        <div class="panel-body">
          ${renderInspectPanel()}
        </div>
      </div>
    </div>
  `;
}

function renderInspectTab(panel, label) {
  return `<button class="inspect-tab ${state.inspectPanel === panel ? 'active' : ''}" type="button" data-action="set-inspect-panel" data-panel="${panel}">${label}</button>`;
}

function renderInspectPanel() {
  if (state.inspectPanel === 'metadata') return renderMetadataPanel();
  if (state.inspectPanel === 'toc') return renderTocPanel();
  if (state.inspectPanel === 'search') return renderSearchPanel();
  return renderPagePanel();
}

function renderMetadataPanel() {
  const info = state.inspectInfo;
  const rows = [
    ['Title', info.title],
    ['Filename', info.filename],
    ['Folder', info.folder || info.project],
    ['Type', info.document_type],
    ['Number', info.document_number],
    ['Revision', info.revision],
    ['Pages', info.total_pages || info.page_count],
    ['Summary', info.summary],
    ['Keywords', Array.isArray(info.keywords) ? info.keywords.join(', ') : info.keywords],
  ].filter(([, value]) => value);
  return `
    <div class="metadata-list">
      ${rows.length ? rows.map(([label, value]) => `
        <div class="meta-row">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `).join('') : '<div class="muted">No LLM metadata stored for this document.</div>'}
    </div>
  `;
}

function renderTocPanel() {
  return `
    <div class="toc-list full">
      ${state.inspectToc.length ? state.inspectToc.map(renderTocItem).join('') : '<div class="muted">No headings detected.</div>'}
    </div>
  `;
}

function renderTocItem(item) {
  const page = item.page_start || 1;
  return `
    <button class="toc-row level-${item.level || 1}" type="button" data-action="load-page" data-page="${page}">
      <span>${escapeHtml(item.heading || item.breadcrumb || 'Heading')}</span>
      <small>${page}${item.page_end && item.page_end !== page ? `-${item.page_end}` : ''}</small>
    </button>
  `;
}

function renderPagePanel() {
  const page = state.inspectPage;
  if (!page) return '<div class="muted">No extracted page text available.</div>';
  return `
    <div class="page-toolbar">
      <button class="ghost small" type="button" data-action="page-prev" ${page.prev_page_num ? '' : 'disabled'}>Previous</button>
      <span class="page-title">Page ${page.page_num || '-'}${page.page_count ? ` of ${page.page_count}` : ''}</span>
      <button class="ghost small" type="button" data-action="page-next" ${page.next_page_num ? '' : 'disabled'}>Next</button>
    </div>
    <div class="breadcrumb-box">${escapeHtml(page.breadcrumb || 'No breadcrumb detected')}</div>
    <pre class="page-text tall">${escapeHtml(page.content || '')}</pre>
  `;
}

function renderSearchPanel() {
  return `
    <form class="inspect-search" id="doc-search-form">
      <input class="field" id="doc-search-input" placeholder="Search inside this document">
      <button class="ghost small" type="submit">Search</button>
    </form>
    <div class="search-results full">
      ${state.inspectSearchResults.length ? state.inspectSearchResults.map((result) => `
        <button class="search-hit" type="button" data-action="load-page" data-page="${result.page_num || 1}">
          <span>${escapeHtml(result.breadcrumb || `Page ${result.page_num || ''}`)}</span>
          <small>${escapeHtml(result.snippet || '')}</small>
        </button>
      `).join('') : '<div class="muted">Search results will appear here.</div>'}
    </div>
  `;
}

function clearInspector() {
  state.inspectDocId = null;
  state.inspectInfo = null;
  state.inspectToc = [];
  state.inspectPage = null;
  state.inspectSearchResults = [];
  state.inspectPanel = 'page';
  state.inspecting = false;
}

async function openInspection(docId) {
  clearInspector();
  state.inspectDocId = Number(docId);
  state.tab = 'inspect';
  state.inspecting = true;
  state.inspectPanel = 'page';
  setHash();
  renderShell();
  renderInspect();
  await loadInspection(docId);
  state.inspecting = false;
  setHash();
  renderShell();
  renderInspect();
}

async function loadInspection(docId) {
  try {
    const [info, toc] = await Promise.all([
      api(`/api/documents/${enc(docId)}`),
      api(`/api/documents/${enc(docId)}/toc`),
    ]);
    state.inspectInfo = info;
    state.inspectToc = toc || [];
    await loadInspectPage(info.first_page_num || 1, false, false);
  } catch (err) {
    state.inspecting = false;
    showToast(err.message, 'error');
  }
}

async function loadInspectPage(pageNum, shouldRender = true, switchPanel = true) {
  if (!state.inspectDocId || !pageNum) return;
  try {
    state.inspectPage = await api(`/api/documents/${enc(state.inspectDocId)}/pages/${enc(pageNum)}`);
    if (switchPanel) state.inspectPanel = 'page';
    if (shouldRender) {
      setHash();
      renderInspect();
    }
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function searchInspectDocument(event) {
  event.preventDefault();
  if (!state.inspectDocId) return;
  const q = $('doc-search-input')?.value.trim();
  if (!q) {
    state.inspectSearchResults = [];
    renderInspect();
    return;
  }
  try {
    state.inspectSearchResults = await api(`/api/documents/${enc(state.inspectDocId)}/search?q=${enc(q)}`);
    renderInspect();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function createChatThread() {
  if (!state.folder) return;
  const created = await api(`/api/folders/${enc(state.folder)}/chats`, {
    method: 'POST',
    body: JSON.stringify({ title: 'New chat' }),
  });
  state.threadId = created.id;
  await loadChats();
  return created;
}

async function newChat() {
  try {
    await createChatThread();
    renderAsk();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function deleteChat() {
  if (!state.threadId) return;
  if (!confirm('Delete this chat? This permanently removes the conversation and cannot be undone.')) return;
  try {
    await api(`/api/chats/${enc(state.threadId)}`, { method: 'DELETE' });
    state.threadId = '';
    state.messages = [];
    await loadChats();
    renderAsk();
    showToast('Chat deleted');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const input = $('message-input');
  const content = input?.value.trim();
  if (!content || state.sending) return;
  try {
    state.sending = true;
    if (!state.threadId) await createChatThread();
    state.messages.push({ role: 'user', content, sources: [] });
    const live = { role: 'assistant', content: '', sources: [], live: { steps: [] } };
    state.messages.push(live);
    renderAsk({ scroll: 'answer' });
    const status = await streamChat(content, live);
    if (status === 'no-start') {
      // Streaming endpoint never began (e.g. unsupported) — fall back to the
      // blocking endpoint. The server only persists the user message after that
      // point, so no double-insert.
      const data = await api(`/api/chats/${enc(state.threadId)}/messages`, {
        method: 'POST',
        body: JSON.stringify({ content }),
      });
      state.messages = data.messages || [];
      state.chats = [data.thread, ...state.chats.filter((t) => t.id !== data.thread.id)];
    }
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    state.sending = false;
    renderAsk({ scroll: 'answer' });
    loadUsage();
  }
}

// Consume the streaming chat endpoint (SSE frames over a POST body). Returns:
//   'ok'           — final answer received and rendered
//   'stream-error' — stream began but errored mid-way (user msg already saved)
//   'no-start'     — stream never began; caller may fall back to blocking POST
async function streamChat(content, live) {
  let resp;
  try {
    resp = await fetch(`/api/chats/${enc(state.threadId)}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    });
  } catch (e) {
    return 'no-start';
  }
  if (!resp.ok || !resp.body) return 'no-start';

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let done = null;
  let broke = false;
  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (e) {
      broke = true;
      break;
    }
    if (chunk.done) break;
    buf += decoder.decode(chunk.value, { stream: true });
    const frames = buf.split('\n\n');
    buf = frames.pop();
    for (const frame of frames) {
      const dline = frame.split('\n').find((l) => l.startsWith('data:'));
      if (!dline) continue;
      let ev;
      try {
        ev = JSON.parse(dline.slice(5).trim());
      } catch (e) {
        continue;
      }
      if (ev.type === 'start') {
        updateLiveBubble(live);
      } else if (ev.type === 'thinking') {
        live.live.steps.push({ kind: 'think', text: ev.text });
        updateLiveBubble(live);
      } else if (ev.type === 'tool') {
        live.live.steps.push({ kind: 'tool', text: ev.query || ev.summary || ev.name, _tool: ev.name });
        updateLiveBubble(live);
      } else if (ev.type === 'tool_result') {
        // Annotate the matching live tool step with its outcome.
        const step = [...live.live.steps].reverse().find((s) => s.kind === 'tool' && !s._done);
        if (step) {
          step._done = true;
          const bits = [`${(ev.ms / 1000).toFixed(1)}s`];
          bits.push(ev.new_sources > 0 ? `${ev.new_sources} new` : 'no new results');
          if (ev.repeated) bits.push('repeat');
          if (ev.error) bits.push('error');
          step.text = `${step.text}  ·  ${bits.join(' · ')}`;
        }
        updateLiveBubble(live);
      } else if (ev.type === 'round_end') {
        /* boundary marker — used by the persisted trace, not the live view */
      } else if (ev.type === 'error') {
        done = { error: ev.error };
      } else if (ev.type === 'done') {
        done = ev;
      }
    }
  }
  if (done && done.messages) {
    state.messages = done.messages || [];
    state.chats = [done.thread, ...state.chats.filter((t) => t.id !== done.thread.id)];
    return 'ok';
  }
  if (done && done.error) showToast(done.error, 'error');
  else if (broke) showToast('The connection dropped before the answer finished.', 'error');
  await loadMessages();
  return 'stream-error';
}

async function switchThread(id) {
  state.threadId = id;
  await loadMessages();
  renderAsk();
}

function bindDropZone() {
  const zone = $('drop-zone');
  if (!zone) return;
  zone.addEventListener('dragover', (event) => {
    event.preventDefault();
    zone.classList.add('dragging');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    zone.classList.remove('dragging');
    stageFiles(event.dataTransfer.files);
  });
}

function stageFiles(files) {
  state.uploadFiles = Array.from(files || []);
  renderIngest();
}

function clearStaged() {
  state.uploadFiles = [];
  renderIngest();
}

async function uploadStaged() {
  if (!state.uploadFiles.length || !state.folder) return;
  const form = new FormData();
  for (const file of state.uploadFiles) form.append('files', file);
  try {
    await api(`/api/folders/${enc(state.folder)}/upload`, { method: 'POST', body: form });
    state.uploadFiles = [];
    showToast('Upload complete');
    await refreshActiveData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function ingestAll(btn) {
  if (!state.pending.length) return;
  setActionBusy(btn, 'Starting…');
  try {
    await api(`/api/folders/${enc(state.folder)}/ingest`, { method: 'POST', body: JSON.stringify({}) });
    showToast('Ingestion started');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    // Stay on the Ingest tab: pending rows flip to a live "ingesting" status.
    await refreshActiveData();
  }
}

async function ingestSingle(folder, filename, noSplit = false, btn) {
  setActionBusy(btn, 'Ingesting…');
  try {
    const query = noSplit ? '?no_split=true' : '';
    const res = await api(`/api/folders/${enc(folder)}/ingest/${enc(filename)}${query}`, { method: 'POST', body: JSON.stringify({}) });
    if (res && res.status === 'already_running') showToast('Already ingesting this file');
    else showToast(noSplit ? 'Ingestion started (no split)' : 'Ingestion started');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    await refreshActiveData();
  }
}

async function deleteDoc(docId, title, btn) {
  if (!docId) return;
  const label = title ? `"${title}"` : 'this document';
  if (!confirm(`Delete ${label}? This removes all of its data from the database. The source file stays on disk and will reappear under pending.`)) return;
  setActionBusy(btn, 'Deleting…');
  showToast('Deleting document…');
  try {
    await api(`/api/documents/${enc(docId)}`, { method: 'DELETE' });
    showToast('Document deleted');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    await refreshActiveData();
  }
}

async function deletePending(folder, filename, btn) {
  if (!filename) return;
  if (!confirm(`Delete pending file "${filename}"? This permanently removes the file from disk.`)) return;
  setActionBusy(btn, 'Deleting…');
  showToast('Deleting pending file…');
  try {
    await api(`/api/folders/${enc(folder)}/pending/${enc(filename)}/discard`, { method: 'POST', body: JSON.stringify({}) });
    showToast('Pending file deleted');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    await refreshActiveData();
  }
}

async function reingestJob(jobId) {
  if (!jobId) return;
  try {
    await api(`/api/ingestion/jobs/${enc(jobId)}/retry`, { method: 'POST', body: JSON.stringify({}) });
    showToast('Re-ingestion started');
    state.tab = 'jobs';
    setHash();
    await refreshActiveData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(idx ? 1 : 0)} ${units[idx]}`;
}

async function promptNewFolder() {
  const name = prompt('Folder name (use / for sub-folders, e.g. tender-A/specs)');
  if (!name) return;
  try {
    await api('/api/folders', { method: 'POST', body: JSON.stringify({ name }) });
    expandAncestors(name);
    saveExpanded();
    state.folder = name;
    await loadFolders();
    await refreshActiveData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function promptNewSubfolder(parent) {
  if (!parent) return promptNewFolder();
  const child = prompt(`New sub-folder inside "${parent}"`);
  if (!child) return;
  const trimmed = child.trim().replace(/^\/+|\/+$/g, '');
  if (!trimmed) return;
  const name = `${parent}/${trimmed}`;
  try {
    await api('/api/folders', { method: 'POST', body: JSON.stringify({ name }) });
    state.expanded.add(parent);
    expandAncestors(name);
    saveExpanded();
    state.folder = name;
    await loadFolders();
    await refreshActiveData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function openMoveDialog(docId) {
  const doc = state.docs.find((d) => d.id === docId);
  if (!doc) return;
  state.moveDoc = doc;
  renderMoveDialog();
}

function closeMoveDialog() {
  state.moveDoc = null;
  const dlg = $('move-dialog');
  if (dlg) dlg.remove();
}

function renderMoveDialog() {
  if (!state.moveDoc) return;
  const doc = state.moveDoc;
  const docFolder = doc.folder || doc.project || '';
  const root = docFolder.split('/')[0];
  const candidates = state.folders
    .filter((f) => f.folder === root || f.folder.startsWith(`${root}/`))
    .filter((f) => f.folder !== docFolder)
    .sort((a, b) => a.folder.localeCompare(b.folder));

  const existing = $('move-dialog');
  if (existing) existing.remove();
  const dlg = document.createElement('div');
  dlg.id = 'move-dialog';
  dlg.className = 'modal-overlay';
  dlg.innerHTML = `
    <div class="modal">
      <div class="modal-head">
        <h3>Move document</h3>
        <button class="ghost small" type="button" data-action="cancel-move" aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <div class="muted">${escapeHtml(doc.title || doc.filename || '')}</div>
        <div class="muted">Currently in: <code>${escapeHtml(docFolder)}</code></div>
        <label class="field-label" for="move-target">Target sub-folder under <code>${escapeHtml(root)}</code></label>
        <select class="field" id="move-target">
          ${candidates.length
            ? candidates.map((f) => `<option value="${escapeHtml(f.folder)}">${escapeHtml(f.folder)}</option>`).join('')
            : '<option value="" disabled>No other sub-folders under this root</option>'}
        </select>
        <div class="muted">Documents can only move within the same root folder (<code>${escapeHtml(root)}</code>).</div>
      </div>
      <div class="modal-foot">
        <button class="ghost" type="button" data-action="cancel-move">Cancel</button>
        <button class="primary" type="button" data-action="confirm-move" ${candidates.length ? '' : 'disabled'}>Move</button>
      </div>
    </div>
  `;
  dlg.addEventListener('click', (e) => {
    if (e.target === dlg) closeMoveDialog();
  });
  document.body.appendChild(dlg);
}

async function confirmMove() {
  if (!state.moveDoc) return;
  const select = $('move-target');
  if (!select || !select.value) return;
  const target = select.value;
  const doc = state.moveDoc;
  closeMoveDialog();
  try {
    await api('/api/documents/bulk-move', {
      method: 'POST',
      body: JSON.stringify({ doc_ids: [doc.id], target_folder: target }),
    });
    showToast(`Moved to ${target}`);
    await loadFolders();
    await refreshActiveData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function startPolling() {
  setInterval(async () => {
    if (!state.folder) return;
    try {
      const previousJobs = JSON.stringify(state.jobs);
      const hadActiveJobs = state.jobs.some((job) => job.status === 'queued' || job.status === 'running');
      await loadJobs();
      const hasActiveJobs = state.jobs.some((job) => job.status === 'queued' || job.status === 'running');
      const changed = previousJobs !== JSON.stringify(state.jobs);
      if (changed && (state.tab === 'jobs' || state.tab === 'ingest') && (hadActiveJobs || hasActiveJobs)) {
        renderActiveView();
      }
    } catch (_) {}
  }, 4000);
}

function bindGlobalEvents() {
  for (const tab of VALID_TABS) {
    const btn = $(`${tab}-tab`);
    if (btn) btn.addEventListener('click', () => setTab(tab));
  }
  $('new-folder-btn').addEventListener('click', promptNewFolder);
  $('folder-search').addEventListener('input', renderShell);
  // Collapsible left (folder/tender) pane — persisted across sessions.
  const shell = document.querySelector('.app-shell');
  if (localStorage.getItem('folders_collapsed') === '1') shell.classList.add('folders-collapsed');
  $('sidebar-toggle')?.addEventListener('click', () => {
    const collapsed = shell.classList.toggle('folders-collapsed');
    localStorage.setItem('folders_collapsed', collapsed ? '1' : '0');
  });
  document.addEventListener('click', handleClick);
  document.addEventListener('change', handleChange);
  document.addEventListener('submit', handleSubmit);
  window.addEventListener('hashchange', async () => {
    readHash();
    expandAncestors(state.folder);
    renderShell();
    await refreshActiveData();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.moveDoc) closeMoveDialog();
  });
}

function handleClick(event) {
  const target = event.target.closest('[data-action]');
  if (!target || target.disabled) return;
  const { action } = target.dataset;
  event.preventDefault();
  if (action === 'select-folder') void selectFolder(target.dataset.folder || '');
  if (action === 'toggle-folder') {
    const folder = target.dataset.folder || '';
    if (!folder) return;
    if (state.expanded.has(folder)) state.expanded.delete(folder);
    else state.expanded.add(folder);
    saveExpanded();
    renderShell();
  }
  if (action === 'new-subfolder') void promptNewSubfolder(target.dataset.parent || '');
  if (action === 'move-doc') openMoveDialog(Number(target.dataset.docId));
  if (action === 'cancel-move') closeMoveDialog();
  if (action === 'confirm-move') void confirmMove();
  if (action === 'new-chat') void newChat();
  if (action === 'delete-chat') void deleteChat();
  if (action === 'toggle-trace') {
    const wrap = target.closest('.search-trace');
    const panel = wrap?.querySelector('.trace-panel');
    if (!panel) return;
    const willShow = panel.classList.contains('hidden');
    panel.classList.toggle('hidden');
    if (willShow && !panel.dataset.rendered) {
      const msg = (state.messages || []).find((m) => String(m.id) === wrap.dataset.msg);
      if (msg && msg.trace) { panel.dataset.rendered = '1'; void renderSearchTrace(panel, msg.trace); }
    }
  }
  if (action === 'refresh') void refreshActiveData();
  if (action === 'go-documents') setTab('documents');
  if (action === 'go-ingest') setTab('ingest');
  if (action === 'upload-staged') void uploadStaged();
  if (action === 'clear-staged') clearStaged();
  if (action === 'ingest-all') void ingestAll(target);
  if (action === 'ingest-single') void ingestSingle(target.dataset.folder || state.folder, target.dataset.filename || '', false, target);
  if (action === 'ingest-single-nosplit') void ingestSingle(target.dataset.folder || state.folder, target.dataset.filename || '', true, target);
  if (action === 'delete-pending') void deletePending(target.dataset.folder || state.folder, target.dataset.filename || '', target);
  if (action === 'delete-doc') void deleteDoc(target.dataset.docId || '', target.dataset.docTitle || '', target);
  if (action === 'reingest-job') void reingestJob(target.dataset.jobId || '');
  if (action === 'refresh-jobs') void loadJobs().then(renderJobs);
  if (action === 'inspect-doc') void openInspection(target.dataset.docId);
  if (action === 'set-inspect-panel') {
    state.inspectPanel = target.dataset.panel || 'page';
    setHash();
    renderInspect();
  }
  if (action === 'load-page') void loadInspectPage(Number(target.dataset.page));
  if (action === 'page-prev') void loadInspectPage(state.inspectPage?.prev_page_num);
  if (action === 'page-next') void loadInspectPage(state.inspectPage?.next_page_num);
}

function handleChange(event) {
  const target = event.target;
  if (target?.dataset?.action === 'switch-thread') void switchThread(target.value);
  if (target?.dataset?.action === 'stage-files') stageFiles(target.files);
}

function handleSubmit(event) {
  if (event.target?.id === 'chat-form') void sendMessage(event);
  if (event.target?.id === 'doc-search-form') void searchInspectDocument(event);
}

async function init() {
  bindGlobalEvents();
  readHash();
  state.expanded = loadExpanded();
  await loadFolders();
  expandAncestors(state.folder);
  setHash();
  await refreshActiveData();
  startPolling();
}

// ===================== Check tab =====================
const CHECK_DOC_TYPES = ['Drawing', 'Datasheet', 'Vendor document', 'Procedure', 'Specification', 'Calculation', 'Material requisition', 'Inspection / test report', 'Report', 'Other'];
const SEV_COLORS = { critical: '#D7263D', major: '#E8A33D', minor: '#F4D35E', observation: '#4C9BE8' };
const checkState = { sub: 'new', runId: null, sse: null, resultsLoaded: false, starting: false };

function renderCheck() {
  const view = $('check-view');
  if (!state.folder) {
    view.innerHTML = `<div class="empty-state">Select a folder (the reference library) to check documents against.</div>`;
    return;
  }
  if (checkState.sub === 'run' && checkState.runId) { renderCheckRun(); return; }
  renderCheckNew();
}

function renderCheckNew() {
  const view = $('check-view');
  view.innerHTML = `
    ${pageHeader('Check documents', `against ${state.folder}`)}
    <div class="check-card">
      <div class="check-grid">
        <div class="check-col">
          <div class="check-col-head">Documents to check <span class="muted">your submission</span></div>
          <label class="dropzone"><input id="chk-submitted" type="file" multiple><span class="dz-text">Drop files or <b>browse</b></span></label>
          <ul id="chk-submitted-list" class="filelist"></ul>
        </div>
        <div class="check-col">
          <div class="check-col-head">Reference <span class="muted">tender / PO / PR — combine any</span></div>
          <div class="ref-choice">
            <label class="radio"><input type="checkbox" id="chk-ref-this" checked> This folder: <b>${escapeHtml(state.folder)}</b></label>
            <label class="radio"><input type="checkbox" id="chk-ref-other"> Another folder</label>
            <select id="chk-ref-folder" class="hidden"></select>
            <label class="radio"><input type="checkbox" id="chk-ref-upload-on"> Upload files</label>
            <div id="chk-ref-upload" class="hidden">
              <label class="dropzone"><input id="chk-reference" type="file" multiple><span class="dz-text">Drop reference files or <b>browse</b></span></label>
              <ul id="chk-reference-list" class="filelist"></ul>
            </div>
          </div>
        </div>
      </div>
      <div class="check-row">
        <label class="field"><span>Document type</span><select id="chk-doctype"><option value="">Select…</option>${CHECK_DOC_TYPES.map((d) => `<option>${d}</option>`).join('')}</select></label>
        <label class="field"><span>Prepared by <span class="muted">optional</span></span><input id="chk-originator" type="text" placeholder="e.g. ACME Pumps"></label>
      </div>
      <label class="field"><span>What to check <span class="muted">in your words</span></span><textarea id="chk-prompt" rows="3" placeholder="e.g. Verify design pressure, materials and BOM quantities against the PO; flag deviations."></textarea></label>
      <label class="switch-row"><input id="chk-revision" type="checkbox"> This is a new revision — check that prior comments were incorporated</label>
      <div id="chk-prior" class="hidden"><label class="dropzone"><input id="chk-prior-file" type="file"><span class="dz-text">Old commented PDF — <b>browse</b></span></label><ul id="chk-prior-list" class="filelist"></ul></div>
      <div class="check-actions"><button id="chk-run" class="primary">Run check</button></div>
    </div>
    <div class="check-recent"><h3>Recent checks</h3><div id="chk-recent" class="muted">Loading…</div></div>
  `;
  bindFileList('chk-submitted', 'chk-submitted-list');
  bindFileList('chk-reference', 'chk-reference-list');
  bindFileList('chk-prior-file', 'chk-prior-list');
  $('chk-ref-other').addEventListener('change', (e) => $('chk-ref-folder').classList.toggle('hidden', !e.target.checked));
  $('chk-ref-upload-on').addEventListener('change', (e) => $('chk-ref-upload').classList.toggle('hidden', !e.target.checked));
  $('chk-ref-folder').innerHTML = state.folders
    .filter((f) => f.folder !== state.folder)
    .map((f) => `<option value="${escapeHtml(f.folder)}">${escapeHtml(f.folder)}</option>`).join('');
  $('chk-revision').addEventListener('change', (e) => $('chk-prior').classList.toggle('hidden', !e.target.checked));
  $('chk-run').addEventListener('click', startCheck);
  loadRecentChecks();
}

function bindFileList(inputId, listId) {
  const input = $(inputId); const list = $(listId);
  if (!input || !list) return;
  const render = () => {
    list.innerHTML = Array.from(input.files || []).map((f) => `<li>${escapeHtml(f.name)} <span class="muted">${(f.size / 1024) | 0} KB</span></li>`).join('');
  };
  input.addEventListener('change', render);

  // The input is a hidden child of a label.dropzone; clicking opens the picker,
  // but dragging files onto it did nothing. Wire drag-and-drop onto the zone.
  const zone = input.closest('.dropzone');
  if (zone) {
    ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation(); zone.classList.add('dragover');
    }));
    ['dragleave', 'dragend'].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation(); zone.classList.remove('dragover');
    }));
    zone.addEventListener('drop', (e) => {
      e.preventDefault(); e.stopPropagation(); zone.classList.remove('dragover');
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || !files.length) return;
      try { input.files = files; } catch (_) { /* unsupported: ignore */ }
      render();
    });
  }
}

async function loadRecentChecks() {
  try {
    const runs = await api(`/api/runs?project_number=${enc(state.folder)}`);
    const el = $('chk-recent'); if (!el) return;
    if (!runs.length) { el.innerHTML = '<div class="muted">No checks yet for this folder.</div>'; return; }
    el.innerHTML = runs.map((r) => `<div class="recent-row"><span class="status-pill" data-s="${r.status}">${r.status}</span><span>${escapeHtml(r.document_type || 'document')}</span><span class="muted">${escapeHtml(r.created_at || '')}</span><a href="#" class="open-run" data-runid="${r.id}">open</a></div>`).join('');
    el.querySelectorAll('.open-run').forEach((a) => a.addEventListener('click', (e) => { e.preventDefault(); openCheckRun(a.dataset.runid); }));
  } catch (_) { /* ignore */ }
}

async function startCheck() {
  const btn = $('chk-run'); btn.disabled = true;
  try {
    const submitted = Array.from($('chk-submitted').files || []);
    if (!submitted.length) { showToast('Add at least one document to check', 'error'); btn.disabled = false; return; }

    // Additive reference sources: any combination of folders + uploads.
    const refFolders = [];
    if ($('chk-ref-this').checked) refFolders.push(state.folder);
    if ($('chk-ref-other').checked && $('chk-ref-folder').value) refFolders.push($('chk-ref-folder').value);
    const uploadRefs = $('chk-ref-upload-on').checked ? Array.from($('chk-reference').files || []) : [];
    if (!refFolders.length && !uploadRefs.length) {
      showToast('Select at least one reference source (a folder or uploaded files)', 'error');
      btn.disabled = false; return;
    }
    const run = await api('/api/runs', { method: 'POST', body: JSON.stringify({
      project_number: state.folder,
      document_type: $('chk-doctype').value || null,
      originator: $('chk-originator').value || null,
      guiding_prompt: $('chk-prompt').value || null,
      reference_mode: uploadRefs.length ? 'both' : 'existing',
      reference_projects: refFolders,
      is_revision: $('chk-revision').checked,
    }) });
    for (const f of submitted) await uploadCheckFile(run.id, 'submitted', f);
    for (const f of uploadRefs) await uploadCheckFile(run.id, 'reference', f);
    if ($('chk-revision').checked && $('chk-prior-file').files[0]) await uploadCheckFile(run.id, 'prior_commented', $('chk-prior-file').files[0]);
    openCheckRun(run.id);
  } catch (e) { showToast(e.message, 'error'); btn.disabled = false; }
}

async function uploadCheckFile(runId, role, file) {
  const form = new FormData(); form.append('role', role); form.append('file', file);
  await api(`/api/runs/${runId}/uploads`, { method: 'POST', body: form });
}

function openCheckRun(runId) { checkState.sub = 'run'; checkState.runId = runId; checkState.starting = false; renderCheckRun(); }

function renderCheckRun() {
  checkState.resultsLoaded = false;
  const view = $('check-view'); const runId = checkState.runId;
  view.innerHTML = `
    ${pageHeader('Check run', '', '<button class="ghost" id="chk-back">&larr; New check</button>')}
    <div id="chk-status" class="run-status"></div>
    <div id="chk-uploads"></div>
    <div id="chk-start" class="hidden"><button id="chk-start-btn" class="primary">Run check</button></div>
    <div id="chk-progress" class="run-log hidden"><h3>Progress</h3><ul id="chk-log"></ul></div>
    <div id="chk-thinking" class="run-thinking hidden"><div class="rt-label">Model reasoning <span class="rt-live">live</span></div><div id="chk-thinking-body" class="rt-body"></div></div>
    <div id="chk-results" class="hidden"></div>
  `;
  $('chk-back').addEventListener('click', () => { if (checkState.sse) { checkState.sse.close(); checkState.sse = null; } checkState.sub = 'new'; renderCheck(); });
  $('chk-start-btn')?.addEventListener('click', () => startRun(runId));
  pollCheckRun(runId);
}

async function startRun(runId) {
  checkState.starting = true;
  const btn = $('chk-start-btn'); if (btn) btn.disabled = true;
  try { await api(`/api/runs/${runId}/start`, { method: 'POST' }); connectCheckSSE(runId); pollCheckRun(runId); }
  catch (e) {
    // Surface a manual "Run check" fallback if auto-start fails.
    showToast(e.message, 'error'); checkState.starting = false;
    if (btn) btn.disabled = false; $('chk-start')?.classList.remove('hidden');
  }
}

function connectCheckSSE(runId) {
  if (checkState.sse) return;
  $('chk-progress').classList.remove('hidden');
  const sse = new EventSource(`/api/runs/${runId}/stream`);
  checkState.sse = sse;
  sse.addEventListener('progress', (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'thinking') { appendThinking(ev.delta || ''); return; }
    const li = document.createElement('li'); li.textContent = ev.stage || ev.type || ''; $('chk-log')?.appendChild(li);
  });
  sse.addEventListener('end', () => { sse.close(); checkState.sse = null; loadCheckResults(runId); pollCheckRun(runId); loadUsage(); });
}

function appendThinking(text) {
  if (!text) return;
  const panel = $('chk-thinking'); const body = $('chk-thinking-body');
  if (!panel || !body) return;
  panel.classList.remove('hidden');
  body.textContent += text;
  body.scrollTop = body.scrollHeight;
}

async function pollCheckRun(runId) {
  let run; try { run = await api(`/api/runs/${runId}`); } catch (e) { return; }
  if (checkState.runId !== runId) return;
  const sp = $('chk-status'); if (sp) sp.innerHTML = `<span class="status-pill" data-s="${run.status}">${run.status}</span>`;
  const up = $('chk-uploads'); if (up) up.innerHTML = `<table class="grid">${run.uploads.map((u) => `<tr><td>${u.role}</td><td>${escapeHtml(u.filename)}</td><td class="ing ${u.ingest_status === 'done' ? 'ok' : u.ingest_status === 'failed' ? 'bad' : 'wait'}">${u.ingest_status}${u.page_count ? ' · ' + u.page_count + 'p' : ''}</td></tr>`).join('')}</table>`;
  const subs = run.uploads.filter((u) => u.role === 'submitted');
  const ready = subs.length && run.uploads.every((u) => ['done', 'failed'].includes(u.ingest_status)) && subs.some((u) => u.ingest_status === 'done' && u.doc_id);
  if (run.status === 'created') {
    // Auto-start once uploads finish ingesting — no second click needed.
    if (ready && !checkState.starting) { startRun(runId); }
    else if (!ready) { setTimeout(() => pollCheckRun(runId), 1500); }
  }
  else if (run.status === 'queued' || run.status === 'running') { $('chk-start').classList.add('hidden'); connectCheckSSE(runId); setTimeout(() => pollCheckRun(runId), 2000); }
  else if (run.status === 'done') { $('chk-start').classList.add('hidden'); loadCheckResults(runId); }
  else if (run.status === 'failed') { $('chk-progress').classList.remove('hidden'); const li = document.createElement('li'); li.textContent = 'Failed: ' + (run.error || 'unknown'); $('chk-log')?.appendChild(li); }
}

async function loadCheckResults(runId) {
  if (checkState.resultsLoaded) return; checkState.resultsLoaded = true;
  const res = await api(`/api/runs/${runId}/results`);
  const el = $('chk-results'); el.classList.remove('hidden');
  const findings = res.findings || []; const comments = res.comment_results || [];
  // Default to the PDF preview filling the pane (findings collapsed); remember
  // the user's choice. '#zoom=page-fit' makes pdf.js fit each full page in the
  // viewport so the page height is never cropped.
  const layout = localStorage.getItem('chk_layout') || 'pdf-full';
  const layoutClass = layout === 'pdf-full' ? ' findings-collapsed'
    : layout === 'findings-full' ? ' pdf-collapsed' : '';
  const pdfUrl = encodeURIComponent('/api/runs/' + runId + '/annotated.pdf') + '#zoom=page-fit';
  el.innerHTML = `
    <div class="check-split${layoutClass}" id="chk-split">
      <div class="split-col findings-col">
        <div class="split-col-head"><span class="split-title">Findings <span class="muted">(${findings.length})</span></span><button class="split-collapse" data-act="collapse-findings" title="Collapse findings" aria-label="Collapse findings">‹</button></div>
        <ul class="findings">${findings.map(checkFindingCard).join('') || '<li class="muted">No findings.</li>'}</ul>
      </div>
      <div class="split-handle" id="chk-split-handle" title="Drag to resize"></div>
      <div class="split-col pdf-pane">
        <div class="split-col-head"><button class="split-collapse" data-act="collapse-pdf" title="Collapse preview" aria-label="Collapse preview">›</button><a href="/api/runs/${runId}/annotated.pdf" target="_blank" class="muted">Open annotated PDF &#8599;</a></div>
        <iframe src="/static/pdfjs/web/viewer.html?file=${pdfUrl}"></iframe>
      </div>
    </div>
    ${comments.length ? `<h3>Prior comment incorporation</h3><ul class="findings">${comments.map(checkCommentCard).join('')}</ul>` : ''}
    <div class="check-trace"><button class="chip-btn" id="chk-trace-btn">Show debug trace</button><div id="chk-trace-body" class="trace-body hidden"></div></div>
  `;
  el.querySelectorAll('.finding [data-act]').forEach((b) => b.addEventListener('click', async () => {
    const li = b.closest('.finding');
    await api(`/api/findings/${li.dataset.id}/status?status=${b.dataset.act}`, { method: 'POST' });
    li.querySelector('.fstatus').textContent = b.dataset.act; li.classList.toggle('dim', b.dataset.act === 'dismissed');
  }));
  $('chk-trace-btn')?.addEventListener('click', () => toggleTrace(runId));
  bindSplit();
}

function bindSplit() {
  const split = $('chk-split'); const handle = $('chk-split-handle');
  if (!split) return;
  // Restore saved width.
  const saved = parseInt(localStorage.getItem('chk_findings_w') || '', 10);
  if (saved >= 180) split.style.setProperty('--findings-w', saved + 'px');

  // Drag-to-resize the findings column.
  if (handle) {
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      if (split.classList.contains('findings-collapsed') || split.classList.contains('pdf-collapsed')) return;
      const startX = e.clientX;
      const startW = $('chk-split').querySelector('.findings-col').getBoundingClientRect().width;
      const maxW = split.getBoundingClientRect().width - 220;
      const onMove = (ev) => {
        const w = Math.max(180, Math.min(maxW, startW + (ev.clientX - startX)));
        split.style.setProperty('--findings-w', w + 'px');
      };
      const onUp = () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.body.classList.remove('resizing');
        const w = parseInt(getComputedStyle(split).getPropertyValue('--findings-w'), 10);
        if (w >= 180) localStorage.setItem('chk_findings_w', String(w));
      };
      document.body.classList.add('resizing');
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  // Collapse toggles — collapsing one expands the other (at most one collapsed).
  const saveLayout = () => {
    const l = split.classList.contains('findings-collapsed') ? 'pdf-full'
      : split.classList.contains('pdf-collapsed') ? 'findings-full' : 'split';
    localStorage.setItem('chk_layout', l);
  };
  split.querySelector('[data-act="collapse-findings"]')?.addEventListener('click', () => {
    split.classList.remove('pdf-collapsed');
    split.classList.toggle('findings-collapsed');
    saveLayout();
  });
  split.querySelector('[data-act="collapse-pdf"]')?.addEventListener('click', () => {
    split.classList.remove('findings-collapsed');
    split.classList.toggle('pdf-collapsed');
    saveLayout();
  });
}

async function toggleTrace(runId) {
  const body = $('chk-trace-body'); const btn = $('chk-trace-btn');
  if (!body) return;
  if (!body.classList.contains('hidden')) { body.classList.add('hidden'); btn.textContent = 'Show debug trace'; return; }
  btn.textContent = 'Loading…';
  try {
    const t = await api(`/api/runs/${enc(runId)}/trace`);
    body.innerHTML = renderTrace(t);
    body.classList.remove('hidden');
    btn.textContent = 'Hide debug trace';
  } catch (e) { btn.textContent = 'Show debug trace'; showToast(e.message, 'error'); }
}

function renderTrace(t) {
  if (!t || (!t.thinking && !(t.verify || []).length && !t.limits)) {
    return '<div class="muted">No trace recorded for this run.</div>';
  }
  const lim = t.limits || {};
  const limRow = `<div class="trace-limits">`
    + `<span><b>${lim.candidates ?? '?'}</b> raised → <b>${lim.confirmed ?? '?'}</b> kept (<b>${lim.possible ?? 0}</b> possible), <b>${lim.dropped ?? 0}</b> pruned</span>`
    + `<span>submitted: <b>${(lim.submitted_chars ?? 0).toLocaleString()}</b> chars${lim.submitted_truncated ? ' <span class="trace-warn">(TRUNCATED)</span>' : ''}</span>`
    + (lim.compare_output_truncated ? '<span class="trace-warn">output budget reached — findings may be missing</span>' : '')
    + (t.model ? `<span>model: ${escapeHtml(t.model)}${t.effort ? ' · effort ' + escapeHtml(t.effort) : ''}</span>` : '')
    + `</div>`;
  const verify = (t.verify || []).map((v) => {
    const cls = v.verdict === 'kept' ? 'ok' : v.verdict === 'dropped' ? 'bad' : 'wait';
    return `<li><span class="tag ${cls}">${escapeHtml(v.verdict || '')}</span> <span class="muted small">p${v.page ?? '?'} · ${escapeHtml(v.confidence || '')}</span> ${escapeHtml(v.title || '')}${v.reason ? ` <span class="muted small">— ${escapeHtml(v.reason)}</span>` : ''}</li>`;
  }).join('');
  return limRow
    + (t.thinking ? `<div class="trace-section"><div class="trace-h">Model reasoning</div><div class="trace-think">${escapeHtml(t.thinking)}</div></div>` : '')
    + (verify ? `<div class="trace-section"><div class="trace-h">Self-verification (kept / dropped / error)</div><ul class="trace-verify">${verify}</ul></div>` : '')
    + ((t.warnings || []).length ? `<div class="trace-section"><div class="trace-h">Warnings</div><ul class="trace-verify">${t.warnings.map((w) => `<li class="muted small">${escapeHtml(w)}</li>`).join('')}</ul></div>` : '');
}

function checkFindingCard(f) {
  // Compact index only — the full comment text lives on the annotated PDF
  // (shown in the viewer), so it is not duplicated here.
  const col = SEV_COLORS[f.severity] || '#888';
  const possible = f.confidence === 'low' ? '<span class="tag wait">possible</span>' : '';
  return `<li class="finding" data-id="${f.id}"><div class="finding-head"><span class="dot" style="background:${col}"></span><strong>${escapeHtml(f.title || f.category)}</strong><span class="tag">${f.category}</span><span class="tag">${f.severity}</span>${possible}<span class="muted small">p${f.page_num ?? '?'}</span></div><div class="actions"><button data-act="accepted">Accept</button><button data-act="dismissed">Dismiss</button><span class="muted small fstatus">${f.status}</span></div></li>`;
}

function checkCommentCard(c) {
  const cls = { incorporated: 'ok', partially: 'wait', not_incorporated: 'bad', not_applicable: 'muted' }[c.verdict] || '';
  return `<li class="finding"><div class="finding-head"><strong>${escapeHtml(c.prior_comment_text || '(comment)')}</strong><span class="tag ${cls}">${escapeHtml(c.verdict)}</span><span class="muted small">p${c.prior_page ?? '?'}</span></div><div>${escapeHtml(c.detail || '')}</div></li>`;
}

(async function initChecker() {
  try {
    const me = await api('/api/me');
    state.isAdmin = !!me.is_admin;
    const chip = $('user-chip'); if (chip) { chip.textContent = me.display_name || me.email || ''; chip.classList.remove('hidden'); }
    const out = $('signout-btn'); if (out) { out.classList.remove('hidden'); out.addEventListener('click', async (e) => { e.preventDefault(); try { await fetch('/logout', { method: 'POST' }); } catch (_) {} location.href = '/login'; }); }
    if (!state.isAdmin && ['documents', 'ingest', 'jobs'].includes(state.tab)) state.tab = 'ask';
    renderShell();
    renderActiveView();
    loadUsage();
  } catch (_) { /* not logged in / endpoint missing */ }
})();

async function loadUsage() {
  try {
    const u = await api('/api/me/usage');
    const chip = $('usage-chip');
    if (!chip) return;
    const inr = Math.round((u && u.inr) || 0);
    if (inr > 0) {
      chip.textContent = `₹${inr.toLocaleString('en-IN')} spent`;
      chip.classList.remove('hidden');
    } else {
      chip.classList.add('hidden');
    }
  } catch (_) { /* usage unavailable */ }
}

init().catch((err) => {
  console.error(err);
  showToast(err.message, 'error');
});
