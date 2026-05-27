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

const VALID_TABS = new Set(['ask', 'documents', 'ingest', 'jobs', 'inspect']);
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
  await Promise.all([loadDocuments(), loadChats(), loadJobs()]);
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
  $('folder-stats').textContent = active
    ? `${active.docs || 0} documents | ${active.pages || 0} pages | ${active.pending || 0} pending`
    : '';

  for (const tab of VALID_TABS) {
    const btn = $(`${tab}-tab`);
    const view = $(`${tab}-view`);
    if (btn) btn.classList.toggle('active', state.tab === tab);
    if (view) view.classList.toggle('hidden', state.tab !== tab);
  }
  $('inspect-tab').classList.toggle('hidden', !state.inspectDocId);
}

function setTab(tab) {
  if (!VALID_TABS.has(tab)) return;
  if (tab === 'inspect' && !state.inspectDocId) return;
  state.tab = tab;
  setHash();
  renderShell();
  renderActiveView();
}

function renderActiveView() {
  renderShell();
  if (state.tab === 'ask') renderAsk();
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

function renderAsk() {
  const view = $('ask-view');
  if (!state.folder) {
    view.innerHTML = `<div class="empty-state">Select a folder to ask questions.</div>`;
    return;
  }

  view.innerHTML = `
    <div class="ask-toolbar">
      ${renderThreadControl()}
      <button class="ghost" type="button" data-action="new-chat">New chat</button>
      <button class="ghost" type="button" data-action="refresh">Refresh</button>
    </div>
    <div class="messages" id="messages">
      ${state.messages.length ? state.messages.map(renderMessage).join('') : renderAskEmpty()}
    </div>
    <form class="chat-input" id="chat-form">
      <textarea id="message-input" placeholder="Ask about this folder" ${state.sending ? 'disabled' : ''}></textarea>
      <button class="primary" type="submit" ${state.sending ? 'disabled' : ''}>Send</button>
    </form>
  `;
  const messages = $('messages');
  messages.scrollTop = messages.scrollHeight;
  renderMath(view);
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
  return `
    <div class="empty-state">
      <div>
        <strong>Ask a question about ${escapeHtml(state.folder)}</strong>
        <div class="muted">Examples: scope of work, vendor requirements, LD clause, inspection requirements</div>
      </div>
    </div>
  `;
}

function renderMessage(message) {
  const sources = Array.isArray(message.sources) ? message.sources : [];
  return `
    <article class="message ${message.role === 'user' ? 'user' : 'assistant'}">
      <div class="message-role">${message.role === 'user' ? 'You' : 'Answer'}</div>
      <div class="message-body">${renderMarkdown(message.content || '')}</div>
      ${sources.length ? `<div class="sources">${sources.map(renderSource).join('')}</div>` : ''}
    </article>
  `;
}

function renderSource(source) {
  const title = source.doc_title || source.filename || 'Source';
  const page = source.page_num ? `Page ${source.page_num}` : '';
  const breadcrumb = source.breadcrumb || '';
  return `
    <div class="source">
      <div class="source-title">[${escapeHtml(source.id || '')}] ${escapeHtml(title)}</div>
      <div class="source-meta">${escapeHtml([page, breadcrumb].filter(Boolean).join(' | '))}</div>
      ${source.snippet ? `<div class="source-meta">${escapeHtml(source.snippet)}</div>` : ''}
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
            <div class="muted">${state.pending.length} pending</div>
          </div>
          <button class="primary small" type="button" data-action="ingest-all" ${state.pending.length ? '' : 'disabled'}>Ingest pending</button>
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

function renderPending(file) {
  return `
    <div class="list-row">
      <span class="row-main">
        <span class="row-title">${escapeHtml(file.relative_path || file.filename)}</span>
        <span class="doc-meta">${escapeHtml(file.folder || state.folder)}</span>
      </span>
      <button class="secondary small" type="button" data-action="ingest-single" data-folder="${escapeHtml(file.folder || state.folder)}" data-filename="${escapeHtml(file.filename)}">Ingest</button>
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

async function sendMessage(event) {
  event.preventDefault();
  const input = $('message-input');
  const content = input?.value.trim();
  if (!content || state.sending) return;
  try {
    state.sending = true;
    if (!state.threadId) await createChatThread();
    state.messages.push({ role: 'user', content, sources: [] });
    state.messages.push({ role: 'assistant', content: 'Working...', sources: [] });
    renderAsk();
    const data = await api(`/api/chats/${enc(state.threadId)}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    });
    state.messages = data.messages || [];
    state.chats = [data.thread, ...state.chats.filter((t) => t.id !== data.thread.id)];
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    state.sending = false;
    renderAsk();
  }
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

async function ingestAll() {
  if (!state.pending.length) return;
  try {
    await api(`/api/folders/${enc(state.folder)}/ingest`, { method: 'POST', body: JSON.stringify({}) });
    showToast('Ingestion started');
    state.tab = 'jobs';
    setHash();
    await refreshActiveData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function ingestSingle(folder, filename) {
  try {
    await api(`/api/folders/${enc(folder)}/ingest/${enc(filename)}`, { method: 'POST', body: JSON.stringify({}) });
    showToast('Ingestion started');
    state.tab = 'jobs';
    setHash();
    await refreshActiveData();
  } catch (err) {
    showToast(err.message, 'error');
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
  if (action === 'refresh') void refreshActiveData();
  if (action === 'go-documents') setTab('documents');
  if (action === 'go-ingest') setTab('ingest');
  if (action === 'upload-staged') void uploadStaged();
  if (action === 'clear-staged') clearStaged();
  if (action === 'ingest-all') void ingestAll();
  if (action === 'ingest-single') void ingestSingle(target.dataset.folder || state.folder, target.dataset.filename || '');
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

init().catch((err) => {
  console.error(err);
  showToast(err.message, 'error');
});
