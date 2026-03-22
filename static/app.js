// =========================================================================
// Esteem Folder Knowledge - Client Application
// =========================================================================

// --- API helpers ---
async function api(path, opts = {}) {
  const res = await fetch('/api' + path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

async function apiUpload(path, files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const res = await fetch('/api' + path, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// --- State ---
let currentProject = null;
let currentDocId = null;
let currentPage = 1;
let currentChatId = null;
let currentFolderSection = 'documents';
let selectedDocumentIds = new Set();
let pollTimer = null;
let thinkingTimer = null;
let workspaceNotice = null;

const SIDEBAR_WIDTH_KEY = 'esteem.folder-knowledge.sidebar-width';
const SUPPORTED_EXT = new Set(['.pdf', '.docx', '.xlsx', '.xls', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.zip', '.tar', '.gz', '.tgz']);
const THINKING_STEPS = [
  'Searching folder documents',
  'Reviewing earlier messages in this chat',
  'Drafting a document-grounded answer',
];

// --- Init ---
window.addEventListener('hashchange', router);
window.addEventListener('load', () => {
  configureMarkdown();
  initSidebarResize();
  loadFolders();
  router();
});

function configureMarkdown() {
  if (window.marked?.setOptions) {
    window.marked.setOptions({
      gfm: true,
      breaks: true,
    });
  }
}

function initSidebarResize() {
  const resizer = document.getElementById('sidebar-resizer');
  const root = document.documentElement;
  const stored = parseInt(localStorage.getItem(SIDEBAR_WIDTH_KEY), 10);
  if (stored && !Number.isNaN(stored)) {
    root.style.setProperty('--sidebar-width', `${clamp(stored, 220, 460)}px`);
  }
  if (!resizer) return;

  let dragging = false;
  const onMove = (event) => {
    if (!dragging) return;
    const width = clamp(event.clientX, 220, 460);
    root.style.setProperty('--sidebar-width', `${width}px`);
  };
  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('dragging');
    const width = parseInt(getComputedStyle(root).getPropertyValue('--sidebar-width'), 10);
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(width));
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
  };

  resizer.addEventListener('mousedown', (event) => {
    dragging = true;
    resizer.classList.add('dragging');
    event.preventDefault();
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });
}

// --- Router ---
function router() {
  const hash = location.hash.slice(1) || '/';
  stopPolling();

  if (hash.startsWith('/doc/')) {
    const parts = hash.split('/');
    const docId = parseInt(parts[2], 10);
    const pageNum = parts[3] === 'page' ? parseInt(parts[4], 10) : 1;
    showViewer(docId, pageNum);
    return;
  }

  if (hash.startsWith('/folder/')) {
    const { folder, section } = parseFolderHash(hash);
    const name = decodeURIComponent(folder);
    currentProject = name;
    currentFolderSection = section;
    showProject(name, section);
    return;
  }

  if (hash.startsWith('/project/')) {
    const legacy = hash.startsWith('/project/') ? `/folder/${hash.slice(9)}` : hash;
    const { folder, section } = parseFolderHash(legacy);
    const name = decodeURIComponent(folder);
    currentProject = name;
    currentFolderSection = section;
    showProject(name, section);
    return;
  }

  if (hash.startsWith('/quality/')) {
    const name = decodeURIComponent(hash.slice(9));
    showQuality(name);
    return;
  }

  showWelcome();
}

function navigate(hash) {
  location.hash = hash;
}

function parseFolderHash(hash) {
  const raw = hash.slice(8);
  const [folderPart, queryString = ''] = raw.split('?');
  const params = new URLSearchParams(queryString);
  const section = params.get('section') || 'documents';
  return {
    folder: folderPart,
    section: ['documents', 'ingestion', 'chat'].includes(section) ? section : 'documents',
  };
}

function folderHash(project, section = 'documents') {
  const encoded = enc(project);
  return section === 'documents'
    ? `#/folder/${encoded}`
    : `#/folder/${encoded}?section=${encodeURIComponent(section)}`;
}

function rootFolder(folder) {
  return String(folder || '').split('/')[0] || '';
}

function renderFolderBreadcrumb(project) {
  const parts = String(project || '').split('/').filter(Boolean);
  const section = currentFolderSection || 'documents';
  let path = '';
  const items = [`<a href="#/">Folders</a>`];

  for (let i = 0; i < parts.length; i++) {
    path = path ? `${path}/${parts[i]}` : parts[i];
    if (i === parts.length - 1) {
      items.push(`<strong>${esc(parts[i])}</strong>`);
    } else {
      items.push(`<a href="${folderHash(path, section)}">${esc(parts[i])}</a>`);
    }
  }

  return items.join('<span>/</span>');
}

// --- Sidebar (tree view) ---
let expandedFolders = new Set();

function toggleFolderExpand(folder, event) {
  event.stopPropagation();
  if (expandedFolders.has(folder)) expandedFolders.delete(folder);
  else expandedFolders.add(folder);
  loadFolders();
}

function buildFolderTree(folders) {
  const map = new Map();
  const roots = [];
  for (const f of folders) map.set(f.folder, { ...f, children: [] });
  for (const f of folders) {
    const node = map.get(f.folder);
    if (f.parent && map.has(f.parent)) {
      map.get(f.parent).children.push(node);
    } else {
      roots.push(node);
    }
  }
  return roots;
}

function renderTreeNode(node, expanded) {
  const hasChildren = node.children.length > 0;
  const isExpanded = expanded.has(node.folder);
  const isActive = currentProject === node.folder;
  const toggle = hasChildren
    ? `<span class="tree-toggle ${isExpanded ? 'open' : ''}" onclick="toggleFolderExpand(${jsq(node.folder)}, event)"></span>`
    : `<span class="tree-toggle leaf"></span>`;
  const badge = (node.docs || node.pending)
    ? `<span class="badge">${node.docs}${node.pending ? `+${node.pending}` : ''}</span>`
    : '';
  let html = `<div class="tree-node">
    <div class="project-item ${isActive ? 'active' : ''}"
         onclick="navigate(folderHash(${jsq(node.folder)}, currentProject === ${jsq(node.folder)} ? currentFolderSection : 'documents'))"
         style="padding-left:${6 + (node.depth || 0) * 18}px">
      ${toggle}
      <span class="name" title="${esc(node.folder)}">${esc(node.display_name)}</span>
      ${badge}
    </div>`;
  if (hasChildren && isExpanded) {
    html += '<div class="tree-children">';
    for (const child of node.children) html += renderTreeNode(child, expanded);
    html += '</div>';
  }
  return html + '</div>';
}

async function loadFolders() {
  try {
    const projects = await api('/folders');
    const list = document.getElementById('project-list');
    if (!list) return;
    if (!projects.length) {
      list.innerHTML = '<div class="empty" style="padding:16px;font-size:12px">No folders yet</div>';
      return;
    }
    if (currentProject) {
      const parts = currentProject.split('/');
      let parent = '';
      for (let i = 0; i < parts.length - 1; i++) {
        parent = parent ? `${parent}/${parts[i]}` : parts[i];
        expandedFolders.add(parent);
      }
    }
    const tree = buildFolderTree(projects);
    list.innerHTML = tree.map(node => renderTreeNode(node, expandedFolders)).join('');
  } catch (error) {
    console.error('Failed to load folders:', error);
  }
}

// --- Views ---
function showWelcome() {
  currentProject = null;
  currentDocId = null;
  currentChatId = null;
  workspaceNotice = null;
  setTopbar(null);
  document.getElementById('content').innerHTML = `
    <div class="welcome">
      <h2>Esteem Folder Knowledge</h2>
      <p>Select a folder to browse documents, ingest new files, and ask grounded questions only against that folder and its sub-folders.</p>
    </div>`;
  loadFolders();
}

async function showProject(project, section = 'documents') {
  if (currentProject && currentProject !== project) {
    selectedDocumentIds.clear();
    workspaceNotice = null;
  }
  currentProject = project;
  currentFolderSection = section;
  currentDocId = null;
  loadFolders();
  setTopbar(
    renderFolderBreadcrumb(project),
    `<button class="btn btn-ghost" onclick="promptNewProject(${jsq(`${project}/`)})">New Sub-folder</button>
     <button class="btn btn-ghost" onclick="promptRenameProject(${jsq(project)})">Rename</button>
     <button class="btn btn-ghost" onclick="navigate('#/quality/${enc(project)}')">Quality</button>
     <button class="btn btn-danger btn-sm" onclick="confirmDeleteProject(${jsq(project)})">Delete Folder</button>`
  );

  const content = document.getElementById('content');
  content.innerHTML = '<div class="spinner"></div>';

  try {
    const moveTargetPromise = section === 'documents'
      ? api(`/folders?scope=${enc(rootFolder(project))}&recursive=true`)
      : Promise.resolve([]);

    const [docData, chats, jobs, folders, moveTargets] = await Promise.all([
      api(`/folders/${enc(project)}/documents`),
      api(`/folders/${enc(project)}/chats`),
      api('/ingestion/jobs'),
      api('/folders'),
      moveTargetPromise,
    ]);
    const childFolders = folders.filter((folder) => folder.parent === project);

    if (currentChatId && !chats.some((chat) => chat.id === currentChatId)) {
      currentChatId = null;
    }
    if (!currentChatId && chats.length) {
      currentChatId = chats[0].id;
    }

    let chatPayload = null;
    if (section === 'chat' && currentChatId) {
      chatPayload = await api(`/chats/${currentChatId}/messages`);
    }

    content.innerHTML = `
      <div class="folder-section-nav">
        ${renderFolderSectionTabs(project, section)}
      </div>
      ${renderWorkspaceNotice()}
      ${renderFolderSection(project, section, {
        documents: docData.documents,
        pending: docData.pending,
        chats,
        chatMessages: chatPayload?.messages || [],
        childFolders,
        moveTargets,
        jobs,
      })}
    `;

    renderMarkdownBlocks(content);
    scrollChatToBottom();
    if (section === 'documents') {
      syncDocumentSelectionState();
      applyDocumentFilters();
    }

    const activeJobs = jobs.filter((job) =>
      inFolderScope(job.project, project) && job.status !== 'completed' && job.status !== 'failed'
    );
    if (activeJobs.length) {
      startPolling(project);
    } else {
      stopPolling();
      if (section === 'ingestion') {
        await pollJobs(project);
      }
    }
  } catch (error) {
    content.innerHTML = `<div class="empty">Error: ${esc(error.message)}</div>`;
  }
}

function getWorkspaceOverviewState(project, { documentsCount, pendingCount, activeJobs }) {
  let headline = 'Start by uploading files';
  let body = `Search and chat cover documents in ${project} and all of its sub-folders.`;
  let tone = 'ready';

  if (activeJobs) {
    headline = `${activeJobs} ingestion job${pluralize(activeJobs)} running`;
    body = 'You can keep browsing while new documents are extracted and indexed in the background.';
    tone = 'active';
  } else if (pendingCount) {
    headline = `${pendingCount} file${pluralize(pendingCount)} ready to ingest`;
    body = 'Pending uploads are stored safely, but they will not appear in search or chat until ingestion completes.';
    tone = 'pending';
  } else if (documentsCount) {
    headline = 'Folder is ready for search and chat';
    body = 'Indexed documents in this workspace are already available for folder-scoped chat and page-level review.';
  }

  return { headline, body, tone };
}

function syncWorkspaceOverview(project, jobs) {
  const overview = document.querySelector('[data-workspace-overview="true"]');
  if (!overview || overview.dataset.project !== project) return;

  const documentsCount = Number(overview.dataset.documentsCount || 0);
  const pendingCount = Number(overview.dataset.pendingCount || 0);
  const activeJobs = jobs.filter((job) => job.status === 'queued' || job.status === 'running').length;
  const failedJobs = jobs.filter((job) => job.status === 'failed').length;
  const state = getWorkspaceOverviewState(project, { documentsCount, pendingCount, activeJobs });

  overview.classList.remove('ready', 'pending', 'active');
  overview.classList.add(state.tone);
  overview.querySelector('[data-workspace-headline]')?.replaceChildren(document.createTextNode(state.headline));
  overview.querySelector('[data-workspace-body]')?.replaceChildren(document.createTextNode(state.body));
  overview.querySelector('[data-workspace-live-jobs]')?.replaceChildren(document.createTextNode(String(activeJobs)));
  overview.querySelector('[data-workspace-failed-jobs]')?.replaceChildren(document.createTextNode(String(failedJobs)));
  overview.querySelector('[data-workspace-failures]')?.classList.toggle('warning', failedJobs > 0);
}

function renderUploadCard(project, pendingCount = 0) {
  const pendingLabel = pendingCount
    ? `${pendingCount} file${pluralize(pendingCount)} waiting to be indexed`
    : 'Uploads land in pending first, then become searchable after ingestion';
  return `
    <div class="card upload-card-shell">
      <div class="card-title">
        <span>Upload &amp; stage files</span>
        <span class="count">${pendingLabel}</span>
      </div>
      <div class="upload-card-grid">
        <div class="upload-area" id="upload-area" role="button" tabindex="0" aria-busy="false"
           aria-describedby="upload-subtitle"
           ondragover="event.preventDefault(); this.classList.add('dragover')"
           ondragleave="this.classList.remove('dragover')"
           ondrop="handleDrop(event, ${jsq(project)})"
           onclick="openFilePicker()"
           onkeydown="handleUploadAreaKeydown(event)">
          <div class="upload-area-icon" aria-hidden="true">+</div>
          <div class="upload-area-title" id="upload-title">Drop files anywhere in this folder workspace</div>
          <div class="upload-area-subtitle" id="upload-subtitle">Supported: PDF, DOCX, spreadsheets, images, ZIP/TAR archives. Files are staged first so you can review them before indexing.</div>
          <input type="file" id="file-input" accept=".pdf,.docx,.xlsx,.xls,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.gif,.zip,.tar,.gz,.tgz" multiple
                 onchange="handleFiles(this.files, ${jsq(project)}); this.value='';">
          <input type="file" id="folder-input" webkitdirectory style="display:none"
                 onchange="handleFolderUpload(this.files, ${jsq(project)}); this.value='';">
        </div>
        <div class="upload-actions-panel">
          <div class="upload-actions-row">
            <button class="btn btn-primary" id="upload-files-btn" onclick="event.stopPropagation(); openFilePicker()">Choose files</button>
            <button class="btn btn-ghost" id="upload-folder-btn" onclick="event.stopPropagation(); openFolderPicker()">Choose folder</button>
          </div>
          <div class="upload-flow-note">
            <strong>Simple flow</strong>
            <p>1. Upload into pending. 2. Review what is staged. 3. Run ingestion to extract text, split large PDFs, and index everything for search and chat.</p>
          </div>
          <div class="upload-supported-types">
            <span class="tag">PDF</span>
            <span class="tag">DOCX</span>
            <span class="tag">XLSX</span>
            <span class="tag">Images</span>
            <span class="tag">ZIP/TAR</span>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderFolderSectionTabs(project, activeSection) {
  const tabs = [
    ['documents', 'Documents'],
    ['ingestion', 'Ingestion'],
    ['chat', 'Chat'],
  ];
  return tabs.map(([section, label]) => `
    <button
      class="section-tab ${activeSection === section ? 'active' : ''}"
      onclick="navigate('${folderHash(project, section)}')">
      ${label}
    </button>
  `).join('');
}

function renderFolderSection(project, section, data) {
  if (section === 'ingestion') {
    const scopedJobs = data.jobs.filter((job) => inFolderScope(job.project, project));
    return `
      <div class="workspace-stack">
        ${renderWorkspaceOverview(project, section, data)}
        ${renderUploadCard(project, data.pending.length)}
        ${renderPendingUploads(project, data.pending)}
        <div id="jobs-area" aria-live="polite">${renderJobActivity(scopedJobs)}</div>
      </div>
    `;
  }

  if (section === 'chat') {
    return `
      <div class="chat-only-view">
        ${renderChatShell(project, data.chats, data.chatMessages)}
      </div>
    `;
  }

  return `
    <div class="workspace-stack">
      ${renderWorkspaceOverview(project, section, data)}
      ${renderUploadCard(project, data.pending.length)}
      ${renderChildFoldersCard(project, data.childFolders)}
      ${renderDocumentsCard(project, data.documents, data.moveTargets, data.pending.length)}
    </div>
  `;
}

function renderWorkspaceOverview(project, section, data) {
  const documentsCount = data.documents.length;
  const pendingCount = data.pending.length;
  const childCount = data.childFolders.length;
  const chatCount = data.chats.length;
  const scopedJobs = data.jobs.filter((job) => inFolderScope(job.project, project));
  const activeJobs = scopedJobs.filter((job) => job.status === 'queued' || job.status === 'running').length;
  const failedJobs = scopedJobs.filter((job) => job.status === 'failed').length;
  const state = getWorkspaceOverviewState(project, { documentsCount, pendingCount, activeJobs });

  return `
    <div class="workspace-overview ${state.tone}"
         data-workspace-overview="true"
         data-project="${esc(project)}"
         data-documents-count="${documentsCount}"
         data-pending-count="${pendingCount}">
      <div class="workspace-overview-main">
        <div class="workspace-overview-label">Folder workspace</div>
        <h2 data-workspace-headline>${esc(state.headline)}</h2>
        <p data-workspace-body>${esc(state.body)}</p>
        <div class="workspace-actions">
          ${section !== 'ingestion' ? `<button class="btn btn-primary" onclick="navigate(${jsq(folderHash(project, 'ingestion'))})">Open ingestion</button>` : `<button class="btn btn-primary" onclick="openFilePicker()">Upload more</button>`}
          ${section !== 'documents' ? `<button class="btn btn-ghost" onclick="navigate(${jsq(folderHash(project, 'documents'))})">View documents</button>` : ''}
          ${section !== 'chat' ? `<button class="btn btn-ghost" onclick="navigate(${jsq(folderHash(project, 'chat'))})">Open chat</button>` : ''}
        </div>
      </div>
      <div class="workspace-stat-grid">
        <div class="workspace-stat">
          <span class="workspace-stat-label">Indexed</span>
          <strong>${documentsCount}</strong>
        </div>
        <div class="workspace-stat">
          <span class="workspace-stat-label">Pending</span>
          <strong>${pendingCount}</strong>
        </div>
        <div class="workspace-stat">
          <span class="workspace-stat-label">Live jobs</span>
          <strong data-workspace-live-jobs>${activeJobs}</strong>
        </div>
        <div class="workspace-stat">
          <span class="workspace-stat-label">Sub-folders</span>
          <strong>${childCount}</strong>
        </div>
        <div class="workspace-stat">
          <span class="workspace-stat-label">Chats</span>
          <strong>${chatCount}</strong>
        </div>
        <div class="workspace-stat ${failedJobs ? 'warning' : ''}" data-workspace-failures>
          <span class="workspace-stat-label">Recent failures</span>
          <strong data-workspace-failed-jobs>${failedJobs}</strong>
        </div>
      </div>
    </div>
  `;
}

function renderChildFoldersCard(project, folders) {
  if (!folders.length) return '';
  return `
    <div class="card">
      <div class="card-title">Sub-folders <span class="count">${folders.length}</span></div>
      <table class="table">
        <tr><th>Name</th><th>Documents</th><th>Pending</th></tr>
        ${folders.map((folder) => `
          <tr class="clickable" onclick="navigate(${jsq(folderHash(folder.folder))})">
            <td><strong>${esc(folder.display_name || folder.folder)}</strong></td>
            <td>${folder.docs}</td>
            <td>${folder.pending || 0}</td>
          </tr>
        `).join('')}
      </table>
    </div>
  `;
}

function renderPendingUploads(project, pending) {
  const totalSize = pending.reduce((sum, file) => sum + (Number(file.size_mb) || 0), 0);
  return `
    <div class="card">
      <div class="card-title">
        <span>Pending Uploads <span class="count">${pending.length}</span></span>
        <div class="title-actions">
          ${pending.length ? `<span class="count">${formatMegabytes(totalSize)} total</span>` : ''}
          <button class="btn btn-primary btn-sm" ${pending.length ? '' : 'disabled'} onclick="ingestAll(${jsq(project)}, ${pending.length})">Ingest All</button>
        </div>
      </div>
      <div class="card-intro">Uploads are staged here first so you can confirm what will be indexed. Once ingested, they become searchable and available to chat.</div>
      ${pending.length ? `
        <table class="table">
          <tr><th>File</th><th>Folder</th><th>Size</th><th></th></tr>
          ${pending.map((file) => `
            <tr>
              <td>
                <strong>${esc(file.filename)}</strong>
                <div class="table-subtext">Ready to extract and index</div>
              </td>
              <td class="muted mono">${esc(relativeFolderLabel(project, file.folder))}</td>
              <td class="muted">${formatMegabytes(file.size_mb)}</td>
              <td>
                <button class="btn btn-ghost btn-sm"
                        onclick="ingestSingle(${jsq(project)}, ${jsq(file.relative_path)}, ${jsq(file.filename)})">Ingest now</button>
              </td>
            </tr>
          `).join('')}
        </table>
      ` : `
        <div class="empty empty-compact">
          No pending uploads right now. Add files above and they will appear here before ingestion starts.
        </div>
      `}
    </div>
  `;
}

function renderDocumentsCard(project, documents, moveTargets = [], pendingCount = 0) {
  const typeOptions = [...new Set(documents
    .map((doc) => doc.document_type)
    .filter(Boolean))]
    .sort((a, b) => a.localeCompare(b));
  const folderOptions = [...new Set(documents.map((doc) => doc.folder))]
    .sort((a, b) => a.localeCompare(b));
  const targetOptions = moveTargets
    .map((folder) => folder.folder)
    .filter(Boolean)
    .sort((a, b) => a.localeCompare(b));

  return `
    <div class="card">
      <div class="card-title">Documents <span class="count">${documents.length}</span></div>
      ${documents.length ? `
        <div class="bulk-actions">
          <div class="bulk-actions-left">
            <strong id="document-selection-count">0 selected</strong>
            <button class="btn btn-ghost btn-sm" onclick="clearDocumentSelection()">Clear selection</button>
          </div>
          <div class="bulk-actions-right">
            <select id="bulk-move-target">
              <option value="">Move to folder...</option>
              ${targetOptions.map((folder) => `<option value="${esc(folder)}">${esc(folder)}</option>`).join('')}
            </select>
            <button class="btn btn-primary btn-sm" onclick="bulkMoveDocuments(${jsq(project)})">Move selected</button>
            <button class="btn btn-danger btn-sm" onclick="bulkDeleteDocuments(${jsq(project)})">Delete selected</button>
          </div>
        </div>
        <div class="table-filters">
          <div class="table-filter-controls">
            <input
              type="text"
              id="doc-filter-query"
              placeholder="Filter by doc #, title, filename, or folder..."
              oninput="applyDocumentFilters()">
            <select id="doc-filter-type" onchange="applyDocumentFilters()">
              <option value="">All types</option>
              ${typeOptions.map((type) => `<option value="${esc(type)}">${esc(type)}</option>`).join('')}
            </select>
            <select id="doc-filter-folder" onchange="applyDocumentFilters()">
              <option value="">All folders</option>
              ${folderOptions.map((folder) => `<option value="${esc(folder)}">${esc(folder)}</option>`).join('')}
            </select>
            <button class="btn btn-ghost btn-sm" onclick="resetDocumentFilters()">Clear</button>
          </div>
          <div class="table-filter-summary" id="document-filter-summary">
            Showing ${documents.length} of ${documents.length} documents
          </div>
        </div>
        <div class="table-wrap">
          <table class="table">
            <tr><th><input type="checkbox" id="document-select-all" onclick="toggleVisibleDocuments(this.checked)"></th><th>Doc #</th><th>Title</th><th>Folder</th><th>Pages</th><th>Sections</th><th>Type</th><th></th></tr>
            ${documents.map((doc) => {
              const splitTag = doc.split_info
                ? `<span class="tag" title="Split from ${esc(doc.split_info.parent)} pages ${doc.split_info.page_start}-${doc.split_info.page_end}">
                     Part ${doc.split_info.part} &middot; p${doc.split_info.page_start}-${doc.split_info.page_end}
                   </span>`
                : '';
              const searchText = [
                doc.id,
                doc.title,
                doc.filename,
                doc.folder,
                doc.document_type || '',
              ].join(' ').toLowerCase();
              return `
                <tr
                  class="clickable document-row"
                  data-doc-id="${doc.id}"
                  data-doc-search="${esc(searchText)}"
                  data-doc-type="${esc(doc.document_type || '')}"
                  data-doc-folder="${esc(doc.folder)}"
                  onclick="navigate('#/doc/${doc.id}')">
                  <td onclick="event.stopPropagation()">
                    <input
                      type="checkbox"
                      class="document-select"
                      data-doc-id="${doc.id}"
                      onchange="toggleDocumentSelection(${doc.id}, this.checked)">
                  </td>
                  <td class="mono doc-id-cell"><strong>${doc.id}</strong></td>
                  <td>
                    <strong>${esc(doc.title)}</strong>
                    ${splitTag}
                    <br>
                    <span class="muted mono">${esc(doc.filename)}</span>
                  </td>
                  <td class="muted mono">${esc(doc.folder)}</td>
                  <td>${doc.total_pages}</td>
                  <td>${doc.sections}</td>
                  <td>${doc.document_type ? `<span class="tag">${esc(doc.document_type)}</span>` : '<span class="muted">-</span>'}</td>
                  <td style="white-space:nowrap">
                    <a class="btn btn-ghost btn-sm" href="/api/documents/${doc.id}/pdf" onclick="event.stopPropagation()" title="Download PDF">PDF</a>
                    <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation(); promptRenameDoc(${doc.id}, ${jsq(doc.title)})" title="Rename">Rename</button>
                    <button class="btn btn-ghost btn-sm" style="color:#dc3545"
                            onclick="event.stopPropagation(); confirmDeleteDoc(${doc.id}, ${jsq(doc.title)})">Delete</button>
                  </td>
                </tr>
              `;
            }).join('')}
          </table>
        </div>
        <div class="empty" id="document-filter-empty" style="display:none;padding:20px">
          No documents match the current filters.
        </div>
      ` : `
        <div class="empty">
          <strong>No documents indexed yet.</strong>
          <p class="empty-detail">${pendingCount
            ? `${pendingCount} file${pluralize(pendingCount)} ${pendingCount === 1 ? 'is' : 'are'} staged in pending uploads. Run ingestion to make ${pendingCount === 1 ? 'it' : 'them'} searchable.`
            : 'Upload files above, then run ingestion to build this folder knowledge base.'}</p>
        </div>
      `}
    </div>
  `;
}

function renderWorkspaceNotice() {
  if (!workspaceNotice) return '';
  const role = workspaceNotice.type === 'error' ? 'alert' : 'status';
  return `
    <div class="workspace-notice ${workspaceNotice.type}" id="workspace-notice" role="${role}">
      <div class="workspace-notice-body">
        <strong>${esc(workspaceNotice.message)}</strong>
        ${workspaceNotice.detail ? `<div class="workspace-notice-detail">${esc(workspaceNotice.detail)}</div>` : ''}
      </div>
      <button class="btn btn-ghost btn-sm" onclick="dismissWorkspaceNotice()">Dismiss</button>
    </div>
  `;
}

function showWorkspaceNotice(type, message, detail = '') {
  workspaceNotice = { type, message, detail };
}

function dismissWorkspaceNotice() {
  workspaceNotice = null;
  document.getElementById('workspace-notice')?.remove();
}

function applyDocumentFilters() {
  const query = (document.getElementById('doc-filter-query')?.value || '').trim().toLowerCase();
  const type = document.getElementById('doc-filter-type')?.value || '';
  const folder = document.getElementById('doc-filter-folder')?.value || '';
  const rows = [...document.querySelectorAll('.document-row')];
  if (!rows.length) return;

  let visibleCount = 0;
  let visibleSelectedCount = 0;
  rows.forEach((row) => {
    const matchesQuery = !query || (row.dataset.docSearch || '').includes(query);
    const matchesType = !type || (row.dataset.docType || '') === type;
    const matchesFolder = !folder || (row.dataset.docFolder || '') === folder;
    const visible = matchesQuery && matchesType && matchesFolder;
    row.style.display = visible ? '' : 'none';
    if (visible) {
      visibleCount += 1;
      if (selectedDocumentIds.has(Number(row.dataset.docId))) {
        visibleSelectedCount += 1;
      }
    }
  });

  const summary = document.getElementById('document-filter-summary');
  if (summary) {
    summary.textContent = `Showing ${visibleCount} of ${rows.length} documents`;
  }

  const empty = document.getElementById('document-filter-empty');
  if (empty) {
    empty.style.display = visibleCount ? 'none' : '';
  }

  const selectAll = document.getElementById('document-select-all');
  if (selectAll) {
    selectAll.checked = visibleCount > 0 && visibleCount === visibleSelectedCount;
    selectAll.indeterminate = visibleSelectedCount > 0 && visibleSelectedCount < visibleCount;
  }
}

function resetDocumentFilters() {
  const query = document.getElementById('doc-filter-query');
  const type = document.getElementById('doc-filter-type');
  const folder = document.getElementById('doc-filter-folder');
  if (query) query.value = '';
  if (type) type.value = '';
  if (folder) folder.value = '';
  applyDocumentFilters();
}

function syncDocumentSelectionState() {
  document.querySelectorAll('.document-select').forEach((checkbox) => {
    checkbox.checked = selectedDocumentIds.has(Number(checkbox.dataset.docId));
  });
  document.querySelectorAll('.document-row').forEach((row) => {
    row.classList.toggle('selected', selectedDocumentIds.has(Number(row.dataset.docId)));
  });
  const count = document.getElementById('document-selection-count');
  if (count) {
    count.textContent = `${selectedDocumentIds.size} selected`;
  }
}

function toggleDocumentSelection(docId, checked) {
  if (checked) {
    selectedDocumentIds.add(docId);
  } else {
    selectedDocumentIds.delete(docId);
  }
  syncDocumentSelectionState();
  applyDocumentFilters();
}

function toggleVisibleDocuments(checked) {
  document.querySelectorAll('.document-row').forEach((row) => {
    if (row.style.display === 'none') return;
    const docId = Number(row.dataset.docId);
    if (checked) {
      selectedDocumentIds.add(docId);
    } else {
      selectedDocumentIds.delete(docId);
    }
  });
  syncDocumentSelectionState();
  applyDocumentFilters();
}

function clearDocumentSelection() {
  selectedDocumentIds.clear();
  syncDocumentSelectionState();
  applyDocumentFilters();
}

async function bulkMoveDocuments(project) {
  if (!selectedDocumentIds.size) {
    alert('Select one or more documents first.');
    return;
  }
  const targetFolder = document.getElementById('bulk-move-target')?.value;
  if (!targetFolder) {
    alert('Choose a target folder first.');
    return;
  }
  try {
    await api('/documents/bulk-move', {
      method: 'POST',
      body: JSON.stringify({
        doc_ids: [...selectedDocumentIds],
        target_folder: targetFolder,
      }),
    });
    selectedDocumentIds.clear();
    await showProject(project, 'documents');
  } catch (error) {
    alert('Move failed: ' + error.message);
  }
}

async function bulkDeleteDocuments(project) {
  if (!selectedDocumentIds.size) {
    alert('Select one or more documents first.');
    return;
  }
  if (!confirm(`Delete ${selectedDocumentIds.size} selected document(s)?`)) {
    return;
  }
  try {
    await api('/documents/bulk-delete', {
      method: 'POST',
      body: JSON.stringify({ doc_ids: [...selectedDocumentIds] }),
    });
    selectedDocumentIds.clear();
    await showProject(project, 'documents');
  } catch (error) {
    alert('Delete failed: ' + error.message);
  }
}

function renderChatShell(project, chats, messages) {
  return `
    <div class="chat-shell">
      <aside class="chat-history">
        <div class="chat-history-header">
          <div>
            <strong>Chats</strong>
            <div class="chat-subtitle">${chats.length} thread${chats.length === 1 ? '' : 's'}</div>
          </div>
          <button class="btn btn-sm btn-primary" onclick="newChat(${jsq(project)})">New</button>
        </div>
        <div class="chat-history-list">
          ${chats.length ? chats.map((chat) => `
            <button class="chat-thread ${chat.id === currentChatId ? 'active' : ''}" onclick="selectChat('${chat.id}')">
              <span class="chat-thread-title">${esc(chat.title || 'New chat')}</span>
              <span class="chat-thread-preview">${esc((chat.last_message || 'No messages yet').slice(0, 120))}</span>
              <span class="chat-thread-meta">${chat.message_count || 0} message${chat.message_count === 1 ? '' : 's'}</span>
            </button>
          `).join('') : '<div class="empty" style="padding:24px 16px;font-size:12px">No chats yet</div>'}
        </div>
      </aside>
      <section class="chat-main">
        <div class="chat-header">
          <div>
            <h3>${currentChatId ? esc((chats.find((chat) => chat.id === currentChatId)?.title) || 'New chat') : 'Folder chat'}</h3>
            <div class="chat-subtitle">Answers must be grounded only in documents from <strong>${esc(project)}</strong> and its sub-folders.</div>
          </div>
          ${currentChatId ? `<button class="btn btn-sm btn-ghost" onclick="newChat(${jsq(project)})">Start fresh</button>` : ''}
        </div>
        <div class="chat-messages" id="chat-messages">
          ${messages.length ? renderChatMessages(messages) : `
            <div class="chat-empty" id="chat-empty">
              <h3>Ask about this folder</h3>
              <p>The assistant will search only the documents inside <strong>${esc(project)}</strong> and its sub-folders, keep the thread history in context, and say so when the documents do not support an answer.</p>
            </div>
          `}
        </div>
        <div class="chat-composer">
          <textarea id="chat-input"
                    placeholder="Ask a question about this folder's documents..."
                    onkeydown="handleChatKeydown(event)"></textarea>
          <div class="chat-composer-footer">
            <div class="chat-composer-hint">Markdown is supported in replies, including math such as <code>\\(E=mc^2\\)</code> or <code>$$x^2$$</code>.</div>
            <button class="btn btn-primary" id="send-chat-btn" onclick="sendChatMessage()">Send</button>
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderChatMessages(messages) {
  return messages.map((message) => `
    <div class="chat-message ${message.role}">
      <div class="chat-bubble">
        <div class="chat-message-header">
          <span class="chat-role">${message.role === 'assistant' ? 'Assistant' : 'You'}</span>
          <span>${formatDate(message.created_at)}</span>
        </div>
        ${message.role === 'assistant'
          ? `<div class="render-markdown">${renderMarkdown(message.content)}</div>`
          : `<div>${esc(message.content).replace(/\n/g, '<br>')}</div>`}
        ${message.role === 'assistant' && message.sources?.length ? renderSourceList(message.sources) : ''}
      </div>
    </div>
  `).join('');
}

function renderSourceList(sources) {
  return `
    <div class="chat-source-list">
      ${sources.map((source) => `
        <div class="chat-source">
          <strong>[${source.id}]</strong>
          <a href="#/doc/${source.doc_id}/page/${source.page_num}">${esc(source.doc_title)}</a>
          <span>${source.folder ? ` &middot; ${esc(source.folder)}` : ''} &middot; p${source.page_num}${source.breadcrumb ? ` &middot; ${esc(source.breadcrumb)}` : ''}</span>
        </div>
      `).join('')}
    </div>
  `;
}

async function showViewer(docId, pageNum) {
  currentDocId = docId;
  currentPage = pageNum || 1;

  try {
    const [doc, toc] = await Promise.all([
      api(`/documents/${docId}`),
      api(`/documents/${docId}/toc`),
    ]);

    const splitNote = doc.split_info
      ? ` <span class="tag">Part ${doc.split_info.part} of ${esc(doc.split_info.parent)} &middot; p${doc.split_info.page_start}-${doc.split_info.page_end}</span>`
      : '';
    setTopbar(
      `<a href="#/" style="color:#666">Folders</a><span>/</span>` +
      `<a href="#/folder/${enc(doc.folder)}">${esc(doc.folder)}</a><span>/</span>` +
      `<strong>${esc(doc.title)}</strong>${splitNote}`,
      `<a class="btn btn-ghost btn-sm" href="/api/documents/${docId}/pdf">Download</a>
       <button class="btn btn-ghost btn-sm" onclick="navigate('#/folder/${enc(doc.folder)}')">Back</button>`
    );

    const content = document.getElementById('content');
    content.style.padding = '0';
    content.innerHTML = `
      <div class="viewer">
        <div class="viewer-toc">
          <div class="viewer-search">
            <input type="text" placeholder="Search in document..."
                   id="doc-search" onkeydown="if(event.key==='Enter') searchInDoc(${docId}, this.value)">
          </div>
          <div id="toc-list">${renderToc(toc, docId)}</div>
          <div id="search-results"></div>
        </div>
        <div class="viewer-content">
          <div class="page-header" id="page-header"></div>
          <div class="page-body" id="page-body"><div class="spinner" style="margin:40px auto"></div></div>
        </div>
      </div>
    `;

    loadPage(docId, currentPage, doc.total_pages);
  } catch (error) {
    document.getElementById('content').innerHTML =
      `<div class="empty">Error: ${esc(error.message)}</div>`;
  }
}

function renderToc(sections, docId) {
  if (!sections.length) {
    return '<div class="empty" style="padding:12px;font-size:12px">No sections</div>';
  }
  return sections.map((section) => `
    <div class="toc-item l${Math.min(section.level, 4)}"
         title="${esc(section.heading)}"
         onclick="loadPage(${docId}, ${section.page_start})">
      ${esc(section.heading)}<span class="toc-pages">p${section.page_start}</span>
    </div>
  `).join('');
}

async function loadPage(docId, pageNum, totalPages) {
  currentPage = pageNum;
  try {
    const page = await api(`/documents/${docId}/pages/${pageNum}`);
    totalPages = totalPages || page.total_pages;

    document.getElementById('page-header').innerHTML = `
      <div><span class="page-bc">${esc(page.breadcrumb || '')}</span></div>
      <div class="page-nav">
        <button class="btn btn-ghost btn-sm" ${pageNum <= 1 ? 'disabled' : ''}
                onclick="loadPage(${docId}, ${pageNum - 1}, ${totalPages})">Prev</button>
        <span class="page-num">Page ${pageNum} / ${totalPages}</span>
        <button class="btn btn-ghost btn-sm" ${pageNum >= totalPages ? 'disabled' : ''}
                onclick="loadPage(${docId}, ${pageNum + 1}, ${totalPages})">Next</button>
      </div>
    `;
    document.getElementById('page-body').innerHTML = `<pre>${esc(page.content)}</pre>`;
    history.replaceState(null, '', `#/doc/${docId}/page/${pageNum}`);

    document.querySelectorAll('.toc-item').forEach((el) => el.classList.remove('active'));
    document.querySelectorAll('.toc-item').forEach((el) => {
      const pg = parseInt(el.querySelector('.toc-pages')?.textContent?.slice(1), 10);
      if (pg === pageNum) el.classList.add('active');
    });
  } catch (error) {
    document.getElementById('page-body').innerHTML =
      `<div class="empty">Error: ${esc(error.message)}</div>`;
  }
}

async function searchInDoc(docId, query) {
  const tocList = document.getElementById('toc-list');
  const searchResults = document.getElementById('search-results');
  if (!query.trim()) {
    searchResults.innerHTML = '';
    tocList.style.display = '';
    return;
  }
  try {
    const results = await api(`/documents/${docId}/search?q=${encodeURIComponent(query)}`);
    tocList.style.display = 'none';
    if (!results.length) {
      searchResults.innerHTML = '<div class="empty" style="padding:12px;font-size:12px">No results</div>';
      return;
    }
    searchResults.innerHTML = results.map((result) => `
      <div class="toc-item" onclick="loadPage(${docId}, ${result.page_num}); document.getElementById('toc-list').style.display=''; document.getElementById('search-results').innerHTML='';">
        <strong>p${result.page_num}</strong> ${result.snippet.replace(/>>>/g, '<b>').replace(/<<</g, '</b>')}
      </div>
    `).join('');
  } catch (error) {
    console.error(error);
  }
}

async function showQuality(project) {
  setTopbar(
    `<a href="#/" style="color:#666">Folders</a><span>/</span>` +
    `<a href="#/folder/${enc(project)}">${esc(project)}</a><span>/</span><strong>Quality</strong>`,
    ''
  );
  const content = document.getElementById('content');
  content.style.padding = '20px';
  content.innerHTML = '<div class="spinner"></div>';

  try {
    const data = await api(`/folders/${enc(project)}/quality`);
    let html = '';

    html += `<div class="card"><div class="card-title">Quality Flags <span class="count">${data.flags.length}</span></div>`;
    if (data.flags.length) {
      for (const flag of data.flags) {
        const resolvedClass = flag.resolved ? ' resolved' : '';
        html += `<div class="flag-row${resolvedClass}">
          <span class="flag-type ${flag.flag_type}">${flag.flag_type}</span>
          <a href="#/doc/${flag.doc_id}">${esc(flag.doc_title)}</a>
          ${flag.page_num ? `<span class="muted">p${flag.page_num}</span>` : ''}
          <span style="flex:1">${esc(flag.reason || '')}</span>
          ${!flag.resolved ? `<button class="btn btn-ghost btn-sm" onclick="resolveFlag(${flag.id})">Resolve</button>` : ''}
        </div>`;
      }
    } else {
      html += '<div class="empty">No quality flags</div>';
    }
    html += '</div>';

    html += `<div class="card"><div class="card-title">Recent Corrections <span class="count">${data.corrections.length}</span></div>`;
    if (data.corrections.length) {
      html += '<table class="table"><tr><th>Document</th><th>Category</th><th>Action</th><th>When</th></tr>';
      for (const correction of data.corrections) {
        html += `<tr>
          <td><a href="#/doc/${correction.doc_id}">${esc(correction.doc_title)}</a></td>
          <td><span class="tag">${esc(correction.category)}</span></td>
          <td>${esc(correction.action)}</td>
          <td class="muted">${esc(correction.created_at || '')}</td>
        </tr>`;
      }
      html += '</table>';
    } else {
      html += '<div class="empty">No corrections logged</div>';
    }
    html += '</div>';
    content.innerHTML = html;
  } catch (error) {
    content.innerHTML = `<div class="empty">Error: ${esc(error.message)}</div>`;
  }
}

// --- Chat actions ---
async function newChat(project) {
  try {
    const created = await api(`/folders/${enc(project)}/chats`, {
      method: 'POST',
      body: JSON.stringify({ title: 'New chat' }),
    });
    currentChatId = created.id;
    showProject(project, currentFolderSection);
  } catch (error) {
    alert('Could not create chat: ' + error.message);
  }
}

function selectChat(threadId) {
  currentChatId = threadId;
  if (currentProject) showProject(currentProject, currentFolderSection);
}

function handleChatKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

async function sendChatMessage() {
  const textarea = document.getElementById('chat-input');
  const sendButton = document.getElementById('send-chat-btn');
  const text = textarea?.value.trim();
  if (!text || !currentProject) return;

  try {
    if (!currentChatId) {
      const created = await api(`/folders/${enc(currentProject)}/chats`, {
        method: 'POST',
        body: JSON.stringify({ title: 'New chat' }),
      });
      currentChatId = created.id;
      await showProject(currentProject, currentFolderSection);
    }
  } catch (error) {
    alert('Could not start chat: ' + error.message);
    return;
  }

  setComposerBusy(true);
  const messageText = text;
  textarea.value = '';
  appendLocalUserMessage(messageText);
  startThinkingIndicator();
  scrollChatToBottom();

  try {
    await api(`/chats/${currentChatId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content: messageText }),
    });
    stopThinkingIndicator();
    await showProject(currentProject, currentFolderSection);
  } catch (error) {
    stopThinkingIndicator();
    if (sendButton) sendButton.disabled = false;
    if (textarea) textarea.disabled = false;
    alert('Chat failed: ' + error.message);
    await showProject(currentProject, currentFolderSection);
  }
}

function appendLocalUserMessage(text) {
  const chatMessages = document.getElementById('chat-messages');
  const empty = document.getElementById('chat-empty');
  if (!chatMessages) return;
  if (empty) empty.remove();
  const wrapper = document.createElement('div');
  wrapper.className = 'chat-message user';
  wrapper.innerHTML = `
    <div class="chat-bubble">
      <div class="chat-message-header">
        <span class="chat-role">You</span>
        <span>${formatDate(new Date().toISOString())}</span>
      </div>
      <div>${esc(text).replace(/\n/g, '<br>')}</div>
    </div>
  `;
  chatMessages.appendChild(wrapper);
}

function startThinkingIndicator() {
  stopThinkingIndicator();
  const chatMessages = document.getElementById('chat-messages');
  const empty = document.getElementById('chat-empty');
  if (!chatMessages) return;
  if (empty) empty.remove();
  const wrapper = document.createElement('div');
  wrapper.className = 'chat-message assistant';
  wrapper.id = 'thinking-message';
  wrapper.innerHTML = `
    <div class="chat-bubble">
      <div class="chat-message-header">
        <span class="chat-role">Assistant</span>
        <span>Working</span>
      </div>
      <div class="thinking">
        <span class="thinking-dots"><span></span><span></span><span></span></span>
        <span id="thinking-status">${THINKING_STEPS[0]}</span>
      </div>
    </div>
  `;
  chatMessages.appendChild(wrapper);
  let index = 0;
  thinkingTimer = window.setInterval(() => {
    index = (index + 1) % THINKING_STEPS.length;
    const label = document.getElementById('thinking-status');
    if (label) label.textContent = THINKING_STEPS[index];
  }, 1300);
}

function stopThinkingIndicator() {
  if (thinkingTimer) {
    clearInterval(thinkingTimer);
    thinkingTimer = null;
  }
  document.getElementById('thinking-message')?.remove();
  setComposerBusy(false);
}

function setComposerBusy(busy) {
  const textarea = document.getElementById('chat-input');
  const button = document.getElementById('send-chat-btn');
  if (textarea) textarea.disabled = busy;
  if (button) button.disabled = busy;
}

function scrollChatToBottom() {
  const chatMessages = document.getElementById('chat-messages');
  if (chatMessages) {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

// --- Folder actions ---
function promptNewProject(defaultValue = '') {
  document.getElementById('modal-content').innerHTML = `
    <h3>New Folder</h3>
    <input type="text" id="new-project-name" placeholder="Folder path" value="${esc(defaultValue)}" autofocus
           onkeydown="if(event.key==='Enter') createProject()">
    <div class="actions">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="createProject()">Create</button>
    </div>
  `;
  document.getElementById('modal-overlay').style.display = 'flex';
  setTimeout(() => document.getElementById('new-project-name')?.focus(), 100);
}

async function createProject() {
  const name = document.getElementById('new-project-name').value.trim();
  if (!name) return;
  try {
    await api('/folders', { method: 'POST', body: JSON.stringify({ name }) });
    closeModal();
    await loadFolders();
    navigate('#/folder/' + encodeURIComponent(name));
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

function promptRenameProject(project) {
  document.getElementById('modal-content').innerHTML = `
    <h3>Rename Folder</h3>
    <input type="text" id="rename-input" value="${esc(project)}" autofocus
           onkeydown="if(event.key==='Enter') doRenameProject(${jsq(project)})">
    <div class="actions">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doRenameProject(${jsq(project)})">Rename</button>
    </div>
  `;
  document.getElementById('modal-overlay').style.display = 'flex';
  setTimeout(() => {
    const input = document.getElementById('rename-input');
    input?.focus();
    input?.select();
  }, 100);
}

async function doRenameProject(oldName) {
  const newName = document.getElementById('rename-input').value.trim();
  if (!newName || newName === oldName) {
    closeModal();
    return;
  }
  try {
    await api(`/folders/${enc(oldName)}`, {
      method: 'PATCH',
      body: JSON.stringify({ name: newName }),
    });
    closeModal();
    currentProject = newName;
    await loadFolders();
    navigate('#/folder/' + encodeURIComponent(newName));
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

function promptRenameDoc(docId, currentTitle) {
  document.getElementById('modal-content').innerHTML = `
    <h3>Rename Document</h3>
    <input type="text" id="rename-input" value="${esc(currentTitle)}" autofocus
           onkeydown="if(event.key==='Enter') doRenameDoc(${docId})">
    <div class="actions">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doRenameDoc(${docId})">Rename</button>
    </div>
  `;
  document.getElementById('modal-overlay').style.display = 'flex';
  setTimeout(() => {
    const input = document.getElementById('rename-input');
    input?.focus();
    input?.select();
  }, 100);
}

async function doRenameDoc(docId) {
  const title = document.getElementById('rename-input').value.trim();
  if (!title) {
    closeModal();
    return;
  }
  try {
    await api(`/documents/${docId}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    });
    closeModal();
    if (currentProject) showProject(currentProject, currentFolderSection);
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

function confirmDeleteProject(project) {
  if (confirm(`Delete folder "${project}" and ALL its documents and chats?`)) {
    api(`/folders/${enc(project)}`, { method: 'DELETE' })
      .then(() => {
        currentChatId = null;
        navigate('#/');
        loadFolders();
      })
      .catch((error) => alert('Error: ' + error.message));
  }
}

function confirmDeleteDoc(docId, title) {
  if (confirm(`Delete document "${title}"?`)) {
    api(`/documents/${docId}`, { method: 'DELETE' })
      .then(() => {
        if (currentProject) showProject(currentProject, currentFolderSection);
      })
      .catch((error) => alert('Error: ' + error.message));
  }
}

async function resolveFlag(flagId) {
  await api(`/quality/${flagId}/resolve`, { method: 'PATCH' });
  if (location.hash.startsWith('#/quality/')) router();
}

// --- Upload ---
function supported(name) {
  const idx = name.lastIndexOf('.');
  return idx >= 0 && SUPPORTED_EXT.has(name.slice(idx).toLowerCase());
}

function handleDrop(event, project) {
  event.preventDefault();
  event.currentTarget.classList.remove('dragover');
  const { accepted, skipped } = splitSupportedFiles(event.dataTransfer.files);
  if (!accepted.length) {
    showWorkspaceNotice('warning', 'No supported files found in this drop.', summarizeSkippedFiles(skipped));
    if (currentProject) showProject(currentProject, currentFolderSection);
    return;
  }
  uploadFiles(accepted, project, { skipped });
}

function handleFiles(fileList, project) {
  const { accepted, skipped } = splitSupportedFiles(fileList);
  if (!accepted.length) {
    showWorkspaceNotice('warning', 'No supported files selected.', summarizeSkippedFiles(skipped));
    if (currentProject) showProject(currentProject, currentFolderSection);
    return;
  }
  uploadFiles(accepted, project, { skipped });
}

async function uploadFiles(files, project, options = {}) {
  setUploadBusyState(
    true,
    `Uploading ${files.length} file${pluralize(files.length)}...`,
    'Files will show up in pending uploads as soon as transfer completes.'
  );
  try {
    const result = await apiUpload(`/folders/${enc(project)}/upload`, files);
    const skippedSummary = summarizeSkippedFiles(options.skipped || []);
    showWorkspaceNotice(
      'success',
      `Added ${result.count} file${pluralize(result.count)} to pending uploads.`,
      skippedSummary || 'Review the staged files below, then run ingestion when you are ready.'
    );
    await showProject(project, currentFolderSection);
  } catch (error) {
    showWorkspaceNotice('error', 'Upload failed.', error.message);
    await showProject(project, currentFolderSection);
  }
}

function handleFolderUpload(fileList, project) {
  const files = [];
  const paths = [];
  const skipped = [];
  for (const f of fileList) {
    if (!supported(f.name)) {
      skipped.push(f.name);
      continue;
    }
    files.push(f);
    // webkitRelativePath is "SelectedFolder/sub/file.pdf"
    // Strip the first component (the folder the user picked)
    const parts = f.webkitRelativePath.split('/');
    paths.push(parts.slice(1).join('/'));
  }
  if (!files.length) {
    showWorkspaceNotice('warning', 'No supported files found in this folder.', summarizeSkippedFiles(skipped));
    if (currentProject) showProject(currentProject, currentFolderSection);
    return;
  }
  uploadFilesWithPaths(files, paths, project, { skipped });
}

async function uploadFilesWithPaths(files, paths, project, options = {}) {
  setUploadBusyState(
    true,
    `Uploading ${files.length} file${pluralize(files.length)} from folder...`,
    'Original sub-folder paths will be preserved inside this workspace.'
  );
  try {
    const form = new FormData();
    for (const f of files) form.append('files', f);
    for (const p of paths) form.append('paths', p);
    const res = await fetch('/api/folders/' + enc(project) + '/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    const result = await res.json();
    const skippedSummary = summarizeSkippedFiles(options.skipped || []);
    showWorkspaceNotice(
      'success',
      `Added ${result.count} file${pluralize(result.count)} from the selected folder.`,
      skippedSummary || 'Folder structure was preserved in pending uploads.'
    );
    await showProject(project, currentFolderSection);
  } catch (error) {
    showWorkspaceNotice('error', 'Folder upload failed.', error.message);
    await showProject(project, currentFolderSection);
  }
}

// --- Ingestion ---
async function ingestAll(project, pendingCount = 0) {
  try {
    await api(`/folders/${enc(project)}/ingest`, { method: 'POST' });
    showWorkspaceNotice(
      'info',
      'Ingestion started.',
      pendingCount
        ? `Processing ${pendingCount} staged file${pluralize(pendingCount)} in the background.`
        : 'Processing pending uploads in the background.'
    );
    await showProject(project, currentFolderSection);
  } catch (error) {
    showWorkspaceNotice('error', 'Could not start ingestion.', error.message);
    await showProject(project, currentFolderSection);
  }
}

async function ingestSingle(project, filename, label = filename) {
  try {
    await api(`/folders/${enc(project)}/ingest/${encPath(filename)}`, { method: 'POST' });
    showWorkspaceNotice('info', `Started ingesting ${label}.`, 'You can stay on this page while progress updates live.');
    await showProject(project, currentFolderSection);
  } catch (error) {
    showWorkspaceNotice('error', `Could not ingest ${label}.`, error.message);
    await showProject(project, currentFolderSection);
  }
}

function startPolling(project) {
  stopPolling();
  pollJobs(project);
  pollTimer = setInterval(() => pollJobs(project), 2000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollJobs(project) {
  try {
    const jobs = await api('/ingestion/jobs');
    const projectJobs = jobs.filter((job) => inFolderScope(job.project, project));
    syncWorkspaceOverview(project, projectJobs);

    const area = document.getElementById('jobs-area');
    if (area && !projectJobs.length) {
      area.innerHTML = '';
    }
    if (area && projectJobs.length) {
      area.innerHTML = renderJobActivity(projectJobs);
    }

    if (!projectJobs.length) {
      stopPolling();
      return;
    }

    const allDone = projectJobs.every((job) => job.status === 'completed' || job.status === 'failed');
    if (allDone) {
      stopPolling();
    }
    if (allDone && projectJobs.some((job) => job.status === 'completed')) {
      setTimeout(() => showProject(project, currentFolderSection), 1000);
    }
  } catch (error) {
    console.error('Poll error:', error);
  }
}

// --- Helpers ---
function renderMarkdown(markdown) {
  if (!markdown) return '';
  if (!window.marked) {
    return esc(markdown).replace(/\n/g, '<br>');
  }
  const raw = window.marked.parse(markdown);
  if (window.DOMPurify) {
    return window.DOMPurify.sanitize(raw);
  }
  return raw;
}

function renderJobActivity(jobs) {
  if (!jobs.length) return '';
  const active = jobs.filter((job) => job.status === 'queued' || job.status === 'running').length;
  const completed = jobs.filter((job) => job.status === 'completed').length;
  const failed = jobs.filter((job) => job.status === 'failed').length;
  return `
    <div class="card">
      <div class="card-title">
        <span>Ingestion Activity <span class="count">${jobs.length} recent job${pluralize(jobs.length)}</span></span>
        <span class="count">${active ? 'Live updates every 2 seconds' : 'Recent activity'}</span>
      </div>
      <div class="job-summary">
        <div class="job-summary-pill active">${active} active</div>
        <div class="job-summary-pill completed">${completed} completed</div>
        <div class="job-summary-pill failed">${failed} failed</div>
      </div>
      <div class="job-list">
        ${jobs.map((job) => renderJobCard(job)).join('')}
      </div>
    </div>
  `;
}

function renderJobCard(job) {
  const progress = jobProgress(job);
  const statusLabel = job.status ? `${job.status[0].toUpperCase()}${job.status.slice(1)}` : 'Unknown';
  return `
    <div class="job-card ${esc(job.status || '')}">
      <div class="job-card-top">
        <div class="job-card-title">
          <span class="filename">${esc(job.filename)}</span>
          <span class="job-stage">${esc(job.stage || 'Waiting to start')}</span>
        </div>
        <span class="status ${esc(job.status || '')}">${esc(statusLabel)}</span>
      </div>
      <div class="job-progress" aria-hidden="true"><span style="width:${progress}%"></span></div>
      <div class="job-card-footer">
        <span class="count">${progress}%</span>
        ${job.doc_id ? `<a href="#/doc/${job.doc_id}">Open document</a>` : '<span class="muted">Document link appears after indexing</span>'}
        ${job.started_at ? `<span class="muted">${esc(formatDate(job.started_at))}</span>` : ''}
      </div>
      ${job.error ? `<div class="job-error">${esc(job.error)}</div>` : ''}
    </div>
  `;
}

function jobProgress(job) {
  if (job.status === 'completed') return 100;
  if (job.status === 'failed') return 100;
  const stage = String(job.stage || '').toLowerCase();
  if (job.status === 'queued') return 8;
  if (stage.includes('checking for document boundaries')) return 18;
  if (stage.includes('split into')) return 35;
  if (stage.includes('ingesting into database')) return 72;
  if (stage.includes('replaying corrections')) return 82;
  if (stage.includes('extracting metadata')) return 90;
  if (stage.includes('computing embeddings')) return 96;
  if (stage.includes('extracting')) return 55;
  if (stage.includes('done') || stage.includes('skipped')) return 100;
  return job.status === 'running' ? 42 : 0;
}

function renderMarkdownBlocks(root = document) {
  root.querySelectorAll('.render-markdown').forEach((node) => {
    if (window.renderMathInElement) {
      window.renderMathInElement(node, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
      });
    }
  });
}

function setTopbar(breadcrumb, actions) {
  const topbar = document.getElementById('topbar');
  const content = document.getElementById('content');
  if (breadcrumb) {
    topbar.style.display = 'flex';
    document.getElementById('breadcrumb').innerHTML = breadcrumb;
    document.getElementById('topbar-actions').innerHTML = actions || '';
  } else {
    topbar.style.display = 'none';
  }
  content.style.padding = '20px';
}

function closeModal() {
  document.getElementById('modal-overlay').style.display = 'none';
}

function openFilePicker() {
  document.getElementById('file-input')?.click();
}

function openFolderPicker() {
  document.getElementById('folder-input')?.click();
}

function handleUploadAreaKeydown(event) {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  event.preventDefault();
  openFilePicker();
}

function setUploadBusyState(busy, title, subtitle) {
  const area = document.getElementById('upload-area');
  const titleNode = document.getElementById('upload-title');
  const subtitleNode = document.getElementById('upload-subtitle');
  const filesBtn = document.getElementById('upload-files-btn');
  const folderBtn = document.getElementById('upload-folder-btn');
  const fileInput = document.getElementById('file-input');
  const folderInput = document.getElementById('folder-input');
  if (area) {
    area.classList.toggle('busy', busy);
    area.setAttribute('aria-busy', String(busy));
  }
  if (titleNode) titleNode.textContent = title;
  if (subtitleNode) subtitleNode.textContent = subtitle;
  [filesBtn, folderBtn, fileInput, folderInput].forEach((node) => {
    if (node) node.disabled = busy;
  });
}

function esc(value) {
  if (value == null) return '';
  const div = document.createElement('div');
  div.textContent = String(value);
  return div.innerHTML;
}

function jsq(value) {
  return `'${String(value ?? '')
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '')}'`;
}

function enc(value) {
  return encodeURIComponent(value);
}

function encPath(value) {
  return String(value ?? '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function inFolderScope(candidate, scope) {
  return candidate === scope || candidate.startsWith(scope + '/');
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function splitSupportedFiles(fileList) {
  const accepted = [];
  const skipped = [];
  for (const file of [...fileList]) {
    if (supported(file.name)) accepted.push(file);
    else skipped.push(file.name);
  }
  return { accepted, skipped };
}

function summarizeSkippedFiles(skipped) {
  if (!skipped.length) return '';
  const preview = skipped.slice(0, 3).join(', ');
  const extra = skipped.length > 3 ? ` +${skipped.length - 3} more` : '';
  return `Skipped ${skipped.length} unsupported file${pluralize(skipped.length)}: ${preview}${extra}`;
}

function pluralize(count) {
  return count === 1 ? '' : 's';
}

function formatMegabytes(value) {
  const amount = Number(value) || 0;
  return `${amount.toFixed(amount >= 10 ? 0 : 1)} MB`;
}

function relativeFolderLabel(project, folder) {
  if (!folder || folder === project) return 'This folder';
  return folder.slice(project.length + 1) || folder;
}

function formatDate(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}
