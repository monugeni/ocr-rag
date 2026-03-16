// =========================================================================
// Esteem Project Knowledge - Client Application
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
let pollTimer = null;
let thinkingTimer = null;

const SIDEBAR_WIDTH_KEY = 'esteem.project-knowledge.sidebar-width';
const SUPPORTED_EXT = new Set(['.pdf', '.docx', '.xlsx', '.xls', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.zip', '.tar', '.gz', '.tgz']);
const THINKING_STEPS = [
  'Searching project documents',
  'Reviewing earlier messages in this chat',
  'Drafting a document-grounded answer',
];

// --- Init ---
window.addEventListener('hashchange', router);
window.addEventListener('load', () => {
  configureMarkdown();
  initSidebarResize();
  loadProjects();
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

  if (hash.startsWith('/project/')) {
    const name = decodeURIComponent(hash.slice(9));
    currentProject = name;
    showProject(name);
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

// --- Sidebar ---
async function loadProjects() {
  try {
    const projects = await api('/projects');
    const list = document.getElementById('project-list');
    if (!list) return;
    if (!projects.length) {
      list.innerHTML = '<div class="empty" style="padding:16px;font-size:12px">No projects yet</div>';
      return;
    }
    list.innerHTML = projects.map((project) => `
      <div class="project-item ${currentProject === project.project ? 'active' : ''}"
           onclick="navigate('#/project/${enc(project.project)}')">
        <span class="name" title="${esc(project.project)}">${esc(project.project)}</span>
        <span class="badge">${project.docs}${project.pending ? `+${project.pending}` : ''}</span>
      </div>
    `).join('');
  } catch (error) {
    console.error('Failed to load projects:', error);
  }
}

// --- Views ---
function showWelcome() {
  currentProject = null;
  currentDocId = null;
  currentChatId = null;
  setTopbar(null);
  document.getElementById('content').innerHTML = `
    <div class="welcome">
      <h2>Esteem Project Knowledge</h2>
      <p>Select a project to browse documents, ingest new files, and ask grounded questions only against that project's indexed content.</p>
    </div>`;
  loadProjects();
}

async function showProject(project) {
  currentProject = project;
  currentDocId = null;
  loadProjects();
  setTopbar(
    `<a href="#/" style="color:#666">Projects</a><span>/</span><strong>${esc(project)}</strong>`,
    `<button class="btn btn-ghost" onclick="promptRenameProject(${jsq(project)})">Rename</button>
     <button class="btn btn-ghost" onclick="navigate('#/quality/${enc(project)}')">Quality</button>
     <button class="btn btn-danger btn-sm" onclick="confirmDeleteProject(${jsq(project)})">Delete Project</button>`
  );

  const content = document.getElementById('content');
  content.innerHTML = '<div class="spinner"></div>';

  try {
    const [docData, chats, jobs] = await Promise.all([
      api(`/projects/${enc(project)}/documents`),
      api(`/projects/${enc(project)}/chats`),
      api('/ingestion/jobs'),
    ]);

    if (currentChatId && !chats.some((chat) => chat.id === currentChatId)) {
      currentChatId = null;
    }
    if (!currentChatId && chats.length) {
      currentChatId = chats[0].id;
    }

    let chatPayload = null;
    if (currentChatId) {
      chatPayload = await api(`/chats/${currentChatId}/messages`);
    }

    content.innerHTML = `
      <div class="project-workspace">
        <section class="project-library workspace-stack">
          ${renderUploadCard(project)}
          ${renderPendingUploads(project, docData.pending)}
          <div id="jobs-area"></div>
          ${renderDocumentsCard(project, docData.documents)}
        </section>
        <section class="project-chat">
          ${renderChatShell(project, chats, chatPayload?.messages || [])}
        </section>
      </div>
    `;

    renderMarkdownBlocks(content);
    scrollChatToBottom();

    const activeJobs = jobs.filter((job) =>
      job.project === project && job.status !== 'completed' && job.status !== 'failed'
    );
    if (activeJobs.length) {
      startPolling(project);
    } else {
      await pollJobs(project);
    }
  } catch (error) {
    content.innerHTML = `<div class="empty">Error: ${esc(error.message)}</div>`;
  }
}

function renderUploadCard(project) {
  return `
    <div class="card">
      <div class="card-title">
        <span>Project Workspace</span>
        <span class="count">Documents and chat stay scoped to this project only</span>
      </div>
      <div class="upload-area" id="upload-area"
           ondragover="event.preventDefault(); this.classList.add('dragover')"
           ondragleave="this.classList.remove('dragover')"
           ondrop="handleDrop(event, ${jsq(project)})"
           onclick="document.getElementById('file-input').click()">
        Drop files here or click to upload
        <span class="muted" style="font-size:11px;display:block;margin-top:4px">PDF, DOCX, XLSX, images, ZIP/TAR archives</span>
        <input type="file" id="file-input" accept=".pdf,.docx,.xlsx,.xls,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.gif,.zip,.tar,.gz,.tgz" multiple
               onchange="handleFiles(this.files, ${jsq(project)})">
      </div>
    </div>
  `;
}

function renderPendingUploads(project, pending) {
  if (!pending.length) return '';
  return `
    <div class="card">
      <div class="card-title">
        <span>Pending Uploads <span class="count">${pending.length}</span></span>
        <button class="btn btn-primary btn-sm" onclick="ingestAll(${jsq(project)})">Ingest All</button>
      </div>
      <table class="table">
        <tr><th>Filename</th><th>Size</th><th></th></tr>
        ${pending.map((file) => `
          <tr>
            <td>${esc(file.filename)}</td>
            <td class="muted">${file.size_mb} MB</td>
            <td>
              <button class="btn btn-ghost btn-sm"
                      onclick="ingestSingle(${jsq(project)}, ${jsq(file.filename)})">Ingest</button>
            </td>
          </tr>
        `).join('')}
      </table>
    </div>
  `;
}

function renderDocumentsCard(project, documents) {
  return `
    <div class="card">
      <div class="card-title">Documents <span class="count">${documents.length}</span></div>
      ${documents.length ? `
        <table class="table">
          <tr><th>Title</th><th>Pages</th><th>Sections</th><th>Type</th><th></th></tr>
          ${documents.map((doc) => {
            const splitTag = doc.split_info
              ? `<span class="tag" title="Split from ${esc(doc.split_info.parent)} pages ${doc.split_info.page_start}-${doc.split_info.page_end}">
                   Part ${doc.split_info.part} &middot; p${doc.split_info.page_start}-${doc.split_info.page_end}
                 </span>`
              : '';
            return `
              <tr class="clickable" onclick="navigate('#/doc/${doc.id}')">
                <td>
                  <strong>${esc(doc.title)}</strong>
                  ${splitTag}
                  <br>
                  <span class="muted mono">${esc(doc.filename)}</span>
                </td>
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
      ` : '<div class="empty">No documents ingested yet. Upload files above to start building this project knowledge base.</div>'}
    </div>
  `;
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
            <h3>${currentChatId ? esc((chats.find((chat) => chat.id === currentChatId)?.title) || 'New chat') : 'Project chat'}</h3>
            <div class="chat-subtitle">Answers must be grounded only in documents from <strong>${esc(project)}</strong>.</div>
          </div>
          ${currentChatId ? `<button class="btn btn-sm btn-ghost" onclick="newChat(${jsq(project)})">Start fresh</button>` : ''}
        </div>
        <div class="chat-messages" id="chat-messages">
          ${messages.length ? renderChatMessages(messages) : `
            <div class="chat-empty" id="chat-empty">
              <h3>Ask about this project</h3>
              <p>The assistant will search only the documents inside <strong>${esc(project)}</strong>, keep the thread history in context, and say so when the documents do not support an answer.</p>
            </div>
          `}
        </div>
        <div class="chat-composer">
          <textarea id="chat-input"
                    placeholder="Ask a question about this project's documents..."
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
          <span> &middot; p${source.page_num}${source.breadcrumb ? ` &middot; ${esc(source.breadcrumb)}` : ''}</span>
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
      `<a href="#/" style="color:#666">Projects</a><span>/</span>` +
      `<a href="#/project/${enc(doc.project)}">${esc(doc.project)}</a><span>/</span>` +
      `<strong>${esc(doc.title)}</strong>${splitNote}`,
      `<a class="btn btn-ghost btn-sm" href="/api/documents/${docId}/pdf">Download</a>
       <button class="btn btn-ghost btn-sm" onclick="navigate('#/project/${enc(doc.project)}')">Back</button>`
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
    `<a href="#/" style="color:#666">Projects</a><span>/</span>` +
    `<a href="#/project/${enc(project)}">${esc(project)}</a><span>/</span><strong>Quality</strong>`,
    ''
  );
  const content = document.getElementById('content');
  content.style.padding = '20px';
  content.innerHTML = '<div class="spinner"></div>';

  try {
    const data = await api(`/projects/${enc(project)}/quality`);
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
    const created = await api(`/projects/${enc(project)}/chats`, {
      method: 'POST',
      body: JSON.stringify({ title: 'New chat' }),
    });
    currentChatId = created.id;
    showProject(project);
  } catch (error) {
    alert('Could not create chat: ' + error.message);
  }
}

function selectChat(threadId) {
  currentChatId = threadId;
  if (currentProject) showProject(currentProject);
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
      const created = await api(`/projects/${enc(currentProject)}/chats`, {
        method: 'POST',
        body: JSON.stringify({ title: 'New chat' }),
      });
      currentChatId = created.id;
      await showProject(currentProject);
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
    await showProject(currentProject);
  } catch (error) {
    stopThinkingIndicator();
    if (sendButton) sendButton.disabled = false;
    if (textarea) textarea.disabled = false;
    alert('Chat failed: ' + error.message);
    await showProject(currentProject);
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

// --- Project actions ---
function promptNewProject() {
  document.getElementById('modal-content').innerHTML = `
    <h3>New Project</h3>
    <input type="text" id="new-project-name" placeholder="Project name" autofocus
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
    await api('/projects', { method: 'POST', body: JSON.stringify({ name }) });
    closeModal();
    await loadProjects();
    navigate('#/project/' + encodeURIComponent(name));
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

function promptRenameProject(project) {
  document.getElementById('modal-content').innerHTML = `
    <h3>Rename Project</h3>
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
    await api(`/projects/${enc(oldName)}`, {
      method: 'PATCH',
      body: JSON.stringify({ name: newName }),
    });
    closeModal();
    currentProject = newName;
    await loadProjects();
    navigate('#/project/' + encodeURIComponent(newName));
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
    if (currentProject) showProject(currentProject);
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

function confirmDeleteProject(project) {
  if (confirm(`Delete project "${project}" and ALL its documents and chats?`)) {
    api(`/projects/${enc(project)}`, { method: 'DELETE' })
      .then(() => {
        currentChatId = null;
        navigate('#/');
        loadProjects();
      })
      .catch((error) => alert('Error: ' + error.message));
  }
}

function confirmDeleteDoc(docId, title) {
  if (confirm(`Delete document "${title}"?`)) {
    api(`/documents/${docId}`, { method: 'DELETE' })
      .then(() => {
        if (currentProject) showProject(currentProject);
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
  const files = [...event.dataTransfer.files].filter((file) => supported(file.name));
  if (files.length) uploadFiles(files, project);
}

function handleFiles(fileList, project) {
  const files = [...fileList].filter((file) => supported(file.name));
  if (files.length) uploadFiles(files, project);
}

async function uploadFiles(files, project) {
  const area = document.getElementById('upload-area');
  if (area) area.textContent = `Uploading ${files.length} file(s)...`;
  try {
    await apiUpload(`/projects/${enc(project)}/upload`, files);
    showProject(project);
  } catch (error) {
    alert('Upload failed: ' + error.message);
    showProject(project);
  }
}

// --- Ingestion ---
async function ingestAll(project) {
  try {
    await api(`/projects/${enc(project)}/ingest`, { method: 'POST' });
    startPolling(project);
  } catch (error) {
    alert('Ingestion failed: ' + error.message);
  }
}

async function ingestSingle(project, filename) {
  try {
    await api(`/projects/${enc(project)}/ingest/${enc(filename)}`, { method: 'POST' });
    startPolling(project);
  } catch (error) {
    alert('Ingestion failed: ' + error.message);
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
    const projectJobs = jobs.filter((job) => job.project === project);
    const area = document.getElementById('jobs-area');
    if (!area) return;

    if (!projectJobs.length) {
      area.innerHTML = '';
      return;
    }

    area.innerHTML = projectJobs.map((job) => `
      <div class="job-card">
        <span class="filename">${esc(job.filename)}</span>
        <span class="stage">${esc(job.stage)}</span>
        <span class="status ${job.status}">${job.status}</span>
        ${job.error ? `<span style="color:#dc3545;font-size:12px">${esc(job.error)}</span>` : ''}
      </div>
    `).join('');

    const allDone = projectJobs.every((job) => job.status === 'completed' || job.status === 'failed');
    if (allDone && projectJobs.some((job) => job.status === 'completed')) {
      stopPolling();
      setTimeout(() => showProject(project), 1000);
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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
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
