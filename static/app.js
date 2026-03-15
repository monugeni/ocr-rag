// =========================================================================
// OCR-RAG Web GUI — Client Application
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
let pollTimer = null;

// --- Router ---
window.addEventListener('hashchange', router);
window.addEventListener('load', () => { loadProjects(); router(); });

function router() {
  const hash = location.hash.slice(1) || '/';
  stopPolling();

  if (hash.startsWith('/doc/')) {
    const parts = hash.split('/');
    const docId = parseInt(parts[2]);
    const pageNum = parts[3] === 'page' ? parseInt(parts[4]) : 1;
    showViewer(docId, pageNum);
  } else if (hash.startsWith('/project/')) {
    const name = decodeURIComponent(hash.slice(9));
    currentProject = name;
    showProject(name);
  } else if (hash.startsWith('/quality/')) {
    const name = decodeURIComponent(hash.slice(9));
    showQuality(name);
  } else {
    showWelcome();
  }
}

function navigate(hash) { location.hash = hash; }

// --- Sidebar: Projects ---
async function loadProjects() {
  try {
    const projects = await api('/projects');
    const list = document.getElementById('project-list');
    if (!projects.length) {
      list.innerHTML = '<div class="empty" style="padding:16px;font-size:12px">No projects yet</div>';
      return;
    }
    list.innerHTML = projects.map(p => `
      <div class="project-item ${currentProject === p.project ? 'active' : ''}"
           onclick="navigate('#/project/${encodeURIComponent(p.project)}')">
        <span class="name" title="${esc(p.project)}">${esc(p.project)}</span>
        <span class="badge">${p.docs}${p.pending ? '+' + p.pending : ''}</span>
      </div>
    `).join('');
  } catch (e) {
    console.error('Failed to load projects:', e);
  }
}

// --- Views ---
function showWelcome() {
  currentProject = null;
  currentDocId = null;
  setTopbar(null);
  document.getElementById('content').innerHTML = `
    <div class="welcome"><h2>Welcome</h2>
    <p>Select a project from the sidebar, or create a new one.</p></div>`;
  loadProjects();
}

async function showProject(project) {
  currentProject = project;
  currentDocId = null;
  loadProjects();
  setTopbar(
    `<a href="#/" style="color:#666">Projects</a><span>/</span><strong>${esc(project)}</strong>`,
    `<button class="btn btn-ghost" onclick="promptRenameProject('${esc(project)}')">Rename</button>
     <button class="btn btn-ghost" onclick="navigate('#/quality/${encodeURIComponent(project)}')">Quality</button>
     <button class="btn btn-danger btn-sm" onclick="confirmDeleteProject('${esc(project)}')">Delete Project</button>`
  );

  const content = document.getElementById('content');
  content.innerHTML = '<div class="spinner"></div>';

  try {
    const data = await api(`/projects/${enc(project)}/documents`);
    let html = '';

    // Upload area
    html += `
      <div class="upload-area" id="upload-area"
           ondragover="event.preventDefault(); this.classList.add('dragover')"
           ondragleave="this.classList.remove('dragover')"
           ondrop="handleDrop(event, '${esc(project)}')"
           onclick="document.getElementById('file-input').click()">
        Drop files here or click to upload
        <span class="muted" style="font-size:11px;display:block;margin-top:4px">PDF, DOCX, XLSX, images, ZIP/TAR archives</span>
        <input type="file" id="file-input" accept=".pdf,.docx,.xlsx,.xls,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.gif,.zip,.tar,.gz,.tgz" multiple
               onchange="handleFiles(this.files, '${esc(project)}')">
      </div>`;

    // Pending uploads
    if (data.pending.length) {
      html += `<div class="card">
        <div class="card-title">Pending Uploads <span class="count">${data.pending.length}</span>
          <button class="btn btn-primary btn-sm" onclick="ingestAll('${esc(project)}')">Ingest All</button>
        </div>
        <table class="table"><tr><th>Filename</th><th>Size</th><th></th></tr>`;
      for (const p of data.pending) {
        html += `<tr>
          <td>${esc(p.filename)}</td><td class="muted">${p.size_mb} MB</td>
          <td><button class="btn btn-ghost btn-sm"
               onclick="ingestSingle('${esc(project)}','${esc(p.filename)}')">Ingest</button></td>
        </tr>`;
      }
      html += '</table></div>';
    }

    // Ingestion jobs
    html += '<div id="jobs-area"></div>';

    // Documents table
    html += `<div class="card">
      <div class="card-title">Documents <span class="count">${data.documents.length}</span></div>`;
    if (data.documents.length) {
      html += `<table class="table"><tr><th>Title</th><th>Pages</th><th>Sections</th><th>Type</th><th></th></tr>`;
      for (const d of data.documents) {
        const splitTag = d.split_info
          ? `<span class="tag" style="background:#e8f4f8;color:#0077aa;margin-left:6px;font-size:10px" title="Split from ${esc(d.split_info.parent)} pages ${d.split_info.page_start}-${d.split_info.page_end}">Part ${d.split_info.part} &middot; p${d.split_info.page_start}-${d.split_info.page_end}</span>`
          : '';
        html += `<tr class="clickable" onclick="navigate('#/doc/${d.id}')">
          <td><strong>${esc(d.title)}</strong>${splitTag}<br><span class="muted mono">${esc(d.filename)}</span></td>
          <td>${d.total_pages}</td><td>${d.sections}</td>
          <td>${d.document_type ? `<span class="tag">${esc(d.document_type)}</span>` : '<span class="muted">-</span>'}</td>
          <td style="white-space:nowrap">
            <a class="btn btn-ghost btn-sm" href="/api/documents/${d.id}/pdf" onclick="event.stopPropagation()" title="Download PDF">PDF</a>
            <button class="btn btn-ghost btn-sm" onclick="event.stopPropagation(); promptRenameDoc(${d.id}, '${esc(d.title)}')" title="Rename">Rename</button>
            <button class="btn btn-ghost btn-sm" style="color:#dc3545"
               onclick="event.stopPropagation(); confirmDeleteDoc(${d.id}, '${esc(d.title)}')">Delete</button>
          </td>
        </tr>`;
      }
      html += '</table>';
    } else {
      html += '<div class="empty">No documents ingested yet. Upload PDFs above.</div>';
    }
    html += '</div>';
    content.innerHTML = html;

    // Only poll if there are active (non-terminal) jobs
    const jobs = await api('/ingestion/jobs');
    const activeJobs = jobs.filter(j => j.project === project && j.status !== 'completed' && j.status !== 'failed');
    if (activeJobs.length) {
      startPolling(project);
    }
  } catch (e) {
    content.innerHTML = `<div class="empty">Error: ${esc(e.message)}</div>`;
  }
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
      ? ` <span class="tag" style="background:#e8f4f8;color:#0077aa;font-size:10px">Part ${doc.split_info.part} of ${esc(doc.split_info.parent)} &middot; p${doc.split_info.page_start}-${doc.split_info.page_end}</span>`
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
      </div>`;

    loadPage(docId, currentPage, doc.total_pages);
  } catch (e) {
    document.getElementById('content').innerHTML =
      `<div class="empty">Error: ${esc(e.message)}</div>`;
  }
}

function renderToc(sections, docId) {
  if (!sections.length) return '<div class="empty" style="padding:12px;font-size:12px">No sections</div>';
  return sections.map(s => `
    <div class="toc-item l${Math.min(s.level, 4)}"
         title="${esc(s.heading)}"
         onclick="loadPage(${docId}, ${s.page_start})">
      ${esc(s.heading)}<span class="toc-pages">p${s.page_start}</span>
    </div>
  `).join('');
}

async function loadPage(docId, pageNum, totalPages) {
  currentPage = pageNum;
  try {
    const p = await api(`/documents/${docId}/pages/${pageNum}`);
    totalPages = totalPages || p.total_pages;

    document.getElementById('page-header').innerHTML = `
      <div>
        <span class="page-bc">${esc(p.breadcrumb || '')}</span>
      </div>
      <div class="page-nav">
        <button class="btn btn-ghost btn-sm" ${pageNum <= 1 ? 'disabled' : ''}
                onclick="loadPage(${docId}, ${pageNum - 1}, ${totalPages})">Prev</button>
        <span class="page-num">Page ${pageNum} / ${totalPages}</span>
        <button class="btn btn-ghost btn-sm" ${pageNum >= totalPages ? 'disabled' : ''}
                onclick="loadPage(${docId}, ${pageNum + 1}, ${totalPages})">Next</button>
      </div>`;
    document.getElementById('page-body').innerHTML = `<pre>${esc(p.content)}</pre>`;

    // Update hash without re-routing
    history.replaceState(null, '', `#/doc/${docId}/page/${pageNum}`);

    // Highlight active TOC item
    document.querySelectorAll('.toc-item').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.toc-item').forEach(el => {
      const pg = parseInt(el.querySelector('.toc-pages')?.textContent?.slice(1));
      if (pg === pageNum) el.classList.add('active');
    });
  } catch (e) {
    document.getElementById('page-body').innerHTML =
      `<div class="empty">Error: ${esc(e.message)}</div>`;
  }
}

async function searchInDoc(docId, query) {
  if (!query.trim()) {
    document.getElementById('search-results').innerHTML = '';
    document.getElementById('toc-list').style.display = '';
    return;
  }
  try {
    const results = await api(`/documents/${docId}/search?q=${encodeURIComponent(query)}`);
    document.getElementById('toc-list').style.display = 'none';
    const sr = document.getElementById('search-results');
    if (!results.length) {
      sr.innerHTML = '<div class="empty" style="padding:12px;font-size:12px">No results</div>';
      return;
    }
    sr.innerHTML = results.map(r => `
      <div class="toc-item" onclick="loadPage(${docId}, ${r.page_num}); document.getElementById('toc-list').style.display=''; document.getElementById('search-results').innerHTML='';">
        <strong>p${r.page_num}</strong> ${r.snippet.replace(/>>>/g, '<b>').replace(/<<</g, '</b>')}
      </div>
    `).join('');
  } catch (e) {
    console.error(e);
  }
}

async function showQuality(project) {
  setTopbar(
    `<a href="#/" style="color:#666">Projects</a><span>/</span>` +
    `<a href="#/project/${enc(project)}">${esc(project)}</a><span>/</span><strong>Quality</strong>`, ''
  );
  const content = document.getElementById('content');
  content.style.padding = '20px';
  content.innerHTML = '<div class="spinner"></div>';

  try {
    const data = await api(`/projects/${enc(project)}/quality`);
    let html = '';

    // Flags
    html += `<div class="card"><div class="card-title">Quality Flags <span class="count">${data.flags.length}</span></div>`;
    if (data.flags.length) {
      for (const f of data.flags) {
        const resolved = f.resolved ? ' resolved' : '';
        html += `<div class="flag-row${resolved}">
          <span class="flag-type ${f.flag_type}">${f.flag_type}</span>
          <a href="#/doc/${f.doc_id}">${esc(f.doc_title)}</a>
          ${f.page_num ? `<span class="muted">p${f.page_num}</span>` : ''}
          <span style="flex:1">${esc(f.reason || '')}</span>
          ${!f.resolved ? `<button class="btn btn-ghost btn-sm" onclick="resolveFlag(${f.id})">Resolve</button>` : ''}
        </div>`;
      }
    } else {
      html += '<div class="empty">No quality flags</div>';
    }
    html += '</div>';

    // Corrections log
    html += `<div class="card"><div class="card-title">Recent Corrections <span class="count">${data.corrections.length}</span></div>`;
    if (data.corrections.length) {
      html += '<table class="table"><tr><th>Document</th><th>Category</th><th>Action</th><th>When</th></tr>';
      for (const c of data.corrections) {
        html += `<tr>
          <td><a href="#/doc/${c.doc_id}">${esc(c.doc_title)}</a></td>
          <td><span class="tag">${esc(c.category)}</span></td>
          <td>${esc(c.action)}</td>
          <td class="muted">${esc(c.created_at || '')}</td>
        </tr>`;
      }
      html += '</table>';
    } else {
      html += '<div class="empty">No corrections logged</div>';
    }
    html += '</div>';
    content.innerHTML = html;
  } catch (e) {
    content.innerHTML = `<div class="empty">Error: ${esc(e.message)}</div>`;
  }
}

// --- Actions ---
function promptNewProject() {
  document.getElementById('modal-content').innerHTML = `
    <h3>New Project</h3>
    <input type="text" id="new-project-name" placeholder="Project name" autofocus
           onkeydown="if(event.key==='Enter') createProject()">
    <div class="actions">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="createProject()">Create</button>
    </div>`;
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
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

function promptRenameProject(project) {
  document.getElementById('modal-content').innerHTML = `
    <h3>Rename Project</h3>
    <input type="text" id="rename-input" value="${esc(project)}" autofocus
           onkeydown="if(event.key==='Enter') doRenameProject('${esc(project)}')">
    <div class="actions">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doRenameProject('${esc(project)}')">Rename</button>
    </div>`;
  document.getElementById('modal-overlay').style.display = 'flex';
  setTimeout(() => { const el = document.getElementById('rename-input'); el?.focus(); el?.select(); }, 100);
}

async function doRenameProject(oldName) {
  const newName = document.getElementById('rename-input').value.trim();
  if (!newName || newName === oldName) { closeModal(); return; }
  try {
    await api(`/projects/${enc(oldName)}`, {
      method: 'PATCH', body: JSON.stringify({ name: newName })
    });
    closeModal();
    currentProject = newName;
    await loadProjects();
    navigate('#/project/' + encodeURIComponent(newName));
  } catch (e) { alert('Error: ' + e.message); }
}

function promptRenameDoc(docId, currentTitle) {
  document.getElementById('modal-content').innerHTML = `
    <h3>Rename Document</h3>
    <input type="text" id="rename-input" value="${esc(currentTitle)}" autofocus
           onkeydown="if(event.key==='Enter') doRenameDoc(${docId})">
    <div class="actions">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="doRenameDoc(${docId})">Rename</button>
    </div>`;
  document.getElementById('modal-overlay').style.display = 'flex';
  setTimeout(() => { const el = document.getElementById('rename-input'); el?.focus(); el?.select(); }, 100);
}

async function doRenameDoc(docId) {
  const title = document.getElementById('rename-input').value.trim();
  if (!title) { closeModal(); return; }
  try {
    await api(`/documents/${docId}`, {
      method: 'PATCH', body: JSON.stringify({ title })
    });
    closeModal();
    if (currentProject) showProject(currentProject);
  } catch (e) { alert('Error: ' + e.message); }
}

function confirmDeleteProject(project) {
  if (confirm(`Delete project "${project}" and ALL its documents?`)) {
    api(`/projects/${enc(project)}`, { method: 'DELETE' }).then(() => {
      navigate('#/');
      loadProjects();
    }).catch(e => alert('Error: ' + e.message));
  }
}

function confirmDeleteDoc(docId, title) {
  if (confirm(`Delete document "${title}"?`)) {
    api(`/documents/${docId}`, { method: 'DELETE' }).then(() => {
      if (currentProject) showProject(currentProject);
    }).catch(e => alert('Error: ' + e.message));
  }
}

async function resolveFlag(flagId) {
  await api(`/quality/${flagId}/resolve`, { method: 'PATCH' });
  if (location.hash.startsWith('#/quality/')) router();
}

// --- Upload ---
const SUPPORTED_EXT = new Set(['.pdf','.docx','.xlsx','.xls','.jpg','.jpeg','.png','.tiff','.tif','.bmp','.gif','.zip','.tar','.gz','.tgz']);
function _supported(name) {
  const i = name.lastIndexOf('.');
  return i >= 0 && SUPPORTED_EXT.has(name.slice(i).toLowerCase());
}

function handleDrop(e, project) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  const files = [...e.dataTransfer.files].filter(f => _supported(f.name));
  if (files.length) uploadFiles(files, project);
}

function handleFiles(fileList, project) {
  const files = [...fileList].filter(f => _supported(f.name));
  if (files.length) uploadFiles(files, project);
}

async function uploadFiles(files, project) {
  const area = document.getElementById('upload-area');
  area.textContent = `Uploading ${files.length} file(s)...`;
  try {
    await apiUpload(`/projects/${enc(project)}/upload`, files);
    showProject(project);
  } catch (e) {
    alert('Upload failed: ' + e.message);
    showProject(project);
  }
}

// --- Ingestion ---
async function ingestAll(project) {
  try {
    await api(`/projects/${enc(project)}/ingest`, { method: 'POST' });
    startPolling(project);
  } catch (e) {
    alert('Ingestion failed: ' + e.message);
  }
}

async function ingestSingle(project, filename) {
  try {
    await api(`/projects/${enc(project)}/ingest/${enc(filename)}`, { method: 'POST' });
    startPolling(project);
  } catch (e) {
    alert('Ingestion failed: ' + e.message);
  }
}

function startPolling(project) {
  stopPolling();
  pollJobs(project);
  pollTimer = setInterval(() => pollJobs(project), 2000);
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollJobs(project) {
  try {
    const jobs = await api('/ingestion/jobs');
    const projectJobs = jobs.filter(j => j.project === project);
    const area = document.getElementById('jobs-area');
    if (!area) return;

    if (!projectJobs.length) {
      area.innerHTML = '';
      return;
    }

    area.innerHTML = projectJobs.map(j => `
      <div class="job-card">
        <span class="filename">${esc(j.filename)}</span>
        <span class="stage">${esc(j.stage)}</span>
        <span class="status ${j.status}">${j.status}</span>
        ${j.error ? `<span style="color:#dc3545;font-size:12px">${esc(j.error)}</span>` : ''}
      </div>
    `).join('');

    // If all done, refresh the project view
    const allDone = projectJobs.every(j => j.status === 'completed' || j.status === 'failed');
    if (allDone && projectJobs.some(j => j.status === 'completed')) {
      stopPolling();
      setTimeout(() => showProject(project), 1000);
    }
  } catch (e) {
    console.error('Poll error:', e);
  }
}

// --- Helpers ---
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

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function enc(s) { return encodeURIComponent(s); }
