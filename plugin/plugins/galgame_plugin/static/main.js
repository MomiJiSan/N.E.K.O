const PLUGIN_ID = 'galgame_plugin';
const RUNS_URL = '/runs';
const UI_API_BASE = `/plugin/${PLUGIN_ID}/ui-api`;
const TEXTRACTOR_INSTALL_URL = `${UI_API_BASE}/textractor/install`;
const INSTALL_TASK_STORAGE_KEY = `${PLUGIN_ID}:textractor_install_task_id`;
const INSTALL_TERMINAL_STATUSES = new Set(['completed', 'failed', 'canceled']);

let latestAgentReply = '暂无交互';
let latestStatus = null;
let latestInstallTaskState = null;
let textractorInstallInProgress = false;
let currentInstallTaskId = '';
let installEventSource = null;
let installReconnectTimer = null;
let handledInstallTerminalKey = '';

async function createRun(entryId, args = {}) {
  const createResp = await fetch(RUNS_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      plugin_id: PLUGIN_ID,
      entry_id: entryId,
      args,
    }),
  });

  if (!createResp.ok) {
    throw new Error(`创建任务失败: HTTP ${createResp.status}`);
  }

  const createData = await createResp.json();
  const runId = createData.run_id || createData.id;
  if (!runId) {
    throw new Error('未获取到 run_id');
  }
  return runId;
}

async function exportRunResult(runId) {
  const exportResp = await fetch(`${RUNS_URL}/${runId}/export`);
  if (!exportResp.ok) {
    return {};
  }
  const exportData = await exportResp.json();
  const items = exportData.items || [];
  const resultItem = items.find((item) => item.type === 'json' && item.json) || items[0];
  const pluginResponse = resultItem ? (resultItem.json || {}) : {};
  if (pluginResponse.success === false || pluginResponse.error) {
    throw new Error(pluginResponse.error?.message || pluginResponse.message || '插件调用失败');
  }
  return pluginResponse.data || {};
}

async function callPlugin(entryId, args = {}) {
  const runId = await createRun(entryId, args);
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    const pollResp = await fetch(`${RUNS_URL}/${runId}`);
    if (!pollResp.ok) {
      continue;
    }

    const runRecord = await pollResp.json();
    if (runRecord.status === 'succeeded') {
      return await exportRunResult(runId);
    }

    if (['failed', 'canceled', 'timeout'].includes(runRecord.status)) {
      throw new Error(runRecord.error?.message || runRecord.message || runRecord.status);
    }
  }

  throw new Error('插件调用超时');
}

async function safeCall(entryId, args = {}, fallback = {}) {
  try {
    return await callPlugin(entryId, args);
  } catch (error) {
    return { ...fallback, degraded: true, diagnostic: error instanceof Error ? error.message : String(error) };
  }
}

function escapeHtml(text) {
  if (text == null) return '';
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

function setFlash(message, type = 'info') {
  const node = document.getElementById('flashMessage');
  node.hidden = !message;
  node.textContent = message || '';
  node.className = `flash-message ${type}`;
}

function renderGrid(nodeId, rows) {
  const container = document.getElementById(nodeId);
  container.innerHTML = rows.map((row) => `
    <div class="data-row">
      <dt>${escapeHtml(row.label)}</dt>
      <dd>${escapeHtml(row.value)}</dd>
    </div>
  `).join('');
}

function renderStackList(nodeId, items, formatter) {
  const node = document.getElementById(nodeId);
  if (!items.length) {
    node.className = 'stack-list empty-state';
    node.textContent = '暂无数据';
    return;
  }
  node.className = 'stack-list';
  node.innerHTML = items.map(formatter).join('');
}

function isInstallTaskTerminal(state) {
  return Boolean(state) && INSTALL_TERMINAL_STATUSES.has(String(state.status || ''));
}

function persistInstallTaskId(taskId) {
  if (!taskId) {
    return;
  }
  try {
    localStorage.setItem(INSTALL_TASK_STORAGE_KEY, taskId);
  } catch (_) {
    // Ignore storage failures in embedded browsers.
  }
}

function readPersistedInstallTaskId() {
  try {
    return localStorage.getItem(INSTALL_TASK_STORAGE_KEY) || '';
  } catch (_) {
    return '';
  }
}

function clearInstallReconnectTimer() {
  if (installReconnectTimer) {
    clearTimeout(installReconnectTimer);
    installReconnectTimer = null;
  }
}

function closeInstallStream() {
  if (installEventSource) {
    installEventSource.close();
    installEventSource = null;
  }
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const digits = size >= 100 || unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(digits)} ${units[unitIndex]}`;
}

function formatInstallPhase(phase) {
  const normalized = String(phase || '').trim();
  const mapping = {
    queued: '排队中',
    metadata: '获取版本信息',
    downloading: 'HTTPS 下载中',
    extracting: '解压安装中',
    verifying: '校验安装中',
    completed: '安装完成',
    failed: '安装失败',
    canceled: '已取消',
  };
  return mapping[normalized] || normalized || '等待中';
}

function renderInstallTaskState() {
  const card = document.getElementById('textractorInstallState');
  const statusText = document.getElementById('textractorInstallStatusText');
  const percentText = document.getElementById('textractorInstallPercent');
  const messageText = document.getElementById('textractorInstallMessage');
  const detailText = document.getElementById('textractorInstallDetail');
  const progressBar = document.getElementById('textractorInstallBar');

  if (!latestInstallTaskState) {
    card.hidden = true;
    statusText.textContent = '等待安装任务';
    percentText.textContent = '0%';
    messageText.textContent = '';
    detailText.textContent = '';
    progressBar.style.width = '0%';
    return;
  }

  const progress = Math.max(0, Math.min(1, Number(latestInstallTaskState.progress || 0)));
  const percent = Math.round(progress * 100);
  const details = [];
  if (latestInstallTaskState.total_bytes) {
    details.push(`${formatBytes(latestInstallTaskState.downloaded_bytes)} / ${formatBytes(latestInstallTaskState.total_bytes)}`);
  } else if (latestInstallTaskState.downloaded_bytes) {
    details.push(formatBytes(latestInstallTaskState.downloaded_bytes));
  }
  if (latestInstallTaskState.resume_from) {
    details.push(`续传自 ${formatBytes(latestInstallTaskState.resume_from)}`);
  }
  if (latestInstallTaskState.asset_name) {
    details.push(latestInstallTaskState.asset_name);
  }
  if (latestInstallTaskState.task_id) {
    details.push(`task ${latestInstallTaskState.task_id}`);
  }

  card.hidden = false;
  statusText.textContent = `${formatInstallPhase(latestInstallTaskState.phase)} · ${latestInstallTaskState.status || ''}`;
  percentText.textContent = `${percent}%`;
  messageText.textContent = latestInstallTaskState.message || '';
  detailText.textContent = details.join(' · ');
  progressBar.style.width = `${percent}%`;
}

function applyInstallTaskState(state, { allowRefresh = true } = {}) {
  if (!state) {
    return;
  }
  latestInstallTaskState = state;
  currentInstallTaskId = state.task_id || state.run_id || currentInstallTaskId;
  if (currentInstallTaskId) {
    persistInstallTaskId(currentInstallTaskId);
  }
  textractorInstallInProgress = !isInstallTaskTerminal(state);
  if (latestStatus) {
    renderTextractor(latestStatus);
  } else {
    renderInstallTaskState();
  }

  if (!isInstallTaskTerminal(state)) {
    return;
  }

  closeInstallStream();
  clearInstallReconnectTimer();
  const terminalKey = `${currentInstallTaskId}:${state.status || ''}:${state.updated_at || ''}`;
  if (handledInstallTerminalKey === terminalKey) {
    return;
  }
  handledInstallTerminalKey = terminalKey;

  if (state.status === 'completed') {
    setFlash(state.message || 'Textractor 安装完成', 'success');
  } else {
    setFlash(state.error || state.message || 'Textractor 安装失败', 'error');
  }

  if (allowRefresh) {
    refreshAll().catch((error) => {
      setFlash(error instanceof Error ? error.message : String(error), 'error');
    });
  }
}

async function fetchInstallTaskState(taskId) {
  const response = await fetch(`${TEXTRACTOR_INSTALL_URL}/${encodeURIComponent(taskId)}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`读取安装状态失败: HTTP ${response.status}`);
  }
  return await response.json();
}

async function fetchLatestInstallTaskState() {
  const response = await fetch(`${TEXTRACTOR_INSTALL_URL}/latest`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`读取最近安装状态失败: HTTP ${response.status}`);
  }
  return await response.json();
}

function scheduleInstallReconnect(taskId) {
  clearInstallReconnectTimer();
  installReconnectTimer = setTimeout(async () => {
    try {
      const recovered = await fetchInstallTaskState(taskId);
      if (recovered) {
        applyInstallTaskState(recovered, { allowRefresh: false });
        if (!isInstallTaskTerminal(recovered)) {
          connectInstallStream(taskId);
        }
        return;
      }
    } catch (_) {
      // Keep retrying until we observe a terminal state or the server becomes reachable again.
    }

    if (!latestInstallTaskState || !isInstallTaskTerminal(latestInstallTaskState)) {
      scheduleInstallReconnect(taskId);
    }
  }, 1500);
}

function connectInstallStream(taskId) {
  closeInstallStream();
  clearInstallReconnectTimer();
  const stream = new EventSource(`${TEXTRACTOR_INSTALL_URL}/${encodeURIComponent(taskId)}/stream`);
  installEventSource = stream;

  stream.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      applyInstallTaskState(payload);
    } catch (error) {
      setFlash(error instanceof Error ? error.message : String(error), 'error');
    }
  };

  stream.onerror = async () => {
    if (installEventSource !== stream) {
      return;
    }
    stream.close();
    installEventSource = null;
    if (latestInstallTaskState && isInstallTaskTerminal(latestInstallTaskState)) {
      return;
    }
    try {
      const recovered = await fetchInstallTaskState(taskId);
      if (recovered) {
        applyInstallTaskState(recovered, { allowRefresh: false });
      }
    } catch (_) {
      // Ignore transient recovery failures and retry shortly.
    }
    scheduleInstallReconnect(taskId);
  };
}

async function restoreTextractorInstallState() {
  const persistedTaskId = readPersistedInstallTaskId();
  if (persistedTaskId) {
    try {
      const persistedState = await fetchInstallTaskState(persistedTaskId);
      if (persistedState) {
        applyInstallTaskState(persistedState, { allowRefresh: false });
        if (!isInstallTaskTerminal(persistedState)) {
          connectInstallStream(persistedTaskId);
        }
        return;
      }
    } catch (_) {
      // Fall through to latest-task recovery.
    }
  }

  try {
    const latestState = await fetchLatestInstallTaskState();
    if (!latestState) {
      return;
    }
    applyInstallTaskState(latestState, { allowRefresh: false });
    const latestTaskId = latestState.task_id || latestState.run_id || '';
    if (latestTaskId && !isInstallTaskTerminal(latestState)) {
      connectInstallStream(latestTaskId);
    }
  } catch (_) {
    // No active/recent install task to restore.
  }
}

function renderStatus(status) {
  latestStatus = status;
  document.getElementById('summaryText').textContent = status.summary || '无摘要';
  document.getElementById('modeSelect').value = status.mode || 'companion';
  document.getElementById('pushToggle').checked = Boolean(status.push_notifications);
  document.getElementById('bindInput').value = status.bound_game_id || '';
  const memoryReaderRuntime = status.memory_reader_runtime || {};
  const textractor = status.textractor || {};
  const memoryReaderProcess = memoryReaderRuntime.process_name
    ? `${memoryReaderRuntime.process_name} (${memoryReaderRuntime.pid || 0})`
    : '';

  renderGrid('statusGrid', [
    { label: 'connection_state', value: status.connection_state || '' },
    { label: 'active_data_source', value: status.active_data_source || '' },
    { label: 'mode', value: status.mode || '' },
    { label: 'push_notifications', value: String(Boolean(status.push_notifications)) },
    { label: 'bound_game_id', value: status.bound_game_id || '(auto)' },
    { label: 'active_session_id', value: status.active_session_id || '' },
    { label: 'last_seq', value: String(status.last_seq || 0) },
    { label: 'stream_reset_pending', value: String(Boolean(status.stream_reset_pending)) },
    { label: 'available_game_ids', value: (status.available_game_ids || []).join(', ') || '(none)' },
    { label: 'memory_reader_enabled', value: String(Boolean(status.memory_reader_enabled)) },
    { label: 'memory_reader_status', value: memoryReaderRuntime.status || '' },
    { label: 'memory_reader_detail', value: memoryReaderRuntime.detail || '' },
    { label: 'memory_reader_process', value: memoryReaderProcess },
    { label: 'textractor_installed', value: String(Boolean(textractor.installed)) },
    { label: 'textractor_detail', value: textractor.detail || '' },
    { label: 'textractor_path', value: textractor.detected_path || '' },
    { label: 'textractor_target_dir', value: textractor.target_dir || '' },
    { label: 'last_error', value: status.last_error?.message || '' },
  ]);
  renderTextractor(status);
}

function renderTextractor(status) {
  const textractor = status.textractor || {};
  const runtime = status.memory_reader_runtime || {};
  const banner = document.getElementById('textractorPrompt');
  const kicker = document.getElementById('textractorPromptKicker');
  const title = document.getElementById('textractorPromptTitle');
  const body = document.getElementById('textractorPromptBody');
  const path = document.getElementById('textractorPathText');
  const button = document.getElementById('textractorInstallBtn');
  const installable = Boolean(textractor.install_supported) && Boolean(textractor.can_install);
  const installed = Boolean(textractor.installed);
  const runtimeBlocked = runtime.detail === 'invalid_textractor_path';
  const installState = latestInstallTaskState;

  banner.hidden = false;
  banner.className = 'textractor-banner';
  button.hidden = !installable;
  button.disabled = textractorInstallInProgress;
  button.textContent = textractorInstallInProgress ? '后台安装中...' : '一键安装 Textractor';

  if (!textractor.install_supported) {
    banner.classList.add('neutral');
    kicker.textContent = 'Textractor';
    title.textContent = '当前平台无需 Textractor 自动安装';
    body.textContent = 'Textractor 读内存兜底仅在 Windows 上启用。当前环境仍可继续使用文件桥模式。';
    path.textContent = '';
    button.hidden = true;
    renderInstallTaskState();
    return;
  }

  if (installed) {
    banner.classList.add('success');
    kicker.textContent = 'Textractor Ready';
    title.textContent = runtimeBlocked ? 'Textractor 已安装，等待读内存链路接管' : 'Textractor 已就绪';
    body.textContent = runtimeBlocked
      ? '插件已经检测到 TextractorCLI.exe。当前只要没有有效文件桥会话，Windows 读内存兜底就可以自动接管。'
      : 'Windows 读内存兜底所需的 TextractorCLI.exe 已检测到，文件桥缺席时可自动尝试接管。';
    path.textContent = textractor.detected_path ? `检测路径: ${textractor.detected_path}` : '';
    button.hidden = true;
  } else {
    kicker.textContent = 'Textractor Missing';
    title.textContent = runtimeBlocked
      ? '未检测到 Textractor，读内存兜底暂时不可用'
      : '尚未检测到 Textractor';
    body.textContent = runtimeBlocked
      ? '当前没有有效文件桥数据时，插件会尝试切到 Windows 读内存兜底，但缺少 TextractorCLI.exe。点击下方按钮可自动下载安装。'
      : '如果你希望在文件桥没有数据时由 Windows 读内存自动接管，可以直接在这里一键安装 Textractor。';
    path.textContent = textractor.expected_executable_path
      ? `预期安装位置: ${textractor.expected_executable_path}`
      : '';
  }

  if (installState && !isInstallTaskTerminal(installState)) {
    banner.classList.add('neutral');
    kicker.textContent = 'Textractor Install';
    title.textContent = 'Textractor 正在后台安装';
    body.textContent = '下载通过 HTTPS 进行，当前页面会通过 SSE 接收实时进度。就算页面刷新，也会尝试恢复到最近的安装状态。';
    button.hidden = false;
    button.disabled = true;
    button.textContent = '后台安装中...';
  } else if (installState && installState.status === 'failed' && installable) {
    banner.classList.add('neutral');
    kicker.textContent = 'Textractor Install';
    title.textContent = 'Textractor 安装失败，可直接重试';
    body.textContent = installState.error || installState.message || '后台安装任务失败，你可以再次点击按钮重试。';
    button.hidden = false;
    button.disabled = false;
    button.textContent = '重试安装 Textractor';
  } else if (installState && installState.status === 'completed' && !installed) {
    banner.classList.add('neutral');
    kicker.textContent = 'Textractor Install';
    title.textContent = 'Textractor 已安装完成，正在刷新插件状态';
    body.textContent = installState.message || '安装任务已结束，正在等待插件状态刷新。';
  }

  renderInstallTaskState();
}

function renderSnapshot(snapshot) {
  const state = snapshot.snapshot || {};
  renderGrid('snapshotGrid', [
    { label: 'game_id', value: snapshot.game_id || '' },
    { label: 'session_id', value: snapshot.session_id || '' },
    { label: 'speaker', value: state.speaker || '' },
    { label: 'text', value: state.text || '' },
    { label: 'scene_id', value: state.scene_id || '' },
    { label: 'line_id', value: state.line_id || '' },
    { label: 'route_id', value: state.route_id || '' },
    { label: 'is_menu_open', value: String(Boolean(state.is_menu_open)) },
    { label: 'snapshot_ts', value: snapshot.snapshot_ts || '' },
    { label: 'stale', value: String(Boolean(snapshot.stale)) },
  ]);
}

function renderHistory(history) {
  renderStackList('linesList', history.stable_lines || [], (item) => `
    <article class="list-card">
      <p class="list-kicker">${escapeHtml(item.speaker || '旁白')} · ${escapeHtml(item.scene_id || '')}</p>
      <h3>${escapeHtml(item.line_id || '')}</h3>
      <p>${escapeHtml(item.text || '')}</p>
    </article>
  `);

  renderStackList('choicesList', history.choices || [], (item) => `
    <article class="list-card">
      <p class="list-kicker">${escapeHtml(item.action || '')} · #${escapeHtml(item.index || 0)}</p>
      <h3>${escapeHtml(item.choice_id || '')}</h3>
      <p>${escapeHtml(item.text || '')}</p>
    </article>
  `);

  renderStackList('eventsList', history.events || [], (item) => `
    <article class="list-card compact">
      <p class="list-kicker">seq ${escapeHtml(item.seq || 0)} · ${escapeHtml(item.type || '')}</p>
      <h3>${escapeHtml(item.line_id || item.scene_id || '')}</h3>
      <p>${escapeHtml(JSON.stringify(item.payload || {}))}</p>
    </article>
  `);
}

function renderAgentStatus(payload) {
  document.getElementById('agentReplyText').textContent = latestAgentReply;
  renderGrid('agentStatusGrid', [
    { label: 'status', value: payload.status || 'standby' },
    { label: 'result', value: payload.result || '' },
    { label: 'recent_pushes', value: String((payload.recent_pushes || []).length) },
  ]);

  renderStackList('pushesList', payload.recent_pushes || [], (item) => `
    <article class="list-card compact">
      <p class="list-kicker">${escapeHtml(item.kind || '')} · ${escapeHtml(item.ts || '')}</p>
      <h3>${escapeHtml(item.scene_id || '')}</h3>
      <p>${escapeHtml(item.content || '')}</p>
    </article>
  `);
}

function renderExplain(payload) {
  const node = document.getElementById('explainPanel');
  node.className = 'result-panel';
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.line_id || '')} · degraded=${escapeHtml(Boolean(payload.degraded))}</p>
    <h3>${escapeHtml(payload.speaker || '旁白')}</h3>
    <p>${escapeHtml(payload.text || '')}</p>
    <p class="result-main">${escapeHtml(payload.explanation || payload.diagnostic || '暂无解释')}</p>
    <p class="result-note">${escapeHtml(payload.diagnostic || '')}</p>
  `;
}

function renderSummary(payload) {
  const node = document.getElementById('summaryPanel');
  node.className = 'result-panel';
  const points = payload.key_points || [];
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.scene_id || '')} · degraded=${escapeHtml(Boolean(payload.degraded))}</p>
    <p class="result-main">${escapeHtml(payload.summary || payload.diagnostic || '暂无总结')}</p>
    <p class="result-note">${escapeHtml(payload.diagnostic || '')}</p>
    <div class="chip-row">
      ${points.map((item) => `<span class="chip">${escapeHtml(item.type || '')}: ${escapeHtml(item.text || '')}</span>`).join('')}
    </div>
  `;
}

function renderSuggest(payload) {
  const node = document.getElementById('suggestPanel');
  node.className = 'result-panel';
  const choices = payload.choices || [];
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.scene_id || '')} · degraded=${escapeHtml(Boolean(payload.degraded))}</p>
    <div class="stack-list">
      ${choices.length ? choices.map((item) => `
        <article class="list-card compact">
          <p class="list-kicker">rank ${escapeHtml(item.rank || 0)} · ${escapeHtml(item.choice_id || '')}</p>
          <h3>${escapeHtml(item.text || '')}</h3>
          <p>${escapeHtml(item.reason || '')}</p>
        </article>
      `).join('') : `<div class="empty-inline">${escapeHtml(payload.diagnostic || '暂无建议')}</div>`}
    </div>
  `;
}

async function refreshAll() {
  setFlash('', 'info');
  try {
    const [status, snapshot, history, agentStatus] = await Promise.all([
      callPlugin('galgame_get_status'),
      callPlugin('galgame_get_snapshot'),
      callPlugin('galgame_get_history', { limit: 20, include_events: true }),
      safeCall('galgame_agent_command', { action: 'query_status' }, { action: 'query_status', result: '', status: 'standby', recent_pushes: [] }),
    ]);
    renderStatus(status);
    renderSnapshot(snapshot);
    renderHistory(history);
    renderAgentStatus(agentStatus);

    const currentLineId = snapshot.snapshot?.line_id || '';
    const currentSceneId = snapshot.snapshot?.scene_id || '';
    const hasChoices = Boolean(snapshot.snapshot?.is_menu_open) && Array.isArray(snapshot.snapshot?.choices) && snapshot.snapshot.choices.length > 0;
    const [explain, summary, suggest] = await Promise.all([
      currentLineId
        ? safeCall('galgame_explain_line', { line_id: currentLineId }, { line_id: currentLineId, speaker: '', text: '', explanation: '', evidence: [] })
        : Promise.resolve({ degraded: true, line_id: '', speaker: '', text: '', explanation: '', evidence: [], diagnostic: 'missing line_id' }),
      safeCall('galgame_summarize_scene', currentSceneId ? { scene_id: currentSceneId } : {}, { scene_id: currentSceneId, summary: '', key_points: [] }),
      hasChoices
        ? safeCall('galgame_suggest_choice', {}, { scene_id: currentSceneId, choices: [] })
        : Promise.resolve({ degraded: true, scene_id: currentSceneId, choices: [], diagnostic: 'no visible choices' }),
    ]);

    renderExplain(explain);
    renderSummary(summary);
    renderSuggest(suggest);
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function installTextractor(force = false) {
  const button = document.getElementById('textractorInstallBtn');
  textractorInstallInProgress = true;
  closeInstallStream();
  clearInstallReconnectTimer();
  button.disabled = true;
  button.textContent = '准备安装...';
  setFlash('已创建后台安装任务，接下来会通过 HTTPS 下载 Textractor，并通过 SSE 推送进度。', 'info');

  try {
    const response = await fetch(TEXTRACTOR_INSTALL_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force }),
    });
    if (!response.ok) {
      throw new Error(`创建 Textractor 安装任务失败: HTTP ${response.status}`);
    }

    const payload = await response.json();
    const taskId = payload.task_id || payload.run_id;
    if (!taskId) {
      throw new Error('未获取到 Textractor 安装 task_id');
    }

    currentInstallTaskId = taskId;
    persistInstallTaskId(taskId);
    if (payload.state) {
      applyInstallTaskState(payload.state, { allowRefresh: false });
    }
    connectInstallStream(taskId);

    const initialState = await fetchInstallTaskState(taskId);
    if (initialState) {
      applyInstallTaskState(initialState, { allowRefresh: false });
    }
  } catch (error) {
    textractorInstallInProgress = false;
    if (latestStatus) {
      renderTextractor(latestStatus);
    }
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function saveMode() {
  const mode = document.getElementById('modeSelect').value;
  const pushNotifications = document.getElementById('pushToggle').checked;
  try {
    await callPlugin('galgame_set_mode', {
      mode,
      push_notifications: pushNotifications,
    });
    setFlash('模式已保存', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function bindGame() {
  const gameId = document.getElementById('bindInput').value.trim();
  try {
    await callPlugin('galgame_bind_game', { game_id: gameId });
    setFlash(gameId ? `已绑定 ${gameId}` : '已恢复自动选择', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function setStandby(standby) {
  try {
    const payload = await callPlugin('galgame_agent_command', {
      action: 'set_standby',
      standby,
    });
    latestAgentReply = payload.result || latestAgentReply;
    setFlash(standby ? '已切换到待机' : '已恢复活跃', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function askAgent(action) {
  const prompt = document.getElementById('agentPromptInput').value.trim();
  if (!prompt) {
    setFlash('请输入要发送给 Agent 的文本', 'error');
    return;
  }

  try {
    const payload = await callPlugin('galgame_agent_command', action === 'query_context'
      ? { action, context_query: prompt }
      : { action, message: prompt });
    latestAgentReply = payload.result || 'Agent 未返回文本';
    setFlash('Agent 已响应', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function initialize() {
  await refreshAll();
  await restoreTextractorInstallState();
}

document.getElementById('refreshBtn').addEventListener('click', refreshAll);
document.getElementById('saveModeBtn').addEventListener('click', saveMode);
document.getElementById('bindBtn').addEventListener('click', bindGame);
document.getElementById('clearBindBtn').addEventListener('click', async () => {
  document.getElementById('bindInput').value = '';
  await bindGame();
});
document.getElementById('standbyOnBtn').addEventListener('click', () => setStandby(true));
document.getElementById('standbyOffBtn').addEventListener('click', () => setStandby(false));
document.getElementById('queryContextBtn').addEventListener('click', () => askAgent('query_context'));
document.getElementById('sendMessageBtn').addEventListener('click', () => askAgent('send_message'));
document.getElementById('textractorInstallBtn').addEventListener('click', () => installTextractor(false));

initialize();
