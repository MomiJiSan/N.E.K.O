const PLUGIN_ID = 'galgame_plugin';
const RUNS_URL = '/runs';
const UI_API_BASE = `/plugin/${PLUGIN_ID}/ui-api`;
const TESSERACT_INSTALL_URL = `${UI_API_BASE}/tesseract/install`;
const TEXTRACTOR_INSTALL_URL = `${UI_API_BASE}/textractor/install`;
const INSTALL_TERMINAL_STATUSES = new Set(['completed', 'failed', 'canceled']);

const INSTALL_UI = {
  tesseract: {
    kind: 'tesseract',
    label: 'Tesseract',
    url: TESSERACT_INSTALL_URL,
    storageKey: `${PLUGIN_ID}:tesseract_install_task_id`,
    domPrefix: 'tesseract',
    actionText: '一键安装 Tesseract',
    retryText: '重试安装 Tesseract',
    runningText: '后台安装中...',
    queuedFlash: '已创建后台安装任务，接下来会通过 HTTPS 下载 Tesseract 和语言包，并通过 SSE 推送实时进度。',
    successFlash: 'Tesseract 安装完成',
    failureFlash: 'Tesseract 安装失败',
  },
  textractor: {
    kind: 'textractor',
    label: 'Textractor',
    url: TEXTRACTOR_INSTALL_URL,
    storageKey: `${PLUGIN_ID}:textractor_install_task_id`,
    domPrefix: 'textractor',
    actionText: '一键安装 Textractor',
    retryText: '重试安装 Textractor',
    runningText: '后台安装中...',
    queuedFlash: '已创建后台安装任务，接下来会通过 HTTPS 下载 Textractor，并通过 SSE 推送实时进度。',
    successFlash: 'Textractor 安装完成',
    failureFlash: 'Textractor 安装失败',
  },
};

function createInstallRuntimeState() {
  return {
    state: null,
    inProgress: false,
    currentTaskId: '',
    eventSource: null,
    reconnectTimer: null,
    handledTerminalKey: '',
  };
}

const installRuntime = {
  tesseract: createInstallRuntimeState(),
  textractor: createInstallRuntimeState(),
};

const DEFAULT_CAPTURE_PROFILE = {
  left_inset_ratio: 0.05,
  right_inset_ratio: 0.05,
  top_ratio: 0.30,
  bottom_inset_ratio: 0.30,
};

let latestAgentReply = '暂无交互';
let latestStatus = null;

function getInstallConfig(kind) {
  const config = INSTALL_UI[kind];
  if (!config) {
    throw new Error(`unsupported install kind: ${kind}`);
  }
  return config;
}

function getInstallState(kind) {
  return installRuntime[kind];
}

function getInstallNodes(kind) {
  const prefix = getInstallConfig(kind).domPrefix;
  return {
    card: document.getElementById(`${prefix}InstallState`),
    statusText: document.getElementById(`${prefix}InstallStatusText`),
    percentText: document.getElementById(`${prefix}InstallPercent`),
    messageText: document.getElementById(`${prefix}InstallMessage`),
    detailText: document.getElementById(`${prefix}InstallDetail`),
    progressBar: document.getElementById(`${prefix}InstallBar`),
    button: document.getElementById(`${prefix}InstallBtn`),
  };
}

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
    return {
      ...fallback,
      degraded: true,
      diagnostic: error instanceof Error ? error.message : String(error),
    };
  }
}

function escapeHtml(text) {
  if (text == null) {
    return '';
  }
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

function persistInstallTaskId(kind, taskId) {
  if (!taskId) {
    return;
  }
  try {
    localStorage.setItem(getInstallConfig(kind).storageKey, taskId);
  } catch (_) {
    // Ignore storage failures in embedded browsers.
  }
}

function readPersistedInstallTaskId(kind) {
  try {
    return localStorage.getItem(getInstallConfig(kind).storageKey) || '';
  } catch (_) {
    return '';
  }
}

function clearInstallReconnectTimer(kind) {
  const state = getInstallState(kind);
  if (state.reconnectTimer) {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
}

function closeInstallStream(kind) {
  const state = getInstallState(kind);
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
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
    metadata: '获取安装信息',
    downloading: 'HTTPS 下载中',
    installing: '安装器执行中',
    extracting: '解压安装中',
    languages: '下载语言包中',
    verifying: '校验安装中',
    completed: '安装完成',
    failed: '安装失败',
    canceled: '已取消',
  };
  return mapping[normalized] || normalized || '等待中';
}

function formatCaptureProfile(profile) {
  const source = profile || {};
  const rows = [
    ['left', source.left_inset_ratio],
    ['right', source.right_inset_ratio],
    ['top', source.top_ratio],
    ['bottom', source.bottom_inset_ratio],
  ].filter(([, value]) => typeof value === 'number' && Number.isFinite(value));
  if (!rows.length) {
    return '';
  }
  return rows.map(([label, value]) => `${label}=${Number(value).toFixed(2)}`).join(' | ');
}

function setInputValueIfIdle(node, value) {
  if (!node) {
    return;
  }
  if (document.activeElement === node) {
    return;
  }
  node.value = value;
}

function profileValueForInputs(runtimeProfile) {
  const merged = {
    ...DEFAULT_CAPTURE_PROFILE,
    ...(runtimeProfile || {}),
  };
  return {
    left: Number(merged.left_inset_ratio).toFixed(2),
    right: Number(merged.right_inset_ratio).toFixed(2),
    top: Number(merged.top_ratio).toFixed(2),
    bottom: Number(merged.bottom_inset_ratio).toFixed(2),
  };
}

function renderInstallTaskState(kind) {
  const state = getInstallState(kind).state;
  const { card, statusText, percentText, messageText, detailText, progressBar } = getInstallNodes(kind);
  const { label } = getInstallConfig(kind);

  if (!state) {
    card.hidden = true;
    statusText.textContent = `等待 ${label} 安装任务`;
    percentText.textContent = '0%';
    messageText.textContent = '';
    detailText.textContent = '';
    progressBar.style.width = '0%';
    return;
  }

  const progress = Math.max(0, Math.min(1, Number(state.progress || 0)));
  const percent = Math.round(progress * 100);
  const details = [];
  if (state.total_bytes) {
    details.push(`${formatBytes(state.downloaded_bytes)} / ${formatBytes(state.total_bytes)}`);
  } else if (state.downloaded_bytes) {
    details.push(formatBytes(state.downloaded_bytes));
  }
  if (state.resume_from) {
    details.push(`续传自 ${formatBytes(state.resume_from)}`);
  }
  if (state.asset_name) {
    details.push(state.asset_name);
  }
  if (state.task_id) {
    details.push(`task ${state.task_id}`);
  }

  card.hidden = false;
  statusText.textContent = `${formatInstallPhase(state.phase)} · ${state.status || ''}`;
  percentText.textContent = `${percent}%`;
  messageText.textContent = state.message || '';
  detailText.textContent = details.join(' · ');
  progressBar.style.width = `${percent}%`;
}

function applyInstallTaskState(kind, state, { allowRefresh = true } = {}) {
  if (!state) {
    return;
  }
  const installState = getInstallState(kind);
  installState.state = state;
  installState.currentTaskId = state.task_id || state.run_id || installState.currentTaskId;
  if (installState.currentTaskId) {
    persistInstallTaskId(kind, installState.currentTaskId);
  }
  installState.inProgress = !isInstallTaskTerminal(state);

  if (latestStatus) {
    renderStatus(latestStatus);
  } else {
    renderInstallTaskState(kind);
  }

  if (!isInstallTaskTerminal(state)) {
    return;
  }

  closeInstallStream(kind);
  clearInstallReconnectTimer(kind);
  const terminalKey = `${installState.currentTaskId}:${state.status || ''}:${state.updated_at || ''}`;
  if (installState.handledTerminalKey === terminalKey) {
    return;
  }
  installState.handledTerminalKey = terminalKey;

  const config = getInstallConfig(kind);
  if (state.status === 'completed') {
    setFlash(state.message || config.successFlash, 'success');
  } else {
    setFlash(state.error || state.message || config.failureFlash, 'error');
  }

  if (allowRefresh) {
    refreshAll().catch((error) => {
      setFlash(error instanceof Error ? error.message : String(error), 'error');
    });
  }
}

async function fetchInstallTaskState(kind, taskId) {
  const response = await fetch(`${getInstallConfig(kind).url}/${encodeURIComponent(taskId)}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`读取 ${getInstallConfig(kind).label} 安装状态失败: HTTP ${response.status}`);
  }
  return await response.json();
}

async function fetchLatestInstallTaskState(kind) {
  const response = await fetch(`${getInstallConfig(kind).url}/latest`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`读取最近 ${getInstallConfig(kind).label} 安装状态失败: HTTP ${response.status}`);
  }
  return await response.json();
}

function scheduleInstallReconnect(kind, taskId) {
  const state = getInstallState(kind);
  clearInstallReconnectTimer(kind);
  state.reconnectTimer = setTimeout(async () => {
    try {
      const recovered = await fetchInstallTaskState(kind, taskId);
      if (recovered) {
        applyInstallTaskState(kind, recovered, { allowRefresh: false });
        if (!isInstallTaskTerminal(recovered)) {
          connectInstallStream(kind, taskId);
        }
        return;
      }
    } catch (_) {
      // Keep retrying until we observe a terminal state or the server becomes reachable again.
    }

    if (!state.state || !isInstallTaskTerminal(state.state)) {
      scheduleInstallReconnect(kind, taskId);
    }
  }, 1500);
}

function connectInstallStream(kind, taskId) {
  const state = getInstallState(kind);
  closeInstallStream(kind);
  clearInstallReconnectTimer(kind);
  const stream = new EventSource(`${getInstallConfig(kind).url}/${encodeURIComponent(taskId)}/stream`);
  state.eventSource = stream;

  stream.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      applyInstallTaskState(kind, payload);
    } catch (error) {
      setFlash(error instanceof Error ? error.message : String(error), 'error');
    }
  };

  stream.onerror = async () => {
    if (state.eventSource !== stream) {
      return;
    }
    stream.close();
    state.eventSource = null;
    if (state.state && isInstallTaskTerminal(state.state)) {
      return;
    }
    try {
      const recovered = await fetchInstallTaskState(kind, taskId);
      if (recovered) {
        applyInstallTaskState(kind, recovered, { allowRefresh: false });
      }
    } catch (_) {
      // Ignore transient recovery failures and retry shortly.
    }
    scheduleInstallReconnect(kind, taskId);
  };
}

async function restoreInstallState(kind) {
  const persistedTaskId = readPersistedInstallTaskId(kind);
  if (persistedTaskId) {
    try {
      const persistedState = await fetchInstallTaskState(kind, persistedTaskId);
      if (persistedState) {
        applyInstallTaskState(kind, persistedState, { allowRefresh: false });
        if (!isInstallTaskTerminal(persistedState)) {
          connectInstallStream(kind, persistedTaskId);
        }
        return;
      }
    } catch (_) {
      // Fall through to latest-task recovery.
    }
  }

  try {
    const latestState = await fetchLatestInstallTaskState(kind);
    if (!latestState) {
      return;
    }
    applyInstallTaskState(kind, latestState, { allowRefresh: false });
    const latestTaskId = latestState.task_id || latestState.run_id || '';
    if (latestTaskId && !isInstallTaskTerminal(latestState)) {
      connectInstallStream(kind, latestTaskId);
    }
  } catch (_) {
    // No active/recent install task to restore.
  }
}

async function restoreTextractorInstallState() {
  await restoreInstallState('textractor');
}

async function restoreTesseractInstallState() {
  await restoreInstallState('tesseract');
}

function renderStatus(status) {
  latestStatus = status;
  document.getElementById('summaryText').textContent = status.summary || '无摘要';
  document.getElementById('modeSelect').value = status.mode || 'companion';
  document.getElementById('pushToggle').checked = Boolean(status.push_notifications);
  document.getElementById('bindInput').value = status.bound_game_id || '';

  const memoryReaderRuntime = status.memory_reader_runtime || {};
  const ocrRuntime = status.ocr_reader_runtime || {};
  const textractor = status.textractor || {};
  const tesseract = status.tesseract || {};

  const memoryReaderProcess = memoryReaderRuntime.process_name
    ? `${memoryReaderRuntime.process_name} (${memoryReaderRuntime.pid || 0})`
    : '';
  const ocrTarget = ocrRuntime.process_name
    ? `${ocrRuntime.process_name} (${ocrRuntime.pid || 0})`
    : '';
  const missingLanguages = (tesseract.missing_languages || []).join(', ');

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
    { label: 'ocr_reader_enabled', value: String(Boolean(status.ocr_reader_enabled)) },
    { label: 'ocr_reader_status', value: ocrRuntime.status || '' },
    { label: 'ocr_reader_detail', value: ocrRuntime.detail || '' },
    { label: 'ocr_reader_target', value: ocrTarget || '' },
    { label: 'memory_reader_enabled', value: String(Boolean(status.memory_reader_enabled)) },
    { label: 'memory_reader_status', value: memoryReaderRuntime.status || '' },
    { label: 'memory_reader_detail', value: memoryReaderRuntime.detail || '' },
    { label: 'memory_reader_process', value: memoryReaderProcess || '' },
    { label: 'tesseract_installed', value: String(Boolean(tesseract.installed)) },
    { label: 'tesseract_detail', value: tesseract.detail || '' },
    { label: 'tesseract_missing_languages', value: missingLanguages || '(none)' },
    { label: 'textractor_installed', value: String(Boolean(textractor.installed)) },
    { label: 'textractor_detail', value: textractor.detail || '' },
    { label: 'last_error', value: status.last_error?.message || '' },
  ]);

  renderOcrRuntime(status);
  renderTesseract(status);
  renderTextractor(status);
  renderOcrProfile(status);
}

function renderOcrRuntime(status) {
  const runtime = status.ocr_reader_runtime || {};
  renderGrid('ocrRuntimeGrid', [
    { label: 'status', value: runtime.status || '' },
    { label: 'detail', value: runtime.detail || '' },
    { label: 'process_name', value: runtime.process_name || '' },
    { label: 'pid', value: String(runtime.pid || 0) },
    { label: 'window_title', value: runtime.window_title || '' },
    { label: 'game_id', value: runtime.game_id || '' },
    { label: 'session_id', value: runtime.session_id || '' },
    { label: 'last_seq', value: String(runtime.last_seq || 0) },
    { label: 'last_event_ts', value: runtime.last_event_ts || '' },
    { label: 'capture_profile', value: formatCaptureProfile(runtime.capture_profile) || '(default)' },
    { label: 'tesseract_path', value: runtime.tesseract_path || '' },
    { label: 'languages', value: runtime.languages || '' },
    { label: 'takeover_reason', value: runtime.takeover_reason || '' },
  ]);
}

function renderTesseract(status) {
  const tesseract = status.tesseract || {};
  const runtime = status.ocr_reader_runtime || {};
  const banner = document.getElementById('tesseractPrompt');
  const kicker = document.getElementById('tesseractPromptKicker');
  const title = document.getElementById('tesseractPromptTitle');
  const body = document.getElementById('tesseractPromptBody');
  const path = document.getElementById('tesseractPathText');
  const button = document.getElementById('tesseractInstallBtn');
  const installState = getInstallState('tesseract').state;
  const installable = Boolean(tesseract.install_supported) && Boolean(tesseract.can_install);
  const installed = Boolean(tesseract.installed);
  const missingLanguages = tesseract.missing_languages || [];

  banner.hidden = false;
  banner.className = 'install-banner install-banner-tesseract';
  button.hidden = !installable;
  button.disabled = getInstallState('tesseract').inProgress;
  button.textContent = getInstallState('tesseract').inProgress
    ? getInstallConfig('tesseract').runningText
    : getInstallConfig('tesseract').actionText;

  if (!tesseract.install_supported) {
    banner.classList.add('neutral');
    kicker.textContent = 'OCR 主 fallback';
    title.textContent = '当前平台暂不支持自动安装 Tesseract';
    body.textContent = 'OCR Reader 目前固定为 Windows-only，本地 Tesseract 也只在 Windows 上提供自动安装入口。';
    path.textContent = '';
    button.hidden = true;
    renderInstallTaskState('tesseract');
    return;
  }

  if (installed) {
    banner.classList.add('success');
    kicker.textContent = 'OCR Ready';
    title.textContent = runtime.status === 'active'
      ? 'Tesseract 已就绪，OCR 正在接管'
      : 'Tesseract 已就绪，等待 OCR 接管';
    body.textContent = runtime.detail === 'bridge_sdk_available'
      ? '当前 Bridge SDK 可用，OCR Reader 保持热备，不会抢占主桥接。'
      : runtime.detail === 'memory_reader_active'
        ? '当前 Memory Reader 仍有有效文本，OCR Reader 保持热备，等待更合适的接管时机。'
        : '本地 Tesseract 与默认语言包 chi_sim+jpn+eng 已齐全。无 SDK 且无有效内存文本时，OCR Reader 会作为当前正式主 fallback 接管。';
    path.textContent = tesseract.detected_path
      ? `检测路径: ${tesseract.detected_path}`
      : '';
    button.hidden = true;
  } else if (tesseract.detail === 'missing_languages') {
    banner.classList.add('warning');
    kicker.textContent = 'OCR Missing Languages';
    title.textContent = '已检测到 Tesseract，但语言包不完整';
    body.textContent = `当前缺少 ${(missingLanguages || []).join(', ') || '语言包'}，OCR 主 fallback 暂不可用。安装流程会按默认语言 chi_sim+jpn+eng 补齐运行所需文件。`;
    path.textContent = tesseract.tessdata_dir
      ? `tessdata 目录: ${tesseract.tessdata_dir}`
      : '';
  } else {
    banner.classList.add('warning');
    kicker.textContent = 'OCR Missing';
    title.textContent = '未检测到 Tesseract，OCR 主 fallback 尚未就绪';
    body.textContent = '当前无 SDK 场景下的正式主 fallback 是 OCR Reader。安装本地 Tesseract 后，插件才会进入窗口截图 + OCR 接管链路。';
    path.textContent = tesseract.expected_executable_path
      ? `预期安装位置: ${tesseract.expected_executable_path}`
      : '';
  }

  if (installState && !isInstallTaskTerminal(installState)) {
    banner.className = 'install-banner install-banner-tesseract neutral';
    kicker.textContent = 'Tesseract Install';
    title.textContent = 'Tesseract 正在后台安装';
    body.textContent = '安装器和语言包下载都通过 HTTPS 进行，当前页面会通过 SSE 接收实时进度；即使刷新页面，也会尝试恢复最近的安装状态。';
    button.hidden = false;
    button.disabled = true;
    button.textContent = getInstallConfig('tesseract').runningText;
  } else if (installState && installState.status === 'failed' && installable) {
    banner.className = 'install-banner install-banner-tesseract neutral';
    kicker.textContent = 'Tesseract Install';
    title.textContent = 'Tesseract 安装失败，可直接重试';
    body.textContent = installState.error || installState.message || '后台安装任务失败，你可以再次点击按钮重试。';
    button.hidden = false;
    button.disabled = false;
    button.textContent = getInstallConfig('tesseract').retryText;
  } else if (installState && installState.status === 'completed' && !installed) {
    banner.className = 'install-banner install-banner-tesseract neutral';
    kicker.textContent = 'Tesseract Install';
    title.textContent = 'Tesseract 安装已完成，正在刷新 OCR 状态';
    body.textContent = installState.message || '安装任务已结束，正在等待插件状态刷新。';
  }

  renderInstallTaskState('tesseract');
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
  const installState = getInstallState('textractor').state;
  const installable = Boolean(textractor.install_supported) && Boolean(textractor.can_install);
  const installed = Boolean(textractor.installed);
  const runtimeBlocked = runtime.detail === 'invalid_textractor_path';

  banner.hidden = false;
  banner.className = 'install-banner install-banner-textractor';
  button.hidden = !installable;
  button.disabled = getInstallState('textractor').inProgress;
  button.textContent = getInstallState('textractor').inProgress
    ? getInstallConfig('textractor').runningText
    : getInstallConfig('textractor').actionText;

  if (!textractor.install_supported) {
    banner.classList.add('neutral');
    kicker.textContent = '实验性兜底';
    title.textContent = '当前平台无需 Textractor 自动安装';
    body.textContent = 'Textractor 读内存兜底仅在 Windows 上启用，而且当前优先级已经低于 OCR Reader。';
    path.textContent = '';
    button.hidden = true;
    renderInstallTaskState('textractor');
    return;
  }

  if (installed) {
    banner.classList.add('success');
    kicker.textContent = 'Experimental Fallback';
    title.textContent = runtimeBlocked
      ? 'Textractor 已安装，等待 Memory Reader 手动/实验性接管'
      : 'Textractor 已就绪，但仅作为实验性兜底';
    body.textContent = runtimeBlocked
      ? '当前已检测到 TextractorCLI.exe。它仍保留在链路中，但优先级固定低于 OCR Reader，不再是当前首发主验收线。'
      : 'TextractorCLI.exe 已检测到。Memory Reader 仍然保留，但不再作为主叙事，也不会压过 OCR Reader。';
    path.textContent = textractor.detected_path ? `检测路径: ${textractor.detected_path}` : '';
    button.hidden = true;
  } else {
    banner.classList.add('neutral');
    kicker.textContent = 'Experimental Fallback';
    title.textContent = runtimeBlocked
      ? '未检测到 Textractor，Memory Reader 实验性 fallback 暂不可用'
      : '尚未检测到 Textractor';
    body.textContent = runtimeBlocked
      ? '如果你后续仍想继续尝试 Memory Reader，可以在这里补装 Textractor；但当前正式主 fallback 仍然是 OCR Reader。'
      : 'Textractor 仅影响实验性 Memory Reader 链路，不影响当前 Bridge SDK > OCR Reader 的正式运行顺序。';
    path.textContent = textractor.expected_executable_path
      ? `预期安装位置: ${textractor.expected_executable_path}`
      : '';
  }

  if (installState && !isInstallTaskTerminal(installState)) {
    banner.className = 'install-banner install-banner-textractor neutral';
    kicker.textContent = 'Textractor Install';
    title.textContent = 'Textractor 正在后台安装';
    body.textContent = '下载通过 HTTPS 进行，页面会通过 SSE 接收实时进度。Textractor 完成后只会补强实验性 Memory Reader 路径。';
    button.hidden = false;
    button.disabled = true;
    button.textContent = getInstallConfig('textractor').runningText;
  } else if (installState && installState.status === 'failed' && installable) {
    banner.className = 'install-banner install-banner-textractor neutral';
    kicker.textContent = 'Textractor Install';
    title.textContent = 'Textractor 安装失败，可直接重试';
    body.textContent = installState.error || installState.message || '后台安装任务失败，你可以再次点击按钮重试。';
    button.hidden = false;
    button.disabled = false;
    button.textContent = getInstallConfig('textractor').retryText;
  } else if (installState && installState.status === 'completed' && !installed) {
    banner.className = 'install-banner install-banner-textractor neutral';
    kicker.textContent = 'Textractor Install';
    title.textContent = 'Textractor 安装已完成，正在刷新插件状态';
    body.textContent = installState.message || '安装任务已结束，正在等待插件状态刷新。';
  }

  renderInstallTaskState('textractor');
}

function renderOcrProfile(status) {
  const runtime = status.ocr_reader_runtime || {};
  const processInput = document.getElementById('ocrProfileProcessInput');
  const leftInput = document.getElementById('ocrProfileLeftInput');
  const rightInput = document.getElementById('ocrProfileRightInput');
  const topInput = document.getElementById('ocrProfileTopInput');
  const bottomInput = document.getElementById('ocrProfileBottomInput');
  const hint = document.getElementById('ocrProfileRuntimeHint');
  const profileValues = profileValueForInputs(runtime.capture_profile);

  if (runtime.process_name) {
    hint.textContent = [
      `当前 OCR 目标: ${runtime.process_name} (${runtime.pid || 0})`,
      runtime.window_title ? `窗口: ${runtime.window_title}` : '',
      runtime.detail ? `状态: ${runtime.detail}` : '',
      runtime.takeover_reason ? `接管原因: ${runtime.takeover_reason}` : '',
    ].filter(Boolean).join(' | ');
  } else {
    hint.textContent = '当前还没有附着的 OCR 目标进程。你也可以先手动填写 process_name，把截图校准预先存起来。';
  }

  if (!processInput.value || document.activeElement !== processInput) {
    setInputValueIfIdle(processInput, runtime.process_name || processInput.value);
  }
  setInputValueIfIdle(leftInput, profileValues.left);
  setInputValueIfIdle(rightInput, profileValues.right);
  setInputValueIfIdle(topInput, profileValues.top);
  setInputValueIfIdle(bottomInput, profileValues.bottom);
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
      safeCall(
        'galgame_agent_command',
        { action: 'query_status' },
        { action: 'query_status', result: '', status: 'standby', recent_pushes: [] },
      ),
    ]);
    renderStatus(status);
    renderSnapshot(snapshot);
    renderHistory(history);
    renderAgentStatus(agentStatus);

    const currentLineId = snapshot.snapshot?.line_id || '';
    const currentSceneId = snapshot.snapshot?.scene_id || '';
    const hasChoices = Boolean(snapshot.snapshot?.is_menu_open)
      && Array.isArray(snapshot.snapshot?.choices)
      && snapshot.snapshot.choices.length > 0;
    const [explain, summary, suggest] = await Promise.all([
      currentLineId
        ? safeCall(
          'galgame_explain_line',
          { line_id: currentLineId },
          { line_id: currentLineId, speaker: '', text: '', explanation: '', evidence: [] },
        )
        : Promise.resolve({
          degraded: true,
          line_id: '',
          speaker: '',
          text: '',
          explanation: '',
          evidence: [],
          diagnostic: 'missing line_id',
        }),
      safeCall(
        'galgame_summarize_scene',
        currentSceneId ? { scene_id: currentSceneId } : {},
        { scene_id: currentSceneId, summary: '', key_points: [] },
      ),
      hasChoices
        ? safeCall('galgame_suggest_choice', {}, { scene_id: currentSceneId, choices: [] })
        : Promise.resolve({
          degraded: true,
          scene_id: currentSceneId,
          choices: [],
          diagnostic: 'no visible choices',
        }),
    ]);

    renderExplain(explain);
    renderSummary(summary);
    renderSuggest(suggest);
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function startInstall(kind, force = false) {
  const config = getInstallConfig(kind);
  const state = getInstallState(kind);
  const { button } = getInstallNodes(kind);
  state.inProgress = true;
  closeInstallStream(kind);
  clearInstallReconnectTimer(kind);
  button.disabled = true;
  button.textContent = '准备安装...';
  setFlash(config.queuedFlash, 'info');

  try {
    const response = await fetch(config.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force }),
    });
    if (!response.ok) {
      throw new Error(`创建 ${config.label} 安装任务失败: HTTP ${response.status}`);
    }

    const payload = await response.json();
    const taskId = payload.task_id || payload.run_id;
    if (!taskId) {
      throw new Error(`未获取到 ${config.label} 安装 task_id`);
    }

    state.currentTaskId = taskId;
    persistInstallTaskId(kind, taskId);
    if (payload.state) {
      applyInstallTaskState(kind, payload.state, { allowRefresh: false });
    }
    connectInstallStream(kind, taskId);

    const initialState = await fetchInstallTaskState(kind, taskId);
    if (initialState) {
      applyInstallTaskState(kind, initialState, { allowRefresh: false });
    }
  } catch (error) {
    state.inProgress = false;
    if (latestStatus) {
      renderStatus(latestStatus);
    } else {
      renderInstallTaskState(kind);
    }
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function installTextractor(force = false) {
  await startInstall('textractor', force);
}

async function installTesseract(force = false) {
  await startInstall('tesseract', force);
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
    const payload = await callPlugin(
      'galgame_agent_command',
      action === 'query_context'
        ? { action, context_query: prompt }
        : { action, message: prompt },
    );
    latestAgentReply = payload.result || 'Agent 未返回文本';
    setFlash('Agent 已响应', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

function readProfileNumber(id, label) {
  const raw = document.getElementById(id).value.trim();
  const value = Number(raw);
  if (!raw) {
    throw new Error(`${label} 不能为空`);
  }
  if (!Number.isFinite(value)) {
    throw new Error(`${label} 必须是数字`);
  }
  if (value < 0 || value >= 1) {
    throw new Error(`${label} 必须在 0.00 到 0.99 之间`);
  }
  return value;
}

async function saveOcrCaptureProfile() {
  try {
    const processName = document.getElementById('ocrProfileProcessInput').value.trim();
    const leftInsetRatio = readProfileNumber('ocrProfileLeftInput', 'left_inset_ratio');
    const rightInsetRatio = readProfileNumber('ocrProfileRightInput', 'right_inset_ratio');
    const topRatio = readProfileNumber('ocrProfileTopInput', 'top_ratio');
    const bottomInsetRatio = readProfileNumber('ocrProfileBottomInput', 'bottom_inset_ratio');
    const payload = await callPlugin('galgame_set_ocr_capture_profile', {
      process_name: processName,
      left_inset_ratio: leftInsetRatio,
      right_inset_ratio: rightInsetRatio,
      top_ratio: topRatio,
      bottom_inset_ratio: bottomInsetRatio,
      clear: false,
    });
    setFlash(payload.summary || 'OCR 截图校准已保存', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function clearOcrCaptureProfile() {
  try {
    const processName = document.getElementById('ocrProfileProcessInput').value.trim();
    const payload = await callPlugin('galgame_set_ocr_capture_profile', {
      process_name: processName,
      clear: true,
    });
    setFlash(payload.summary || 'OCR 截图校准已清空', 'success');
    await refreshAll();
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function initialize() {
  await refreshAll();
  await Promise.all([
    restoreTesseractInstallState(),
    restoreTextractorInstallState(),
  ]);
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
document.getElementById('tesseractInstallBtn').addEventListener('click', () => installTesseract(false));
document.getElementById('textractorInstallBtn').addEventListener('click', () => installTextractor(false));
document.getElementById('ocrProfileSaveBtn').addEventListener('click', saveOcrCaptureProfile);
document.getElementById('ocrProfileClearBtn').addEventListener('click', clearOcrCaptureProfile);

initialize();
