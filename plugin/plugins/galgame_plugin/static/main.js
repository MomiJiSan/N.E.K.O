const PLUGIN_ID = 'galgame_plugin';
const RUNS_URL = '/runs';
const UI_API_BASE = `/plugin/${PLUGIN_ID}/ui-api`;
const RAPIDOCR_INSTALL_URL = `${UI_API_BASE}/rapidocr/install`;
const TESSERACT_INSTALL_URL = `${UI_API_BASE}/tesseract/install`;
const TEXTRACTOR_INSTALL_URL = `${UI_API_BASE}/textractor/install`;
const INSTALL_TERMINAL_STATUSES = new Set(['completed', 'failed', 'canceled']);

const INSTALL_UI = {
  rapidocr: {
    kind: 'rapidocr',
    label: 'RapidOCR',
    url: RAPIDOCR_INSTALL_URL,
    storageKey: `${PLUGIN_ID}:rapidocr_install_task_id`,
    domPrefix: 'rapidocr',
    actionText: '一键安装 RapidOCR',
    retryText: '重试安装 RapidOCR',
    runningText: '后台安装中...',
    queuedFlash: '已创建后台安装任务，接下来会安装插件隔离的 RapidOCR 运行时，并通过 SSE 推送实时进度。',
    successFlash: 'RapidOCR 安装完成',
    failureFlash: 'RapidOCR 安装失败',
  },
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
  rapidocr: createInstallRuntimeState(),
  tesseract: createInstallRuntimeState(),
  textractor: createInstallRuntimeState(),
};

const DEFAULT_CAPTURE_PROFILE = {
  left_inset_ratio: 0.05,
  right_inset_ratio: 0.05,
  top_ratio: 0.62,
  bottom_inset_ratio: 0.08,
};

const AUTO_REFRESH_INTERVAL_MS = 3000;
const FIELD_LABELS_ZH = {
  connection_state: '连接状态',
  active_data_source: '当前数据源',
  mode: '模式',
  push_notifications: '推送通知',
  bound_game_id: '绑定游戏 ID',
  active_session_id: '当前会话 ID',
  last_seq: '最新序号',
  stream_reset_pending: '等待重置流',
  available_game_ids: '可用游戏 ID',
  ocr_reader_enabled: 'OCR Reader 已启用',
  ocr_reader_status: 'OCR Reader 状态',
  ocr_reader_detail: 'OCR Reader 详情',
  ocr_reader_target: 'OCR Reader 目标',
  ocr_backend_kind: 'OCR 后端类型',
  ocr_backend_detail: 'OCR 后端详情',
  rapidocr_enabled: 'RapidOCR 已启用',
  rapidocr_installed: 'RapidOCR 已安装',
  rapidocr_detail: 'RapidOCR 详情',
  memory_reader_enabled: 'Memory Reader 已启用',
  memory_reader_status: 'Memory Reader 状态',
  memory_reader_detail: 'Memory Reader 详情',
  memory_reader_process: 'Memory Reader 进程',
  tesseract_installed: 'Tesseract 已安装',
  tesseract_detail: 'Tesseract 详情',
  tesseract_missing_languages: 'Tesseract 缺失语言',
  textractor_installed: 'Textractor 已安装',
  textractor_detail: 'Textractor 详情',
  last_error: '最近错误',
  status: '状态',
  detail: '详情',
  process_name: '进程名',
  pid: '进程 ID',
  window_title: '窗口标题',
  game_id: '游戏 ID',
  session_id: '会话 ID',
  last_event_ts: '最近事件时间',
  capture_profile: '截图配置',
  backend_kind: '后端类型',
  backend_detail: '后端详情',
  backend_path: '后端路径',
  backend_model: '后端模型',
  tesseract_path: 'Tesseract 路径',
  languages: '语言',
  takeover_reason: '接管原因',
  target_selection_mode: '目标选择模式',
  target_selection_detail: '目标选择详情',
  effective_window_key: '生效窗口键',
  effective_window_title: '生效窗口标题',
  effective_process_name: '生效进程名',
  candidate_count: '候选窗口数',
  excluded_candidate_count: '排除窗口数',
  last_exclude_reason: '最近排除原因',
  speaker: '说话人',
  text: '文本',
  scene_id: '场景 ID',
  line_id: '台词 ID',
  route_id: '路线 ID',
  is_menu_open: '菜单是否打开',
  snapshot_ts: '快照时间',
  stale: '是否过期',
  result: '结果',
  recent_pushes: '最近推送数',
  activity: '活动',
  reason: '原因',
  input_source: '输入源',
  scene_stage: '场景阶段',
  push_policy: '推送策略',
  actionable: '可操作',
  standby_requested: '已请求待机',
  memory_counts: '记忆计数',
};

let latestAgentReply = '暂无交互';
let latestStatus = null;
let latestOcrWindowSnapshot = null;
let refreshInFlight = null;
let autoRefreshTimer = null;

const latestInsights = {
  explainKey: '',
  explainPayload: null,
  summaryKey: '',
  summaryPayload: null,
  suggestKey: '',
  suggestPayload: null,
};

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

function buildExplainFallback(lineId = '', diagnostic = 'missing line_id') {
  return {
    degraded: true,
    line_id: lineId,
    speaker: '',
    text: '',
    explanation: '',
    evidence: [],
    diagnostic,
  };
}

function buildSummaryFallback(sceneId = '', diagnostic = 'missing scene_id') {
  return {
    degraded: true,
    scene_id: sceneId,
    summary: '',
    key_points: [],
    diagnostic,
  };
}

function buildSuggestFallback(sceneId = '', diagnostic = 'no visible choices') {
  return {
    degraded: true,
    scene_id: sceneId,
    choices: [],
    diagnostic,
  };
}

function renderGrid(nodeId, rows) {
  const container = document.getElementById(nodeId);
  container.innerHTML = rows.map((row) => `
    <div class="data-row">
      <dt>${escapeHtml(FIELD_LABELS_ZH[row.label] || row.label)}</dt>
      <dd>${escapeHtml(row.value)}</dd>
    </div>
  `).join('');
}

function renderStackList(nodeId, items, formatter) {
  const node = document.getElementById(nodeId);
  if (!items.length) {
    node.className = 'stack-list scroll-region empty-state';
    node.textContent = '暂无数据';
    return;
  }
  node.className = 'stack-list scroll-region';
  node.innerHTML = items.map(formatter).join('');
}

function isInstallTaskTerminal(state) {
  return Boolean(state) && INSTALL_TERMINAL_STATUSES.has(String(state.status || ''));
}

function installStatusPriority(status) {
  const normalized = String(status || '').trim();
  if (normalized === 'completed') {
    return 3;
  }
  if (normalized === 'failed') {
    return 2;
  }
  if (normalized === 'canceled') {
    return 1;
  }
  return 0;
}

function selectPreferredInstallState(primary, secondary) {
  if (!primary) {
    return secondary;
  }
  if (!secondary) {
    return primary;
  }

  const primaryTerminal = isInstallTaskTerminal(primary);
  const secondaryTerminal = isInstallTaskTerminal(secondary);
  if (primaryTerminal !== secondaryTerminal) {
    return primaryTerminal ? secondary : primary;
  }

  const primaryUpdated = Number(primary.updated_at || primary.started_at || 0);
  const secondaryUpdated = Number(secondary.updated_at || secondary.started_at || 0);
  if (primaryUpdated !== secondaryUpdated) {
    return primaryUpdated >= secondaryUpdated ? primary : secondary;
  }

  const primaryPriority = installStatusPriority(primary.status);
  const secondaryPriority = installStatusPriority(secondary.status);
  if (primaryPriority !== secondaryPriority) {
    return primaryPriority >= secondaryPriority ? primary : secondary;
  }

  return primary;
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

function clearPersistedInstallTaskId(kind) {
  try {
    localStorage.removeItem(getInstallConfig(kind).storageKey);
  } catch (_) {
    // Ignore storage failures in embedded browsers.
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
    if (isInstallTaskTerminal(state)) {
      clearPersistedInstallTaskId(kind);
    } else {
      persistInstallTaskId(kind, installState.currentTaskId);
    }
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
    refreshAll({ preserveFlash: true, forceInsights: true }).catch((error) => {
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
  let persistedState = null;
  let latestState = null;

  if (persistedTaskId) {
    try {
      persistedState = await fetchInstallTaskState(kind, persistedTaskId);
      if (!persistedState) {
        clearPersistedInstallTaskId(kind);
      }
    } catch (_) {
      persistedState = null;
    }
  }

  try {
    latestState = await fetchLatestInstallTaskState(kind);
  } catch (_) {
    latestState = null;
  }

  const restoredState = selectPreferredInstallState(persistedState, latestState);
  if (!restoredState) {
    return;
  }

  applyInstallTaskState(kind, restoredState, { allowRefresh: false });
  const restoredTaskId = restoredState.task_id || restoredState.run_id || '';
  if (restoredTaskId && !isInstallTaskTerminal(restoredState)) {
    connectInstallStream(kind, restoredTaskId);
  }
}

async function restoreTextractorInstallState() {
  await restoreInstallState('textractor');
}

async function restoreRapidOcrInstallState() {
  await restoreInstallState('rapidocr');
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
  const rapidocr = status.rapidocr || {};
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
    { label: 'ocr_backend_kind', value: ocrRuntime.backend_kind || '' },
    { label: 'ocr_backend_detail', value: ocrRuntime.backend_detail || '' },
    { label: 'rapidocr_enabled', value: String(Boolean(status.rapidocr_enabled)) },
    { label: 'rapidocr_installed', value: String(Boolean(rapidocr.installed)) },
    { label: 'rapidocr_detail', value: rapidocr.detail || '' },
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
  renderRapidOcr(status);
  renderTesseract(status);
  renderTextractor(status);
  renderOcrWindowTargetStatus(status);
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
    { label: 'backend_kind', value: runtime.backend_kind || '' },
    { label: 'backend_detail', value: runtime.backend_detail || '' },
    { label: 'backend_path', value: runtime.backend_path || '' },
    { label: 'backend_model', value: runtime.backend_model || '' },
    { label: 'tesseract_path', value: runtime.tesseract_path || '' },
    { label: 'languages', value: runtime.languages || '' },
    { label: 'takeover_reason', value: runtime.takeover_reason || '' },
    { label: 'target_selection_mode', value: runtime.target_selection_mode || '' },
    { label: 'target_selection_detail', value: runtime.target_selection_detail || '' },
    { label: 'effective_window_key', value: runtime.effective_window_key || '' },
    { label: 'effective_window_title', value: runtime.effective_window_title || '' },
    { label: 'effective_process_name', value: runtime.effective_process_name || '' },
    { label: 'candidate_count', value: String(runtime.candidate_count || 0) },
    { label: 'excluded_candidate_count', value: String(runtime.excluded_candidate_count || 0) },
    { label: 'last_exclude_reason', value: runtime.last_exclude_reason || '' },
  ]);
}

function formatOcrWindowReason(reason) {
  const mapping = {
    excluded_self_window: '已排除 N.E.K.O 自身窗口',
    excluded_overlay_window: '已排除 overlay / launcher / helper',
    excluded_helper_window: '已排除系统或宿主辅助窗口',
    excluded_small_or_hidden_window: '已排除过小或不可用窗口',
  };
  return mapping[reason] || reason || 'unknown';
}

function formatOcrWindowSelectionDetail(detail) {
  const mapping = {
    auto_candidate_scan: '自动扫描可用窗口',
    manual_target_active: '手动锁定已启用',
    manual_target_exact: '命中手动锁定窗口',
    manual_target_rebound: '已按签名重新绑定手动窗口',
    manual_target_unavailable_fallback_to_auto: '手动窗口不可用，已回退到自动选择',
    no_eligible_window: '当前没有可用游戏窗口',
    memory_reader_pid: '优先沿用 Memory Reader 命中的 PID',
    memory_reader_process: '优先沿用 Memory Reader 命中的进程',
    attached_hwnd: '优先复用当前已附着窗口',
    attached_pid: '优先复用当前已附着进程',
    foreground_window: '优先使用当前前台候选窗口',
    scored_candidate: '按候选排序选择窗口',
  };
  return mapping[detail] || detail || '';
}

function renderOcrWindowTargetStatus(status) {
  const runtime = status.ocr_reader_runtime || {};
  const snapshot = latestOcrWindowSnapshot || {};
  const modeText = document.getElementById('ocrWindowTargetModeText');
  const hint = document.getElementById('ocrWindowRuntimeHint');
  const autoButton = document.getElementById('ocrWindowAutoBtn');
  const mode = runtime.target_selection_mode || snapshot.target_selection_mode || 'auto';
  const manualTarget = runtime.manual_target || snapshot.manual_target || {};
  const effectiveTitle = runtime.effective_window_title || runtime.window_title || '';
  const effectiveProcess = runtime.effective_process_name || runtime.process_name || '';
  const detail = formatOcrWindowSelectionDetail(runtime.target_selection_detail || '');
  const hintParts = [
    effectiveProcess ? `当前目标: ${effectiveProcess}${runtime.pid ? ` (${runtime.pid})` : ''}` : '',
    effectiveTitle ? `窗口: ${effectiveTitle}` : '',
    detail,
    runtime.last_exclude_reason ? `最近排除: ${formatOcrWindowReason(runtime.last_exclude_reason)}` : '',
  ].filter(Boolean);

  modeText.textContent = mode === 'manual'
    ? `当前模式: manual${manualTarget.process_name ? ` | 锁定 ${manualTarget.process_name}` : ''}`
    : '当前模式: auto';
  hint.textContent = hintParts.join(' | ') || '等待 OCR 窗口候选列表。';
  autoButton.disabled = mode !== 'manual';
}

function renderOcrWindowTargetSnapshot(snapshot, status = latestStatus) {
  latestOcrWindowSnapshot = snapshot;
  const runtime = (status || {}).ocr_reader_runtime || {};
  const node = document.getElementById('ocrWindowList');
  const excludedNode = document.getElementById('ocrExcludedWindowList');
  const windows = snapshot.windows || [];
  const excludedWindows = snapshot.excluded_windows || [];

  if (!windows.length) {
    node.className = 'stack-list scroll-region empty-state window-candidate-list';
    node.textContent = '暂无可用游戏窗口';
  } else {
    node.className = 'stack-list scroll-region window-candidate-list';
    node.innerHTML = windows.map((item) => {
      const chips = [
        item.is_attached ? '<span class="status-chip active">当前附着</span>' : '',
        item.is_foreground ? '<span class="status-chip">前台窗口</span>' : '',
        item.is_manual_target ? '<span class="status-chip active">手动锁定</span>' : '',
      ].filter(Boolean).join('');
      return `
        <article class="list-card compact">
          <p class="list-kicker">${escapeHtml(item.process_name || '未知进程')} · pid ${escapeHtml(item.pid || 0)}</p>
          <h3>${escapeHtml(item.title || '未命名窗口')}</h3>
          <p class="result-note mono">${escapeHtml(item.window_key || '')}</p>
          <div class="window-candidate-actions">
            <div class="window-candidate-meta">${chips}</div>
            <button class="secondary" data-window-key="${escapeHtml(item.window_key || '')}">锁定此窗口</button>
          </div>
        </article>
      `;
    }).join('');
    node.querySelectorAll('[data-window-key]').forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.getAttribute('data-window-key') || '';
        setOcrWindowTarget(key).catch((error) => {
          setFlash(error instanceof Error ? error.message : String(error), 'error');
        });
      });
    });
  }

  if (!excludedWindows.length) {
    excludedNode.className = 'stack-list scroll-region empty-state window-candidate-list';
    excludedNode.textContent = '暂无排除窗口';
  } else {
    excludedNode.className = 'stack-list scroll-region window-candidate-list';
    excludedNode.innerHTML = excludedWindows.map((item) => `
      <article class="list-card compact">
        <p class="list-kicker">${escapeHtml(item.process_name || '未知进程')} · ${escapeHtml(formatOcrWindowReason(item.exclude_reason || ''))}</p>
        <h3>${escapeHtml(item.title || '未命名窗口')}</h3>
        <p class="result-note mono">${escapeHtml(item.window_key || '')}</p>
      </article>
    `).join('');
  }

  renderOcrWindowTargetStatus(status || { ocr_reader_runtime: runtime });
}

function renderRapidOcr(status) {
  const rapidocr = status.rapidocr || {};
  const runtime = status.ocr_reader_runtime || {};
  const banner = document.getElementById('rapidocrPrompt');
  const kicker = document.getElementById('rapidocrPromptKicker');
  const title = document.getElementById('rapidocrPromptTitle');
  const body = document.getElementById('rapidocrPromptBody');
  const path = document.getElementById('rapidocrPathText');
  const button = document.getElementById('rapidocrInstallBtn');
  const installState = getInstallState('rapidocr').state;
  const installable = Boolean(rapidocr.install_supported) && Boolean(rapidocr.can_install);
  const installed = Boolean(rapidocr.installed);
  const usingRapidOcr = runtime.backend_kind === 'rapidocr';
  const usingFallback = runtime.backend_kind === 'tesseract';

  banner.hidden = false;
  banner.className = 'install-banner install-banner-rapidocr';
  button.hidden = !installable;
  button.disabled = getInstallState('rapidocr').inProgress;
  button.textContent = getInstallState('rapidocr').inProgress
    ? getInstallConfig('rapidocr').runningText
    : getInstallConfig('rapidocr').actionText;

  if (!rapidocr.install_supported) {
    banner.classList.add('neutral');
    kicker.textContent = 'OCR 主后端';
    title.textContent = '当前平台暂不支持自动安装 RapidOCR';
    body.textContent = 'RapidOCR 主后端目前只支持 Windows 本地运行时安装。';
    path.textContent = '';
    button.hidden = true;
    renderInstallTaskState('rapidocr');
    return;
  }

  if (installed) {
    banner.classList.add(usingRapidOcr ? 'success' : 'neutral');
    kicker.textContent = usingRapidOcr ? 'OCR 主后端已接管' : 'OCR 主后端已就绪';
    title.textContent = usingRapidOcr
      ? 'RapidOCR 已接管当前 OCR Reader'
      : 'RapidOCR 已就绪，等待作为 OCR 主后端工作';
    body.textContent = usingRapidOcr
      ? `当前主后端: ${runtime.backend_kind || 'rapidocr'}，模型 ${runtime.backend_model || rapidocr.selected_model || ''}。`
      : usingFallback
        ? `RapidOCR 已安装，但本帧 OCR 回退到了 Tesseract。原因: ${runtime.backend_detail || rapidocr.detail || '未知'}。`
        : 'RapidOCR 已安装完成。无 SDK 且无有效内存文本时，它会优先于 Tesseract 作为 OCR Reader 的主后端。';
    path.textContent = rapidocr.detected_path
      ? `检测路径: ${rapidocr.detected_path}${rapidocr.model_cache_dir ? ` | 模型目录: ${rapidocr.model_cache_dir}` : ''}`
      : '';
    button.hidden = true;
  } else if (rapidocr.detail === 'missing_models') {
    banner.classList.add('warning');
    kicker.textContent = 'OCR 主后端缺少模型';
    title.textContent = 'RapidOCR 运行时存在，但模型状态不完整';
    body.textContent = 'RapidOCR 包已存在，但缺少安装完成状态或模型缓存标记。重新安装会执行预热校验并修复状态。';
    path.textContent = rapidocr.target_dir ? `目标目录: ${rapidocr.target_dir}` : '';
  } else if (rapidocr.detail === 'broken_runtime') {
    banner.classList.add('warning');
    kicker.textContent = 'OCR 主后端异常';
    title.textContent = 'RapidOCR 运行时已损坏或导入失败';
    body.textContent = '建议重新执行一键安装。安装流程会重新落地插件隔离运行时，并在完成后做一次空白图推理自检。';
    path.textContent = rapidocr.detected_path
      ? `检测路径: ${rapidocr.detected_path}`
      : '';
  } else {
    banner.classList.add('warning');
    kicker.textContent = 'OCR 主后端未就绪';
    title.textContent = 'RapidOCR 尚未就绪';
    body.textContent = 'RapidOCR 现在是 OCR Reader 的内部主后端。安装完成后会优先于 Tesseract 参与 OCR，Tesseract 仅保留为兼容兜底。';
    path.textContent = rapidocr.target_dir
      ? `预期安装位置: ${rapidocr.target_dir}`
      : '';
  }

  if (installState && !isInstallTaskTerminal(installState)) {
    banner.className = 'install-banner install-banner-rapidocr neutral';
    kicker.textContent = 'RapidOCR 安装';
    title.textContent = 'RapidOCR 正在后台安装';
    body.textContent = '页面会通过 SSE 接收实时安装进度。安装完成后会自动刷新插件状态，并优先切回 RapidOCR 主后端。';
    button.hidden = false;
    button.disabled = true;
    button.textContent = getInstallConfig('rapidocr').runningText;
  } else if (installState && installState.status === 'failed' && installable) {
    banner.className = 'install-banner install-banner-rapidocr neutral';
    kicker.textContent = 'RapidOCR 安装';
    title.textContent = 'RapidOCR 安装失败，可直接重试';
    body.textContent = installState.error || installState.message || '后台安装任务失败，你可以再次点击按钮重试。';
    button.hidden = false;
    button.disabled = false;
    button.textContent = getInstallConfig('rapidocr').retryText;
  } else if (installState && installState.status === 'completed' && !installed) {
    banner.className = 'install-banner install-banner-rapidocr neutral';
    kicker.textContent = 'RapidOCR 安装';
    title.textContent = 'RapidOCR 安装已完成，正在刷新 OCR 状态';
    body.textContent = installState.message || '安装任务已结束，正在等待插件状态刷新。';
  }

  renderInstallTaskState('rapidocr');
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
    kicker.textContent = 'OCR 兼容兜底';
    title.textContent = '当前平台暂不支持自动安装 Tesseract';
    body.textContent = 'Tesseract 目前只保留为 OCR Reader 的兼容兜底，本地自动安装也只在 Windows 上提供。';
    path.textContent = '';
    button.hidden = true;
    renderInstallTaskState('tesseract');
    return;
  }

  if (installed) {
    banner.classList.add('success');
    kicker.textContent = 'OCR 兼容兜底';
    title.textContent = runtime.backend_kind === 'tesseract'
      ? 'Tesseract 正在作为兼容兜底工作'
      : 'Tesseract 已就绪，等待必要时回退';
    body.textContent = runtime.backend_kind === 'tesseract'
      ? `当前 OCR Reader 使用 Tesseract。原因: ${runtime.backend_detail || runtime.detail || '兼容兜底'}.`
      : '本地 Tesseract 与默认语言包 chi_sim+jpn+eng 已齐全。它会在 RapidOCR 缺失、损坏或运行时异常时接管 OCR。';
    path.textContent = tesseract.detected_path
      ? `检测路径: ${tesseract.detected_path}`
      : '';
    button.hidden = true;
  } else if (tesseract.detail === 'missing_languages') {
    banner.classList.add('warning');
    kicker.textContent = '兜底语言缺失';
    title.textContent = '已检测到 Tesseract，但兼容兜底语言包不完整';
    body.textContent = `当前缺少 ${(missingLanguages || []).join(', ') || '语言包'}。安装流程会按默认语言 chi_sim+jpn+eng 补齐兼容兜底所需文件。`;
    path.textContent = tesseract.tessdata_dir
      ? `tessdata 目录: ${tesseract.tessdata_dir}`
      : '';
  } else {
    banner.classList.add('warning');
    kicker.textContent = '兼容兜底未就绪';
    title.textContent = '未检测到 Tesseract，兼容兜底尚未就绪';
    body.textContent = '这不会阻止 RapidOCR 作为主后端工作，但当 RapidOCR 缺失或运行异常时，将无法自动回退到本地 Tesseract。';
    path.textContent = tesseract.expected_executable_path
      ? `预期安装位置: ${tesseract.expected_executable_path}`
      : '';
  }

  if (installState && !isInstallTaskTerminal(installState)) {
    banner.className = 'install-banner install-banner-tesseract neutral';
    kicker.textContent = 'Tesseract 安装';
    title.textContent = 'Tesseract 正在后台安装';
    body.textContent = '安装器和语言包下载都通过 HTTPS 进行，当前页面会通过 SSE 接收实时进度；即使刷新页面，也会尝试恢复最近的安装状态。';
    button.hidden = false;
    button.disabled = true;
    button.textContent = getInstallConfig('tesseract').runningText;
  } else if (installState && installState.status === 'failed' && installable) {
    banner.className = 'install-banner install-banner-tesseract neutral';
    kicker.textContent = 'Tesseract 安装';
    title.textContent = 'Tesseract 安装失败，可直接重试';
    body.textContent = installState.error || installState.message || '后台安装任务失败，你可以再次点击按钮重试。';
    button.hidden = false;
    button.disabled = false;
    button.textContent = getInstallConfig('tesseract').retryText;
  } else if (installState && installState.status === 'completed' && !installed) {
    banner.className = 'install-banner install-banner-tesseract neutral';
    kicker.textContent = 'Tesseract 安装';
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
    kicker.textContent = '实验性兜底';
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
    kicker.textContent = '实验性兜底';
    title.textContent = runtimeBlocked
      ? '未检测到 Textractor，Memory Reader 实验性兜底暂不可用'
      : '尚未检测到 Textractor';
    body.textContent = runtimeBlocked
      ? '如果你后续仍想继续尝试 Memory Reader，可以在这里补装 Textractor；但当前正式主兜底仍然是 OCR Reader。'
      : 'Textractor 仅影响实验性 Memory Reader 链路，不影响当前 Bridge SDK > OCR Reader 的正式运行顺序。';
    path.textContent = textractor.expected_executable_path
      ? `预期安装位置: ${textractor.expected_executable_path}`
      : '';
  }

  if (installState && !isInstallTaskTerminal(installState)) {
    banner.className = 'install-banner install-banner-textractor neutral';
    kicker.textContent = 'Textractor 安装';
    title.textContent = 'Textractor 正在后台安装';
    body.textContent = '下载通过 HTTPS 进行，页面会通过 SSE 接收实时进度。Textractor 完成后只会补强实验性 Memory Reader 路径。';
    button.hidden = false;
    button.disabled = true;
    button.textContent = getInstallConfig('textractor').runningText;
  } else if (installState && installState.status === 'failed' && installable) {
    banner.className = 'install-banner install-banner-textractor neutral';
    kicker.textContent = 'Textractor 安装';
    title.textContent = 'Textractor 安装失败，可直接重试';
    body.textContent = installState.error || installState.message || '后台安装任务失败，你可以再次点击按钮重试。';
    button.hidden = false;
    button.disabled = false;
    button.textContent = getInstallConfig('textractor').retryText;
  } else if (installState && installState.status === 'completed' && !installed) {
    banner.className = 'install-banner install-banner-textractor neutral';
    kicker.textContent = 'Textractor 安装';
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
  node.className = 'result-panel scroll-region';
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
  node.className = 'result-panel scroll-region';
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
  node.className = 'result-panel scroll-region';
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

function formatInsightMeta(payload) {
  const inputSource = payload.input_source || (latestStatus && latestStatus.active_data_source) || 'unknown';
  const semantic = payload.semantic_granularity
    || (payload.semantic_degraded ? 'weaker_than_bridge_sdk' : 'bridge_sdk_level');
  const fallback = payload.fallback_used ? '是' : '否';
  return `输入源=${inputSource} | degraded=${Boolean(payload.degraded)} | 语义粒度=${semantic} | 使用回退=${fallback}`;
}

function renderAgentStatus(payload) {
  document.getElementById('agentReplyText').textContent = latestAgentReply;
  const memoryCounts = payload.memory_counts || {};
  renderGrid('agentStatusGrid', [
    { label: 'status', value: payload.status || 'standby' },
    { label: 'activity', value: payload.activity || 'idle' },
    { label: 'reason', value: payload.reason || '' },
    { label: 'input_source', value: payload.input_source || (latestStatus && latestStatus.active_data_source) || 'unknown' },
    { label: 'scene_stage', value: payload.scene_stage || 'unknown' },
    { label: 'scene_id', value: payload.scene_id || '' },
    { label: 'line_id', value: payload.line_id || '' },
    { label: 'push_policy', value: payload.push_policy || 'disabled' },
    { label: 'actionable', value: String(Boolean(payload.actionable)) },
    { label: 'standby_requested', value: String(Boolean(payload.standby_requested)) },
    {
      label: 'memory_counts',
      value: `scene=${memoryCounts.scene_memory || 0} choice=${memoryCounts.choice_memory || 0} failure=${memoryCounts.failure_memory || 0}`,
    },
    { label: 'result', value: payload.result || '' },
    { label: 'recent_pushes', value: String((payload.recent_pushes || []).length) },
  ]);

  renderStackList('pushesList', payload.recent_pushes || [], (item) => `
    <article class="list-card compact">
      <p class="list-kicker">${escapeHtml(item.kind || '')} | ${escapeHtml(item.ts || '')}</p>
      <h3>${escapeHtml(item.scene_id || '')}</h3>
      <p>${escapeHtml(item.content || '')}</p>
    </article>
  `);
}

function renderExplain(payload) {
  const node = document.getElementById('explainPanel');
  node.className = 'result-panel scroll-region';
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.line_id || '')} | ${escapeHtml(formatInsightMeta(payload))}</p>
    <h3>${escapeHtml(payload.speaker || 'Narration')}</h3>
    <p>${escapeHtml(payload.text || '')}</p>
    <p class="result-main">${escapeHtml(payload.explanation || payload.diagnostic || 'No explanation yet')}</p>
    <p class="result-note">${escapeHtml(payload.diagnostic || '')}</p>
  `;
}

function renderSummary(payload) {
  const node = document.getElementById('summaryPanel');
  node.className = 'result-panel scroll-region';
  const points = payload.key_points || [];
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.scene_id || '')} | ${escapeHtml(formatInsightMeta(payload))}</p>
    <p class="result-main">${escapeHtml(payload.summary || payload.diagnostic || 'No summary yet')}</p>
    <p class="result-note">${escapeHtml(payload.diagnostic || '')}</p>
    <div class="chip-row">
      ${points.map((item) => `<span class="chip">${escapeHtml(item.type || '')}: ${escapeHtml(item.text || '')}</span>`).join('')}
    </div>
  `;
}

function renderSuggest(payload) {
  const node = document.getElementById('suggestPanel');
  node.className = 'result-panel scroll-region';
  const choices = payload.choices || [];
  node.innerHTML = `
    <p class="list-kicker">${escapeHtml(payload.scene_id || '')} | ${escapeHtml(formatInsightMeta(payload))}</p>
    <div class="stack-list">
      ${choices.length ? choices.map((item) => `
        <article class="list-card compact">
          <p class="list-kicker">rank ${escapeHtml(item.rank || 0)} | ${escapeHtml(item.choice_id || '')}</p>
          <h3>${escapeHtml(item.text || '')}</h3>
          <p>${escapeHtml(item.reason || '')}</p>
        </article>
      `).join('') : `<div class="empty-inline">${escapeHtml(payload.diagnostic || 'No suggestion yet')}</div>`}
    </div>
  `;
}

async function refreshInsights(snapshot, { force = false } = {}) {
  const state = snapshot.snapshot || {};
  const currentLineId = state.line_id || '';
  const currentSceneId = state.scene_id || '';
  const choices = Array.isArray(state.choices) ? state.choices : [];
  const hasChoices = Boolean(state.is_menu_open) && choices.length > 0;
  const explainKey = currentLineId || 'missing-line';
  const summaryKey = currentSceneId || 'missing-scene';
  const suggestKey = hasChoices
    ? `${currentSceneId}::${choices.map((item) => `${item.choice_id || ''}:${item.text || ''}`).join('|')}`
    : `${currentSceneId}::no-choices`;

  const explainPromise = currentLineId
    ? (force || latestInsights.explainKey !== explainKey || !latestInsights.explainPayload)
      ? safeCall(
        'galgame_explain_line',
        { line_id: currentLineId },
        buildExplainFallback(currentLineId),
      )
      : Promise.resolve(latestInsights.explainPayload)
    : Promise.resolve(buildExplainFallback('', 'missing line_id'));

  const summaryPromise = currentSceneId
    ? (force || latestInsights.summaryKey !== summaryKey || !latestInsights.summaryPayload)
      ? safeCall(
        'galgame_summarize_scene',
        { scene_id: currentSceneId },
        buildSummaryFallback(currentSceneId),
      )
      : Promise.resolve(latestInsights.summaryPayload)
    : Promise.resolve(buildSummaryFallback('', 'missing scene_id'));

  const suggestPromise = hasChoices
    ? (force || latestInsights.suggestKey !== suggestKey || !latestInsights.suggestPayload)
      ? safeCall(
        'galgame_suggest_choice',
        {},
        buildSuggestFallback(currentSceneId),
      )
      : Promise.resolve(latestInsights.suggestPayload)
    : Promise.resolve(buildSuggestFallback(currentSceneId, 'no visible choices'));

  const [explain, summary, suggest] = await Promise.all([
    explainPromise,
    summaryPromise,
    suggestPromise,
  ]);

  latestInsights.explainKey = explainKey;
  latestInsights.explainPayload = explain;
  latestInsights.summaryKey = summaryKey;
  latestInsights.summaryPayload = summary;
  latestInsights.suggestKey = suggestKey;
  latestInsights.suggestPayload = suggest;

  renderExplain(explain);
  renderSummary(summary);
  renderSuggest(suggest);
}

function stopAutoRefresh() {
  if (autoRefreshTimer !== null) {
    window.clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
}

function startAutoRefresh() {
  stopAutoRefresh();
  autoRefreshTimer = window.setInterval(() => {
    if (document.hidden) {
      return;
    }
    refreshAll({ preserveFlash: true, silent: true }).catch(() => {});
  }, AUTO_REFRESH_INTERVAL_MS);
}

async function refreshAll(options = {}) {
  if (refreshInFlight) {
    return refreshInFlight;
  }

  const { preserveFlash = false, silent = false, forceInsights = false } = options;
  refreshInFlight = (async () => {
    if (!preserveFlash && !silent) {
      setFlash('', 'info');
    }
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
      await refreshInsights(snapshot, { force: forceInsights });
    } catch (error) {
      if (silent) {
        console.warn('[galgame_plugin ui] refresh failed', error);
        return;
      }
      setFlash(error instanceof Error ? error.message : String(error), 'error');
    }
  })();

  try {
    await refreshInFlight;
  } finally {
    refreshInFlight = null;
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

async function installRapidOcr(force = false) {
  await startInstall('rapidocr', force);
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
    await refreshAll({ preserveFlash: true, forceInsights: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function bindGame() {
  const gameId = document.getElementById('bindInput').value.trim();
  try {
    await callPlugin('galgame_bind_game', { game_id: gameId });
    setFlash(gameId ? `已绑定 ${gameId}` : '已恢复自动选择', 'success');
    await refreshAll({ preserveFlash: true, forceInsights: true });
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
    await refreshAll({ preserveFlash: true, forceInsights: true });
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
    await refreshAll({ preserveFlash: true, forceInsights: true });
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
    await refreshAll({ preserveFlash: true, forceInsights: true });
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
    await refreshAll({ preserveFlash: true, forceInsights: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function refreshOcrWindowTargets({ includeExcluded = true, silent = false } = {}) {
  try {
    const payload = await callPlugin('galgame_list_ocr_windows', {
      include_excluded: Boolean(includeExcluded),
    });
    renderOcrWindowTargetSnapshot(payload, latestStatus);
  } catch (error) {
    if (silent) {
      console.warn('[galgame_plugin ui] refresh OCR window targets failed', error);
      return;
    }
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function setOcrWindowTarget(windowKey) {
  try {
    const payload = await callPlugin('galgame_set_ocr_window_target', {
      window_key: windowKey,
      clear: false,
    });
    setFlash(payload.summary || 'OCR 目标窗口已锁定', 'success');
    await refreshAll({ preserveFlash: true, forceInsights: true });
    await refreshOcrWindowTargets({ includeExcluded: true, silent: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function clearOcrWindowTarget() {
  try {
    const payload = await callPlugin('galgame_set_ocr_window_target', {
      clear: true,
    });
    setFlash(payload.summary || 'OCR 已恢复自动选窗', 'success');
    await refreshAll({ preserveFlash: true, forceInsights: true });
    await refreshOcrWindowTargets({ includeExcluded: true, silent: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function initialize() {
  await refreshAll({ forceInsights: true });
  await refreshOcrWindowTargets({ includeExcluded: true, silent: true });
  await Promise.all([
    restoreRapidOcrInstallState(),
    restoreTesseractInstallState(),
    restoreTextractorInstallState(),
  ]);
  startAutoRefresh();
}

document.getElementById('refreshBtn').addEventListener('click', async () => {
  await refreshAll({ forceInsights: true });
  await refreshOcrWindowTargets({ includeExcluded: true, silent: true });
});
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
document.getElementById('rapidocrInstallBtn').addEventListener('click', () => installRapidOcr(false));
document.getElementById('tesseractInstallBtn').addEventListener('click', () => installTesseract(false));
document.getElementById('textractorInstallBtn').addEventListener('click', () => installTextractor(false));
document.getElementById('ocrWindowRefreshBtn').addEventListener('click', () => {
  refreshOcrWindowTargets({ includeExcluded: true }).catch((error) => {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  });
});
document.getElementById('ocrWindowAutoBtn').addEventListener('click', clearOcrWindowTarget);
document.getElementById('ocrProfileSaveBtn').addEventListener('click', saveOcrCaptureProfile);
document.getElementById('ocrProfileClearBtn').addEventListener('click', clearOcrCaptureProfile);

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    refreshAll({ preserveFlash: true, silent: true }).catch(() => {});
  }
});

window.addEventListener('focus', () => {
  refreshAll({ preserveFlash: true, silent: true }).catch(() => {});
});

initialize();
