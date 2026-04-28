const PLUGIN_ID = 'galgame_plugin';
const RUNS_URL = '/runs';
const UI_API_BASE = `/plugin/${PLUGIN_ID}/ui-api`;
const RAPIDOCR_INSTALL_URL = `${UI_API_BASE}/rapidocr/install`;
const DXCAM_INSTALL_URL = `${UI_API_BASE}/dxcam/install`;
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
  dxcam: {
    kind: 'dxcam',
    label: 'DXcam',
    url: DXCAM_INSTALL_URL,
    storageKey: `${PLUGIN_ID}:dxcam_install_task_id`,
    domPrefix: 'dxcam',
    actionText: '一键安装 DXcam',
    retryText: '重试安装 DXcam',
    runningText: '后台安装中...',
    queuedFlash: '已创建后台安装任务，接下来会安装 DXcam 截图依赖，并通过 SSE 推送实时进度。',
    successFlash: 'DXcam 安装完成',
    failureFlash: 'DXcam 安装失败',
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
  dxcam: createInstallRuntimeState(),
  tesseract: createInstallRuntimeState(),
  textractor: createInstallRuntimeState(),
};

const DEFAULT_CAPTURE_PROFILE = {
  left_inset_ratio: 0.05,
  right_inset_ratio: 0.05,
  top_ratio: 0.62,
  bottom_inset_ratio: 0.08,
};
const AIHONG_PROCESS_NAMES = new Set(['thelamentinggeese.exe']);
const OCR_PROFILE_STAGE_LABELS_ZH = {
  default: '通用区域',
  dialogue_stage: '对白区',
  menu_stage: '菜单区',
};
const OCR_CAPTURE_SAVE_SCOPE_LABELS_ZH = {
  window_bucket: '当前窗口分辨率',
  process_fallback: '进程通用回退',
};
const OCR_CAPTURE_MATCH_SOURCE_LABELS_ZH = {
  bucket_exact: '当前窗口精确命中',
  bucket_aspect_nearest: '相近宽高比回退',
  process_fallback: '进程通用回退',
  builtin_preset: '内建预设',
  config_default: '插件默认配置',
};
const AIHONG_CAPTURE_PRESETS = {
  dialogue_stage: {
    left_inset_ratio: 0.05,
    right_inset_ratio: 0.24,
    top_ratio: 0.73,
    bottom_inset_ratio: 0.10,
  },
  menu_stage: {
    left_inset_ratio: 0.20,
    right_inset_ratio: 0.20,
    top_ratio: 0.40,
    bottom_inset_ratio: 0.34,
  },
};

const AUTO_REFRESH_IDLE_INTERVAL_MS = 5000;
const AUTO_REFRESH_ACTIVE_INTERVAL_MS = 2000;
const AUTO_REFRESH_INTERVAL_MS = AUTO_REFRESH_ACTIVE_INTERVAL_MS;
const FOCUS_PAUSE_REFRESH_INTERVAL_MS = 1000;
const ERROR_REFRESH_INTERVAL_MS = 10000;
const OCR_WINDOW_REFRESH_TTL_MS = 3000;
const FIELD_LABELS_ZH = {
  connection_state: '连接状态',
  active_data_source: '当前数据源',
  reader_mode: '文本读取模式',
  mode: '模式',
  push_notifications: '推送通知',
  advance_speed: '推进速度',
  bound_game_id: '绑定游戏 ID',
  active_session_id: '当前会话 ID',
  last_seq: '最新序号',
  stream_reset_pending: '等待重置流',
  available_game_ids: '可用游戏 ID',
  performance_cpu_percent: '插件 CPU',
  performance_memory_mb: '插件内存',
  performance_memory_percent: '插件内存占比',
  performance_thread_count: '插件线程数',
  performance_process: '插件进程',
  performance_detail: '性能指标状态',
  ocr_reader_enabled: 'OCR Reader 已启用',
  ocr_reader_status: 'OCR Reader 状态',
  ocr_reader_detail: 'OCR Reader 详情',
  ocr_reader_target: 'OCR Reader 目标',
  ocr_poll_interval_seconds: 'OCR 识别间隔',
  ocr_trigger_mode: 'OCR 触发方式',
  ocr_backend_selection: 'OCR 后端选择',
  ocr_capture_backend_selection: '截图后端选择',
  ocr_backend_kind: 'OCR 后端类型',
  ocr_backend_detail: 'OCR 后端详情',
  rapidocr_enabled: 'RapidOCR 已启用',
  rapidocr_installed: 'RapidOCR 已安装',
  rapidocr_detail: 'RapidOCR 详情',
  dxcam_installed: 'DXcam 已安装',
  dxcam_detail: 'DXcam 详情',
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
  width: '窗口宽度',
  height: '窗口高度',
  aspect_ratio: '窗口宽高比',
  game_id: '游戏 ID',
  session_id: '会话 ID',
  last_event_ts: '最近事件时间',
  capture_stage: '截图阶段',
  capture_profile: '截图配置',
  capture_profile_match_source: '截图配置来源',
  capture_profile_bucket_key: '截图配置桶',
  capture_backend_kind: '截图后端',
  capture_backend_detail: '截图后端详情',
  last_capture_image_hash: '最近截图 Hash',
  consecutive_same_capture_frames: '连续相同截图',
  stale_capture_backend: '截图源未更新',
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
  foreground_refresh_at: '前台刷新时间',
  foreground_refresh_detail: '前台刷新详情',
  foreground_hwnd: '当前前台 hwnd',
  target_hwnd: '目标 hwnd',
  last_poll_started_at: '最近 OCR poll 开始',
  last_poll_completed_at: '最近 OCR poll 完成',
  last_poll_duration_seconds: '最近 OCR poll 耗时',
  last_poll_emitted_event: '最近 OCR poll 产生事件',
  last_tick_skipped: '最近 tick 被跳过',
  last_tick_skip_reason: 'tick 跳过原因',
  pending_visual_scene_count: '待提交场景变化',
  last_auto_recalibrate_attempts: '自动校准 OCR 次数',
  last_auto_recalibrate_duration_seconds: '自动校准耗时',
  last_auto_recalibrate_limited: '自动校准达到限制',
  last_auto_recalibrate_error: '自动校准错误',
  last_capture_total_duration_seconds: '最近 OCR 总耗时',
  last_capture_frame_duration_seconds: '截图耗时',
  last_capture_background_duration_seconds: '背景 Hash 耗时',
  last_capture_image_hash_duration_seconds: '截图 Hash 耗时',
  last_ocr_extract_duration_seconds: 'OCR 推理耗时',
  last_backend_plan_duration_seconds: '后端选择耗时',
  last_window_scan_duration_seconds: '窗口扫描耗时',
  last_capture_background_hash_skipped: '已跳过背景 Hash',
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
  agent_user_status: 'Agent 用户状态',
  agent_pause_kind: 'Agent 暂停类型',
  agent_pause_message: 'Agent 暂停说明',
  agent_can_resume_by_button: '可用按钮恢复',
  agent_can_resume_by_focus: '可由窗口聚焦恢复',
  inbound_queue_size: '入站队列',
  outbound_queue_size: '出站队列',
  last_interruption: '最近打断',
  last_outbound_message: '最近出站消息',
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

const CONNECTION_STATE_LABELS_ZH = {
  active: '运行中',
  idle: '空闲',
  stale: '已过期',
};

const MODE_LABELS_ZH = {
  silent: '静默模式',
  companion: '伴读模式',
  choice_advisor: '自动推进模式',
};
const ADVANCE_SPEED_LABELS_ZH = {
  slow: '慢',
  medium: '中等',
  fast: '快速',
};

const AGENT_USER_STATUS_LABELS_ZH = {
  running: '运行中',
  read_only: '只读伴读',
  paused_by_user: '用户待机',
  paused_window_not_foreground: '游戏窗口未前台',
  ocr_unavailable: 'OCR 不可用',
  waiting_choice: '等待/处理选项',
  acting: '正在操作',
  error: '错误',
};

const DATA_SOURCE_LABELS_ZH = {
  bridge_sdk: 'Bridge SDK',
  ocr_reader: 'OCR 读取',
  memory_reader: '内存读取',
};

const READER_MODE_LABELS_ZH = {
  auto: '自动（内存优先，空则 OCR）',
  memory_reader: '内存读取',
  ocr_reader: 'OCR',
};

let latestAgentReply = '暂无交互';
let latestAgentStatus = null;
let latestStatus = null;
let latestOcrWindowSnapshot = null;
let refreshInFlight = null;
let ocrWindowRefreshInFlight = null;
let lastOcrWindowRefreshAt = 0;
let emptyOcrWindowFocusForceRefreshDone = false;
let autoRefreshTimer = null;
let autoRefreshIntervalMs = AUTO_REFRESH_INTERVAL_MS;
let activeInstallTab = 'rapidocr';
let settingsDirty = false;
let settingsSaveInFlight = false;

const SETTINGS_CONTROL_IDS = new Set([
  'modeSelect',
  'pushToggle',
  'advanceSpeedSelect',
  'readerModeSelect',
  'ocrPollIntervalInput',
  'ocrTriggerModeSelect',
]);

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

function configureUseButton(id, { active = false, disabled = false, text = '', title = '' } = {}) {
  const button = document.getElementById(id);
  if (!button) {
    return;
  }
  if (text) {
    button.textContent = text;
  }
  button.disabled = Boolean(disabled || active);
  button.classList.toggle('active', Boolean(active));
  button.title = title || '';
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

function textValue(value) {
  return String(value == null ? '' : value).trim();
}

function lineText(line = {}) {
  return textValue(line && typeof line === 'object' ? line.text : '');
}

function compactLineText(value) {
  return textValue(value).replace(/\s+/g, '');
}

function getCurrentLineTexts(status = {}) {
  const runtime = status.ocr_reader_runtime || {};
  const effectiveText = lineText(status.effective_current_line);
  const stableText = lineText(runtime.last_stable_line) || effectiveText;
  const observedText = lineText(runtime.last_observed_line);
  const rawText = textValue(runtime.last_raw_ocr_text);
  return {
    rawText,
    observedText,
    stableText,
    effectiveText,
  };
}

function formatStableBlockReason(reason) {
  const mapping = {
    waiting_for_repeat: '刚读到新文字，正在确认是不是同一句台词',
    duplicate_stable_text: '这句台词已经显示过',
    duplicate_raw_text: '识别结果和上一轮相同，暂不重复写入',
    duplicate_observed_text: '候选台词和上一轮相同，正在等待变化',
    duplicate_candidate_text: '候选台词和上一轮相同，正在等待变化',
    empty_text: '文字识别暂时没有读到有效文本',
    no_text: '文字识别暂时没有读到有效文本',
    no_valid_text: '文字识别暂时没有读到有效文本',
    low_confidence: '文字识别置信度较低，暂不写入台词',
    overlay_text: '识别结果像系统界面文字，暂不写入台词',
    game_overlay_text: '识别结果像游戏菜单或系统界面，暂不写入台词',
    waiting_for_change: '画面文字没有变化，正在等待新台词',
    waiting_for_new_text: '正在等待新台词出现',
    capture_failed: '截图或识别失败，暂时不能确认台词',
  };
  const normalized = textValue(reason);
  return mapping[normalized] || normalized;
}

function normalizePrimaryDiagnosis(diagnosis) {
  if (!diagnosis || typeof diagnosis !== 'object') {
    return null;
  }
  const severity = textValue(diagnosis.severity);
  const title = textValue(diagnosis.title);
  const body = textValue(diagnosis.message || diagnosis.body);
  if (!title && !body) {
    return null;
  }
  const actions = Array.isArray(diagnosis.actions)
    ? diagnosis.actions.map((action) => {
      if (!action || typeof action !== 'object') {
        return null;
      }
      const id = textValue(action.id);
      if (!id) {
        return null;
      }
      return {
        id,
        label: textValue(action.label || action.title || id) || id,
      };
    }).filter(Boolean)
    : [];
  return {
    severity: ['ok', 'info', 'warning', 'error'].includes(severity) ? severity : 'info',
    title: title || '运行诊断',
    body,
    actions,
  };
}

function buildPrimaryDiagnosis(status = {}) {
  const backendDiagnosis = normalizePrimaryDiagnosis(status.primary_diagnosis);
  if (backendDiagnosis) {
    return backendDiagnosis;
  }

  const runtime = status.ocr_reader_runtime || {};
  const detail = textValue(runtime.target_selection_detail);
  const contextState = textValue(status.ocr_context_state || runtime.ocr_context_state);
  const lastExcludeReason = textValue(runtime.last_exclude_reason);
  const lastCaptureError = textValue(runtime.last_capture_error);
  const lastError = textValue(status.last_error && status.last_error.message);
  const agentPauseKind = textValue(status.agent_pause_kind);
  const agentUserStatus = textValue(status.agent_user_status);
  const { observedText, stableText, effectiveText } = getCurrentLineTexts(status);
  const observedKey = compactLineText(observedText);
  const stableKey = compactLineText(stableText);
  const hasEffectiveWindow = Boolean(textValue(runtime.effective_window_key));
  const candidateCount = Number(runtime.candidate_count || 0);
  const hasOcrRuntimeSignal = Boolean(
    status.ocr_reader_enabled
    || runtime.status
    || runtime.detail
    || contextState
    || detail
    || Object.prototype.hasOwnProperty.call(runtime, 'candidate_count')
  );

  if (lastError) {
    return {
      severity: 'error',
      title: '插件运行出错',
      body: `${lastError}。可以先刷新状态；如果仍然出现，请查看调试详情。`,
      actions: [
        { id: 'refresh_all', label: '刷新全部' },
        { id: 'debug_details', label: '查看调试详情' },
      ],
    };
  }

  if (detail === 'memory_reader_window_minimized' || lastExcludeReason === 'excluded_minimized_window') {
    return {
      severity: 'warning',
      title: '游戏窗口最小化了',
      body: '检测到游戏，但窗口最小化，文字识别不能截图。请恢复游戏窗口后继续。',
      actions: [
        { id: 'refresh_ocr_windows', label: '我已恢复，刷新窗口' },
        { id: 'select_ocr_window', label: '选择游戏窗口' },
      ],
    };
  }

  if (contextState === 'capture_failed' || lastCaptureError) {
    return {
      severity: 'error',
      title: '截图或文字识别失败',
      body: lastCaptureError || '截图或识别后端返回错误，新台词不会更新。',
      actions: [
        { id: 'recalibrate_ocr', label: '重新截图校准' },
        { id: 'capture_backend', label: '切换截图方式' },
        { id: 'debug_details', label: '查看调试详情' },
      ],
    };
  }

  if (runtime.stale_capture_backend) {
    return {
      severity: 'warning',
      title: '截图画面没有更新',
      body: '当前截图源可能停在旧画面。请切回游戏窗口，或切换截图方式后再试。',
      actions: [
        { id: 'focus_game', label: '切回游戏窗口' },
        { id: 'capture_backend', label: '切换 DXcam' },
        { id: 'refresh_ocr_windows', label: '刷新窗口' },
      ],
    };
  }

  if (
    detail === 'no_eligible_window'
    || (!hasEffectiveWindow && candidateCount === 0 && hasOcrRuntimeSignal)
  ) {
    return {
      severity: 'warning',
      title: '没找到能识别的游戏窗口',
      body: '游戏可能未启动、被最小化，或当前窗口不是游戏。请确认游戏窗口可见后刷新。',
      actions: [
        { id: 'refresh_ocr_windows', label: '刷新窗口' },
        { id: 'select_ocr_window', label: '选择游戏窗口' },
      ],
    };
  }

  if (detail === 'foreground_window_needs_manual_confirmation' || detail === 'auto_detect_needs_manual_fallback') {
    return {
      severity: 'warning',
      title: '需要手动选择游戏窗口',
      body: '自动检测不够确定。手动选择一次可以避免识别到插件页面或其他窗口。',
      actions: [
        { id: 'select_ocr_window', label: '选择游戏窗口' },
        { id: 'refresh_ocr_windows', label: '刷新窗口' },
      ],
    };
  }

  if (observedKey && observedKey !== stableKey) {
    return {
      severity: 'info',
      title: '刚读到新文字',
      body: '文字识别已经看到候选台词，正在确认这是不是同一句台词。',
      actions: [
        { id: 'line_details', label: '查看识别详情' },
      ],
    };
  }

  if (agentPauseKind === 'window_not_foreground' || agentUserStatus === 'paused_window_not_foreground') {
    return {
      severity: 'info',
      title: '游戏不在前台',
      body: '自动推进已暂停。切回游戏窗口后会继续，伴读信息仍会刷新。',
      actions: [
        { id: 'focus_game', label: '切回游戏窗口' },
      ],
    };
  }

  if (agentPauseKind === 'read_only' || agentUserStatus === 'read_only') {
    return {
      severity: 'info',
      title: '当前是伴读模式',
      body: '会显示台词和建议，但不会自动点击。需要自动推进时请切换模式。',
      actions: [
        { id: 'choice_advisor', label: '切换到自动推进模式' },
      ],
    };
  }

  if (effectiveText || stableText) {
    const target = formatOcrTargetForUser(status);
    return {
      severity: 'ok',
      title: '正在识别台词',
      body: target ? `当前目标：${target}。已读到台词，页面会持续刷新。` : '已读到台词，页面会持续刷新。',
      actions: [
        { id: 'refresh_all', label: '刷新全部' },
      ],
    };
  }

  return {
    severity: 'info',
    title: '等待游戏状态',
    body: status.summary || '暂时没有足够信息判断当前卡点。请先打开游戏，或刷新窗口列表。',
    actions: [
      { id: 'refresh_all', label: '刷新全部' },
      { id: 'select_ocr_window', label: '选择游戏窗口' },
    ],
  };
}

function renderPrimaryDiagnosis(status = {}) {
  const node = document.getElementById('primaryDiagnosisPanel');
  if (!node) {
    return;
  }
  const diagnosis = buildPrimaryDiagnosis(status);
  const kicker = document.getElementById('primaryDiagnosisKicker');
  const title = document.getElementById('primaryDiagnosisTitle');
  const body = document.getElementById('primaryDiagnosisBody');
  const actions = document.getElementById('primaryDiagnosisActions');
  node.className = `primary-diagnosis ${diagnosis.severity || 'info'}`;
  kicker.textContent = '运行诊断';
  title.textContent = diagnosis.title;
  body.textContent = diagnosis.body;
  actions.innerHTML = (diagnosis.actions || []).map((action, index) => `
    <button class="${index === 0 ? 'primary' : 'secondary'}" data-primary-action="${escapeHtml(action.id)}">
      ${escapeHtml(action.label)}
    </button>
  `).join('');
}

function buildFirstRunSteps(status = {}) {
  const runtime = status.ocr_reader_runtime || {};
  const memoryRuntime = status.memory_reader_runtime || {};
  const snapshotWindows = latestOcrWindowSnapshot && Array.isArray(latestOcrWindowSnapshot.windows)
    ? latestOcrWindowSnapshot.windows
    : [];
  const availableGameIds = Array.isArray(status.available_game_ids) ? status.available_game_ids : [];
  const detail = textValue(runtime.target_selection_detail);
  const lastExcludeReason = textValue(runtime.last_exclude_reason);
  const hasGame = Boolean(
    textValue(status.active_session_id)
    || availableGameIds.length
    || Number(runtime.pid || 0)
    || textValue(runtime.process_name)
    || textValue(runtime.window_title)
    || Number(memoryRuntime.pid || 0)
    || textValue(memoryRuntime.process_name)
  );
  const hasWindow = Boolean(
    textValue(runtime.effective_window_key)
    || Number(runtime.candidate_count || 0) > 0
    || snapshotWindows.length > 0
  );
  const hasConfirmedWindow = Boolean(
    textValue(runtime.effective_window_key)
    && detail !== 'no_eligible_window'
    && detail !== 'memory_reader_window_minimized'
    && lastExcludeReason !== 'excluded_minimized_window'
  );
  const { observedText, stableText, effectiveText } = getCurrentLineTexts(status);
  const hasLine = Boolean(effectiveText || stableText || observedText);

  return [
    {
      done: hasGame,
      title: '启动或恢复游戏',
      body: hasGame ? '已发现游戏状态。' : '打开游戏，并停在有文字的画面。',
    },
    {
      done: hasWindow,
      title: '刷新窗口',
      body: hasWindow ? '已找到可检查的窗口。' : '回到插件页，点击“刷新窗口”。',
    },
    {
      done: hasConfirmedWindow,
      title: '选择游戏窗口',
      body: hasConfirmedWindow ? '已确认识别窗口。' : '如果没有自动选中，请手动选择游戏窗口。',
    },
    {
      done: hasLine,
      title: '开始识别',
      body: hasLine ? '已读到台词。' : '开始自动识别，或在游戏中推进到下一句台词。',
    },
  ];
}

function renderFirstRunGuide(status = {}) {
  const node = document.getElementById('firstRunGuide');
  const stepsNode = document.getElementById('firstRunSteps');
  if (!node || !stepsNode) {
    return;
  }
  const steps = buildFirstRunSteps(status);
  const allDone = steps.every((step) => step.done);
  node.hidden = allDone;
  if (allDone) {
    stepsNode.replaceChildren();
    return;
  }
  const firstIncompleteIndex = steps.findIndex((step) => !step.done);
  stepsNode.innerHTML = steps.map((step, index) => {
    const stateClass = step.done ? 'done' : (index === firstIncompleteIndex ? 'active' : 'pending');
    const marker = step.done ? '完成' : String(index + 1);
    return `
      <article class="first-run-step ${stateClass}">
        <span class="first-run-step-marker">${escapeHtml(marker)}</span>
        <div>
          <h3>${escapeHtml(step.title)}</h3>
          <p>${escapeHtml(step.body)}</p>
        </div>
      </article>
    `;
  }).join('');
}

function renderCurrentLineOverview(status = {}) {
  const node = document.getElementById('currentLineOverview');
  if (!node) {
    return;
  }
  const runtime = status.ocr_reader_runtime || {};
  const title = document.getElementById('currentLineOverviewTitle');
  const statusChip = document.getElementById('currentLineOverviewStatus');
  const hint = document.getElementById('currentLineOverviewHint');
  const grid = document.getElementById('currentLineOverviewGrid');
  const { rawText, observedText, stableText, effectiveText } = getCurrentLineTexts(status);
  const displayStable = stableText || effectiveText;
  const observedKey = compactLineText(observedText);
  const stableKey = compactLineText(displayStable);
  const hasMismatch = Boolean(observedKey && observedKey !== stableKey);
  const blockReason = formatStableBlockReason(runtime.stable_ocr_block_reason);
  const repeatCount = Number(runtime.stable_ocr_repeat_count || 0);

  node.classList.toggle('waiting', !rawText && !observedText && !displayStable);
  statusChip.className = 'status-chip';
  if (hasMismatch) {
    title.textContent = '刚读到新文字';
    statusChip.textContent = '确认中';
    statusChip.classList.add('warning');
    hint.textContent = blockReason || `正在确认这是不是同一句台词${repeatCount ? `，已连续看到 ${repeatCount} 次` : ''}。`;
  } else if (displayStable) {
    title.textContent = '已确认当前台词';
    statusChip.textContent = '已确认';
    statusChip.classList.add('active');
    hint.textContent = '这句台词已经进入正式上下文，解释、总结和建议会以它为基础更新。';
  } else if (rawText || observedText) {
    title.textContent = '正在筛选识别结果';
    statusChip.textContent = '筛选中';
    statusChip.classList.add('warning');
    hint.textContent = blockReason || '文字识别已有结果，但还没有写入正式台词。';
  } else {
    title.textContent = '等待识别结果';
    statusChip.textContent = '等待刷新';
    hint.textContent = buildOcrMissingLineDiagnostic(status);
  }

  const rows = [
    {
      label: '最新 OCR 原文',
      value: rawText,
      empty: '还没有 OCR 原文',
    },
    {
      label: '刚读到的候选台词',
      value: observedText,
      empty: '还没有候选台词',
    },
    {
      label: '已确认台词',
      value: displayStable,
      empty: '还没有已确认台词',
    },
  ];

  grid.innerHTML = rows.map((row) => `
    <article class="current-line-item${row.value ? '' : ' empty'}">
      <p class="list-kicker">${escapeHtml(row.label)}</p>
      <p>${escapeHtml(row.value || row.empty)}</p>
    </article>
  `).join('');
}

function pipelineStateLabel(state) {
  const mapping = {
    ok: '正常',
    info: '等待',
    warning: '注意',
    error: '异常',
  };
  return mapping[state] || '等待';
}

function buildOcrPipelineSteps(status = {}) {
  const runtime = status.ocr_reader_runtime || {};
  const rapidocr = status.rapidocr || {};
  const tesseract = status.tesseract || {};
  const detail = textValue(runtime.target_selection_detail);
  const contextState = textValue(status.ocr_context_state || runtime.ocr_context_state);
  const lastExcludeReason = textValue(runtime.last_exclude_reason);
  const lastCaptureError = textValue(runtime.last_capture_error);
  const { rawText, observedText, stableText, effectiveText } = getCurrentLineTexts(status);
  const displayStable = stableText || effectiveText;
  const observedKey = compactLineText(observedText);
  const stableKey = compactLineText(displayStable);
  const hasObservedMismatch = Boolean(observedKey && observedKey !== stableKey);
  const blockReason = formatStableBlockReason(runtime.stable_ocr_block_reason);
  const targetName = [
    textValue(runtime.effective_process_name || runtime.process_name),
    textValue(runtime.effective_window_title || runtime.window_title),
  ].filter(Boolean).join(' / ');
  const captureBackend = textValue(runtime.capture_backend_kind || status.ocr_capture_backend_selection || 'auto');
  const ocrBackend = textValue(runtime.backend_kind || status.ocr_backend_selection || 'auto');

  let windowStep = {
    key: 'window',
    state: 'info',
    title: '等待游戏窗口',
    body: '请打开游戏，或点击“选择游戏窗口”。',
    meta: detail ? formatOcrWindowSelectionDetail(detail) : '',
  };
  if (detail === 'memory_reader_window_minimized' || lastExcludeReason === 'excluded_minimized_window') {
    windowStep = {
      key: 'window',
      state: 'warning',
      title: '游戏窗口最小化',
      body: 'OCR 不能截取最小化窗口。',
      meta: formatOcrWindowReason(lastExcludeReason || 'excluded_minimized_window'),
    };
  } else if (textValue(runtime.effective_window_key)) {
    windowStep = {
      key: 'window',
      state: 'ok',
      title: '已确认游戏窗口',
      body: targetName || '已找到可用 OCR 目标窗口。',
      meta: runtime.target_is_foreground ? '前台窗口' : '非前台窗口',
    };
  } else if (Number(runtime.candidate_count || 0) > 0) {
    windowStep = {
      key: 'window',
      state: 'info',
      title: '发现候选窗口',
      body: `发现 ${Number(runtime.candidate_count || 0)} 个候选窗口。`,
      meta: '需要时可手动选择',
    };
  } else if (detail === 'no_eligible_window' || Object.prototype.hasOwnProperty.call(runtime, 'candidate_count')) {
    windowStep = {
      key: 'window',
      state: 'warning',
      title: '没有可识别窗口',
      body: '游戏可能未启动、被最小化，或当前窗口不是游戏。',
      meta: formatOcrWindowSelectionDetail(detail),
    };
  }

  let captureStep = {
    key: 'capture',
    state: 'info',
    title: '等待截图',
    body: '确认窗口后才会截图。',
    meta: captureBackend,
  };
  if (contextState === 'capture_failed' || lastCaptureError) {
    captureStep = {
      key: 'capture',
      state: 'error',
      title: '截图失败',
      body: lastCaptureError || '截图后端返回错误。',
      meta: captureBackend,
    };
  } else if (runtime.stale_capture_backend) {
    captureStep = {
      key: 'capture',
      state: 'warning',
      title: '截图画面未更新',
      body: '截图源可能停在旧画面。',
      meta: `${captureBackend}${runtime.consecutive_same_capture_frames ? ` | 连续 ${runtime.consecutive_same_capture_frames} 帧相同` : ''}`,
    };
  } else if (runtime.last_capture_completed_at || runtime.last_capture_image_hash || runtime.capture_backend_kind) {
    captureStep = {
      key: 'capture',
      state: 'ok',
      title: '截图后端可用',
      body: runtime.last_capture_completed_at ? `最近截图：${runtime.last_capture_completed_at}` : '截图方式已选择。',
      meta: captureBackend,
    };
  }

  let ocrStep = {
    key: 'ocr',
    state: 'info',
    title: '等待文字识别',
    body: '截图完成后会进入文字识别。',
    meta: ocrBackend,
  };
  if (!status.ocr_reader_enabled) {
    ocrStep = {
      key: 'ocr',
      state: 'warning',
      title: 'OCR Reader 未启用',
      body: '当前不会通过 OCR 读取台词。',
      meta: ocrBackend,
    };
  } else if (runtime.backend_detail === 'backend_unavailable' || runtime.detail === 'backend_unavailable') {
    ocrStep = {
      key: 'ocr',
      state: 'error',
      title: 'OCR 后端不可用',
      body: '请安装 RapidOCR 或 Tesseract。',
      meta: ocrBackend,
    };
  } else if (runtime.backend_kind || rawText || observedText || displayStable) {
    ocrStep = {
      key: 'ocr',
      state: 'ok',
      title: '文字识别可用',
      body: rawText ? '最近已有 OCR 原文。' : 'OCR 后端已选定。',
      meta: ocrBackend,
    };
  } else if (!rapidocr.installed && !tesseract.installed) {
    ocrStep = {
      key: 'ocr',
      state: 'warning',
      title: 'OCR 组件可能缺失',
      body: '建议优先安装 RapidOCR。',
      meta: ocrBackend,
    };
  }

  const observedStep = observedText
    ? {
      key: 'observed',
      state: hasObservedMismatch ? 'warning' : 'ok',
      title: hasObservedMismatch ? '候选台词确认中' : '已读到候选台词',
      body: observedText,
      meta: hasObservedMismatch ? (blockReason || '等待稳定确认') : '候选与已确认台词一致',
    }
    : {
      key: 'observed',
      state: rawText ? 'info' : 'warning',
      title: rawText ? '正在筛选 OCR 原文' : '还没有候选台词',
      body: rawText || '等待 OCR 读到有效文字。',
      meta: blockReason || '',
    };

  const stableStep = displayStable
    ? {
      key: 'stable',
      state: 'ok',
      title: '已确认台词',
      body: displayStable,
      meta: status.effective_current_line?.source || runtime.last_stable_line?.source || '',
    }
    : {
      key: 'stable',
      state: observedText ? 'warning' : 'info',
      title: observedText ? '等待稳定确认' : '还没有已确认台词',
      body: observedText ? '候选台词还没有进入正式上下文。' : '读到有效台词后会显示在这里。',
      meta: blockReason || '',
    };

  let agentStep = {
    key: 'agent',
    state: 'info',
    title: '等待 Agent 状态',
    body: 'Agent 会根据模式决定是否自动推进。',
    meta: status.mode || '',
  };
  if (status.agent_user_status === 'error') {
    agentStep = {
      key: 'agent',
      state: 'error',
      title: 'Agent 异常',
      body: status.agent_reason || status.agent_diagnostic || 'Agent 返回错误状态。',
      meta: status.agent_pause_kind || '',
    };
  } else if (status.agent_pause_kind === 'window_not_foreground' || status.agent_user_status === 'paused_window_not_foreground') {
    agentStep = {
      key: 'agent',
      state: 'warning',
      title: '游戏不在前台',
      body: status.agent_pause_message || '自动推进已暂停，切回游戏后继续。',
      meta: AGENT_USER_STATUS_LABELS_ZH[status.agent_user_status] || status.agent_user_status || '',
    };
  } else if (status.agent_pause_kind === 'read_only' || status.agent_user_status === 'read_only') {
    agentStep = {
      key: 'agent',
      state: 'info',
      title: '伴读模式',
      body: '会显示台词和建议，但不会自动点击。',
      meta: AGENT_USER_STATUS_LABELS_ZH[status.agent_user_status] || '',
    };
  } else if (status.agent_user_status || status.agent_status) {
    agentStep = {
      key: 'agent',
      state: 'ok',
      title: 'Agent 状态正常',
      body: status.agent_activity || status.agent_reason || '按当前模式运行。',
      meta: AGENT_USER_STATUS_LABELS_ZH[status.agent_user_status] || status.agent_user_status || status.agent_status || '',
    };
  }

  return [windowStep, captureStep, ocrStep, observedStep, stableStep, agentStep];
}

function renderOcrPipelinePanel(status = {}) {
  const node = document.getElementById('ocrPipelinePanel');
  const stepsNode = document.getElementById('ocrPipelineSteps');
  const summaryNode = document.getElementById('ocrPipelineSummary');
  if (!node || !stepsNode || !summaryNode) {
    return;
  }
  const steps = buildOcrPipelineSteps(status);
  const worstState = steps.some((step) => step.state === 'error')
    ? 'error'
    : steps.some((step) => step.state === 'warning')
      ? 'warning'
      : steps.every((step) => step.state === 'ok')
        ? 'ok'
        : 'info';
  summaryNode.className = `status-chip ${worstState === 'ok' ? 'active' : worstState}`;
  summaryNode.textContent = worstState === 'ok'
    ? '链路正常'
    : worstState === 'error'
      ? '链路异常'
      : worstState === 'warning'
        ? '需要处理'
        : '等待状态';
  stepsNode.innerHTML = steps.map((step) => `
    <article class="ocr-pipeline-step ${escapeHtml(step.state)}">
      <span class="ocr-pipeline-dot">${escapeHtml(pipelineStateLabel(step.state))}</span>
      <div>
        <p class="list-kicker">${escapeHtml(step.key)}</p>
        <h3>${escapeHtml(step.title)}</h3>
        <p>${escapeHtml(step.body || '')}</p>
        ${step.meta ? `<p class="result-note">${escapeHtml(step.meta)}</p>` : ''}
      </div>
    </article>
  `).join('');
}

function installTaskDisplayState(kind) {
  const state = getInstallState(kind);
  const task = state && state.state ? state.state : {};
  if (state?.inProgress) {
    return {
      state: 'running',
      labelText: '安装中',
      needsAttention: true,
    };
  }
  if (task.status === 'failed') {
    return {
      state: 'failed',
      labelText: '安装失败',
      needsAttention: true,
    };
  }
  if (task.status === 'completed') {
    return {
      state: 'installed',
      labelText: '已安装',
      needsAttention: false,
    };
  }
  return null;
}

function dependencySummaryItem(kind, status = {}) {
  const taskState = installTaskDisplayState(kind);
  if (taskState) {
    return {
      kind,
      label: getInstallConfig(kind).label,
      ...taskState,
    };
  }

  if (kind === 'rapidocr') {
    const rapidocr = status.rapidocr || {};
    if (!rapidocr.install_supported) {
      return { kind, label: 'RapidOCR', state: 'neutral', labelText: '不支持自动安装', needsAttention: false };
    }
    return rapidocr.installed
      ? { kind, label: 'RapidOCR', state: 'installed', labelText: '已安装', needsAttention: false }
      : { kind, label: 'RapidOCR', state: 'missing', labelText: '未安装', needsAttention: true };
  }

  if (kind === 'dxcam') {
    const dxcam = status.dxcam || {};
    return dxcam.installed
      ? { kind, label: 'DXcam', state: 'installed', labelText: '已安装', needsAttention: false }
      : { kind, label: 'DXcam', state: 'missing', labelText: '未安装', needsAttention: true };
  }

  if (kind === 'tesseract') {
    const tesseract = status.tesseract || {};
    const missingLanguages = Array.isArray(tesseract.missing_languages) ? tesseract.missing_languages : [];
    if (tesseract.installed && !missingLanguages.length) {
      return { kind, label: 'Tesseract', state: 'installed', labelText: '已安装', needsAttention: false };
    }
    if (tesseract.installed && missingLanguages.length) {
      return {
        kind,
        label: 'Tesseract',
        state: 'warning',
        labelText: `缺少语言包 ${missingLanguages.join(', ')}`,
        needsAttention: true,
      };
    }
    return { kind, label: 'Tesseract', state: 'missing', labelText: '未安装', needsAttention: true };
  }

  const textractor = status.textractor || {};
  return textractor.installed
    ? { kind, label: 'Textractor', state: 'installed', labelText: '已安装', needsAttention: false }
    : { kind, label: 'Textractor', state: 'optional', labelText: '未安装（可选）', needsAttention: false };
}

function renderInstallCompactSummary(status = {}) {
  const summary = document.getElementById('installCompactSummary');
  const module = document.getElementById('dependencyModule');
  if (!summary) {
    return;
  }
  const items = ['rapidocr', 'dxcam', 'tesseract', 'textractor'].map((kind) => dependencySummaryItem(kind, status));
  summary.innerHTML = items.map((item) => `
    <span class="install-summary-chip ${escapeHtml(item.state || 'neutral')}">
      ${escapeHtml(item.label)} ${escapeHtml(item.labelText || item.label || '')}
    </span>
  `).join('');
  const needsAttention = items.some((item) => item.needsAttention);
  if (module && needsAttention) {
    module.open = true;
  }
}

function formatOcrTargetForUser(status = {}) {
  const runtime = status.ocr_reader_runtime || {};
  const processName = runtime.process_name || runtime.effective_process_name || '';
  const title = runtime.window_title || runtime.effective_window_title || '';
  const pid = Number(runtime.pid || 0);
  const parts = [];
  if (processName) {
    parts.push(processName);
  }
  if (title) {
    parts.push(title);
  }
  if (pid) {
    parts.push(`pid ${pid}`);
  }
  return parts.join(' / ');
}

function syncAgentResumeButton(status = {}) {
  const button = document.getElementById('standbyOffBtn');
  const userStatus = status.agent_user_status || '';
  const pauseKind = status.agent_pause_kind || '';
  button.disabled = false;
  if (status.agent_can_resume_by_button || userStatus === 'paused_by_user') {
    button.textContent = '恢复活跃';
    button.dataset.resumeAction = 'standby';
  } else if (pauseKind === 'window_not_foreground' || userStatus === 'paused_window_not_foreground') {
    button.textContent = '请切回游戏窗口';
    button.dataset.resumeAction = 'focus';
  } else if (pauseKind === 'read_only' || userStatus === 'read_only') {
    button.textContent = '只读模式';
    button.dataset.resumeAction = 'read_only';
  } else {
    button.textContent = '恢复活跃';
    button.dataset.resumeAction = 'noop';
  }
}

function isSettingsControlElement(element) {
  return Boolean(element && SETTINGS_CONTROL_IDS.has(element.id || ''));
}

function shouldPreserveSettingsControls() {
  return settingsDirty || settingsSaveInFlight || isSettingsControlElement(document.activeElement);
}

function syncSettingsValue(id, value) {
  if (shouldPreserveSettingsControls()) {
    return;
  }
  const node = document.getElementById(id);
  if (node) {
    node.value = value;
  }
}

function syncSettingsChecked(id, checked) {
  if (shouldPreserveSettingsControls()) {
    return;
  }
  const node = document.getElementById(id);
  if (node) {
    node.checked = Boolean(checked);
  }
}

function renderAgentUserNotice(status = {}) {
  const node = document.getElementById('agentUserNotice');
  const title = document.getElementById('agentUserNoticeTitle');
  const body = document.getElementById('agentUserNoticeBody');
  const target = document.getElementById('agentUserNoticeTarget');
  const userStatus = status.agent_user_status || '';
  const pauseKind = status.agent_pause_kind || 'none';
  const label = AGENT_USER_STATUS_LABELS_ZH[userStatus] || userStatus || '等待状态';
  const targetText = formatOcrTargetForUser(status);
  const mode = status.mode || '';
  const waitingInAutoMode = userStatus === 'read_only' && mode === 'choice_advisor';
  const displayLabel = waitingInAutoMode ? '等待可操作状态' : label;
  const displayPauseMessage = waitingInAutoMode && !status.agent_pause_message
    ? '自动推进已开启，正在等待游戏会话、OCR 台词或目标窗口进入可操作状态。'
    : status.agent_pause_message;

  node.hidden = false;
  title.textContent = displayLabel;
  body.textContent = displayPauseMessage
    || (userStatus === 'read_only' && status.mode === 'companion'
      ? '游戏窗口已在前台，但伴读模式不会自动推进。需要自动推进时请切到自动推进模式。'
      : '')
    || (userStatus === 'running' && status.mode === 'choice_advisor'
      ? '游戏窗口已在前台，Agent 会按自动推进模式继续。OCR 会在后台持续刷新。'
      : '')
    || (userStatus === 'running'
      ? 'Agent 正在按当前模式运行。OCR 会在后台持续刷新。'
      : 'Agent 状态会随游戏窗口、OCR 和模式设置自动更新。');
  target.textContent = targetText ? `目标窗口：${targetText}` : '';

  node.className = 'agent-user-notice neutral';
  if (pauseKind === 'window_not_foreground' || userStatus === 'paused_window_not_foreground') {
    node.classList.add('warning');
  } else if (pauseKind === 'ocr_unavailable' || userStatus === 'ocr_unavailable' || userStatus === 'error') {
    node.classList.add('error');
  } else if (pauseKind === 'read_only' || pauseKind === 'user' || userStatus === 'read_only' || userStatus === 'paused_by_user') {
    node.classList.add('read-only');
  }
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

function buildOcrMissingLineDiagnostic(status = {}) {
  const runtime = status.ocr_reader_runtime || {};
  const rapidocr = status.rapidocr || {};
  const parts = [
    status.ocr_capture_diagnostic_required
      ? 'OCR 截图区/窗口目标可能异常'
      : 'OCR 尚未读到可解释台词',
  ];
  if (status.ocr_capture_diagnostic) {
    parts.push(status.ocr_capture_diagnostic);
  }
  if (status.agent_reason) {
    parts.push(`agent_reason=${status.agent_reason}`);
  }
  if (status.agent_diagnostic) {
    parts.push(status.agent_diagnostic);
  }
  if (runtime.status) {
    parts.push(`status=${runtime.status}`);
  }
  if (runtime.detail) {
    parts.push(`detail=${runtime.detail}`);
  }
  if (runtime.ocr_context_state) {
    parts.push(`context_state=${runtime.ocr_context_state}`);
  }
  if (runtime.backend_kind) {
    parts.push(`backend=${runtime.backend_kind}`);
  }
  if (runtime.backend_detail) {
    parts.push(`backend_detail=${runtime.backend_detail}`);
  }
  if (typeof status.bridge_poll_running === 'boolean') {
    parts.push(`bridge_poll_running=${status.bridge_poll_running}`);
  }
  if (typeof status.bridge_poll_inflight_seconds === 'number') {
    parts.push(`bridge_poll_inflight=${status.bridge_poll_inflight_seconds.toFixed(1)}s`);
  }
  if (typeof status.last_bridge_poll_duration_seconds === 'number') {
    parts.push(`last_poll_duration=${status.last_bridge_poll_duration_seconds.toFixed(1)}s`);
  }
  if (typeof status.pending_ocr_advance_captures === 'number' && status.pending_ocr_advance_captures > 0) {
    parts.push(`pending_ocr=${status.pending_ocr_advance_captures}`);
  }
  if (status.last_ocr_advance_capture_reason) {
    parts.push(`ocr_reason=${status.last_ocr_advance_capture_reason}`);
  }
  if (status.last_error?.message) {
    parts.push(`last_error=${status.last_error.message}`);
  }
  if (rapidocr.detail && rapidocr.detail !== 'installed') {
    parts.push(`rapidocr=${rapidocr.detail}`);
  }
  if (runtime.capture_stage) {
    parts.push(`stage=${runtime.capture_stage}`);
  }
  if (runtime.capture_profile) {
    parts.push(`capture=${formatCaptureProfile(runtime.capture_profile)}`);
  }
  if (runtime.consecutive_no_text_polls) {
    parts.push(`no_text_polls=${runtime.consecutive_no_text_polls}`);
  }
  if (runtime.last_observed_at) {
    parts.push(`last_observed_at=${runtime.last_observed_at}`);
  }
  if (runtime.last_capture_error) {
    parts.push(`last_capture_error=${runtime.last_capture_error}`);
  }
  if (runtime.last_raw_ocr_text) {
    parts.push(`last_raw=${String(runtime.last_raw_ocr_text).slice(0, 80)}`);
  }
  return parts.join(' | ');
}

function normalizeLineText(value = '') {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function formatOcrTriggerMode(value = '') {
  return value === 'after_advance' ? '点击对白后识别' : '按间隔识别';
}

function formatFixedNumber(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '0.0';
}

function isLikelyGameDialogueLine(item = {}) {
  if (!item || item.is_diagnostic) {
    return false;
  }
  const text = normalizeLineText(item.text || '');
  if (!text || text.length < 2 || text.length > 220) {
    return false;
  }
  const lowered = text.toLowerCase();
  const blockedTokens = [
    'agent',
    'capture_failed',
    'context_state=',
    'dxcam:',
    'galgame_',
    'gateway_unavailable',
    'http://',
    'https://',
    'last_error=',
    'ocr_context_unavailable',
    'plugin/',
    'plugin\\',
    'powershell',
    'status=',
    'stability',
    '当前快照',
    '场景 id',
    '场景id',
    '会话 id',
    '会话id',
    '游戏 id',
    '游戏id',
    '菜单是否打开',
    '台词 id',
    '台词id',
    '路线 id',
    '路线id',
    '快照时间',
    '是否过期',
    '退出全屏',
    '收起',
    '全屏',
    'ocr 诊断',
    'recent raw ocr',
    '最近 raw ocr',
  ];
  if (blockedTokens.some((token) => lowered.includes(token))) {
    return false;
  }
  if (text.startsWith('{') || text.startsWith('[') || (text.includes('{') && text.includes('}'))) {
    return false;
  }
  const hasDialoguePunctuation = /[。！？!?…]|——|「|」|『|』|“|”/.test(text);
  const hasWeakDialoguePunctuation = /[，,、：:]/.test(text);
  const hasSpeaker = Boolean(String(item.speaker || '').trim());
  if (hasSpeaker || hasDialoguePunctuation) {
    return true;
  }
  return hasWeakDialoguePunctuation && text.replace(/\s+/g, '').length >= 8;
}

function lineKey(item = {}) {
  const text = normalizeLineText(item.text || '');
  if (text) {
    return [
      item.scene_id || '',
      item.speaker || '',
      text,
    ].join('::');
  }
  return String(item.line_id || '').trim();
}

function mergedHistoryLines(history = {}) {
  const merged = new Map();
  (history.observed_lines || []).forEach((item) => {
    if (!isLikelyGameDialogueLine(item)) {
      return;
    }
    merged.set(lineKey(item), { ...item, stability: item.stability || 'tentative' });
  });
  (history.stable_lines || []).forEach((item) => {
    if (!isLikelyGameDialogueLine(item)) {
      return;
    }
    merged.set(lineKey(item), { ...item, stability: item.stability || 'stable' });
  });
  return Array.from(merged.values());
}

function scrollToBottom(node) {
  if (!node) {
    return;
  }
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      node.scrollTop = node.scrollHeight;
    });
  });
}

function scrollAllRegionsToBottom(root = document) {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      root.querySelectorAll('.scroll-region, .reply-text-scroll').forEach((node) => {
        node.scrollTop = node.scrollHeight;
      });
      if (root.classList?.contains('panel-fullscreen')) {
        root.scrollTop = root.scrollHeight;
      }
    });
  });
}

function isScrollableNode(node) {
  return Boolean(node && node.scrollHeight > node.clientHeight + 1);
}

function canScrollNode(node, deltaY) {
  if (!isScrollableNode(node)) {
    return false;
  }
  if (deltaY < 0) {
    return node.scrollTop > 0;
  }
  if (deltaY > 0) {
    return node.scrollTop + node.clientHeight < node.scrollHeight - 1;
  }
  return true;
}

function eventElement(eventTarget) {
  if (eventTarget instanceof Element) {
    return eventTarget;
  }
  return eventTarget?.parentElement || null;
}

function fullscreenWheelTarget(eventTarget, deltaY) {
  const element = eventElement(eventTarget);
  const panel = element?.closest?.('.panel-fullscreen');
  if (!panel) {
    return null;
  }
  let node = element;
  while (node && node !== panel.parentElement) {
    if (
      node.matches?.('.scroll-region, .reply-text-scroll, .module-body, .list-card, .panel-fullscreen')
      && canScrollNode(node, deltaY)
    ) {
      return node;
    }
    if (node === panel) {
      break;
    }
    node = node.parentElement;
  }
  const nested = Array.from(panel.querySelectorAll('.scroll-region, .reply-text-scroll, .module-body, .list-card'))
    .find((candidate) => canScrollNode(candidate, deltaY));
  return nested || (canScrollNode(panel, deltaY) ? panel : null);
}

function pageWheelTarget(eventTarget, deltaY) {
  let node = eventElement(eventTarget);
  while (node && node !== document.body) {
    if (
      node.matches?.('.scroll-region, .reply-text-scroll, .module-body, .list-card')
      && canScrollNode(node, deltaY)
    ) {
      return node;
    }
    node = node.parentElement;
  }
  const scroller = document.scrollingElement || document.documentElement || document.body;
  return canScrollNode(scroller, deltaY) ? scroller : null;
}

function exitPanelFullscreen() {
  document.querySelectorAll('.panel-fullscreen').forEach((panel) => {
    panel.classList.remove('panel-fullscreen');
    const button = panel.querySelector('.panel-fullscreen-toggle');
    if (button) {
      button.textContent = '全屏';
      button.setAttribute('aria-label', '全屏');
    }
  });
  document.body.classList.remove('panel-fullscreen-active');
}

function togglePanelFullscreen(panel) {
  if (!panel) {
    return;
  }
  const isActive = panel.classList.contains('panel-fullscreen');
  exitPanelFullscreen();
  if (isActive) {
    return;
  }
  panel.open = true;
  panel.classList.add('panel-fullscreen');
  document.body.classList.add('panel-fullscreen-active');
  const button = panel.querySelector('.panel-fullscreen-toggle');
  if (button) {
    button.textContent = '退出全屏';
    button.setAttribute('aria-label', '退出全屏');
  }
  scrollAllRegionsToBottom(panel);
}

function initializePanelFullscreenControls() {
  document.querySelectorAll('.dashboard-module, .settings-module').forEach((panel) => {
    const summary = panel.querySelector(':scope > summary');
    if (!summary || summary.querySelector('.panel-fullscreen-toggle')) {
      return;
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'panel-fullscreen-toggle';
    button.textContent = '全屏';
    button.setAttribute('aria-label', '全屏');
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      togglePanelFullscreen(panel);
    });
    summary.appendChild(button);
  });
}

function latestHistoryLine(history = {}) {
  const lines = mergedHistoryLines(history);
  return lines.length ? lines[lines.length - 1] : null;
}

function sceneSummaryKey(sceneId = '', history = {}, fallbackLine = {}) {
  const stableLines = Array.isArray(history.stable_lines) ? history.stable_lines : [];
  const observedLines = Array.isArray(history.observed_lines) ? history.observed_lines : [];
  const latestLine = latestHistoryLine(history) || fallbackLine || {};
  return [
    sceneId || 'missing-scene',
    stableLines.length,
    observedLines.length,
    lineKey(latestLine),
  ].join('::');
}

function effectiveCurrentLine(snapshot = {}, history = {}, status = {}) {
  const state = snapshot.snapshot || {};
  if (state.line_id && state.text) {
    return { ...state, source: 'snapshot', stability: state.stability || '' };
  }
  if (snapshot.effective_current_line?.line_id && snapshot.effective_current_line?.text) {
    return { ...snapshot.effective_current_line };
  }
  if (status.effective_current_line?.line_id && status.effective_current_line?.text) {
    return { ...status.effective_current_line };
  }
  return latestHistoryLine(history) || {};
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
  scrollToBottom(container);
}

function renderStackList(nodeId, items, formatter) {
  const node = document.getElementById(nodeId);
  if (!items.length) {
    node.className = 'stack-list scroll-region empty-state';
    node.textContent = '暂无数据';
    scrollToBottom(node);
    return;
  }
  node.className = 'stack-list scroll-region';
  node.innerHTML = items.map(formatter).join('');
  scrollToBottom(node);
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

function normalizeProcessName(value) {
  return String(value || '').trim().toLowerCase();
}

function normalizeCaptureProfileSaveScope(value) {
  const normalized = String(value || '').trim();
  return normalized === 'window_bucket' ? 'window_bucket' : 'process_fallback';
}

function normalizeCaptureProfileBucketKey(value) {
  return String(value || '').trim().toLowerCase();
}

function buildCaptureProfileBucketKey(width, height) {
  const normalizedWidth = Math.max(0, Number(width || 0));
  const normalizedHeight = Math.max(0, Number(height || 0));
  if (!normalizedWidth || !normalizedHeight) {
    return '';
  }
  return `${Math.round(normalizedWidth)}x${Math.round(normalizedHeight)}`;
}

function isRatioProfileValue(value) {
  return Boolean(value)
    && typeof value === 'object'
    && ['left_inset_ratio', 'right_inset_ratio', 'top_ratio', 'bottom_inset_ratio']
      .every((key) => typeof value[key] === 'number' && Number.isFinite(value[key]));
}

function isAihongProcessName(value) {
  return AIHONG_PROCESS_NAMES.has(normalizeProcessName(value));
}

function findStoredCaptureProfileEntry(status, processName) {
  const profiles = status?.ocr_capture_profiles || {};
  const direct = profiles[processName];
  if (direct) {
    return direct;
  }
  const normalizedProcessName = normalizeProcessName(processName);
  return Object.entries(profiles).find(([name]) => normalizeProcessName(name) === normalizedProcessName)?.[1] || null;
}

function resolveStoredFallbackCaptureProfile(entry, stage) {
  if (!entry || typeof entry !== 'object') {
    return null;
  }
  if (isRatioProfileValue(entry)) {
    return entry;
  }
  const stageEntry = entry[stage];
  if (isRatioProfileValue(stageEntry)) {
    return stageEntry;
  }
  const defaultEntry = entry.default;
  if (isRatioProfileValue(defaultEntry)) {
    return defaultEntry;
  }
  return null;
}

function resolveStoredBucketCaptureProfile(entry, stage, bucketKey) {
  if (!entry || typeof entry !== 'object' || !bucketKey) {
    return null;
  }
  const buckets = entry.__window_buckets__;
  if (!buckets || typeof buckets !== 'object') {
    return null;
  }
  const normalizedBucketKey = normalizeCaptureProfileBucketKey(bucketKey);
  const directBucket = buckets[normalizedBucketKey];
  const bucketEntry = directBucket
    || Object.entries(buckets).find(([key]) => normalizeCaptureProfileBucketKey(key) === normalizedBucketKey)?.[1]
    || null;
  if (!bucketEntry || typeof bucketEntry !== 'object') {
    return null;
  }
  const bucketStages = bucketEntry.stages;
  if (!bucketStages || typeof bucketStages !== 'object') {
    return null;
  }
  const stageEntry = bucketStages[stage];
  if (isRatioProfileValue(stageEntry)) {
    return stageEntry;
  }
  const defaultEntry = bucketStages.default;
  if (isRatioProfileValue(defaultEntry)) {
    return defaultEntry;
  }
  return null;
}

function resolveRuntimeDefaultSaveScope(status, processName) {
  const runtime = status?.ocr_reader_runtime || {};
  return normalizeProcessName(processName)
    && normalizeProcessName(processName) === normalizeProcessName(runtime.process_name)
    && Number(runtime.width || 0) > 0
    && Number(runtime.height || 0) > 0
    ? 'window_bucket'
    : 'process_fallback';
}

function resolveEditableCaptureProfile(status, processName, stage, saveScope) {
  const runtime = status?.ocr_reader_runtime || {};
  const entry = findStoredCaptureProfileEntry(status, processName);
  const normalizedScope = normalizeCaptureProfileSaveScope(saveScope);
  const runtimeProcessMatches = normalizeProcessName(processName)
    && normalizeProcessName(processName) === normalizeProcessName(runtime.process_name);
  const runtimeBucketKey = normalizeCaptureProfileBucketKey(
    runtime.capture_profile_bucket_key || buildCaptureProfileBucketKey(runtime.width, runtime.height),
  );

  if (normalizedScope === 'window_bucket') {
    const storedBucketProfile = resolveStoredBucketCaptureProfile(entry, stage, runtimeBucketKey);
    if (storedBucketProfile) {
      return storedBucketProfile;
    }
    if (runtimeProcessMatches && runtime.capture_profile && runtimeBucketKey) {
      return runtime.capture_profile;
    }
  } else {
    const storedFallbackProfile = resolveStoredFallbackCaptureProfile(entry, stage);
    if (storedFallbackProfile) {
      return storedFallbackProfile;
    }
    if (
      runtimeProcessMatches
      && runtime.capture_profile
      && !['bucket_exact', 'bucket_aspect_nearest'].includes(String(runtime.capture_profile_match_source || ''))
    ) {
      return runtime.capture_profile;
    }
  }
  if (isAihongProcessName(processName) && AIHONG_CAPTURE_PRESETS[stage]) {
    return AIHONG_CAPTURE_PRESETS[stage];
  }
  return DEFAULT_CAPTURE_PROFILE;
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
  const { card, statusText, percentText, messageText, detailText, progressBar, button } = getInstallNodes(kind);
  const { label } = getInstallConfig(kind);

  if (!state) {
    card.hidden = true;
    card.style.display = '';
    button.hidden = false;
    button.disabled = false;
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
  card.style.display = '';
  statusText.textContent = `${formatInstallPhase(state.phase)} · ${state.status || ''}`;
  percentText.textContent = `${percent}%`;
  messageText.textContent = state.message || '';
  detailText.textContent = details.join(' · ');
  progressBar.style.width = `${percent}%`;
  if (state.status === 'completed') {
    button.hidden = true;
    button.disabled = true;
  } else if (state.status === 'failed') {
    button.hidden = false;
    button.disabled = false;
    button.textContent = getInstallConfig(kind).retryText;
  }
}

function renderPluginUnavailable(error) {
  latestStatus = null;
  const message = error instanceof Error ? error.message : String(error || '插件尚未启动');
  document.getElementById('summaryText').textContent = '插件尚未启动';
  renderPrimaryDiagnosis({
    connection_state: 'plugin_not_started',
    last_error: { message },
    summary: '插件尚未启动',
  });
  renderFirstRunGuide({});
  renderCurrentLineOverview({});
  renderOcrPipelinePanel({});
  renderInstallCompactSummary({});
  renderGrid('statusGrid', [
    { label: 'connection_state', value: 'plugin_not_started' },
    { label: 'status', value: '插件尚未启动' },
    { label: 'last_error', value: message },
  ]);
  renderGrid('ocrRuntimeGrid', [
    { label: 'status', value: '插件尚未启动' },
  ]);
  renderGrid('snapshotGrid', [
    { label: 'status', value: '插件尚未启动' },
  ]);

  for (const kind of ['rapidocr', 'dxcam', 'tesseract', 'textractor']) {
    const config = getInstallConfig(kind);
    const { button, card } = getInstallNodes(kind);
    const banner = document.getElementById(`${config.domPrefix}Prompt`);
    const kicker = document.getElementById(`${config.domPrefix}PromptKicker`);
    const title = document.getElementById(`${config.domPrefix}PromptTitle`);
    const body = document.getElementById(`${config.domPrefix}PromptBody`);
    const path = document.getElementById(`${config.domPrefix}PathText`);
    banner.className = `install-banner install-banner-${kind} neutral`;
    kicker.textContent = config.label;
    title.textContent = '插件尚未启动';
    body.textContent = '当前无法读取插件运行状态。请先启动或重载 galgame_plugin，启动完成后这里会显示安装和运行时状态。';
    path.textContent = message;
    card.hidden = true;
    card.style.display = 'none';
    button.hidden = true;
    button.disabled = true;
  }
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

async function restoreDxcamInstallState() {
  await restoreInstallState('dxcam');
}

async function restoreTesseractInstallState() {
  await restoreInstallState('tesseract');
}

function renderStatus(status) {
  latestStatus = status;
  document.getElementById('summaryText').textContent = buildStatusSummaryText(status);
  syncSettingsValue('modeSelect', status.mode || 'companion');
  syncSettingsChecked('pushToggle', Boolean(status.push_notifications));
  syncSettingsValue('advanceSpeedSelect', status.advance_speed || 'medium');
  syncSettingsValue('readerModeSelect', status.reader_mode || 'auto');
  const ocrPollIntervalInput = document.getElementById('ocrPollIntervalInput');
  if (ocrPollIntervalInput && !shouldPreserveSettingsControls()) {
    const interval = Number(status.ocr_reader_poll_interval_seconds || 2);
    ocrPollIntervalInput.value = Number.isFinite(interval) ? interval.toFixed(1) : '2.0';
  }
  syncSettingsValue('ocrTriggerModeSelect', status.ocr_reader_trigger_mode || 'after_advance');

  const memoryReaderRuntime = status.memory_reader_runtime || {};
  const ocrRuntime = status.ocr_reader_runtime || {};
  const rapidocr = status.rapidocr || {};
  const dxcam = status.dxcam || {};
  const textractor = status.textractor || {};
  const tesseract = status.tesseract || {};
  const performance = status.performance || {};

  const memoryReaderProcess = memoryReaderRuntime.process_name
    ? `${memoryReaderRuntime.process_name} (${memoryReaderRuntime.pid || 0})`
    : '';
  const ocrTarget = ocrRuntime.process_name
    ? `${ocrRuntime.process_name} (${ocrRuntime.pid || 0})`
    : '';
  const missingLanguages = (tesseract.missing_languages || []).join(', ');
  const performanceProcess = performance.process_name
    ? `${performance.process_name} (${performance.pid || 0})`
    : String(performance.pid || '');

  renderPrimaryDiagnosis(status);
  renderFirstRunGuide(status);
  renderCurrentLineOverview(status);
  renderOcrPipelinePanel(status);
  renderInstallCompactSummary(status);

  renderGrid('statusGrid', [
    { label: 'connection_state', value: status.connection_state || '' },
    { label: 'active_data_source', value: status.active_data_source || '' },
    { label: 'reader_mode', value: READER_MODE_LABELS_ZH[status.reader_mode] || status.reader_mode || 'auto' },
    { label: 'mode', value: status.mode || '' },
    {
      label: 'agent_user_status',
      value: AGENT_USER_STATUS_LABELS_ZH[status.agent_user_status] || status.agent_user_status || '',
    },
    { label: 'agent_pause_kind', value: status.agent_pause_kind || '' },
    { label: 'agent_pause_message', value: status.agent_pause_message || '' },
    { label: 'agent_can_resume_by_button', value: String(Boolean(status.agent_can_resume_by_button)) },
    { label: 'agent_can_resume_by_focus', value: String(Boolean(status.agent_can_resume_by_focus)) },
    { label: 'agent_status', value: status.agent_status || '' },
    { label: 'agent_activity', value: status.agent_activity || '' },
    { label: 'agent_reason', value: status.agent_reason || '' },
    { label: 'agent_diagnostic', value: status.agent_diagnostic || '' },
    { label: 'inbound_queue_size', value: String(status.agent_inbound_queue_size || 0) },
    { label: 'outbound_queue_size', value: String(status.agent_outbound_queue_size || 0) },
    { label: 'push_notifications', value: String(Boolean(status.push_notifications)) },
    { label: 'advance_speed', value: ADVANCE_SPEED_LABELS_ZH[status.advance_speed] || status.advance_speed || 'medium' },
    { label: 'bound_game_id', value: status.bound_game_id || '(auto)' },
    { label: 'active_session_id', value: status.active_session_id || '' },
    { label: 'last_seq', value: String(status.last_seq || 0) },
    { label: 'stream_reset_pending', value: String(Boolean(status.stream_reset_pending)) },
    { label: 'available_game_ids', value: (status.available_game_ids || []).join(', ') || '(none)' },
    { label: 'performance_cpu_percent', value: `${formatFixedNumber(performance.cpu_percent, 1)}%` },
    { label: 'performance_memory_mb', value: `${formatFixedNumber(performance.memory_mb, 1)} MB` },
    { label: 'performance_memory_percent', value: `${formatFixedNumber(performance.memory_percent, 2)}%` },
    { label: 'performance_thread_count', value: String(performance.thread_count || 0) },
    { label: 'performance_process', value: performanceProcess || '' },
    { label: 'performance_detail', value: performance.detail || '' },
    { label: 'ocr_reader_enabled', value: String(Boolean(status.ocr_reader_enabled)) },
    { label: 'ocr_poll_interval_seconds', value: String(status.ocr_reader_poll_interval_seconds || '') },
    { label: 'ocr_trigger_mode', value: formatOcrTriggerMode(status.ocr_reader_trigger_mode || 'after_advance') },
    { label: 'pending_ocr_advance_captures', value: String(status.pending_ocr_advance_captures || 0) },
    {
      label: 'pending_ocr_advance_capture_age_seconds',
      value: formatFixedNumber(status.pending_ocr_advance_capture_age_seconds, 1),
    },
    { label: 'last_ocr_advance_capture_reason', value: status.last_ocr_advance_capture_reason || '' },
    { label: 'ocr_reader_status', value: ocrRuntime.status || '' },
    { label: 'ocr_reader_detail', value: ocrRuntime.detail || '' },
    { label: 'ocr_context_state', value: ocrRuntime.ocr_context_state || '' },
    { label: 'target_is_foreground', value: String(Boolean(ocrRuntime.target_is_foreground)) },
    { label: 'effective_current_line', value: status.effective_current_line?.text || '' },
    { label: 'ocr_capture_diagnostic_required', value: String(Boolean(status.ocr_capture_diagnostic_required)) },
    { label: 'ocr_capture_diagnostic', value: status.ocr_capture_diagnostic || '' },
    { label: 'ocr_reader_target', value: ocrTarget || '' },
    { label: 'ocr_backend_selection', value: status.ocr_backend_selection || 'auto' },
    { label: 'ocr_capture_backend_selection', value: status.ocr_capture_backend_selection || 'auto' },
    { label: 'ocr_backend_kind', value: ocrRuntime.backend_kind || '' },
    { label: 'ocr_backend_detail', value: ocrRuntime.backend_detail || '' },
    { label: 'rapidocr_enabled', value: String(Boolean(status.rapidocr_enabled)) },
    { label: 'rapidocr_installed', value: String(Boolean(rapidocr.installed)) },
    { label: 'rapidocr_detail', value: rapidocr.detail || '' },
    { label: 'dxcam_installed', value: String(Boolean(dxcam.installed)) },
    { label: 'dxcam_detail', value: dxcam.detail || '' },
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
  renderAgentUserNotice(status);
  syncAgentResumeButton(status);
  syncAutoRefreshIntervalForStatus(status);
  renderRapidOcr(status);
  renderDxcam(status);
  renderTesseract(status);
  renderTextractor(status);
  renderOcrWindowTargetStatus(status);
  renderOcrProfile(status);
  renderGameBinding(status);
}

function renderGameBinding(status) {
  const currentNode = document.getElementById('currentBoundGameId');
  const detailNode = document.getElementById('currentBoundGameDetail');
  const listNode = document.getElementById('availableGameIds');
  if (!currentNode || !listNode) {
    return;
  }

  const boundGameId = String(status.bound_game_id || '').trim();
  const gameIds = Array.isArray(status.available_game_ids) ? status.available_game_ids : [];
  const boundDescription = describeGameBindingId(boundGameId);
  currentNode.textContent = boundGameId ? `已固定：${boundDescription.title}` : '自动选择游戏窗口';
  if (detailNode) {
    detailNode.textContent = boundGameId
      ? `${boundDescription.detail}。点“恢复自动”后，插件会重新按当前可用目标选择。`
      : '插件会优先选择当前可用目标。需要固定目标时，点击下面的候选项。';
  }

  if (!gameIds.length) {
    listNode.className = 'binding-chip-row empty-inline';
    listNode.textContent = '未发现可绑定游戏。请确认 Bridge/OCR/Memory Reader 已连接到游戏窗口。';
    return;
  }

  const normalizedGameIds = gameIds.map((gameId) => String(gameId || '').trim()).filter(Boolean);
  if (!normalizedGameIds.length) {
    listNode.className = 'binding-chip-row empty-inline';
    listNode.textContent = '可用游戏 ID 为空。';
    return;
  }

  listNode.className = 'binding-chip-row';
  listNode.replaceChildren(
    ...normalizedGameIds
      .map((normalized) => {
        const active = normalized === boundGameId;
        const description = describeGameBindingId(normalized);
        const button = document.createElement('button');
        button.className = `binding-chip${active ? ' active' : ''}`;
        button.dataset.gameId = normalized;
        button.disabled = active;
        const title = document.createElement('span');
        title.className = 'binding-chip-title';
        title.textContent = active ? `当前：${description.title}` : description.title;
        const detail = document.createElement('span');
        detail.className = 'binding-chip-detail';
        detail.textContent = description.detail;
        button.replaceChildren(title, detail);
        return button;
      }),
  );

  listNode.querySelectorAll('[data-game-id]').forEach((button) => {
    button.addEventListener('click', () => {
      const gameId = button.getAttribute('data-game-id') || '';
      withButtonPending(button, '绑定中...', () => bindGame(gameId)).catch(() => {});
    });
  });
}

function describeGameBindingId(gameId) {
  const normalized = String(gameId || '').trim();
  if (!normalized) {
    return {
      title: '自动选择游戏窗口',
      detail: '插件会优先选择当前可用目标',
    };
  }
  const [prefix, ...rest] = normalized.split('-');
  const suffix = rest.join('-') || normalized;
  if (prefix === 'mem') {
    return {
      title: '内存读取目标',
      detail: `ID ${suffix}`,
    };
  }
  if (prefix === 'ocr') {
    return {
      title: 'OCR 窗口目标',
      detail: `ID ${suffix}`,
    };
  }
  return {
    title: '游戏目标',
    detail: normalized,
  };
}

function formatConnectionStateZh(value) {
  const normalized = String(value || '').trim();
  return CONNECTION_STATE_LABELS_ZH[normalized] || normalized || '未知';
}

function formatModeZh(value) {
  const normalized = String(value || '').trim();
  return MODE_LABELS_ZH[normalized] || normalized || '未知模式';
}

function formatDataSourceZh(value) {
  const normalized = String(value || '').trim();
  return DATA_SOURCE_LABELS_ZH[normalized] || normalized || '未知来源';
}

function buildStatusSummaryText(status) {
  if (!status || typeof status !== 'object') {
    return '无摘要';
  }

  const source = String(status.active_data_source || '').trim();
  const sessionId = String(status.active_session_id || '').trim();
  const boundGameId = String(status.bound_game_id || '').trim();
  const connectionState = formatConnectionStateZh(status.connection_state);
  const mode = formatModeZh(status.mode);
  const lastSeq = String(status.last_seq || 0);
  const warningMessage = typeof status.last_error?.message === 'string'
    ? status.last_error.message.trim()
    : '';

  let prefix = '';
  if (source === 'ocr_reader' && sessionId) {
    prefix = '已通过 OCR 读取连接（降级模式）';
  } else if (source === 'memory_reader' && sessionId) {
    prefix = '已通过内存读取连接（降级模式）';
  } else if (source === 'bridge_sdk' && sessionId) {
    prefix = '已通过 Bridge SDK 连接';
  } else if (status.connection_state === 'stale') {
    prefix = '当前桥接快照已过期';
  } else if (status.connection_state === 'active') {
    prefix = '当前桥接链路运行中';
  } else {
    prefix = `当前数据源：${formatDataSourceZh(source)}`;
  }

  const parts = [
    `状态：${connectionState}`,
    `模式：${mode}`,
  ];

  if (boundGameId) {
    parts.push(`绑定：${boundGameId}`);
  }
  if (sessionId) {
    parts.push(`会话：${sessionId}`);
  }
  parts.push(`最新序号：${lastSeq}`);

  if (warningMessage) {
    parts.push(`告警：${warningMessage}`);
  }
  if (status.ocr_capture_diagnostic_required) {
    parts.push(`OCR诊断：${status.ocr_context_state || ocrRuntimeState(status) || '截图区/窗口目标可能异常'}`);
  }
  if (status.agent_diagnostic_required || status.agent_reason) {
    const agentText = status.agent_diagnostic || status.agent_reason || status.agent_status || '';
    if (agentText) {
      parts.push(`Agent：${agentText}`);
    }
  }

  return `${prefix}｜${parts.join('｜')}`;
}

function ocrRuntimeState(status) {
  const runtime = status?.ocr_reader_runtime || {};
  return readOcrRuntimeValue(runtime, 'ocr', 'context_state', 'ocr_context_state') || runtime.detail || '';
}

function ocrRuntimeGroup(runtime, groupName) {
  const group = runtime?.[groupName];
  return group && typeof group === 'object' ? group : {};
}

function readOcrRuntimeValue(runtime, groupName, groupKey, legacyKey = groupKey) {
  const group = ocrRuntimeGroup(runtime, groupName);
  const groupedValue = group[groupKey];
  if (groupedValue !== undefined && groupedValue !== null && groupedValue !== '') {
    return groupedValue;
  }
  const legacyValue = runtime?.[legacyKey];
  return legacyValue !== undefined && legacyValue !== null ? legacyValue : '';
}

function formatOcrRuntimeSeconds(value) {
  if (value === undefined || value === null || value === '') {
    return '';
  }
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue.toFixed(2) : '';
}

function renderOcrRuntime(status) {
  const runtime = status.ocr_reader_runtime || {};
  const fromWindow = (key, legacyKey = key) => readOcrRuntimeValue(runtime, 'window', key, legacyKey);
  const fromCapture = (key, legacyKey = key) => readOcrRuntimeValue(runtime, 'capture', key, legacyKey);
  const fromOcr = (key, legacyKey = key) => readOcrRuntimeValue(runtime, 'ocr', key, legacyKey);
  const fromTiming = (key, legacyKey = key) => readOcrRuntimeValue(runtime, 'timing', key, legacyKey);
  const fromAdvance = (key, legacyKey = key) => readOcrRuntimeValue(runtime, 'advance', key, legacyKey);
  const windowTitle = fromWindow('title', 'window_title');
  const captureStage = fromCapture('stage', 'capture_stage');
  const captureProfile = fromCapture('profile', 'capture_profile');
  const captureProfileMatchSource = fromCapture('profile_match_source', 'capture_profile_match_source');
  const lastCaptureStage = fromCapture('last_stage', 'last_capture_stage');
  const lastCaptureProfile = fromCapture('last_profile', 'last_capture_profile');
  renderGrid('ocrRuntimeGrid', [
    { label: 'status', value: runtime.status || '' },
    { label: 'detail', value: runtime.detail || '' },
    { label: 'process_name', value: fromWindow('process_name') || '' },
    { label: 'pid', value: String(fromWindow('pid') || 0) },
    { label: 'window_title', value: windowTitle || '' },
    { label: 'width', value: String(fromWindow('width') || 0) },
    { label: 'height', value: String(fromWindow('height') || 0) },
    { label: 'aspect_ratio', value: fromWindow('aspect_ratio') ? Number(fromWindow('aspect_ratio')).toFixed(4) : '' },
    { label: 'game_id', value: runtime.game_id || '' },
    { label: 'session_id', value: runtime.session_id || '' },
    { label: 'last_seq', value: String(runtime.last_seq || 0) },
    { label: 'last_event_ts', value: runtime.last_event_ts || '' },
    { label: 'capture_stage', value: OCR_PROFILE_STAGE_LABELS_ZH[captureStage] || captureStage || '通用区域' },
    { label: 'capture_profile', value: formatCaptureProfile(captureProfile) || '(default)' },
    {
      label: 'capture_profile_match_source',
      value: OCR_CAPTURE_MATCH_SOURCE_LABELS_ZH[captureProfileMatchSource] || captureProfileMatchSource || '',
    },
    { label: 'capture_profile_bucket_key', value: fromCapture('profile_bucket_key', 'capture_profile_bucket_key') || '' },
    { label: 'consecutive_no_text_polls', value: String(fromOcr('consecutive_no_text_polls') || 0) },
    { label: 'last_observed_at', value: fromOcr('last_observed_at') || '' },
    { label: 'last_capture_stage', value: OCR_PROFILE_STAGE_LABELS_ZH[lastCaptureStage] || lastCaptureStage || '' },
    { label: 'last_capture_profile', value: formatCaptureProfile(lastCaptureProfile) || '' },
    { label: 'ocr_context_state', value: fromOcr('context_state', 'ocr_context_state') || '' },
    { label: 'last_capture_attempt_at', value: fromOcr('last_capture_attempt_at') || '' },
    { label: 'last_capture_completed_at', value: fromOcr('last_capture_completed_at') || '' },
    { label: 'last_capture_error', value: fromOcr('last_capture_error') || '' },
    { label: 'capture_backend_kind', value: fromCapture('backend_kind', 'capture_backend_kind') || '' },
    { label: 'capture_backend_detail', value: fromCapture('backend_detail', 'capture_backend_detail') || '' },
    { label: 'last_capture_image_hash', value: fromCapture('last_image_hash', 'last_capture_image_hash') || '' },
    { label: 'consecutive_same_capture_frames', value: String(fromCapture('consecutive_same_frames', 'consecutive_same_capture_frames') || 0) },
    { label: 'stale_capture_backend', value: String(Boolean(fromCapture('stale_backend', 'stale_capture_backend'))) },
    { label: 'last_raw_ocr_text', value: fromOcr('last_raw_text', 'last_raw_ocr_text') || '' },
    { label: 'last_observed_line', value: fromOcr('last_observed_line')?.text || '' },
    { label: 'last_stable_line', value: fromOcr('last_stable_line')?.text || '' },
    { label: 'ocr_capture_diagnostic_required', value: String(Boolean(fromCapture('diagnostic_required', 'ocr_capture_diagnostic_required'))) },
    { label: 'backend_kind', value: fromOcr('backend_kind') || '' },
    { label: 'backend_detail', value: fromOcr('backend_detail') || '' },
    { label: 'backend_path', value: fromOcr('backend_path') || '' },
    { label: 'backend_model', value: fromOcr('backend_model') || '' },
    { label: 'tesseract_path', value: fromOcr('tesseract_path') || '' },
    { label: 'languages', value: fromOcr('languages') || '' },
    { label: 'takeover_reason', value: runtime.takeover_reason || '' },
    { label: 'target_selection_mode', value: fromWindow('selection_mode', 'target_selection_mode') || '' },
    { label: 'target_selection_detail', value: fromWindow('selection_detail', 'target_selection_detail') || '' },
    { label: 'effective_window_key', value: fromWindow('effective_window_key') || '' },
    { label: 'effective_window_title', value: fromWindow('effective_window_title') || '' },
    { label: 'effective_process_name', value: fromWindow('effective_process_name') || '' },
    { label: 'target_is_foreground', value: String(Boolean(fromWindow('target_is_foreground'))) },
    { label: 'foreground_refresh_at', value: fromWindow('foreground_refresh_at') || '' },
    { label: 'foreground_refresh_detail', value: fromWindow('foreground_refresh_detail') || '' },
    { label: 'foreground_hwnd', value: String(fromWindow('foreground_hwnd') || 0) },
    { label: 'target_hwnd', value: String(fromWindow('target_hwnd') || 0) },
    { label: 'foreground_advance_monitor_running', value: String(Boolean(fromAdvance('foreground_monitor_running', 'foreground_advance_monitor_running'))) },
    { label: 'foreground_advance_last_seq', value: String(fromAdvance('foreground_last_seq', 'foreground_advance_last_seq') || 0) },
    { label: 'foreground_advance_consumed_seq', value: String(fromAdvance('foreground_consumed_seq', 'foreground_advance_consumed_seq') || 0) },
    { label: 'foreground_advance_last_kind', value: fromAdvance('foreground_last_kind', 'foreground_advance_last_kind') || '' },
    { label: 'foreground_advance_last_delta', value: String(fromAdvance('foreground_last_delta', 'foreground_advance_last_delta') || 0) },
    { label: 'foreground_advance_last_matched', value: String(Boolean(fromAdvance('foreground_last_matched', 'foreground_advance_last_matched'))) },
    { label: 'foreground_advance_last_match_reason', value: fromAdvance('foreground_last_match_reason', 'foreground_advance_last_match_reason') || '' },
    { label: 'last_poll_started_at', value: fromTiming('last_poll_started_at') || '' },
    { label: 'last_poll_completed_at', value: fromTiming('last_poll_completed_at') || '' },
    {
      label: 'last_poll_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_poll_duration_seconds')),
    },
    {
      label: 'last_capture_total_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_capture_total_duration_seconds')),
    },
    {
      label: 'last_capture_frame_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_capture_frame_duration_seconds')),
    },
    {
      label: 'last_capture_background_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_capture_background_duration_seconds')),
    },
    {
      label: 'last_capture_image_hash_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_capture_image_hash_duration_seconds')),
    },
    {
      label: 'last_ocr_extract_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_ocr_extract_duration_seconds')),
    },
    {
      label: 'last_backend_plan_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_backend_plan_duration_seconds')),
    },
    {
      label: 'last_window_scan_duration_seconds',
      value: formatOcrRuntimeSeconds(fromTiming('last_window_scan_duration_seconds')),
    },
    { label: 'last_capture_background_hash_skipped', value: String(Boolean(fromAdvance('last_background_hash_skipped', 'last_capture_background_hash_skipped'))) },
    { label: 'last_poll_emitted_event', value: String(Boolean(fromAdvance('last_poll_emitted_event'))) },
    { label: 'last_tick_skipped', value: String(Boolean(fromAdvance('last_tick_skipped'))) },
    { label: 'last_tick_skip_reason', value: fromAdvance('last_tick_skip_reason') || '' },
    { label: 'pending_visual_scene_count', value: String(fromAdvance('pending_visual_scene_count') || 0) },
    { label: 'last_auto_recalibrate_attempts', value: String(fromAdvance('last_auto_recalibrate_attempts') || 0) },
    {
      label: 'last_auto_recalibrate_duration_seconds',
      value: formatOcrRuntimeSeconds(fromAdvance('last_auto_recalibrate_duration_seconds')),
    },
    { label: 'last_auto_recalibrate_limited', value: String(Boolean(fromAdvance('last_auto_recalibrate_limited'))) },
    { label: 'last_auto_recalibrate_error', value: fromAdvance('last_auto_recalibrate_error') || '' },
    { label: 'candidate_count', value: String(fromWindow('candidate_count') || 0) },
    { label: 'excluded_candidate_count', value: String(fromWindow('excluded_candidate_count') || 0) },
    { label: 'last_exclude_reason', value: fromWindow('last_exclude_reason') || '' },
  ]);
}

function formatOcrWindowReason(reason) {
  const mapping = {
    excluded_self_window: '已排除 N.E.K.O 自身窗口',
    excluded_overlay_window: '已排除 overlay / launcher / helper',
    excluded_helper_window: '已排除系统或宿主辅助窗口',
    excluded_small_or_hidden_window: '已排除过小或不可用窗口',
    excluded_minimized_window: '游戏窗口已最小化，OCR 不能截图，请恢复窗口',
    excluded_non_game_process: '非游戏进程，已忽略',
  };
  return mapping[reason] || reason || 'unknown';
}

function formatOcrWindowSelectionDetail(detail) {
  const mapping = {
    auto_candidate_scan: '正在自动检测游戏窗口',
    manual_target_active: '手动锁定已启用',
    manual_target_exact: '命中手动锁定窗口',
    manual_target_rebound: '已按签名重新绑定手动窗口',
    manual_target_unavailable_fallback_to_auto: '手动窗口不可用，请重新锁定窗口',
    waiting_for_manual_window_target: '自动检测失败，请手动锁定 OCR 目标窗口',
    auto_detect_needs_manual_fallback: '未找到可信自动目标，请手动选择游戏窗口',
    foreground_window_needs_manual_confirmation: '当前前台窗口不像游戏，请手动选择游戏窗口',
    no_eligible_window: '当前没有可用游戏窗口',
    memory_reader_window_minimized: '游戏窗口已最小化，OCR 不能截图，请恢复窗口',
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
  let captureHint = '';
  if (runtime.capture_backend_kind === 'dxcam') {
    captureHint = '使用 DXcam 截图后端';
  } else if (runtime.capture_backend_detail === 'dxcam_unavailable_fallback') {
    captureHint = '未安装 DXcam，正在使用兼容截图；可安装 dxcam 降低遮挡或旧帧影响';
  } else if (runtime.capture_backend_detail === 'dxcam_failed_fallback') {
    captureHint = 'DXcam 截图失败，已自动切到兼容截图';
  } else if (runtime.capture_backend_kind) {
    captureHint = `使用 ${runtime.capture_backend_kind} 兼容截图`;
  }
  if (runtime.stale_capture_backend) {
    captureHint = '截图源没有更新，请切回游戏窗口或切换 DXcam 截图后端';
  }
  const hintParts = [
    effectiveProcess ? `当前目标: ${effectiveProcess}${runtime.pid ? ` (${runtime.pid})` : ''}` : '',
    effectiveTitle ? `窗口: ${effectiveTitle}` : '',
    captureHint,
    detail,
    runtime.last_exclude_reason ? `最近排除: ${formatOcrWindowReason(runtime.last_exclude_reason)}` : '',
  ].filter(Boolean);

  modeText.textContent = mode === 'manual'
    ? `当前模式: manual${manualTarget.process_name ? ` | 锁定 ${manualTarget.process_name}` : ''}`
    : '当前模式: 自动检测优先';
  hint.textContent = hintParts.join(' | ') || '插件会先尝试当前前台/已绑定游戏窗口；无法可信识别时，请手动选择游戏窗口。';
  autoButton.disabled = mode !== 'manual';
}

function renderLockedWindow(status) {
  const runtime = (status || {}).ocr_reader_runtime || {};
  const snapshot = latestOcrWindowSnapshot || {};
  const card = document.getElementById('ocrLockedWindowCard');
  const mode = runtime.target_selection_mode || snapshot.target_selection_mode || 'auto';
  const manualTarget = runtime.manual_target || snapshot.manual_target || {};
  const effectiveTitle = runtime.effective_window_title || runtime.window_title || '';
  const effectiveProcess = runtime.effective_process_name || runtime.process_name || '';

  if (mode === 'manual' && (manualTarget.process_name || effectiveProcess)) {
    const processName = manualTarget.process_name || effectiveProcess;
    const title = manualTarget.title || effectiveTitle;
    const pid = manualTarget.pid || runtime.pid || '';
    const windowKey = manualTarget.window_key || '';
    card.innerHTML = `
      <div class="locked-window-info">
        <p class="list-kicker">${escapeHtml(processName)}${pid ? ` · PID ${escapeHtml(pid)}` : ''}</p>
        <h3>${escapeHtml(title || '未命名窗口')}</h3>
        <p class="result-note mono">${escapeHtml(windowKey)}</p>
        <div class="window-candidate-meta" style="margin-top:8px;">
          <span class="status-chip active">手动锁定</span>
        </div>
      </div>
    `;
  } else if (effectiveProcess || effectiveTitle) {
    const detail = formatOcrWindowSelectionDetail(runtime.target_selection_detail || '');
    card.innerHTML = `
      <div class="locked-window-info">
        <p class="list-kicker">${escapeHtml(effectiveProcess || '自动检测目标')}${runtime.pid ? ` · PID ${escapeHtml(runtime.pid)}` : ''}</p>
        <h3>${escapeHtml(effectiveTitle || '未命名窗口')}</h3>
        <p class="result-note">${escapeHtml(detail || '自动检测到可信游戏窗口')}</p>
        <div class="window-candidate-meta" style="margin-top:8px;">
          <span class="status-chip active">自动检测</span>
          ${runtime.target_is_foreground ? '<span class="status-chip active">前台窗口</span>' : '<span class="status-chip warning">非前台</span>'}
        </div>
      </div>
    `;
  } else {
    card.innerHTML = '<div class="locked-window-empty">尚未确认 OCR 目标窗口。插件会优先尝试前台/已绑定游戏窗口；如果仍没有读到台词，请点击“选择识别窗口”手动锁定。</div>';
  }
}

function renderOcrWindowListToNode(node, windows) {
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
          <div class="window-candidate-header">
            <div class="window-candidate-summary">
              <p class="list-kicker">${escapeHtml(item.process_name || '未知进程')} · pid ${escapeHtml(item.pid || 0)}</p>
              <h3>${escapeHtml(item.title || '未命名窗口')}</h3>
            </div>
            <button class="secondary" data-window-key="${escapeHtml(item.window_key || '')}">锁定此窗口</button>
          </div>
          <p class="result-note mono">${escapeHtml(item.window_key || '')}</p>
          <div class="window-candidate-actions">
            <div class="window-candidate-meta">${chips}</div>
          </div>
        </article>
      `;
    }).join('');
    node.querySelectorAll('[data-window-key]').forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.getAttribute('data-window-key') || '';
        withButtonPending(button, '锁定中...', () => setOcrWindowTarget(key)).catch(() => {});
      });
    });
  }
}

function renderOcrWindowTargetSnapshot(snapshot, status = latestStatus) {
  latestOcrWindowSnapshot = snapshot;
  const runtime = (status || {}).ocr_reader_runtime || {};
  const excludedNode = document.getElementById('ocrExcludedWindowList');
  const windows = snapshot.windows || [];
  const excludedWindows = snapshot.excluded_windows || [];
  emptyOcrWindowFocusForceRefreshDone = windows.length === 0
    ? emptyOcrWindowFocusForceRefreshDone
    : false;

  const modal = document.getElementById('ocrWindowModal');
  if (modal && !modal.hidden) {
    const modalList = document.getElementById('ocrWindowList');
    renderOcrWindowListToNode(modalList, windows);
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
  renderLockedWindow(status || { ocr_reader_runtime: runtime });
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
  const installed = Boolean(rapidocr.installed) || (installState && installState.status === 'completed');
  const selectedBackend = status.ocr_backend_selection || 'auto';
  const usingRapidOcr = runtime.backend_kind === 'rapidocr';
  const usingFallback = runtime.backend_kind === 'tesseract';
  configureUseButton('rapidocrUseBtn', {
    active: selectedBackend === 'rapidocr',
    disabled: !installed,
    text: selectedBackend === 'rapidocr' ? '正在使用 RapidOCR' : '使用 RapidOCR',
    title: installed ? '强制 OCR Reader 使用 RapidOCR' : '请先安装 RapidOCR',
  });
  configureUseButton('ocrBackendAutoBtn', {
    active: selectedBackend === 'auto',
    text: selectedBackend === 'auto' ? 'OCR 自动选择中' : 'OCR 自动',
    title: '按 RapidOCR 优先、Tesseract 兜底自动选择 OCR 后端',
  });

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
      ? `后端: ${runtime.backend_kind || 'rapidocr'}\n模型: ${runtime.backend_model || rapidocr.selected_model || ''}`
      : usingFallback
        ? `RapidOCR 已安装，但本帧 OCR 回退到了 Tesseract。原因: ${runtime.backend_detail || rapidocr.detail || '未知'}。`
        : 'RapidOCR 已安装完成。无 SDK 且无有效内存文本时，它会优先于 Tesseract 作为 OCR Reader 的主后端。';
    path.textContent = [
      rapidocr.detected_path ? `检测路径: ${rapidocr.detected_path}` : '',
      rapidocr.model_cache_dir ? `模型目录: ${rapidocr.model_cache_dir}` : '',
    ].filter(Boolean).join('\n');
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

  if (installed) {
    getInstallNodes('rapidocr').card.hidden = true;
  } else {
    renderInstallTaskState('rapidocr');
  }
}

function renderDxcam(status) {
  const dxcam = status.dxcam || {};
  const runtime = status.ocr_reader_runtime || {};
  const banner = document.getElementById('dxcamPrompt');
  const kicker = document.getElementById('dxcamPromptKicker');
  const title = document.getElementById('dxcamPromptTitle');
  const body = document.getElementById('dxcamPromptBody');
  const path = document.getElementById('dxcamPathText');
  const button = document.getElementById('dxcamInstallBtn');
  const installState = getInstallState('dxcam').state;
  const installable = Boolean(dxcam.install_supported) && Boolean(dxcam.can_install);
  const installed = Boolean(dxcam.installed) || (installState && installState.status === 'completed');
  const selectedCaptureBackend = status.ocr_capture_backend_selection || 'auto';
  const usingDxcam = runtime.capture_backend_kind === 'dxcam';
  const captureBackendText = runtime.capture_backend_kind || (
    selectedCaptureBackend === 'dxcam'
      ? 'DXcam 已选择，等待下一次 OCR 截图确认'
      : '未知'
  );
  configureUseButton('dxcamUseBtn', {
    active: selectedCaptureBackend === 'dxcam',
    disabled: !installed,
    text: selectedCaptureBackend === 'dxcam' ? '正在使用 DXcam' : '使用 DXcam',
    title: installed ? '强制截图后端使用 DXcam' : '请先安装 DXcam',
  });
  configureUseButton('captureBackendAutoBtn', {
    active: selectedCaptureBackend === 'auto',
    text: selectedCaptureBackend === 'auto' ? '截图自动选择中' : '截图自动',
    title: '按 DXcam 优先、ImageGrab/PrintWindow 兜底自动选择截图后端',
  });
  configureUseButton('imagegrabUseBtn', {
    active: selectedCaptureBackend === 'imagegrab',
    text: selectedCaptureBackend === 'imagegrab' ? '正在使用 ImageGrab' : '使用 ImageGrab',
    title: '使用系统截图兼容后端，游戏窗口需要可见',
  });
  configureUseButton('printwindowUseBtn', {
    active: selectedCaptureBackend === 'printwindow',
    text: selectedCaptureBackend === 'printwindow' ? '正在使用 PrintWindow' : '使用 PrintWindow',
    title: '使用 Win32 PrintWindow 兜底，DirectX/Unity 游戏可能旧帧',
  });

  banner.className = 'install-banner install-banner-dxcam';
  button.hidden = !installable;
  button.disabled = getInstallState('dxcam').inProgress;
  button.textContent = getInstallState('dxcam').inProgress
    ? getInstallConfig('dxcam').runningText
    : getInstallConfig('dxcam').actionText;

  if (!dxcam.install_supported) {
    banner.classList.add('neutral');
    kicker.textContent = '截图依赖';
    title.textContent = '当前平台暂不支持自动安装 DXcam';
    body.textContent = 'DXcam 截图后端仅用于 Windows 桌面捕获；当前平台会继续使用兼容截图方案。';
    path.textContent = '';
    button.hidden = true;
    renderInstallTaskState('dxcam');
    return;
  }

  if (installed) {
    banner.classList.add(usingDxcam ? 'success' : 'neutral');
    kicker.textContent = usingDxcam ? '截图依赖已接管' : '截图依赖已就绪';
    title.textContent = usingDxcam
      ? 'DXcam 正在作为截图后端工作'
      : selectedCaptureBackend === 'dxcam'
        ? 'DXcam 已选择，等待下一次 OCR 截图确认'
        : 'DXcam 已安装，等待 OCR Reader 自动使用';
    body.textContent = usingDxcam
      ? '当前截图后端使用 DXcam。它仍要求游戏窗口前台可见，不做后台捕获或绕过。'
      : `DXcam 已安装。当前截图后端: ${captureBackendText}。`;
    path.textContent = dxcam.detected_path ? `检测路径: ${dxcam.detected_path}` : '';
    button.hidden = true;
  } else {
    banner.classList.add('warning');
    kicker.textContent = '截图依赖未就绪';
    title.textContent = '未检测到 DXcam';
    body.textContent = '安装 DXcam 后，auto 截图后端会优先使用 GPU Desktop Duplication 截取前台游戏画面，降低插件页遮挡或 PrintWindow 旧帧问题。';
    path.textContent = '安装位置: 当前插件 Python 环境';
  }

  if (installState && !isInstallTaskTerminal(installState)) {
    banner.className = 'install-banner install-banner-dxcam neutral';
    kicker.textContent = 'DXcam 安装';
    title.textContent = 'DXcam 正在后台安装';
    body.textContent = '页面会通过 SSE 接收实时安装进度。安装完成后会刷新 OCR 截图后端状态。';
    button.hidden = false;
    button.disabled = true;
    button.textContent = getInstallConfig('dxcam').runningText;
  } else if (installState && installState.status === 'failed' && installable) {
    banner.className = 'install-banner install-banner-dxcam neutral';
    kicker.textContent = 'DXcam 安装';
    title.textContent = 'DXcam 安装失败，可直接重试';
    body.textContent = installState.error || installState.message || '后台安装任务失败，你可以再次点击按钮重试。';
    button.hidden = false;
    button.disabled = false;
    button.textContent = getInstallConfig('dxcam').retryText;
  } else if (installState && installState.status === 'completed' && !installed) {
    banner.className = 'install-banner install-banner-dxcam neutral';
    kicker.textContent = 'DXcam 安装';
    title.textContent = 'DXcam 安装已完成，正在刷新截图状态';
    body.textContent = installState.message || '安装任务已结束，正在等待插件状态刷新。';
  }

  if (installed) {
    getInstallNodes('dxcam').card.hidden = true;
  } else {
    renderInstallTaskState('dxcam');
  }
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
  const installed = Boolean(tesseract.installed) || (installState && installState.status === 'completed');
  const missingLanguages = tesseract.missing_languages || [];
  const selectedBackend = status.ocr_backend_selection || 'auto';
  configureUseButton('tesseractUseBtn', {
    active: selectedBackend === 'tesseract',
    disabled: !installed,
    text: selectedBackend === 'tesseract' ? '正在使用 Tesseract' : '使用 Tesseract',
    title: installed ? '强制 OCR Reader 使用 Tesseract' : '请先安装 Tesseract 和所需语言包',
  });

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

  if (installed) {
    getInstallNodes('tesseract').card.hidden = true;
  } else {
    renderInstallTaskState('tesseract');
  }
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
  const installed = Boolean(textractor.installed) || (installState && installState.status === 'completed');
  const runtimeBlocked = runtime.detail === 'invalid_textractor_path';

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
      : 'TextractorCLI.exe 已检测到';
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

  if (installed) {
    getInstallNodes('textractor').card.hidden = true;
  } else {
    renderInstallTaskState('textractor');
  }
}

function renderOcrProfile(status) {
  const runtime = status.ocr_reader_runtime || {};
  const processInput = document.getElementById('ocrProfileProcessInput');
  const stageSelect = document.getElementById('ocrProfileStageSelect');
  const saveScopeSelect = document.getElementById('ocrProfileSaveScopeSelect');
  const leftInput = document.getElementById('ocrProfileLeftInput');
  const rightInput = document.getElementById('ocrProfileRightInput');
  const topInput = document.getElementById('ocrProfileTopInput');
  const bottomInput = document.getElementById('ocrProfileBottomInput');
  const hint = document.getElementById('ocrProfileRuntimeHint');
  const currentProcessName = processInput.value.trim() || runtime.process_name || '';
  const currentStage = stageSelect.value || 'default';
  const defaultSaveScope = resolveRuntimeDefaultSaveScope(status, currentProcessName);
  if (!saveScopeSelect.value || (saveScopeSelect.value === 'window_bucket' && defaultSaveScope === 'process_fallback' && !runtime.width)) {
    saveScopeSelect.value = defaultSaveScope;
  }
  const currentSaveScope = normalizeCaptureProfileSaveScope(saveScopeSelect.value || defaultSaveScope);
  const profileValues = profileValueForInputs(
    resolveEditableCaptureProfile(status, currentProcessName, currentStage, currentSaveScope),
  );
  const autoRecalibrateButton = document.getElementById('ocrProfileAutoRecalibrateBtn');
  let autoRecalibrateReason = '';
  if (!Boolean(runtime.enabled)) {
    autoRecalibrateReason = 'OCR Reader 未启用';
  } else if (runtime.detail === 'unsupported_platform') {
    autoRecalibrateReason = '当前平台不是 Windows';
  } else if (runtime.detail === 'capture_backend_unavailable') {
    autoRecalibrateReason = '当前截图后端不可用';
  } else if (!runtime.process_name || !Number(runtime.width || 0) || !Number(runtime.height || 0)) {
    autoRecalibrateReason = '当前没有已附着的 OCR 目标窗口';
  }

  if (runtime.process_name) {
    hint.textContent = [
      `当前 OCR 目标: ${runtime.process_name} (${runtime.pid || 0})`,
      runtime.window_title ? `窗口: ${runtime.window_title}` : '',
      runtime.width && runtime.height ? `尺寸: ${runtime.width}x${runtime.height}` : '',
      runtime.capture_stage
        ? `运行阶段: ${OCR_PROFILE_STAGE_LABELS_ZH[runtime.capture_stage] || runtime.capture_stage}`
        : '',
      runtime.capture_profile_match_source
        ? `命中来源: ${OCR_CAPTURE_MATCH_SOURCE_LABELS_ZH[runtime.capture_profile_match_source] || runtime.capture_profile_match_source}`
        : '',
      runtime.capture_profile_bucket_key ? `命中桶: ${runtime.capture_profile_bucket_key}` : '',
      runtime.detail ? `状态: ${runtime.detail}` : '',
      runtime.takeover_reason ? `接管原因: ${runtime.takeover_reason}` : '',
      `自动重校准: ${autoRecalibrateReason || '可用'}`,
      isAihongProcessName(currentProcessName || runtime.process_name)
        ? '哀鸿支持按对白区 / 菜单区分别保存'
        : '',
    ].filter(Boolean).join(' | ');
  } else {
    hint.textContent = isAihongProcessName(currentProcessName)
      ? '当前还没有附着的 OCR 目标进程。你可以先手动填写 TheLamentingGeese.exe，并分别预存哀鸿的对白区 / 菜单区截图范围。自动重校准需要先附着到真实游戏窗口。'
      : '当前还没有附着的 OCR 目标进程。你也可以先手动填写 process_name，把截图校准预先存起来。自动重校准需要先附着到真实游戏窗口。';
  }

  if (!processInput.value || document.activeElement !== processInput) {
    setInputValueIfIdle(processInput, runtime.process_name || processInput.value);
  }
  if (!saveScopeSelect.value) {
    saveScopeSelect.value = defaultSaveScope;
  }
  setInputValueIfIdle(leftInput, profileValues.left);
  setInputValueIfIdle(rightInput, profileValues.right);
  setInputValueIfIdle(topInput, profileValues.top);
  setInputValueIfIdle(bottomInput, profileValues.bottom);
  autoRecalibrateButton.disabled = Boolean(autoRecalibrateReason);
  autoRecalibrateButton.title = autoRecalibrateReason || '使用当前附着窗口自动重校准对白区';
}

function renderSnapshot(snapshot) {
  const state = snapshot.snapshot || {};
  renderGrid('snapshotGrid', [
    { label: 'game_id', value: snapshot.game_id || '' },
    { label: 'session_id', value: snapshot.session_id || '' },
    { label: 'speaker', value: state.speaker || '' },
    { label: 'text', value: state.text || '' },
    { label: 'stability', value: state.stability || '' },
    { label: 'scene_id', value: state.scene_id || '' },
    { label: 'line_id', value: state.line_id || '' },
    { label: 'route_id', value: state.route_id || '' },
    { label: 'is_menu_open', value: String(Boolean(state.is_menu_open)) },
    { label: 'snapshot_ts', value: snapshot.snapshot_ts || '' },
    { label: 'stale', value: String(Boolean(snapshot.stale)) },
  ]);
}

function renderHistory(history) {
  const mergedLines = mergedHistoryLines(history);
  const runtime = latestStatus?.ocr_reader_runtime || {};
  const fallbackItems = mergedLines.length ? mergedLines : [{
    speaker: 'OCR',
    scene_id: runtime.ocr_context_state || runtime.detail || '',
    stability: 'diagnostic',
    line_id: runtime.last_poll_completed_at || runtime.last_capture_completed_at || '',
    text: runtime.last_raw_ocr_text
      ? `最近 raw OCR：${runtime.last_raw_ocr_text}`
      : buildOcrMissingLineDiagnostic(latestStatus || {}),
    is_diagnostic: true,
  }];
  renderStackList('linesList', fallbackItems, (item) => `
    <article class="list-card">
      <p class="list-kicker">${escapeHtml(item.is_diagnostic ? 'OCR 诊断' : (item.speaker || '旁白'))} · ${escapeHtml(item.scene_id || '')} · ${escapeHtml(item.stability || '')}</p>
      <h3>${escapeHtml(item.is_diagnostic ? '未写入稳定台词' : (item.line_id || ''))}</h3>
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
  scrollToBottom(node);
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
  scrollToBottom(node);
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
  scrollToBottom(node);
}

function formatInsightMeta(payload) {
  const inputSource = payload.input_source || (latestStatus && latestStatus.active_data_source) || 'unknown';
  const semantic = payload.semantic_granularity
    || (payload.semantic_degraded ? 'weaker_than_bridge_sdk' : 'bridge_sdk_level');
  const fallback = payload.fallback_used ? '是' : '否';
  return `输入源=${inputSource} | degraded=${Boolean(payload.degraded)} | 语义粒度=${semantic} | 使用回退=${fallback}`;
}

function renderAgentStatus(payload) {
  latestAgentStatus = payload || latestAgentStatus;
  const replyNode = document.getElementById('agentReplyText');
  replyNode.textContent = latestAgentReply;
  scrollToBottom(replyNode);
  const memoryCounts = payload.memory_counts || {};
  renderGrid('agentStatusGrid', [
    {
      label: 'agent_user_status',
      value: AGENT_USER_STATUS_LABELS_ZH[payload.agent_user_status] || payload.agent_user_status || '',
    },
    { label: 'status', value: payload.status || 'standby' },
    { label: 'activity', value: payload.activity || 'idle' },
    { label: 'reason', value: payload.reason || '' },
    { label: 'diagnostic', value: payload.debug?.ocr_capture_diagnostic || payload.error || '' },
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
    { label: 'inbound_queue_size', value: String(payload.inbound_queue_size || 0) },
    { label: 'outbound_queue_size', value: String(payload.outbound_queue_size || 0) },
    {
      label: 'last_interruption',
      value: payload.last_interruption?.interrupted_message_id || '',
    },
    {
      label: 'last_outbound_message',
      value: payload.last_outbound_message?.content || '',
    },
    { label: 'result', value: payload.result || '' },
    { label: 'recent_pushes', value: String((payload.recent_pushes || []).length) },
  ]);

  renderStackList('pushesList', payload.recent_pushes || [], (item) => `
    <article class="list-card compact">
      <p class="list-kicker">${escapeHtml(item.kind || '')} | ${escapeHtml(item.status || '')} | ${escapeHtml(item.ts || '')}</p>
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
  scrollToBottom(node);
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
  scrollToBottom(node);
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
  scrollToBottom(node);
}

async function refreshInsights(snapshot, { force = false, history = {}, status = {} } = {}) {
  const mode = String(status.mode || '').trim();
  if (mode === 'silent') {
    const diagnostic = '静默模式：不自动解释、总结或建议。';
    const explain = buildExplainFallback('', diagnostic);
    const summary = buildSummaryFallback('', diagnostic);
    const suggest = buildSuggestFallback('', diagnostic);
    latestInsights.explainKey = 'silent';
    latestInsights.explainPayload = explain;
    latestInsights.summaryKey = 'silent';
    latestInsights.summaryPayload = summary;
    latestInsights.suggestKey = 'silent';
    latestInsights.suggestPayload = suggest;
    renderExplain(explain);
    renderSummary(summary);
    renderSuggest(suggest);
    return;
  }

  const state = snapshot.snapshot || {};
  const fallbackLine = effectiveCurrentLine(snapshot, history, status);
  const currentLineId = state.line_id || fallbackLine.line_id || '';
  const currentSceneId = state.scene_id || fallbackLine.scene_id || '';
  const choices = Array.isArray(state.choices) ? state.choices : [];
  const visibleChoiceMenu = Boolean(state.is_menu_open) && choices.length > 0;
  const hasChoices = mode === 'choice_advisor' && visibleChoiceMenu;
  const explainKey = currentLineId || 'missing-line';
  const summaryKey = sceneSummaryKey(currentSceneId, history, fallbackLine);
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
    : Promise.resolve(buildExplainFallback('', buildOcrMissingLineDiagnostic(status)));

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
    : Promise.resolve(buildSuggestFallback(
      currentSceneId,
      visibleChoiceMenu ? '伴读模式：不自动生成选项建议。' : 'no visible choices',
    ));

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

function renderInsightsPending(message = '解释、总结和选项建议正在后台刷新...') {
  const payload = buildExplainFallback('', message);
  renderExplain(payload);
  renderSummary(buildSummaryFallback('', message));
  renderSuggest(buildSuggestFallback('', message));
}

function runBackgroundTask(label, task) {
  Promise.resolve()
    .then(task)
    .catch((error) => {
      console.warn(`[galgame_plugin ui] ${label} failed`, error);
    });
}

function stopAutoRefresh() {
  if (autoRefreshTimer !== null) {
    window.clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
}

function desiredAutoRefreshInterval(status = latestStatus) {
  if (!status || status.connection_state === 'error' || status.agent_user_status === 'error') {
    return ERROR_REFRESH_INTERVAL_MS;
  }
  if (
    status.agent_user_status === 'paused_window_not_foreground'
    || status.agent_pause_kind === 'window_not_foreground'
  ) {
    return FOCUS_PAUSE_REFRESH_INTERVAL_MS;
  }
  const runtime = status.ocr_reader_runtime || {};
  const { observedText, stableText, effectiveText } = getCurrentLineTexts(status);
  if (
    status.connection_state === 'active'
    || runtime.status === 'active'
    || runtime.status === 'running'
    || observedText
    || stableText
    || effectiveText
  ) {
    return AUTO_REFRESH_ACTIVE_INTERVAL_MS;
  }
  return AUTO_REFRESH_IDLE_INTERVAL_MS;
}

function startAutoRefresh(intervalMs = AUTO_REFRESH_INTERVAL_MS) {
  stopAutoRefresh();
  autoRefreshIntervalMs = intervalMs;
  autoRefreshTimer = window.setInterval(() => {
    if (document.hidden) {
      return;
    }
    refreshAll({ preserveFlash: true, silent: true }).catch(() => {});
  }, intervalMs);
}

function syncAutoRefreshIntervalForStatus(status = latestStatus) {
  const desired = desiredAutoRefreshInterval(status);
  if (autoRefreshTimer !== null && desired !== autoRefreshIntervalMs) {
    startAutoRefresh(desired);
  }
}

function isOcrWindowModalOpen() {
  const modal = document.getElementById('ocrWindowModal');
  return Boolean(modal && !modal.hidden);
}

function shouldRefreshOcrWindowsForStatus(status) {
  const runtime = status?.ocr_reader_runtime || {};
  const detail = String(runtime.target_selection_detail || '');
  const context = String(status?.ocr_context_state || runtime.ocr_context_state || '');
  return (
    context === 'waiting_for_valid_window'
    || detail === 'no_eligible_window'
    || detail === 'foreground_window_needs_manual_confirmation'
    || detail === 'auto_detect_needs_manual_fallback'
    || detail === 'memory_reader_window_minimized'
    || Number(runtime.candidate_count || 0) === 0
    || isOcrWindowModalOpen()
  );
}

function refreshOcrWindowTargetsIfNeeded({
  reason = '',
  force = false,
  silent = true,
} = {}) {
  if (ocrWindowRefreshInFlight) {
    return ocrWindowRefreshInFlight;
  }

  const now = Date.now();
  if (!force && now - lastOcrWindowRefreshAt < OCR_WINDOW_REFRESH_TTL_MS) {
    return Promise.resolve(false);
  }

  ocrWindowRefreshInFlight = refreshOcrWindowTargets({
    includeExcluded: true,
    silent,
    force,
  }).then((refreshed) => {
    if (refreshed) {
      lastOcrWindowRefreshAt = Date.now();
    }
    return Boolean(refreshed);
  }).catch((error) => {
    console.warn(`[galgame_plugin ui] refresh OCR window targets for ${reason || 'unknown'} failed`, error);
    if (!silent) {
      setFlash(error instanceof Error ? error.message : String(error), 'error');
    }
    return false;
  }).finally(() => {
    ocrWindowRefreshInFlight = null;
  });
  return ocrWindowRefreshInFlight;
}

function refreshOcrWindowsOnPageFocus() {
  if (document.hidden) {
    return;
  }
  refreshOcrWindowTargetsIfNeeded({
    reason: 'page_focus',
    silent: true,
  }).catch(() => {});
}

async function refreshAll(options = {}) {
  const {
    preserveFlash = false,
    silent = false,
    forceInsights = false,
    insightMode = 'background',
    showInsightPending = false,
    forceRefresh = false,
  } = options;
  if (refreshInFlight) {
    if (!forceRefresh) {
      return refreshInFlight;
    }
    try {
      await refreshInFlight;
    } catch (error) {
      console.warn('[galgame_plugin ui] ignored stale refresh before forced refresh', error);
    }
  }

  refreshInFlight = (async () => {
    if (!preserveFlash && !silent) {
      setFlash('', 'info');
    }
    try {
      const [status, snapshot, history] = await Promise.all([
        callPlugin('galgame_get_status'),
        callPlugin('galgame_get_snapshot'),
        callPlugin('galgame_get_history', { limit: 20, include_events: true }),
      ]);
      const agentStatus = status.agent || buildAgentStatusFromStatus(status);
      renderStatus(status);
      renderSnapshot(snapshot);
      renderHistory(history);
      renderAgentStatus(agentStatus);
      if (shouldRefreshOcrWindowsForStatus(status)) {
        const snapshotWindows = latestOcrWindowSnapshot && Array.isArray(latestOcrWindowSnapshot.windows)
          ? latestOcrWindowSnapshot.windows
          : [];
        const forceEmptyFocusedRefresh = Boolean(
          !emptyOcrWindowFocusForceRefreshDone
          && !document.hidden
          && document.hasFocus()
          && snapshotWindows.length === 0
          && Number((status.ocr_reader_runtime || {}).candidate_count || 0) === 0
        );
        if (forceEmptyFocusedRefresh) {
          emptyOcrWindowFocusForceRefreshDone = true;
        }
        runBackgroundTask('refresh OCR window targets after status', () => (
          refreshOcrWindowTargetsIfNeeded({
            reason: 'status_needs_window_refresh',
            force: forceEmptyFocusedRefresh,
            silent: true,
          })
        ));
      }
      if (showInsightPending && !latestInsights.explainPayload) {
        renderInsightsPending();
      }
      if (insightMode !== 'none') {
        const insightRefresh = refreshInsights(snapshot, { force: forceInsights, history, status });
        if (insightMode === 'blocking') {
          await insightRefresh;
        } else {
          insightRefresh.catch((error) => {
            console.warn('[galgame_plugin ui] background insight refresh failed', error);
          });
        }
      }
      scrollAllRegionsToBottom();
      return true;
    } catch (error) {
      renderPluginUnavailable(error);
      if (silent) {
        console.warn('[galgame_plugin ui] refresh failed', error);
        return false;
      }
      setFlash(error instanceof Error ? error.message : String(error), 'error');
      return false;
    }
  })();

  try {
    return await refreshInFlight;
  } finally {
    refreshInFlight = null;
  }
}

function buildAgentStatusFromStatus(status = {}) {
  return {
    action: 'peek_status',
    result: status.agent_error || '',
    status: status.agent_status || 'standby',
    agent_user_status: status.agent_user_status || '',
    activity: status.agent_activity || '',
    reason: status.agent_reason || '',
    error: status.agent_error || '',
    inbound_queue_size: status.agent_inbound_queue_size || 0,
    outbound_queue_size: status.agent_outbound_queue_size || 0,
    last_interruption: status.agent_last_interruption || {},
    last_outbound_message: status.agent_last_outbound_message || {},
    debug: {
      ocr_capture_diagnostic: status.agent_diagnostic || status.ocr_capture_diagnostic || '',
    },
    recent_pushes: latestAgentStatus?.recent_pushes || [],
  };
}

async function withButtonPending(buttonOrId, pendingText, fn) {
  const button = typeof buttonOrId === 'string'
    ? document.getElementById(buttonOrId)
    : buttonOrId;
  if (!button) {
    return fn();
  }
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = pendingText;
  try {
    return await fn();
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function startInstall(kind, force = false) {
  const config = getInstallConfig(kind);
  const state = getInstallState(kind);
  const { button } = getInstallNodes(kind);
  state.inProgress = true;
  state.state = {
    kind,
    task_id: '',
    status: 'queued',
    phase: 'queued',
    message: `正在创建 ${config.label} 后台安装任务...`,
    progress: 0.01,
    updated_at: Date.now() / 1000,
  };
  closeInstallStream(kind);
  clearInstallReconnectTimer(kind);
  button.disabled = true;
  button.textContent = '准备安装...';
  if (latestStatus) {
    renderStatus(latestStatus);
  } else {
    renderInstallTaskState(kind);
  }
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
    const message = error instanceof Error ? error.message : String(error);
    state.inProgress = false;
    state.state = {
      kind,
      task_id: state.currentTaskId || '',
      status: 'failed',
      phase: 'failed',
      message,
      error: message,
      progress: 1,
      updated_at: Date.now() / 1000,
    };
    if (latestStatus) {
      renderStatus(latestStatus);
    } else {
      renderInstallTaskState(kind);
    }
    setFlash(message, 'error');
  }
}

async function installTextractor(force = false) {
  await startInstall('textractor', force);
}

async function installRapidOcr(force = false) {
  await startInstall('rapidocr', force);
}

async function installDxcam(force = false) {
  await startInstall('dxcam', force);
}

async function installTesseract(force = false) {
  await startInstall('tesseract', force);
}

async function setOcrBackendSelection({ backendSelection = null, captureBackend = null } = {}) {
  const args = {};
  if (backendSelection) {
    args.backend_selection = backendSelection;
  }
  if (captureBackend) {
    args.capture_backend = captureBackend;
  }
  const label = backendSelection
    ? `OCR 后端切换为 ${backendSelection}`
    : `截图后端切换为 ${captureBackend}`;
  try {
    setFlash(`正在${label}...`, 'info');
    await callPlugin('galgame_set_ocr_backend', args);
    setFlash(`${label} 已保存`, 'success');
    await refreshAll({ preserveFlash: true, forceInsights: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function saveMode() {
  const mode = document.getElementById('modeSelect').value;
  const pushNotifications = document.getElementById('pushToggle').checked;
  const advanceSpeed = document.getElementById('advanceSpeedSelect').value || 'medium';
  const readerMode = document.getElementById('readerModeSelect')?.value || 'auto';
  const ocrPollIntervalRaw = document.getElementById('ocrPollIntervalInput')?.value || '';
  const ocrPollInterval = Number(ocrPollIntervalRaw || 2);
  const ocrTriggerMode = document.getElementById('ocrTriggerModeSelect')?.value || 'after_advance';
  if (!['auto', 'memory_reader', 'ocr_reader'].includes(readerMode)) {
    setFlash('文本读取模式无效。', 'error');
    return;
  }
  if (!Number.isFinite(ocrPollInterval) || ocrPollInterval < 0.5 || ocrPollInterval > 10) {
    setFlash('OCR/DXcam 识别间隔必须在 0.5 到 10 秒之间。', 'error');
    return;
  }
  if (!['interval', 'after_advance'].includes(ocrTriggerMode)) {
    setFlash('OCR 触发方式无效。', 'error');
    return;
  }
  try {
    settingsSaveInFlight = true;
    setFlash('正在保存设置...', 'info');
    await callPlugin('galgame_set_mode', {
      mode,
      push_notifications: pushNotifications,
      advance_speed: advanceSpeed,
      reader_mode: readerMode,
    });
    await callPlugin('galgame_set_ocr_timing', {
      poll_interval_seconds: ocrPollInterval,
      trigger_mode: ocrTriggerMode,
    });
    setFlash('设置已保存', 'success');
    settingsDirty = false;
    settingsSaveInFlight = false;
    await refreshAll({ preserveFlash: true, forceInsights: true, forceRefresh: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  } finally {
    settingsSaveInFlight = false;
  }
}

async function bindGame(gameId = '') {
  const normalized = String(gameId || '').trim();
  try {
    setFlash(normalized ? `正在绑定 ${normalized}...` : '正在恢复自动选择...', 'info');
    await callPlugin('galgame_bind_game', { game_id: normalized });
    setFlash(normalized ? `已绑定 ${normalized}` : '已恢复自动选择', 'success');
    await refreshAll({ preserveFlash: true, forceInsights: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function setStandby(standby) {
  try {
    setFlash(standby ? '正在进入待机...' : '正在恢复活跃...', 'info');
    const payload = await callPlugin('galgame_agent_command', {
      action: 'set_standby',
      standby,
    });
    latestAgentReply = payload.result || latestAgentReply;
    setFlash(standby ? '已切换到待机' : '已恢复活跃', 'success');
    refreshAll({ preserveFlash: true, forceInsights: true }).catch((error) => {
      console.warn('[galgame_plugin ui] refresh after standby change failed', error);
    });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function resumeAgentFromButton() {
  const action = document.getElementById('standbyOffBtn').dataset.resumeAction || 'noop';
  if (action === 'focus') {
    setFlash('当前是窗口失焦暂停。请切回游戏窗口，Agent 会自动继续；恢复活跃只解除手动待机。', 'info');
    refreshAll({ preserveFlash: true, silent: true }).catch(() => {});
    return;
  }
  if (action === 'read_only') {
    setFlash('当前为伴读/静默模式，不会自动点击。需要自动推进时请切到“自动推进”。', 'info');
    return;
  }
  if (action === 'noop') {
    setFlash('Agent 当前没有手动待机。', 'info');
    return;
  }
  await setStandby(false);
}

async function askAgent(action) {
  const prompt = document.getElementById('agentPromptInput').value.trim();
  if (!prompt) {
    setFlash('请输入要发送给 Agent 的文本', 'error');
    return;
  }

  try {
    setFlash(action === 'query_context' ? '正在查询上下文...' : '正在发送给 Agent...', 'info');
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
    const stage = document.getElementById('ocrProfileStageSelect').value || 'default';
    const saveScope = normalizeCaptureProfileSaveScope(
      document.getElementById('ocrProfileSaveScopeSelect').value,
    );
    const leftInsetRatio = readProfileNumber('ocrProfileLeftInput', 'left_inset_ratio');
    const rightInsetRatio = readProfileNumber('ocrProfileRightInput', 'right_inset_ratio');
    const topRatio = readProfileNumber('ocrProfileTopInput', 'top_ratio');
    const bottomInsetRatio = readProfileNumber('ocrProfileBottomInput', 'bottom_inset_ratio');
    const payload = await callPlugin('galgame_set_ocr_capture_profile', {
      process_name: processName,
      stage,
      save_scope: saveScope,
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
    const stage = document.getElementById('ocrProfileStageSelect').value || 'default';
    const saveScope = normalizeCaptureProfileSaveScope(
      document.getElementById('ocrProfileSaveScopeSelect').value,
    );
    const payload = await callPlugin('galgame_set_ocr_capture_profile', {
      process_name: processName,
      stage,
      save_scope: saveScope,
      clear: true,
    });
    setFlash(payload.summary || 'OCR 截图校准已清空', 'success');
    await refreshAll({ preserveFlash: true, forceInsights: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function autoRecalibrateOcrDialogueProfile() {
  try {
    const payload = await callPlugin('galgame_auto_recalibrate_ocr_dialogue_profile', {});
    const sampleText = String(payload.sample_text || '').trim();
    const summary = payload.summary || 'OCR 对白区已自动重校准';
    setFlash(sampleText ? `${summary} | ${sampleText}` : summary, 'success');
    const saveScopeSelect = document.getElementById('ocrProfileSaveScopeSelect');
    saveScopeSelect.value = 'window_bucket';
    await refreshAll({ preserveFlash: true, forceInsights: true });
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function refreshOcrWindowTargets({ includeExcluded = true, silent = false, force = false } = {}) {
  try {
    const payload = await callPlugin('galgame_list_ocr_windows', {
      include_excluded: Boolean(includeExcluded),
      force: Boolean(force),
    });
    renderOcrWindowTargetSnapshot(payload, latestStatus);
    return true;
  } catch (error) {
    if (silent) {
      console.warn('[galgame_plugin ui] refresh OCR window targets failed', error);
      return false;
    }
    setFlash(error instanceof Error ? error.message : String(error), 'error');
    return false;
  }
}

async function openOcrWindowModal() {
  const modal = document.getElementById('ocrWindowModal');
  const modalList = document.getElementById('ocrWindowList');
  modal.hidden = false;
  modalList.className = 'stack-list scroll-region empty-state window-candidate-list';
  modalList.textContent = '正在加载可用游戏窗口...';
  const refreshed = await refreshOcrWindowTargetsIfNeeded({
    reason: 'open_window_modal',
    force: true,
    silent: false,
  });
  if (!refreshed && !latestOcrWindowSnapshot) {
    setFlash('OCR 窗口列表刷新失败，请稍后重试。', 'warning');
  }
  const snapshot = latestOcrWindowSnapshot || {};
  renderOcrWindowListToNode(modalList, snapshot.windows || []);
}

function closeOcrWindowModal() {
  const modal = document.getElementById('ocrWindowModal');
  modal.hidden = true;
}

async function setOcrWindowTarget(windowKey) {
  try {
    setFlash('正在锁定 OCR 识别窗口...', 'info');
    const payload = await callPlugin('galgame_set_ocr_window_target', {
      window_key: windowKey,
      clear: false,
    });
    const target = payload.window_target || {};
    const targetName = target.process_name || target.normalized_title || '目标窗口';
    setFlash(`已锁定 OCR 识别窗口：${targetName}。后台正在刷新识别状态。`, 'success');
    closeOcrWindowModal();
    refreshAll({ preserveFlash: true, forceInsights: true }).catch((error) => {
      console.warn('[galgame_plugin ui] refresh after OCR window lock failed', error);
    });
    refreshOcrWindowTargetsIfNeeded({
      reason: 'lock_window_target',
      force: true,
      silent: true,
    }).catch(() => {});
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function clearOcrWindowTarget() {
  try {
    setFlash('正在清除 OCR 目标窗口...', 'info');
    const payload = await callPlugin('galgame_set_ocr_window_target', {
      clear: true,
    });
    setFlash(payload.summary || '已清除 OCR 目标窗口。插件会重新尝试自动检测；识别不到时再手动选择。', 'success');
    refreshAll({ preserveFlash: true, forceInsights: true }).catch((error) => {
      console.warn('[galgame_plugin ui] refresh after OCR target clear failed', error);
    });
    refreshOcrWindowTargetsIfNeeded({
      reason: 'clear_window_target',
      force: true,
      silent: true,
    }).catch(() => {});
  } catch (error) {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  }
}

async function clearOcrWindowTargetWithFeedback() {
  await withButtonPending('ocrWindowAutoBtn', '清除中...', clearOcrWindowTarget);
}

function expandAndScrollTo(elementId) {
  const node = document.getElementById(elementId);
  if (!node) {
    return;
  }
  const details = node.closest('details');
  if (details) {
    details.open = true;
  }
  node.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function revealDebugDetails() {
  expandAndScrollTo('ocrRuntimeGrid');
}

function revealLineDetails() {
  expandAndScrollTo('currentLineOverview');
}

function revealCaptureBackendSettings() {
  switchInstallTab('dxcam');
  expandAndScrollTo('dxcamPrompt');
}

async function refreshStatusAndWindowsFromAction() {
  setFlash('正在刷新状态和窗口列表...', 'info');
  const loaded = await refreshAll({ preserveFlash: true, forceInsights: true, showInsightPending: true });
  const windowsLoaded = loaded
    ? await refreshOcrWindowTargetsIfNeeded({
      reason: 'primary_diagnosis',
      force: true,
      silent: false,
    })
    : false;
  setFlash(
    loaded
      ? (windowsLoaded ? '状态和窗口列表已刷新。' : '状态已刷新；窗口列表刷新失败，请稍后重试。')
      : '状态刷新失败，请稍后重试。',
    loaded && windowsLoaded ? 'success' : 'warning',
  );
}

async function switchToChoiceAdvisorMode() {
  const modeSelect = document.getElementById('modeSelect');
  if (modeSelect) {
    modeSelect.value = 'choice_advisor';
  }
  await saveMode();
}

async function handleDiagnosisAction(action) {
  switch (action) {
    case 'refresh_all':
      await refreshStatusAndWindowsFromAction();
      break;
    case 'refresh_ocr_windows':
      await refreshStatusAndWindowsFromAction();
      break;
    case 'select_ocr_window':
      await openOcrWindowModal();
      break;
    case 'debug_details':
      revealDebugDetails();
      setFlash('已展开 OCR 运行时调试详情。', 'info');
      break;
    case 'line_details':
      revealLineDetails();
      setFlash('已定位到当前台词识别详情。', 'info');
      break;
    case 'recalibrate_ocr':
      await autoRecalibrateOcrDialogueProfile();
      break;
    case 'capture_backend':
      revealCaptureBackendSettings();
      setFlash('已定位到截图方式设置。可以切换 DXcam、ImageGrab 或 PrintWindow。', 'info');
      break;
    case 'choice_advisor':
      await switchToChoiceAdvisorMode();
      break;
    case 'focus_game':
      setFlash('请切回游戏窗口。窗口回到前台后，插件会在下一轮刷新中继续识别。', 'info');
      break;
    default:
      setFlash('这个操作暂时不可用。', 'warning');
      break;
  }
}

function switchInstallTab(tab) {
  activeInstallTab = tab;
  document.querySelectorAll('.install-tab').forEach((btn) => {
    if (btn.dataset.installTab === tab) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  ['rapidocr', 'dxcam', 'tesseract', 'textractor'].forEach((kind) => {
    const banner = document.getElementById(`${kind}Prompt`);
    if (banner) {
      banner.hidden = kind !== tab;
    }
  });
}

async function initialize() {
  initializePanelFullscreenControls();
  switchInstallTab(activeInstallTab);
  try {
    localStorage.removeItem(`${PLUGIN_ID}:last_ui_state:v1`);
  } catch (error) {
    console.warn('[galgame_plugin ui] clear cached state failed', error);
  }
  renderInsightsPending('等待首轮状态刷新；解释、总结和选项建议会在后台更新。');
  setFlash('正在加载插件状态...', 'info');
  const loaded = await refreshAll({ forceInsights: false, showInsightPending: true });
  if (loaded) {
    setFlash('插件状态已加载；解释/总结、窗口列表和依赖状态正在后台更新。', 'success');
  }
  runBackgroundTask('refresh OCR window targets', () => (
    refreshOcrWindowTargetsIfNeeded({
      reason: 'initialize',
      force: true,
      silent: true,
    })
  ));
  runBackgroundTask('restore install states', () => Promise.all([
    restoreRapidOcrInstallState(),
    restoreDxcamInstallState(),
    restoreTesseractInstallState(),
    restoreTextractorInstallState(),
  ]));
  startAutoRefresh();
}

document.getElementById('refreshBtn').addEventListener('click', async () => {
  await withButtonPending('refreshBtn', '刷新中...', async () => {
    setFlash('正在刷新插件状态...', 'info');
    const loaded = await refreshAll({ forceInsights: true, showInsightPending: true });
    const windowsLoaded = loaded
      ? await refreshOcrWindowTargetsIfNeeded({
        reason: 'manual_refresh',
        force: true,
        silent: true,
      })
      : false;
    if (loaded) {
      setFlash(
        windowsLoaded
          ? '状态和窗口列表已刷新；解释/总结在后台更新。'
          : '状态已刷新；窗口列表刷新失败，请稍后重试。',
        windowsLoaded ? 'success' : 'warning',
      );
    }
  });
});
document.getElementById('primaryDiagnosisPanel').addEventListener('click', (event) => {
  const target = eventElement(event.target);
  const button = target ? target.closest('[data-primary-action]') : null;
  if (!button) {
    return;
  }
  const action = button.getAttribute('data-primary-action') || '';
  withButtonPending(button, '处理中...', () => handleDiagnosisAction(action)).catch((error) => {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  });
});
document.getElementById('firstRunGuide').addEventListener('click', (event) => {
  const target = eventElement(event.target);
  const button = target ? target.closest('[data-first-run-action]') : null;
  if (!button) {
    return;
  }
  const action = button.getAttribute('data-first-run-action') || '';
  withButtonPending(button, '处理中...', () => handleDiagnosisAction(action)).catch((error) => {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  });
});
document.getElementById('saveModeBtn').addEventListener('click', () => {
  withButtonPending('saveModeBtn', '保存中...', saveMode).catch(() => {});
});
SETTINGS_CONTROL_IDS.forEach((id) => {
  const node = document.getElementById(id);
  if (!node) {
    return;
  }
  const markDirty = () => {
    if (!settingsSaveInFlight) {
      settingsDirty = true;
    }
  };
  node.addEventListener('input', markDirty);
  node.addEventListener('change', markDirty);
});
document.getElementById('clearBindBtn').addEventListener('click', async () => {
  await withButtonPending('clearBindBtn', '恢复中...', () => bindGame(''));
});
document.getElementById('standbyOnBtn').addEventListener('click', () => {
  withButtonPending('standbyOnBtn', '切换中...', () => setStandby(true)).catch(() => {});
});
document.getElementById('standbyOffBtn').addEventListener('click', () => {
  withButtonPending('standbyOffBtn', '处理中...', resumeAgentFromButton).catch(() => {});
});
document.getElementById('queryContextBtn').addEventListener('click', () => {
  withButtonPending('queryContextBtn', '查询中...', () => askAgent('query_context')).catch(() => {});
});
document.getElementById('sendMessageBtn').addEventListener('click', () => {
  withButtonPending('sendMessageBtn', '发送中...', () => askAgent('send_message')).catch(() => {});
});
document.getElementById('rapidocrInstallBtn').addEventListener('click', () => installRapidOcr(false));
document.getElementById('dxcamInstallBtn').addEventListener('click', () => installDxcam(false));
document.getElementById('tesseractInstallBtn').addEventListener('click', () => installTesseract(false));
document.getElementById('textractorInstallBtn').addEventListener('click', () => installTextractor(false));
document.getElementById('rapidocrUseBtn').addEventListener('click', () => setOcrBackendSelection({ backendSelection: 'rapidocr' }));
document.getElementById('ocrBackendAutoBtn').addEventListener('click', () => setOcrBackendSelection({ backendSelection: 'auto' }));
document.getElementById('tesseractUseBtn').addEventListener('click', () => setOcrBackendSelection({ backendSelection: 'tesseract' }));
document.getElementById('dxcamUseBtn').addEventListener('click', () => setOcrBackendSelection({ captureBackend: 'dxcam' }));
document.getElementById('captureBackendAutoBtn').addEventListener('click', () => setOcrBackendSelection({ captureBackend: 'auto' }));
document.getElementById('imagegrabUseBtn').addEventListener('click', () => setOcrBackendSelection({ captureBackend: 'imagegrab' }));
document.getElementById('printwindowUseBtn').addEventListener('click', () => setOcrBackendSelection({ captureBackend: 'printwindow' }));
document.getElementById('ocrWindowRefreshBtn').addEventListener('click', () => {
  refreshOcrWindowTargetsIfNeeded({
    reason: 'window_list_refresh_button',
    force: true,
    silent: false,
  }).then((refreshed) => {
    if (!refreshed) {
      setFlash('OCR 窗口列表刷新失败，请稍后重试。', 'warning');
    }
  }).catch((error) => {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  });
});
document.getElementById('ocrWindowAutoBtn').addEventListener('click', () => {
  clearOcrWindowTargetWithFeedback().catch(() => {});
});
document.getElementById('ocrWindowSelectBtn').addEventListener('click', () => {
  openOcrWindowModal().catch((error) => {
    setFlash(error instanceof Error ? error.message : String(error), 'error');
  });
});
document.getElementById('ocrWindowModalClose').addEventListener('click', closeOcrWindowModal);
document.querySelector('#ocrWindowModal .modal-overlay').addEventListener('click', closeOcrWindowModal);
document.addEventListener('pointerdown', (event) => {
  if (!document.body.classList.contains('panel-fullscreen-active')) {
    return;
  }
  const panel = document.querySelector('.panel-fullscreen');
  const target = eventElement(event.target);
  if (panel && target && !panel.contains(target)) {
    exitPanelFullscreen();
  }
});
document.addEventListener('wheel', (event) => {
  const target = document.body.classList.contains('panel-fullscreen-active')
    ? fullscreenWheelTarget(event.target, event.deltaY)
    : pageWheelTarget(event.target, event.deltaY);
  if (!target) {
    return;
  }
  target.scrollTop += event.deltaY;
  event.preventDefault();
}, { passive: false });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    exitPanelFullscreen();
    closeOcrWindowModal();
  }
});
document.getElementById('ocrProfileSaveBtn').addEventListener('click', saveOcrCaptureProfile);
document.getElementById('ocrProfileClearBtn').addEventListener('click', clearOcrCaptureProfile);
document.getElementById('ocrProfileAutoRecalibrateBtn').addEventListener('click', autoRecalibrateOcrDialogueProfile);
document.getElementById('ocrProfileStageSelect').addEventListener('change', () => {
  if (latestStatus) {
    renderOcrProfile(latestStatus);
  }
});
document.getElementById('ocrProfileSaveScopeSelect').addEventListener('change', () => {
  if (latestStatus) {
    renderOcrProfile(latestStatus);
  }
});
document.getElementById('ocrProfileProcessInput').addEventListener('blur', () => {
  if (latestStatus) {
    renderOcrProfile(latestStatus);
  }
});

document.querySelectorAll('.install-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.installTab;
    if (tab) {
      switchInstallTab(tab);
    }
  });
});

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    refreshAll({ preserveFlash: true, silent: true }).catch(() => {});
    refreshOcrWindowsOnPageFocus();
  }
});

window.addEventListener('focus', () => {
  refreshAll({ preserveFlash: true, silent: true }).catch(() => {});
  refreshOcrWindowsOnPageFocus();
});

initialize();
