'use strict';

/* Фронт без фреймворков.
 * Поток: выбор альбома -> подтверждение -> нарезка на батчи -> POST /api/batches ->
 * опрос GET /api/batches/{id} -> отрисовка результата и просмотр с боксами.
 * Настройки батчей и опроса приходят с сервера (/api/config, см. config.py).
 */

const DEFAULT_CONFIG = {
  use_mock: false,
  check_health: false,
  batch_size: 8,
  poll_interval_ms: 1000,
  poll_timeout_sec: 300,
  health_interval_ms: 5000,
  allowed_extensions: ['.jpg', '.jpeg', '.png', '.webp'],
  max_file_size_mb: 20,
  class_names: [],
};

const STATUS_LABEL = {
  idle: 'Не отправлено',
  queued: 'В очереди',
  processing: 'Обработка…',
  failed: 'Ошибка',
};

const MAX_POLL_FAILURES = 5;

const state = {
  config: { ...DEFAULT_CONFIG },
  items: [], // { localId, file, url, status, imageId, boxes, width, height, error }
  keys: new Set(),
  running: false,
  cancelRequested: false,
  run: { batchesTotal: 0, batchesDone: 0, imagesTotal: 0, imagesDone: 0 },
  nextId: 1,
  viewerIndex: -1,
};

const $ = (id) => document.getElementById(id);
const els = {
  dropzone: $('dropzone'),
  fileInput: $('fileInput'),
  notice: $('notice'),
  summary: $('summary'),
  gallery: $('gallery'),
  clearBtn: $('clearBtn'),
  exportBtn: $('exportBtn'),
  retryBtn: $('retryBtn'),
  cancelBtn: $('cancelBtn'),
  startBtn: $('startBtn'),
  progress: $('progress'),
  progressBar: $('progressBar'),
  progressText: $('progressText'),
  confirmDialog: $('confirmDialog'),
  confirmText: $('confirmText'),
  health: $('health'),
  healthText: $('healthText'),
  modeBadge: $('modeBadge'),
  viewer: $('viewer'),
  viewerTitle: $('viewerTitle'),
  viewerCounter: $('viewerCounter'),
  viewerStage: $('viewerStage'),
  viewerInfo: $('viewerInfo'),
  showBoxes: $('showBoxes'),
  prevBtn: $('prevBtn'),
  nextBtn: $('nextBtn'),
  viewerClose: $('viewerClose'),
};

class CancelError extends Error {}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function chunk(list, size) {
  const out = [];
  for (let i = 0; i < list.length; i += size) out.push(list.slice(i, i + size));
  return out;
}

function className(cls) {
  return state.config.class_names[cls] ?? `class ${cls}`;
}

function classColor(cls) {
  return `hsl(${(cls * 67) % 360} 75% 42%)`;
}

async function httpError(response) {
  let message = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (body.detail) {
      message += `: ${typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)}`;
    }
  } catch {
    /* тело не JSON — оставляем код */
  }
  return new Error(message);
}

/* ---------- конфиг и health ---------- */

async function loadConfig() {
  try {
    const response = await fetch('/api/config');
    if (!response.ok) throw await httpError(response);
    state.config = { ...DEFAULT_CONFIG, ...(await response.json()) };
  } catch (err) {
    showNotice(
      `Не удалось загрузить настройки, используются значения по умолчанию (${err.message})`
    );
  }
  els.fileInput.accept = state.config.allowed_extensions.join(',');
  els.modeBadge.hidden = !state.config.use_mock;
  els.modeBadge.textContent = 'MOCK';
}

function setHealth(data) {
  const texts = {
    ok: 'Сервис доступен',
    unavailable: 'Сервис недоступен',
    disabled: 'Проверка отключена',
  };
  els.health.dataset.state = data.status;
  els.healthText.textContent = texts[data.status] ?? data.status;
  els.health.title = data.error ?? '';
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw await httpError(response);
    setHealth(await response.json());
  } catch (err) {
    setHealth({ status: 'unavailable', error: err.message });
  }
}

function startHealthPolling() {
  checkHealth();
  if (state.config.check_health) {
    setInterval(checkHealth, state.config.health_interval_ms);
  }
}

/* ---------- добавление файлов ---------- */

function showNotice(text) {
  els.notice.hidden = !text;
  els.notice.textContent = text ?? '';
}

function addFiles(fileList) {
  const allowed = new Set(state.config.allowed_extensions);
  const maxBytes = state.config.max_file_size_mb * 1024 * 1024;
  const skipped = [];

  for (const file of fileList) {
    const ext = `.${file.name.split('.').pop().toLowerCase()}`;
    const key = `${file.name}|${file.size}|${file.lastModified}`;
    if (!allowed.has(ext)) {
      skipped.push(`${file.name}: формат не поддерживается`);
    } else if (file.size === 0) {
      skipped.push(`${file.name}: пустой файл`);
    } else if (file.size > maxBytes) {
      skipped.push(`${file.name}: больше ${state.config.max_file_size_mb} МБ`);
    } else if (!state.keys.has(key)) {
      state.keys.add(key);
      const item = {
        localId: state.nextId++,
        key,
        file,
        url: URL.createObjectURL(file),
        status: 'idle',
        imageId: null,
        boxes: [],
        width: null,
        height: null,
        error: null,
      };
      state.items.push(item);
      createCard(item);
    }
  }

  const shown = skipped.slice(0, 5);
  if (skipped.length > 5) shown.push(`…и ещё ${skipped.length - 5}`);
  showNotice(shown.length ? `Пропущено файлов: ${skipped.length}\n${shown.join('\n')}` : '');
  updateControls();
}

function clearAll() {
  if (state.running) return;
  state.items.forEach((item) => URL.revokeObjectURL(item.url));
  state.items = [];
  state.keys.clear();
  state.run = { batchesTotal: 0, batchesDone: 0, imagesTotal: 0, imagesDone: 0 };
  els.gallery.replaceChildren();
  els.progress.hidden = true;
  showNotice('');
  updateControls();
}

/* ---------- карточки и контролы ---------- */

function createCard(item) {
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'card';

  const img = new Image();
  img.src = item.url;
  img.alt = item.file.name;
  img.loading = 'lazy';

  const badge = document.createElement('span');
  badge.className = 'badge';

  card.append(img, badge);
  card.addEventListener('click', () => openViewer(state.items.indexOf(item)));

  item.card = card;
  item.badge = badge;
  els.gallery.append(card);
  updateCard(item);
}

function statusText(item) {
  if (item.status === 'done') return `Найдено: ${item.boxes.length}`;
  return STATUS_LABEL[item.status];
}

function updateCard(item) {
  item.card.dataset.status = item.status;
  item.badge.textContent = statusText(item);
  item.card.title = item.error ? `${item.file.name}\n${item.error}` : item.file.name;
}

function countBy(status) {
  return state.items.filter((item) => item.status === status).length;
}

function updateControls() {
  const idle = countBy('idle');
  const failed = countBy('failed');
  const done = countBy('done');

  els.startBtn.disabled = state.running || idle === 0;
  els.retryBtn.hidden = state.running || failed === 0;
  els.cancelBtn.hidden = !state.running;
  els.exportBtn.disabled = done === 0;
  els.clearBtn.disabled = state.running || state.items.length === 0;

  if (state.items.length === 0) {
    els.summary.textContent = 'Нет изображений';
    return;
  }
  const parts = [`Всего: ${state.items.length}`, `обработано: ${done}`];
  if (idle) parts.push(`ожидают: ${idle}`);
  if (failed) parts.push(`с ошибкой: ${failed}`);
  els.summary.textContent = parts.join(' · ');
}

function updateProgress() {
  const { batchesTotal, batchesDone, imagesTotal, imagesDone } = state.run;
  els.progress.hidden = batchesTotal === 0;
  els.progressBar.max = Math.max(batchesTotal, 1);
  els.progressBar.value = batchesDone;
  els.progressText.textContent = `Батчи: ${batchesDone} / ${batchesTotal} · Изображения: ${imagesDone} / ${imagesTotal}`;
}

/* ---------- отправка и опрос ---------- */

async function postBatch(batch) {
  const form = new FormData();
  // batch_id и image_ids генерирует app.py — здесь только файлы.
  batch.forEach((item) => form.append('files', item.file, item.file.name));

  const response = await fetch('/api/batches', { method: 'POST', body: form });
  if (!response.ok) throw await httpError(response);

  const data = await response.json();
  if (!Array.isArray(data.image_ids) || data.image_ids.length !== batch.length) {
    throw new Error('Сервис вернул неверный список image_ids');
  }
  return data;
}

async function pollBatch(batchId) {
  const { poll_interval_ms: interval, poll_timeout_sec: timeout } = state.config;
  const deadline = Date.now() + timeout * 1000;
  let failures = 0;

  for (;;) {
    if (state.cancelRequested) throw new CancelError();
    await sleep(interval);
    if (state.cancelRequested) throw new CancelError();
    if (Date.now() > deadline) throw new Error('Превышено время ожидания результата');

    let data;
    try {
      const response = await fetch(`/api/batches/${encodeURIComponent(batchId)}`);
      if (!response.ok) throw await httpError(response);
      data = await response.json();
      failures = 0;
    } catch (err) {
      if (++failures >= MAX_POLL_FAILURES) throw err;
      continue;
    }

    if (data.status === 'failed') throw new Error(data.error || 'Ошибка обработки батча');
    if (data.ready) return data;
  }
}

async function processBatch(batch) {
  batch.forEach((item) => {
    item.status = 'processing';
    item.error = null;
    updateCard(item);
  });

  const accepted = await postBatch(batch);
  batch.forEach((item, i) => {
    item.imageId = accepted.image_ids[i];
  });

  const result = await pollBatch(accepted.batch_id);
  const byId = new Map(result.results.map((r) => [r.image_id, r]));

  batch.forEach((item) => {
    const r = byId.get(item.imageId);
    if (r) {
      item.boxes = r.boxes ?? [];
      item.width = r.width;
      item.height = r.height;
      item.status = 'done';
    } else {
      item.status = 'failed';
      item.error = 'В ответе нет результата для этого изображения';
    }
    updateCard(item);
  });
}

async function runQueue() {
  if (state.running) return;
  const queued = state.items.filter((item) => item.status === 'queued');
  const batches = chunk(queued, state.config.batch_size);

  state.running = true;
  state.cancelRequested = false;
  state.run = {
    batchesTotal: batches.length,
    batchesDone: 0,
    imagesTotal: queued.length,
    imagesDone: 0,
  };
  updateControls();
  updateProgress();

  for (const batch of batches) {
    if (state.cancelRequested) break;
    try {
      await processBatch(batch);
    } catch (err) {
      const cancelled = err instanceof CancelError;
      batch.forEach((item) => {
        item.status = cancelled ? 'idle' : 'failed';
        item.error = cancelled ? null : err.message;
        updateCard(item);
      });
    }
    state.run.batchesDone++;
    state.run.imagesDone += batch.filter((item) => item.status === 'done').length;
    updateProgress();
    updateControls();
  }

  // Остановка пользователем: всё, что не успели отправить, возвращаем в "не отправлено".
  state.items.forEach((item) => {
    if (item.status === 'queued') {
      item.status = 'idle';
      updateCard(item);
    }
  });

  state.running = false;
  updateControls();
  if (state.viewerIndex >= 0) renderViewer();
}

function queueItems(status) {
  state.items.forEach((item) => {
    if (item.status === status) {
      item.status = 'queued';
      item.error = null;
      updateCard(item);
    }
  });
  runQueue();
}

function askConfirm() {
  const count = countBy('idle');
  if (count === 0) return;
  const size = state.config.batch_size;
  const batches = Math.ceil(count / size);
  els.confirmText.textContent =
    `Будет отправлено изображений: ${count}. Батчей: ${batches} (до ${size} изображений в каждом). ` +
    'Батчи отправляются по очереди.';
  els.confirmDialog.showModal();
}

/* ---------- экспорт ---------- */

function exportJson() {
  const data = state.items
    .filter((item) => item.status === 'done')
    .map((item) => ({
      file: item.file.name,
      image_id: item.imageId,
      width: item.width,
      height: item.height,
      boxes: item.boxes.map((b) => ({ ...b, name: className(b.cls) })),
    }));
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'detections.json';
  link.click();
  URL.revokeObjectURL(url);
}

/* ---------- просмотр с боксами ---------- */

function openViewer(index) {
  if (index < 0) return;
  state.viewerIndex = index;
  renderViewer();
  if (!els.viewer.open) els.viewer.showModal();
}

function stepViewer(delta) {
  const next = state.viewerIndex + delta;
  if (next >= 0 && next < state.items.length) {
    state.viewerIndex = next;
    renderViewer();
  }
}

function renderViewer() {
  const item = state.items[state.viewerIndex];
  if (!item) return;

  els.viewerTitle.textContent = item.file.name;
  els.viewerCounter.textContent = `${state.viewerIndex + 1} / ${state.items.length}`;
  els.prevBtn.disabled = state.viewerIndex === 0;
  els.nextBtn.disabled = state.viewerIndex === state.items.length - 1;

  const wrap = document.createElement('div');
  wrap.className = 'stage-wrap';
  const img = new Image();
  img.src = item.url;
  img.alt = item.file.name;
  wrap.append(img);

  if (els.showBoxes.checked) {
    for (const b of item.boxes) {
      const [x1, y1, x2, y2] = b.xyxy_norm;
      const box = document.createElement('div');
      box.className = 'box';
      box.style.left = `${x1 * 100}%`;
      box.style.top = `${y1 * 100}%`;
      box.style.width = `${(x2 - x1) * 100}%`;
      box.style.height = `${(y2 - y1) * 100}%`;
      box.style.borderColor = classColor(b.cls);

      const label = document.createElement('span');
      label.textContent = `${className(b.cls)} ${Math.round(b.conf * 100)}%`;
      label.style.background = classColor(b.cls);
      box.append(label);
      wrap.append(box);
    }
  }
  els.viewerStage.replaceChildren(wrap);

  const title = document.createElement('h3');
  title.textContent = statusText(item);
  const list = document.createElement('ul');
  for (const b of item.boxes) {
    const li = document.createElement('li');
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = classColor(b.cls);
    const name = document.createElement('span');
    name.textContent = className(b.cls);
    const conf = document.createElement('span');
    conf.className = 'conf';
    conf.textContent = `${Math.round(b.conf * 100)}%`;
    li.append(swatch, name, conf);
    list.append(li);
  }
  els.viewerInfo.replaceChildren(title, list);
  if (item.error) {
    const err = document.createElement('p');
    err.className = 'muted';
    err.textContent = item.error;
    els.viewerInfo.append(err);
  }
}

/* ---------- события ---------- */

els.fileInput.addEventListener('change', () => {
  addFiles(els.fileInput.files);
  els.fileInput.value = '';
});

['dragenter', 'dragover'].forEach((type) =>
  els.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    els.dropzone.classList.add('drag');
  })
);
['dragleave', 'drop'].forEach((type) =>
  els.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove('drag');
  })
);
els.dropzone.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

els.startBtn.addEventListener('click', askConfirm);
els.confirmDialog.addEventListener('close', () => {
  if (els.confirmDialog.returnValue === 'confirm') queueItems('idle');
});
els.retryBtn.addEventListener('click', () => queueItems('failed'));
els.cancelBtn.addEventListener('click', () => {
  state.cancelRequested = true;
});
els.clearBtn.addEventListener('click', clearAll);
els.exportBtn.addEventListener('click', exportJson);

els.prevBtn.addEventListener('click', () => stepViewer(-1));
els.nextBtn.addEventListener('click', () => stepViewer(1));
els.viewerClose.addEventListener('click', () => els.viewer.close());
els.showBoxes.addEventListener('change', renderViewer);
els.viewer.addEventListener('close', () => {
  state.viewerIndex = -1;
});
els.viewer.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') stepViewer(-1);
  if (e.key === 'ArrowRight') stepViewer(1);
});

/* ---------- старт ---------- */

(async function init() {
  await loadConfig();
  startHealthPolling();
  updateControls();
})();
