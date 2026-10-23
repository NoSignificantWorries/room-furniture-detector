(() => {
  const state = {
    config: { batch_size: 4, poll_interval_ms: 1500, use_mock: true },
    files: [],
    batches: new Map(),
    sending: false,
    showLabels: true,
  };

  let fileInput,
    confirmBtn,
    clearBtn,
    healthEl,
    configInfo,
    galleryEl,
    batchesEl,
    overallEl,
    overallBar,
    logEl;
  let showLabelsCheckbox;

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    fileInput = document.getElementById('fileInput');
    confirmBtn = document.getElementById('confirmBtn');
    clearBtn = document.getElementById('clearBtn');
    healthEl = document.getElementById('health');
    configInfo = document.getElementById('configInfo');
    galleryEl = document.getElementById('gallery');
    batchesEl = document.getElementById('batches');
    overallEl = document.getElementById('overall');
    overallBar = document.getElementById('overallBar');
    logEl = document.getElementById('log');
    showLabelsCheckbox = document.getElementById('showLabels');

    await loadConfig();
    await checkHealth();
    setInterval(checkHealth, 10000);

    fileInput.addEventListener('change', onFilesSelected);
    confirmBtn.addEventListener('click', confirmAndSend);
    clearBtn.addEventListener('click', clearAll);

    showLabelsCheckbox.addEventListener('change', (e) => {
      state.showLabels = e.target.checked;
      redrawAll();
    });

    updateControls();
  }

  async function loadConfig() {
    try {
      const response = await fetch('/config', { cache: 'no-store' });
      if (response.ok) {
        state.config = { ...state.config, ...(await response.json()) };
      }
    } catch (error) {
      log(`Не удалось получить /config: ${error.message}`, true);
    }
  }

  async function checkHealth() {
    if (!healthEl) return;
    try {
      const response = await fetch('/health', { cache: 'no-store' });
      const data = await response.json().catch(() => ({}));

      if (data.status === 'ok') {
        healthEl.textContent = 'Сервис: онлайн';
        healthEl.className = 'health ok';
      } else if (data.status === 'disabled') {
        healthEl.textContent = 'Mock-режим (без сервиса)';
        healthEl.className = 'health mock';
      } else {
        healthEl.textContent = 'Сервис: ошибка соединения';
        healthEl.className = 'health error';
      }
    } catch (error) {
      healthEl.textContent = 'Веб-сервер недоступен';
      healthEl.className = 'health error';
    }
  }

  function onFilesSelected(event) {
    if (state.sending) return;

    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    const picked = [...event.target.files].filter((file) => allowedTypes.includes(file.type));

    state.files.forEach((item) => URL.revokeObjectURL(item.url));

    state.files = picked.map((file, index) => ({
      index,
      file,
      url: URL.createObjectURL(file),
      imageId: null,
      batchId: null,
      boxes: [],
      status: 'queued',
    }));

    state.batches.clear();
    batchesEl.innerHTML = '';
    overallEl.textContent = state.files.length ? `Выбрано изображений: ${state.files.length}` : '';
    overallBar.style.width = '0%';

    renderGallery();
    updateControls();
    log(`Выбрано изображений: ${state.files.length}`);
  }

  async function confirmAndSend() {
    if (state.sending || state.files.length === 0) return;

    state.sending = true;
    updateControls();
    log('Пользователь подтвердил отправку. Начинаю отправку батчей.');

    state.batches.clear();
    state.files.forEach((item) => {
      item.imageId = null;
      item.batchId = null;
      item.boxes = [];
      item.status = 'queued';
    });

    updateGallery();
    renderBatches();
    overallBar.style.width = '0%';
    overallEl.textContent = 'Отправка батчей...';

    const batchSize = Math.max(1, Number(state.config.batch_size) || 4);
    const chunks = [];
    for (let i = 0; i < state.files.length; i += batchSize) {
      chunks.push(state.files.slice(i, i + batchSize));
    }

    for (let batchIndex = 0; batchIndex < chunks.length; batchIndex += 1) {
      const items = chunks[batchIndex];
      const form = new FormData();
      items.forEach((item) => form.append('files', item.file, item.file.name));

      try {
        const response = await fetch('/batches', { method: 'POST', body: form });
        const data = await response.json().catch(() => ({}));

        if (!response.ok) throw new Error(data.detail || data.error || `HTTP ${response.status}`);

        const batchId = data.batch_id;
        const imageIds = Array.isArray(data.image_ids) ? data.image_ids : [];

        items.forEach((item, index) => {
          item.batchId = batchId;
          item.imageId = imageIds[index] || null;
          item.status = 'processing';
        });

        state.batches.set(batchId, {
          id: batchId,
          items,
          status: data.status || 'processing',
          ready: false,
          results: [],
          error: null,
        });
        log(`Батч ${batchIndex + 1}/${chunks.length} принят: ${batchId}`);
      } catch (error) {
        const localId = `local-${batchIndex + 1}-${Date.now()}`;
        items.forEach((item) => (item.status = 'failed'));
        state.batches.set(localId, {
          id: localId,
          items,
          status: 'failed',
          ready: false,
          results: [],
          error: error.message,
        });
        log(`Батч ${batchIndex + 1}/${chunks.length} не отправлен: ${error.message}`, true);
      }

      renderBatches();
      updateGallery();
      updateOverall();
    }

    await pollBatches();
    state.sending = false;
    updateControls();
    updateOverall();
    log('Опрос батчей завершён.');
  }

  async function pollBatches() {
    const interval = Math.max(250, Number(state.config.poll_interval_ms) || 1500);

    while (true) {
      const pending = [...state.batches.values()].filter(
        (batch) => batch.status !== 'done' && batch.status !== 'failed'
      );
      if (pending.length === 0) break;

      await Promise.all(
        pending.map(async (batch) => {
          try {
            const response = await fetch(`/batches/${encodeURIComponent(batch.id)}`, {
              cache: 'no-store',
            });
            const data = await response.json().catch(() => ({}));

            if (!response.ok)
              throw new Error(data.detail || data.error || `HTTP ${response.status}`);

            batch.status = data.status || batch.status;
            batch.ready = Boolean(data.ready);
            batch.error = data.error || null;

            if (data.ready && Array.isArray(data.results)) {
              batch.results = data.results;
              applyBatchResults(batch);
            }

            if (batch.status === 'failed') {
              batch.items.forEach((item) => (item.status = 'failed'));
            }
          } catch (error) {
            batch.status = 'failed';
            batch.error = error.message;
            batch.items.forEach((item) => (item.status = 'failed'));
            log(`Ошибка опроса батча ${batch.id}: ${error.message}`, true);
          }
        })
      );

      renderBatches();
      updateGallery();
      updateOverall();
      await sleep(interval);
    }
  }

  function applyBatchResults(batch) {
    const byImageId = new Map();
    batch.items.forEach((item) => {
      if (item.imageId) byImageId.set(item.imageId, item);
    });

    batch.results.forEach((result, index) => {
      const item = byImageId.get(result.image_id) || batch.items[index];
      if (!item) return;

      item.boxes = Array.isArray(result.boxes) ? result.boxes : [];
      item.width = result.width;
      item.height = result.height;
      item.status = 'done';
    });

    batch.items.forEach((item) => {
      if (item.status !== 'failed' && item.status !== 'done') item.status = 'done';
    });
  }

  function renderGallery() {
    galleryEl.innerHTML = '';
    state.files.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.index = String(item.index);

      const img = document.createElement('img');
      img.src = item.url;
      img.alt = item.file.name;

      const canvas = document.createElement('canvas');

      const caption = document.createElement('div');
      caption.className = 'caption';

      const fname = document.createElement('span');
      fname.className = 'fname';
      fname.textContent = item.file.name;

      const istatus = document.createElement('span');
      istatus.className = 'istatus';
      istatus.textContent = statusLabel(item.status);

      caption.append(fname, istatus);
      card.append(img, canvas, caption);
      galleryEl.appendChild(card);

      img.addEventListener('load', () => drawBoxesForFile(item));
      if (img.complete) drawBoxesForFile(item);
    });

    requestAnimationFrame(() => state.files.forEach(drawBoxesForFile));
  }

  function updateGallery() {
    state.files.forEach((item) => {
      const card = galleryEl.querySelector(`.card[data-index="${item.index}"]`);
      if (!card) return;
      const status = card.querySelector('.istatus');
      if (status) status.textContent = statusLabel(item.status);
      drawBoxesForFile(item);
    });
  }

  function drawBoxesForFile(item) {
    const card = galleryEl.querySelector(`.card[data-index="${item.index}"]`);
    if (!card) return;

    const img = card.querySelector('img');
    const canvas = card.querySelector('canvas');
    if (!img || !canvas || !img.complete || !img.naturalWidth) return;

    const width = img.clientWidth;
    const height = img.clientHeight;
    if (!width || !height) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    (item.boxes || []).forEach((box) => {
      const xy = box.xyxy_norm || [];
      if (xy.length !== 4) return;

      const [x1, y1, x2, y2] = xy;
      const x = x1 * width;
      const y = y1 * height;
      const boxWidth = (x2 - x1) * width;
      const boxHeight = (y2 - y1) * height;
      const color = colorForClass(box.cls);

      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, Math.min(6, width / 250));
      ctx.strokeRect(x, y, boxWidth, boxHeight);

      if (state.showLabels) {
        const label = `cls ${box.cls} ${(Number(box.conf || 0) * 100).toFixed(0)}%`;
        const fontSize = Math.max(12, Math.min(24, width / 40));
        ctx.font = `600 ${fontSize}px system-ui, sans-serif`;

        const textWidth = ctx.measureText(label).width;
        const labelHeight = fontSize + 8;
        const labelY = Math.max(0, y - labelHeight);

        ctx.fillStyle = color;
        ctx.fillRect(x, labelY, textWidth + 12, labelHeight);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x + 6, labelY + fontSize);
      }
    });
  }

  function renderBatches() {
    batchesEl.innerHTML = '';
    [...state.batches.values()].forEach((batch, index) => {
      const row = document.createElement('div');
      row.className = 'batch-row';

      const head = document.createElement('div');
      head.className = 'batch-head';
      const title = document.createElement('span');
      title.textContent = `Батч ${index + 1}`;
      const status = document.createElement('span');
      status.className = `batch-status ${batch.status}`;
      status.textContent = batchStatusLabel(batch);
      head.append(title, status);

      const done = batch.items.filter((item) => item.status === 'done').length;
      const total = batch.items.length;

      const progress = document.createElement('div');
      progress.className = 'progress';
      const fill = document.createElement('div');
      fill.style.width = `${total ? Math.round((done / total) * 100) : 0}%`;
      progress.appendChild(fill);

      const meta = document.createElement('div');
      meta.className = 'batch-meta';
      meta.textContent =
        `${done}/${total} изображений обработано` + (batch.error ? ` · ${batch.error}` : '');

      row.append(head, progress, meta);
      batchesEl.appendChild(row);
    });
  }

  function updateOverall() {
    const total = state.files.length;
    if (!total) {
      overallEl.textContent = '';
      overallBar.style.width = '0%';
      return;
    }

    const items = [...state.batches.values()].flatMap((batch) => batch.items);
    const done = items.filter((item) => item.status === 'done').length;
    const failed = items.filter((item) => item.status === 'failed').length;

    overallEl.textContent = `Готово ${done} из ${total}` + (failed ? `, ошибок: ${failed}` : '');
    overallBar.style.width = `${Math.round((done / total) * 100)}%`;
  }

  function clearAll() {
    if (state.sending) return;
    state.files.forEach((item) => URL.revokeObjectURL(item.url));
    state.files = [];
    state.batches.clear();
    fileInput.value = '';
    galleryEl.innerHTML = '';
    batchesEl.innerHTML = '';
    overallEl.textContent = '';
    overallBar.style.width = '0%';
    updateControls();
    log('Альбом очищен.');
  }

  function updateControls() {
    if (!confirmBtn) return;
    confirmBtn.disabled = state.sending || state.files.length === 0;
    clearBtn.disabled = state.sending || state.files.length === 0;
    fileInput.disabled = state.sending;
  }

  function statusLabel(status) {
    const labels = {
      queued: 'в очереди',
      processing: 'обработка',
      done: 'готово',
      failed: 'ошибка',
    };
    return labels[status] || status || '';
  }

  function batchStatusLabel(batch) {
    if (batch.status === 'done') return 'готово';
    if (batch.status === 'failed') return 'ошибка';
    if (batch.status === 'accepted') return 'принят';
    return 'обработка';
  }

  function colorForClass(cls) {
    const palette = [
      '#ef4444',
      '#f97316',
      '#eab308',
      '#22c55e',
      '#06b6d4',
      '#3b82f6',
      '#a855f7',
      '#ec4899',
    ];
    return palette[Math.abs(Number(cls) || 0) % palette.length];
  }

  function redrawAll() {
    state.files.forEach(drawBoxesForFile);
  }
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function log(message, isError = false) {
    if (!logEl) return;
    const line = document.createElement('div');
    line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    if (isError) line.style.color = '#fca5a5';
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }
})();
