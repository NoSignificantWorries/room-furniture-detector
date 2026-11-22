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
    logEl,
    showLabelsCheckbox;

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

    // перерисовка при ресайзе окна — иначе боксы "уедут"
    window.addEventListener('resize', redrawAll);

    updateControls();
  }

  async function loadConfig() {
    try {
      const r = await fetch('/config', { cache: 'no-store' });
      if (r.ok) state.config = { ...state.config, ...(await r.json()) };
    } catch (e) {
      log(`Не удалось получить /config: ${e.message}`, true);
    }
    configInfo.textContent =
      `Размер батча: ${state.config.batch_size} | ` +
      `Интервал опроса: ${state.config.poll_interval_ms} мс | ` +
      `Mock: ${state.config.use_mock ? 'Да' : 'Нет'}`;
  }

  async function checkHealth() {
    if (!healthEl) return;
    try {
      const r = await fetch('/health', { cache: 'no-store' });
      const d = await r.json().catch(() => ({}));
      if (d.status === 'ok') {
        healthEl.textContent = 'Сервис: онлайн';
        healthEl.className = 'health ok';
      } else if (d.status === 'disabled') {
        healthEl.textContent = 'Mock-режим (без сервиса)';
        healthEl.className = 'health mock';
      } else {
        healthEl.textContent = 'Сервис: ошибка соединения';
        healthEl.className = 'health error';
      }
    } catch {
      healthEl.textContent = 'Веб-сервер недоступен';
      healthEl.className = 'health error';
    }
  }

  function onFilesSelected(event) {
    if (state.sending) return;
    const allowed = ['image/jpeg', 'image/png', 'image/webp'];
    const picked = [...event.target.files].filter((f) => allowed.includes(f.type));

    state.files.forEach((it) => URL.revokeObjectURL(it.url));
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
    state.files.forEach((it) => {
      it.imageId = null;
      it.batchId = null;
      it.boxes = [];
      it.status = 'queued';
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

    for (let bi = 0; bi < chunks.length; bi++) {
      const items = chunks[bi];
      const form = new FormData();
      items.forEach((it) => form.append('files', it.file, it.file.name));

      try {
        const r = await fetch('/batches', { method: 'POST', body: form });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || d.error || `HTTP ${r.status}`);

        const batchId = d.batch_id;
        const imageIds = Array.isArray(d.image_ids) ? d.image_ids : [];

        items.forEach((it, idx) => {
          it.batchId = batchId;
          it.imageId = imageIds[idx] || null;
          it.status = 'processing';
        });

        state.batches.set(batchId, {
          id: batchId,
          items,
          status: d.status || 'processing',
          ready: false,
          results: [],
          error: null,
        });
        log(`Батч ${bi + 1}/${chunks.length} принят: ${batchId}`);
      } catch (e) {
        const lid = `local-${bi + 1}-${Date.now()}`;
        items.forEach((it) => (it.status = 'failed'));
        state.batches.set(lid, {
          id: lid,
          items,
          status: 'failed',
          ready: false,
          results: [],
          error: e.message,
        });
        log(`Батч ${bi + 1}/${chunks.length} не отправлен: ${e.message}`, true);
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
        (b) => b.status !== 'done' && b.status !== 'failed'
      );
      if (pending.length === 0) break;

      await Promise.all(
        pending.map(async (b) => {
          try {
            const r = await fetch(`/batches/${encodeURIComponent(b.id)}`, { cache: 'no-store' });
            const d = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(d.detail || d.error || `HTTP ${r.status}`);

            b.status = d.status || b.status;
            b.ready = Boolean(d.ready);
            b.error = d.error || null;

            if (d.ready && Array.isArray(d.results)) {
              b.results = d.results;
              applyBatchResults(b);
            }
            if (b.status === 'failed') b.items.forEach((it) => (it.status = 'failed'));
          } catch (e) {
            b.status = 'failed';
            b.error = e.message;
            b.items.forEach((it) => (it.status = 'failed'));
            log(`Ошибка опроса батча ${b.id}: ${e.message}`, true);
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
    const byId = new Map();
    batch.items.forEach((it) => {
      if (it.imageId) byId.set(it.imageId, it);
    });
    batch.results.forEach((r, idx) => {
      const it = byId.get(r.image_id) || batch.items[idx];
      if (!it) return;
      it.boxes = Array.isArray(r.boxes) ? r.boxes : [];
      it.width = r.width;
      it.height = r.height;
      it.status = 'done';
    });
    batch.items.forEach((it) => {
      if (it.status !== 'failed' && it.status !== 'done') it.status = 'done';
    });
  }

  function renderGallery() {
    galleryEl.innerHTML = '';
    state.files.forEach((it) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.index = String(it.index);

      const img = document.createElement('img');
      img.src = it.url;
      img.alt = it.file.name;

      const canvas = document.createElement('canvas');

      const caption = document.createElement('div');
      caption.className = 'caption';
      const fname = document.createElement('span');
      fname.className = 'fname';
      fname.textContent = it.file.name;
      const istatus = document.createElement('span');
      istatus.className = 'istatus';
      istatus.textContent = statusLabel(it.status);
      caption.append(fname, istatus);

      card.append(img, canvas, caption);
      galleryEl.appendChild(card);

      img.addEventListener('load', () => drawBoxesForFile(it));
      if (img.complete) drawBoxesForFile(it);
    });

    requestAnimationFrame(() => state.files.forEach(drawBoxesForFile));
  }

  function updateGallery() {
    state.files.forEach((it) => {
      const card = galleryEl.querySelector(`.card[data-index="${it.index}"]`);
      if (!card) return;
      const s = card.querySelector('.istatus');
      if (s) s.textContent = statusLabel(it.status);
      drawBoxesForFile(it);
    });
  }

  // ============================================================
  // ГЛАВНАЯ ФУНКЦИЯ ОТРИСОВКИ БОКСОВ
  // Все боксы красные (#ef4444), подпись "class_name conf" (0.XX),
  // текст белый на красной плашке. Позиция — сверху, если есть место,
  // иначе внутри бокса у верхней границы. Учтён DPR (retina).
  // ============================================================
  function drawBoxesForFile(item) {
    const card = galleryEl.querySelector(`.card[data-index="${item.index}"]`);
    if (!card) return;

    const img = card.querySelector('img');
    const canvas = card.querySelector('canvas');
    if (!img || !canvas || !img.complete || !img.naturalWidth) return;

    const cssW = img.clientWidth;
    const cssH = img.clientHeight;
    if (!cssW || !cssH) return;

    // Разрешение backing store под retina; CSS всё равно растянет до cssW×cssH
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(cssW * dpr);
    canvas.width = bw;
    canvas.height = Math.round((bw * cssH) / cssW); // сохраняем пропорции

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // дальше — в CSS-пикселях
    ctx.clearRect(0, 0, cssW, cssH);

    (item.boxes || []).forEach((box) => {
      const xy = box.xyxy_norm || [];
      if (xy.length !== 4) return;

      // нормализуем: clamp [0..1] и перестановка x1<x2, y1<y2
      const [a, b, c, d] = xy.map((v) => {
        const n = Number(v);
        return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 0;
      });
      const x1 = Math.min(a, c),
        y1 = Math.min(b, d);
      const x2 = Math.max(a, c),
        y2 = Math.max(b, d);

      const px = x1 * cssW;
      const py = y1 * cssH;
      const pw = (x2 - x1) * cssW;
      const ph = (y2 - y1) * cssH;
      if (pw <= 0 || ph <= 0) return;

      // 1. Красная рамка
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = Math.max(2, Math.min(cssW, cssH) / 200);
      ctx.strokeRect(px, py, pw, ph);

      // 2. Подпись: белый текст на красной плашке
      if (state.showLabels) {
        const className = String(box.class_name || 'object');
        const conf = Math.max(0, Math.min(1, Number(box.conf) || 0));
        const label = `${className} ${conf.toFixed(2)}`;

        const fontSize = Math.max(12, Math.min(22, cssW / 28));
        ctx.font = `600 ${fontSize}px -apple-system, system-ui, sans-serif`;
        ctx.textBaseline = 'top';

        const padX = 6,
          padY = 3;
        const tw = ctx.measureText(label).width + padX * 2;
        const th = fontSize + padY * 2;

        // сверху если есть место, иначе внутри у верхней границы
        let lx = Math.max(0, Math.min(px, cssW - tw));
        let ly = py - th >= 0 ? py - th : py;

        ctx.fillStyle = '#ef4444';
        ctx.fillRect(lx, ly, tw, th);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, lx + padX, ly + padY);
      }
    });
  }

  function renderBatches() {
    batchesEl.innerHTML = '';
    [...state.batches.values()].forEach((b, idx) => {
      const row = document.createElement('div');
      row.className = 'batch-row';

      const head = document.createElement('div');
      head.className = 'batch-head';
      const title = document.createElement('span');
      title.textContent = `Батч ${idx + 1}`;
      const status = document.createElement('span');
      status.className = `batch-status ${b.status}`;
      status.textContent = batchStatusLabel(b);
      head.append(title, status);

      const done = b.items.filter((it) => it.status === 'done').length;
      const total = b.items.length;

      const prog = document.createElement('div');
      prog.className = 'progress';
      const fill = document.createElement('div');
      fill.style.width = `${total ? Math.round((done / total) * 100) : 0}%`;
      prog.appendChild(fill);

      const meta = document.createElement('div');
      meta.className = 'batch-meta';
      meta.textContent =
        `${done}/${total} изображений обработано` + (b.error ? ` · ${b.error}` : '');

      row.append(head, prog, meta);
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

    const items = [...state.batches.values()].flatMap((b) => b.items);
    const done = items.filter((it) => it.status === 'done').length;
    const failed = items.filter((it) => it.status === 'failed').length;

    overallEl.textContent = `Готово ${done} из ${total}` + (failed ? `, ошибок: ${failed}` : '');
    overallBar.style.width = `${Math.round((done / total) * 100)}%`;
  }

  function clearAll() {
    if (state.sending) return;
    state.files.forEach((it) => URL.revokeObjectURL(it.url));
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

  function statusLabel(s) {
    return (
      { queued: 'в очереди', processing: 'обработка', done: 'готово', failed: 'ошибка' }[s] ||
      s ||
      ''
    );
  }

  function batchStatusLabel(b) {
    if (b.status === 'done') return 'готово';
    if (b.status === 'failed') return 'ошибка';
    if (b.status === 'accepted') return 'принят';
    return 'обработка';
  }

  function redrawAll() {
    state.files.forEach(drawBoxesForFile);
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function log(msg, isError = false) {
    if (!logEl) return;
    const line = document.createElement('div');
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    if (isError) line.style.color = '#fca5a5';
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }
})();
