'use strict';

// ── Конфігурація типів ФП: підписи параметрів і дефолти ──────────────────────
const MF_PARAM_LABELS = {
  trimf:   ['a', 'b', 'c'],
  trapmf:  ['a', 'b', 'c', 'd'],
  gaussmf: ['σ', 'c'],
  gauss2mf: ['σ₁', 'c₁', 'σ₂', 'c₂'],
  gbellmf:  ['a', 'b', 'c'],
  sigmf:    ['a', 'c'],
  dsigmf:   ['a₁', 'c₁', 'a₂', 'c₂'],
  psigmf:   ['a₁', 'c₁', 'a₂', 'c₂'],
  zmf:      ['a', 'b'],
  smf:      ['a', 'b'],
  pimf:     ['a', 'b', 'c', 'd'],
  linsmf:   ['a', 'b'],
  linzmf:   ['a', 'b'],
};

const MF_DEFAULTS = {
  B: {
    trimf:    [0.99,  0.999, 1.0],
    trapmf:   [0.99,  0.995, 0.999, 1.0],
    gaussmf:  [0.005, 0.999],
    gauss2mf: [0.005, 0.997, 0.005, 1.001],
    gbellmf:  [0.005, 2,     0.999],
    sigmf:    [1,     0.5],
    dsigmf:   [1,     0.4,   1,    0.6],
    psigmf:   [1,     0.4,   1,    0.6],
    zmf:      [0.997, 0.999],
    smf:      [0.997, 0.999],
    pimf:     [0.99,  0.995, 0.999, 1.0],
    linsmf:   [0.99,  0.999],
    linzmf:   [0.99,  0.999],
  },
  MT: {
    trimf:    [4.0,  5.0,  6.0],
    trapmf:   [4.0,  4.5,  5.5,  6.0],
    gaussmf:  [0.5,  5.0],
    gauss2mf: [0.5,  4.5,  0.5,  5.5],
    gbellmf:  [0.5,  2,    5.0],
    sigmf:    [3,    5],
    dsigmf:   [3,    4.5,  3,    5.5],
    psigmf:   [3,    4.5,   3,   5.5],
    zmf:      [4.5,  5.5],
    smf:      [4.5,  5.5],
    pimf:     [4.0,  4.5,  5.5,  6.0],
    linsmf:   [4.0,  6.0],
    linzmf:   [4.0,  6.0],
  },
  DT: {
    trimf:    [0.2,  0.3,  0.4],
    trapmf:   [0.2,  0.25, 0.35, 0.4],
    gaussmf:  [0.05, 0.3],
    gauss2mf: [0.05, 0.25, 0.05, 0.35],
    gbellmf:  [0.05, 2,    0.3],
    sigmf:    [30,   0.3],
    dsigmf:   [30,   0.25, 30,   0.35],
    psigmf:   [30,   0.25,  30,  0.35],
    zmf:      [0.25, 0.35],
    smf:      [0.25, 0.35],
    pimf:     [0.2,  0.25, 0.35, 0.4],
    linsmf:   [0.2,  0.4],
    linzmf:   [0.2,  0.4],
  },
};

function getDefaults(mfType, context) {
  return MF_DEFAULTS[context]?.[mfType] ?? MF_DEFAULTS.B[mfType] ?? [];
}

// Індекси параметрів, яким дозволено від'ємне значення (параметр нахилу a)
const MF_FREE_NEG_INDICES = {
  sigmf:  new Set([0]),
  dsigmf: new Set([0, 2]),
  psigmf: new Set([0, 2]),
};

// Структура операцій для кожного типу ТФС
const TFS_STRUCTURE = {
  RR:   () => [makeOpSpec('R', `${window.i18n?.t('form_labels.operation_label', 'РО')} 1`)],
  RK:   () => [makeOpSpec('R', window.i18n?.t('operations.work',    'Виконавча РО')),
               makeOpSpec('K', window.i18n?.t('operations.control', 'Контрольна операція'))],
  RKR1: () => [makeOpSpec('R', window.i18n?.t('operations.work',    'Виконавча РО')),
               makeOpSpec('K', window.i18n?.t('operations.control', 'Контрольна операція')),
               makeOpSpec('R2',window.i18n?.t('operations.fixup',   'Доопрацювання'))],
  RKR:  () => [makeOpSpec('R', window.i18n?.t('operations.work',    'Виконавча РО')),
               makeOpSpec('K', window.i18n?.t('operations.control', 'Контрольна операція')),
               makeOpSpec('R2',window.i18n?.t('operations.fixup',   'Доопрацювання'))],
};

function makeOpSpec(type, title) { return { type, title }; }

// ── Стан ─────────────────────────────────────────────────────────────────────
let availableMFs = [];
let lastResult = null;
window.lastResult = null;

// ── Ініціалізація ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Активуємо Bootstrap tooltips
  document.querySelectorAll('[data-bs-toggle="tooltip"]')
    .forEach(el => new bootstrap.Tooltip(el));

  await loadDictionaries();
  const tfsInitial = document.getElementById('tfs-type').value;
  buildForm(tfsInitial);
  updateTFSIllustration(tfsInitial);

  document.getElementById('tfs-type').addEventListener('change', e => {
    buildForm(e.target.value);
    updateTFSIllustration(e.target.value);
  });
  document.getElementById('btn-evaluate').addEventListener('click', evaluate);
  document.getElementById('directive-time')
    ?.addEventListener('blur', e => normalizeNumberInput(e.target));
  document.getElementById('btn-download').addEventListener('click', downloadJSON);
  document.getElementById('btn-load-example').addEventListener('click', loadDefaultExample);
  document.getElementById('btn-add-op').addEventListener('click', addRROperation);

  document.getElementById('show-method-comparison')
    ?.addEventListener('change', () => {
      if (window.lastPsvData) renderPsv(window.lastPsvData);
    });

  document.getElementById('btn-save-scenario')
    ?.addEventListener('click', saveCurrentScenario);
  document.getElementById('btn-compare-scenarios')
    ?.addEventListener('click', compareSelected);
  loadScenariosList();
  setupFileLoad();
  document.getElementById('btn-clear-scenarios')
    ?.addEventListener('click', clearAllScenarios);

  // Закриття модального порівняння
  const cmpModal = document.getElementById('comparison-modal');
  document.getElementById('btn-close-cmp')
    ?.addEventListener('click', () => cmpModal.style.display = 'none');
  cmpModal?.addEventListener('click', e => {
    if (e.target === cmpModal) cmpModal.style.display = 'none';
  });

  // Перемикач мови
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      await window.i18n.loadTranslations(btn.dataset.lang);
    });
  });

  // Оновлення активної кнопки та перерендер всього при зміні мови
  document.addEventListener('languageChanged', e => {
    const lang = e.detail?.lang || 'uk';
    document.querySelectorAll('.lang-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.lang === lang));
    // Оновлюємо TFS-options у select (динамічні, не охоплені applyTranslations)
    document.querySelectorAll('#tfs-type option[data-i18n]').forEach(opt => {
      opt.textContent = tfsLabel(opt.value);
    });
    // Перебудовуємо форму зі збереженням введених значень
    const tfsSelect = document.getElementById('tfs-type');
    if (tfsSelect?.value) {
      const saved = collectFormData();
      buildForm(tfsSelect.value);
      updateTFSIllustration(tfsSelect.value);
      fillFormFromJSON(saved);
    }
    // Оновлюємо список сценаріїв (оновлює скорочення "оп."/"op.")
    loadScenariosList();
    if (lastResult) displayResults(lastResult);
  });
});

async function loadDictionaries() {
  const [mfs, tfsTypes] = await Promise.all([
    fetch('/api/membership-types').then(r => r.json()),
    fetch('/api/tfs-types').then(r => r.json()),
  ]);
  availableMFs = mfs;

  const sel = document.getElementById('tfs-type');
  tfsTypes.forEach(t => {
    const opt = document.createElement('option');
    opt.value = t;
    opt.dataset.i18n = 'tfs_types.' + t;
    opt.textContent = tfsLabel(t);
    sel.appendChild(opt);
  });
}

function tfsLabel(t) {
  const fallbacks = {
    RR:   'RR — Послідовне виконання',
    RK:   'RK — РО з контролем',
    RKR1: 'RKR1 — РО з контролем і доопрацюванням',
    RKR:  'RKR — Циклічна РКР',
    PAR:  'PAR — Паралельне виконання',
  };
  return window.i18n?.t('tfs_types.' + t, fallbacks[t] || t) ?? fallbacks[t] ?? t;
}

// ── Ілюстрація ТФС ────────────────────────────────────────────────────────────
const TFS_BASENAMES = {
  RR:   'tfs1_RR',
  RK:   'tfs2_RK',
  RKR1: 'tfs3_RKR1',
  RKR:  'tfs4_RKR',
  PAR:  'tfs5_PAR',
};

function updateTFSIllustration(tfsType) {
  const img         = document.getElementById('tfs-img');
  const placeholder = document.getElementById('tfs-img-placeholder');
  if (!img || !placeholder) return;

  const base = TFS_BASENAMES[tfsType];
  if (!base) {
    img.style.display = 'none';
    placeholder.style.display = 'block';
    return;
  }

  const tryLoad = (ext, onFail) => {
    const test = new Image();
    test.onload = () => {
      img.src = test.src;
      img.alt = tfsType;
      img.style.display = '';
      placeholder.style.display = 'none';
    };
    test.onerror = onFail;
    test.src = `/static/img/${base}.${ext}`;
  };

  tryLoad('svg', () => tryLoad('png', () => {
    img.style.display = 'none';
    placeholder.style.display = 'block';
  }));
}

// ── Побудова форми ────────────────────────────────────────────────────────────
function buildForm(tfsType) {
  const container = document.getElementById('operations-container');
  container.innerHTML = '';
  const addBtn = document.getElementById('btn-add-op');
  addBtn.style.display = 'none';

  if (tfsType === 'RR') {
    addBtn.style.display = '';
    appendRROperation(container, 1);
  } else if (tfsType === 'PAR') {
    // Композиціонер
    const composerDiv = document.createElement('div');
    composerDiv.className = 'mb-2 d-flex align-items-center gap-2';
    composerDiv.innerHTML = `
      <label class="fw-semibold small mb-0">${window.i18n?.t('par_composers.label', 'Композиціонер:')}</label>
      <select class="form-select form-select-sm w-auto" id="par-composer">
        <option value="AND">AND — ${window.i18n?.t('par_composers.and_desc', 'усі виконані')}</option>
        <option value="OR">OR — ${window.i18n?.t('par_composers.or_desc', 'хоча б один')}</option>
        <option value="XOR">XOR — ${window.i18n?.t('par_composers.xor_desc', 'рівно один')}</option>
      </select>`;
    container.appendChild(composerDiv);
    addBtn.style.display = '';
    appendRROperation(container, 1);
  } else if (TFS_STRUCTURE[tfsType]) {
    TFS_STRUCTURE[tfsType]().forEach(spec => appendFixedOperation(container, spec));
  } else {
    container.innerHTML = '<p class="text-muted small">Тип не реалізовано.</p>';
  }
}

// РО для ТФС-1 (динамічна кількість)
function appendRROperation(container, index) {
  const tmpl = document.getElementById('tmpl-op-rr');
  const node = tmpl.content.cloneNode(true);
  const block = node.querySelector('.op-block');
  block.dataset.opType = 'R';
  block.querySelector('.op-title').textContent = `${window.i18n?.t('form_labels.operation_label', 'РО')} ${index}`;

  const removeBtn = block.querySelector('.btn-remove-op');
  if (index > 1) removeBtn.style.display = '';
  removeBtn.addEventListener('click', () => {
    block.remove();
    renumberOps();
  });

  ['B1', 'MT', 'DT'].forEach(param => {
    const div = block.querySelector(`.mf-group[data-param="${param}"]`);
    renderMFGroup(div, param);
  });

  container.appendChild(node);
}

function addRROperation() {
  const container = document.getElementById('operations-container');
  const idx = container.querySelectorAll('.op-block').length + 1;
  appendRROperation(container, idx);
}

function renumberOps() {
  const blocks = document.querySelectorAll('#operations-container .op-block');
  blocks.forEach((b, i) => {
    const title = b.querySelector('.op-title');
    if (title && b.dataset.opType === 'R') title.textContent = `${window.i18n?.t('form_labels.operation_label', 'РО')} ${i + 1}`;
    const removeBtn = b.querySelector('.btn-remove-op');
    if (removeBtn) removeBtn.style.display = i === 0 ? 'none' : '';
  });
}

// Фіксована операція (для RK, RKR1)
function appendFixedOperation(container, spec) {
  const isK = spec.type === 'K';
  const tmplId = isK ? 'tmpl-op-k' : 'tmpl-op-rr';
  const tmpl = document.getElementById(tmplId);
  const node = tmpl.content.cloneNode(true);
  const block = node.querySelector('.op-block');
  block.dataset.opType = spec.type;
  block.querySelector('.op-title').textContent = spec.title;

  const params = isK ? ['K11', 'K00', 'MT', 'DT'] : ['B1', 'MT', 'DT'];
  params.forEach(param => {
    const div = block.querySelector(`.mf-group[data-param="${param}"]`);
    if (div) renderMFGroup(div, param);
  });

  container.appendChild(node);
}

// ── Рендер групи "тип ФП + поля параметрів + мінімонографік" ─────────────────
function renderMFGroup(container, paramName) {
  const isProb = ['B1', 'K11', 'K00'].includes(paramName);

  const sel = document.createElement('select');
  sel.className = 'form-select form-select-sm mb-1';
  availableMFs.forEach(mf => {
    const opt = document.createElement('option');
    opt.value = mf;
    opt.textContent = mf;
    sel.appendChild(opt);
  });

  const paramsDiv = document.createElement('div');
  paramsDiv.className = 'mf-params d-flex gap-1 flex-wrap mb-1';

  const previewDiv = document.createElement('div');
  previewDiv.className = 'preview-mini';

  container.appendChild(sel);
  container.appendChild(paramsDiv);
  container.appendChild(previewDiv);

  function refresh() {
    const mfType = sel.value;
    const labels = MF_PARAM_LABELS[mfType] || ['p1', 'p2', 'p3'];
    const defaults = isProb
      ? defaultsForProb(mfType)
      : defaultsForTime(mfType, paramName);

    paramsDiv.innerHTML = '';
    labels.forEach((lbl, i) => {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.className = 'form-control form-control-sm';
      inp.style.width = '70px';
      inp.placeholder = lbl;
      // min/max залежно від ролі параметра
      if (['B1', 'K11', 'K00'].includes(paramName)) {
        inp.min = '0'; inp.max = '1'; inp.step = '0.001';
      } else if (['MT', 'DT'].includes(paramName)) {
        inp.min = '0'; inp.step = '0.01';
      } else {
        inp.step = '0.01';
      }
      inp.value = defaults[i] ?? '';
      inp.addEventListener('change', () => {
        clearValidationErrors();
        previewMF(sel.value, getParamValues(paramsDiv), previewDiv);
      });
      inp.addEventListener('blur', () => normalizeNumberInput(inp));
      paramsDiv.appendChild(inp);
    });
    previewMF(mfType, defaults, previewDiv);
  }

  sel.addEventListener('change', refresh);
  refresh();
}

function defaultsForProb(mfType) {
  return getDefaults(mfType, 'B');
}

function defaultsForTime(mfType, paramName) {
  return getDefaults(mfType, paramName === 'DT' ? 'DT' : 'MT');
}

function getParamValues(paramsDiv) {
  return Array.from(paramsDiv.querySelectorAll('input')).map(i => parseFloat(i.value));
}

function normalizeNumberInput(input) {
  const v = input.value.trim();
  if (v === '' || v === '-' || v === '.') return;
  const n = parseFloat(v);
  if (Number.isFinite(n)) input.value = n.toString();
}

async function previewMF(mfType, params, plotDiv) {
  if (params.some(isNaN)) return;
  try {
    const data = await fetch('/api/mf-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mf_type: mfType, params }),
    }).then(r => r.json());
    if (data.error) return;
    Plotly.react(plotDiv, [{
      x: data.x, y: data.mu, type: 'scatter', mode: 'lines',
      line: { color: '#0d6efd', width: 1.5 },
      fill: 'tozeroy', fillcolor: 'rgba(13,110,253,0.08)',
    }], {
      margin: { l: 20, r: 4, t: 4, b: 20 },
      xaxis: { tickfont: { size: 9 } },
      yaxis: { range: [0, 1.05], tickfont: { size: 9 } },
      showlegend: false,
    }, { responsive: true, displayModeBar: false });
  } catch (_) { /* ігнорувати помилки preview */ }
}

// ── Збір даних форми ──────────────────────────────────────────────────────────
function collectFormData() {
  const tfsType = document.getElementById('tfs-type').value;
  const dirTime = parseFloat(document.getElementById('directive-time').value);
  const blocks = document.querySelectorAll('#operations-container .op-block');

  const operations = Array.from(blocks).map(block => {
    const opType = block.dataset.opType;
    const params = {};
    block.querySelectorAll('.mf-group[data-param]').forEach(group => {
      const paramName = group.dataset.param;
      const mfType = group.querySelector('select').value;
      const values = getParamValues(group.querySelector('.mf-params'));
      params[paramName] = { mf_type: mfType, params: values };
    });
    return { type: opType, params };
  });

  // Для PAR: додаємо composer як перший елемент
  if (tfsType === 'PAR') {
    const sel = document.getElementById('par-composer');
    operations.unshift({ composer: sel ? sel.value : 'AND' });
  }

  const ptimelyMethod = document.getElementById('ptimely-method').value;
  return { tfs_type: tfsType, directive_time: dirTime,
           ptimely_method: ptimelyMethod, operations };
}

// ── Валідація ────────────────────────────────────────────────────────────────

function validateInputs() {
  const errors = [];
  const opLabel = window.i18n?.t('form_labels.operation_label', 'РО') || 'РО';
  document.querySelectorAll('#operations-container .op-block').forEach((block, opIdx) => {
    block.querySelectorAll('.mf-group[data-param]').forEach(group => {
      const role   = group.dataset.param;
      const mfType = group.querySelector('select')?.value;
      const vals   = getParamValues(group.querySelector('.mf-params'));
      if (vals.some(isNaN)) {
        errors.push(`${opLabel} ${opIdx+1}, ${role}: ${window.i18n?.t('validation.fields_empty', 'не всі поля заповнені')}`);
        return;
      }
      const isProb = ['B1','K11','K00'].includes(role);
      const isTime = ['MT','DT'].includes(role);
      if (isProb && vals.some(v => v < 0 || v > 1))
        errors.push(`${opLabel} ${opIdx+1}, ${role}: ${window.i18n?.t('validation.probability_range', 'значення мають бути в [0, 1]')}`);
      const freeNeg = MF_FREE_NEG_INDICES[mfType] || new Set();
      if (isTime && vals.some((v, i) => v < 0 && !freeNeg.has(i)))
        errors.push(`${opLabel} ${opIdx+1}, ${role}: ${window.i18n?.t('validation.non_negative', "значення не можуть бути від'ємними")}`);
      if (mfType === 'trimf' && vals.length === 3) {
        const [a,b,c] = vals;
        if (!(a<=b && b<=c)) errors.push(`${opLabel} ${opIdx+1}, ${role} ${window.i18n?.t('validation.trimf_order', 'trimf: a ≤ b ≤ c')} (${a}, ${b}, ${c})`);
      }
      if (mfType === 'trapmf' && vals.length === 4) {
        const [a,b,c,d] = vals;
        if (!(a<=b && b<=c && c<=d)) errors.push(`${opLabel} ${opIdx+1}, ${role} ${window.i18n?.t('validation.trapmf_order', 'trapmf: a ≤ b ≤ c ≤ d')}`);
      }
      if (mfType === 'gaussmf' && vals.length >= 1 && vals[0] <= 0)
        errors.push(`${opLabel} ${opIdx+1}, ${role} ${window.i18n?.t('validation.gauss_sigma', 'gaussmf: σ > 0')}`);
    });
  });
  const tDir = parseFloat(document.getElementById('directive-time').value);
  if (!isNaN(tDir) && tDir < 0)
    errors.push(window.i18n?.t('validation.directive_time_negative', "Директивний час не може бути від'ємним"));
  return errors;
}

function showValidationErrors(errors) {
  let box = document.getElementById('validation-errors');
  if (!box) {
    box = document.createElement('div');
    box.id = 'validation-errors';
    box.className = 'alert alert-warning mt-2 py-2';
    document.getElementById('error-box').after(box);
  }
  box.innerHTML = `<strong>${window.i18n?.t('validation.header', 'Перевірте дані:')}</strong><ul class="mb-0 ps-3">${
    errors.map(e => `<li>${e}</li>`).join('')}</ul>`;
  box.style.display = '';
}

function clearValidationErrors() {
  const box = document.getElementById('validation-errors');
  if (box) box.style.display = 'none';
}

// ── Обчислення ────────────────────────────────────────────────────────────────
async function evaluate() {
  clearValidationErrors();
  const valErrors = validateInputs();
  if (valErrors.length) { showValidationErrors(valErrors); return; }

  const errorBox = document.getElementById('error-box');
  errorBox.classList.add('d-none');
  document.getElementById('btn-evaluate').disabled = true;
  document.getElementById('btn-evaluate').textContent = window.i18n?.t('buttons.evaluating', 'Обчислення…');

  try {
    const payload = collectFormData();
    const result = await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());

    if (result.error) {
      showError(result.error);
      return;
    }

    lastResult = result;
    window.lastResult = result;
    window.lastPsvData = result.P_sv;
    displayResults(result);

  } catch (e) {
    showError((window.i18n?.t('errors.server_error', "Помилка з'єднання з сервером:")) + ' ' + e.message);
  } finally {
    document.getElementById('btn-evaluate').disabled = false;
    document.getElementById('btn-evaluate').textContent = window.i18n?.t('buttons.evaluate', 'Обчислити');
  }
}

function displayResults(result) {
  document.getElementById('results-section').style.display = '';
  document.getElementById('btn-download').disabled = false;
  plotFuzzySet('plot-P',  result.P,  'P̃');
  plotFuzzySet('plot-MT', result.MT, 'M̃(T), с');
  plotFuzzySet('plot-DT', result.DT, 'D̃(T)');
  renderMetrics(result);
  renderPsv(result.P_sv);
}

// ── Завантаження JSON з файлу ────────────────────────────────────────────────

function setupFileLoad() {
  const btn   = document.getElementById('btn-load-from-file');
  const input = document.getElementById('file-input-json');
  if (!btn || !input) return;
  btn.addEventListener('click', () => input.click());
  input.addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      await loadFromJsonData(data);
    } catch (err) {
      alert(`${window.i18n?.t('errors.load_file_failed', 'Помилка читання файлу:')} ${err.message}`);
    }
    input.value = '';
  });
}

async function loadFromJsonData(data) {
  let inputData, resultData = null;
  if (data.input && data.input.tfs_type) {
    inputData  = data.input;
    resultData = data.result || null;
  } else if (data.tfs_type && data.operations) {
    inputData = data;
  } else {
    alert(window.i18n?.t('errors.invalid_format', 'Невалідний формат. Очікується JSON з полями tfs_type і operations.'));
    return;
  }
  fillFormFromJSON(inputData);
  if (resultData && Object.keys(resultData).length > 0) {
    if (confirm(window.i18n?.t('errors.show_results_confirm', 'Файл містить збережені результати. Показати без повторного обчислення?'))) {
      lastResult = resultData;
      window.lastPsvData = resultData.P_sv ?? null;
      displayResults(resultData);
      return;
    }
  }
  await evaluate();
}

// Розширює масиви x/mu нулями ліворуч і праворуч для контексту (стиль MATLAB)
function expandFuzzySetForPlot(fs, capAtOne = false) {
  const xVals = fs.x, muVals = fs.mu;
  const xMin = xVals[0], xMax = xVals[xVals.length - 1];
  const range = xMax - xMin;
  if (range < 1e-9) return { x: xVals, mu: muVals };

  const pad = range * 0.15;
  const newXMin = Math.max(0, xMin - pad);
  const newXMax = capAtOne ? Math.min(1.0, xMax + pad) : xMax + pad;
  const N = 20;

  const leftXs = [], leftMus = [], rightXs = [], rightMus = [];
  if (newXMin < xMin) {
    const step = (xMin - newXMin) / N;
    for (let i = 0; i < N; i++) { leftXs.push(newXMin + step * i); leftMus.push(0); }
  }
  if (newXMax > xMax) {
    const step = (newXMax - xMax) / N;
    for (let i = 1; i <= N; i++) { rightXs.push(xMax + step * i); rightMus.push(0); }
  }
  return {
    x: [...leftXs, ...xVals, ...rightXs],
    mu: [...leftMus, ...muVals, ...rightMus],
  };
}

function plotFuzzySet(divId, fs, label) {
  const expanded = expandFuzzySetForPlot(fs, divId === 'plot-P');
  Plotly.react(document.getElementById(divId), [{
    x: expanded.x, y: expanded.mu, type: 'scatter', mode: 'lines',
    name: label,
    line: { color: '#0d6efd', width: 2 },
    fill: 'tozeroy', fillcolor: 'rgba(13,110,253,0.10)',
  }], {
    margin: { l: 35, r: 10, t: 10, b: 30 },
    xaxis: { tickfont: { size: 10 } },
    yaxis: { range: [0, 1.08], title: window.i18n?.t('axes.mu_label', 'μ'), tickfont: { size: 10 } },
    showlegend: false,
  }, { responsive: true, displayModeBar: false });
}

function renderMetrics(result) {
  const tbody = document.querySelector('#metrics-table tbody');
  tbody.innerHTML = '';

  // Рядки для трьох нечітких показників
  [
    [window.i18n?.t('metrics_table.row_P',  'P̃ (безпомилковість)'), result.P],
    [window.i18n?.t('metrics_table.row_MT', 'M̃(T), с'),             result.MT],
    [window.i18n?.t('metrics_table.row_DT', 'D̃(T)'),                result.DT],
  ].forEach(([name, fs]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${name}</td>
      <td>${fs.centroid.toFixed(5)}</td>
      <td>[${fs.support[0].toFixed(4)}, ${fs.support[1].toFixed(4)}]</td>
      <td>[${fs.core[0].toFixed(4)}, ${fs.core[1].toFixed(4)}]</td>`;
    tbody.appendChild(tr);
  });

  // Рядок P_св: значення в T_дир, носій і ядро по осі часу
  const tr = document.createElement('tr');
  const psv = result.P_sv;
  if (!psv) {
    tr.innerHTML = `<td>${window.i18n?.t('metrics_table.row_Psv', 'P̃_св(T_дир)')}</td><td>—</td><td>—</td><td>—</td>`;
  } else {
    const val = psv.value_at_directive.toFixed(5);
    const aboveZero = psv.p_sv.reduce((acc, v, i) => v > 0.001 ? [...acc, i] : acc, []);
    const suppStr = aboveZero.length
      ? `[${psv.t_dir[aboveZero[0]].toFixed(2)}, ${psv.t_dir[aboveZero.at(-1)].toFixed(2)}]`
      : '—';
    const coreStr = (psv.t_garant !== null && psv.t_garant !== undefined)
      ? `${window.i18n?.t('annotations.t_garant', 'T_гарант =')} ${psv.t_garant.toFixed(2)} ${window.i18n?.t('axes.seconds_short', 'с')}`
      : '—';
    tr.innerHTML = `<td>${window.i18n?.t('metrics_table.row_Psv', 'P̃_св(T_дир)')}</td><td>${val}</td><td>${suppStr}</td><td>${coreStr}</td>`;
  }
  tbody.appendChild(tr);
}

function updateQualityTooltip(quality) {
  const tip = document.getElementById('quality-tooltip');
  const wrap = document.getElementById('comparison-toggle-wrap');
  if (!quality) {
    tip.style.display = 'none';
    if (wrap) wrap.style.display = 'none';
    return;
  }
  tip.style.display = 'block';
  if (wrap) wrap.style.display = '';
  const icons = { good: '✓', acceptable: '⚠', poor: '✗' };
  const msgKeys = { good: 'quality.msg_good', acceptable: 'quality.msg_acceptable', poor: 'quality.msg_poor' };
  const qLabel  = window.i18n?.t('quality.label_prefix', 'Якість апроксимації:');
  const qMsg    = window.i18n?.t(msgKeys[quality.category], quality.message);
  const rmseLabel = window.i18n?.t('quality.rmse_label', 'RMSE =');
  tip.innerHTML = `<div class="quality-info quality-${quality.category}">
    <strong>${icons[quality.category]} ${qLabel} ${quality.quality_percent.toFixed(1)}%</strong>
    &nbsp;${rmseLabel} ${quality.rmse.toFixed(4)}
    &nbsp;— <em>${qMsg}</em>
  </div>`;
}

function renderPsv(psv) {
  const psvPlot = document.getElementById('plot-psv');
  updateQualityTooltip(psv?.quality ?? null);
  if (!psv) {
    // Використовуємо Plotly-annotation замість innerHTML,
    // щоб div-стан Plotly не губився між викликами.
    Plotly.react(psvPlot, [], {
      xaxis: { visible: false },
      yaxis: { visible: false },
      margin: { l: 10, r: 10, t: 10, b: 10 },
      annotations: [{
        text: window.i18n?.t('annotations.psv_empty', "Задайте директивний час і натисніть «Обчислити»"),
        xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
        showarrow: false, font: { size: 12, color: '#6c757d' },
      }],
    }, { responsive: true, displayModeBar: false });
    return;
  }
  plotPtimely(psv);
}

function pickDtick(rangeWidth) {
  /* Підбір кроку міток осі X: ~8 міток на ширину діапазону. */
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
  const ideal = rangeWidth / 8;
  return steps.reduce((best, s) =>
    Math.abs(s - ideal) < Math.abs(best - ideal) ? s : best, steps[0]);
}

function plotPtimely(psv) {
  const tDir   = parseFloat(document.getElementById('directive-time').value);
  const pAtDir = psv.value_at_directive;
  const tGarant = psv.t_garant;
  const tMin = psv.t_dir[0], tMax = psv.t_dir[psv.t_dir.length - 1];
  const rangeWidth = tMax - tMin;

  // Оновлюємо badge методу в заголовку карти
  const badge = document.getElementById('psv-method-badge');
  if (badge) { badge.textContent = psv.method; badge.style.display = ''; }

  // Адаптивний зсув анотації: якщо точка у правій 60% — текст ліворуч
  function annotAx(x) { return ((x - tMin) / rangeWidth) > 0.6 ? -55 : 55; }

  const shapes = [
    { type: 'line', xref: 'x', yref: 'paper',
      x0: tDir, x1: tDir, y0: 0, y1: 1,
      line: { color: 'red', dash: 'dash', width: 1.5 } },
  ];
  const annotations = [];

  // Зелена анотація T_гарант — вище точки (ay=-25)
  if (tGarant !== null && tGarant !== undefined && tGarant >= tMin && tGarant <= tMax) {
    annotations.push({
      x: tGarant, y: 1.0, xref: 'x', yref: 'y',
      text: `${window.i18n?.t('annotations.t_garant', 'T_гарант =')} ${tGarant.toFixed(2)} ${window.i18n?.t('axes.seconds_short', 'с')}`,
      showarrow: true, arrowhead: 2, arrowcolor: '#2e7d32',
      ax: annotAx(tGarant), ay: -25,
      font: { size: 11, color: 'white' },
      bgcolor: 'rgba(46,125,50,0.88)',
      bordercolor: '#1b5e20', borderwidth: 1,
    });
  }

  // Червона анотація P_св(T_дир) — нижче точки (ay=+30), лише якщо < 0.99
  if (pAtDir !== null && pAtDir < 0.99) {
    shapes.push({
      type: 'line', xref: 'x', yref: 'y',
      x0: tMin, x1: tDir, y0: pAtDir, y1: pAtDir,
      line: { color: 'red', dash: 'dot', width: 1 },
    });
    annotations.push({
      x: tDir, y: pAtDir, xref: 'x', yref: 'y',
      text: `${window.i18n?.t('annotations.psv_at_dir_prefix', 'P_св(')}${tDir.toFixed(1)}) = ${pAtDir.toFixed(3)}`,
      showarrow: true, arrowhead: 2, arrowcolor: '#b71c1c',
      ax: annotAx(tDir), ay: 30,
      font: { size: 11, color: 'white' },
      bgcolor: 'rgba(183,28,28,0.88)',
      bordercolor: '#7f0000', borderwidth: 1,
    });
  }

  const dtick = pickDtick(rangeWidth);
  const tickfmt = rangeWidth < 10 ? '.2f' : rangeWidth < 100 ? '.1f' : '.0f';

  const layout = {
    // t:100 резервує простір зверху для анотацій без зменшення їх розміру
    margin: { l: 50, r: 20, t: 100, b: 44 },
    xaxis: {
      title: { text: window.i18n?.t('axes.t_dir_label', 'T_дир, с'), font: { size: 10 } },
      tickfont: { size: 9 }, dtick, tickformat: tickfmt,
    },
    yaxis: {
      title: { text: window.i18n?.t('axes.psv_label', 'P_св'), font: { size: 10 } },
      range: [0, 1.08], tickfont: { size: 9 }, dtick: 0.1,
    },
    showlegend: false,
    shapes,
    annotations,
  };

  // Заголовок з індикатором якості
  const q = psv.quality;
  const qIcons = { good: '✓', acceptable: '⚠', poor: '✗' };
  layout.title = q
    ? { text: `P̃_св (${psv.method}) ${qIcons[q.category]} ${q.quality_percent.toFixed(1)}%`,
        font: { size: 13 } }
    : undefined;
  if (q) layout.margin.t = Math.max(layout.margin.t, 42);

  // Traces: основна крива + опціональна control-крива
  const showCmp = document.getElementById('show-method-comparison')?.checked && q;
  const traces = [{
    x: psv.t_dir, y: psv.p_sv, type: 'scatter', mode: 'lines',
    line: { color: '#0d6efd', width: 2 },
    name: psv.method,
  }];
  if (showCmp) {
    traces.push({
      x: psv.t_dir, y: q.control_curve.p_sv,
      type: 'scatter', mode: 'lines',
      line: { color: '#888', width: 1.5, dash: 'dash' },
      name: window.i18n?.t('graph_titles.control_curve', 'possibility (контроль)'),
    });
    layout.showlegend = true;
    layout.legend = { orientation: 'h', y: -0.18 };
  }

  Plotly.react(document.getElementById('plot-psv'), traces, layout,
               { responsive: true, displayModeBar: false });
}

function showError(msg) {
  const box = document.getElementById('error-box');
  box.textContent = msg;
  box.classList.remove('d-none');
}

// ── Скачати JSON (bare input — симетрично до імпорту) ─────────────────────────
function downloadJSON() {
  const input = collectFormData();
  const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  const blob = new Blob([JSON.stringify(input, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `fuzzy_scenario_${ts}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}


function getUserId() {
  let uid = localStorage.getItem('userId');
  if (!uid) {
    uid = crypto.randomUUID();
    localStorage.setItem('userId', uid);
  }
  return uid;
}

// Хелпер: fetch з автоматичним X-User-Id (щоб не забути в жодному місці)
function userFetch(url, options = {}) {
  const headers = { ...(options.headers || {}), 'X-User-Id': getUserId() };
  return fetch(url, { ...options, headers });
}

// ── Сценарії ─────────────────────────────────────────────────────────────────


async function clearAllScenarios() {
  if (!confirm(window.i18n?.t('scenarios_panel.clear_all_confirm', 'Видалити ВСІ ваші збережені сценарії? Цю дію не можна скасувати.'))) return;
  const res = await userFetch('/api/scenarios/clear-all', { method: 'POST' });
  if (res.ok) {
    await loadScenariosList();
  } else {
    const err = await res.json();
    alert(`${window.i18n?.t('errors.generic_prefix', 'Помилка:')} ${err.error}`);
  }
}

async function loadScenariosList() {
  const res  = await userFetch('/api/scenarios');
  const list = await res.json();
  const ul   = document.getElementById('scenarios-list');
  ul.innerHTML = '';
  if (!list.length) {
    const li = document.createElement('li');
    li.className = 'empty-state';
    li.dataset.i18n = 'scenarios_panel.empty_state';
    li.textContent = window.i18n?.t('scenarios_panel.empty_state', 'Немає збережених сценаріїв');
    ul.appendChild(li);
    return;
  }
  for (const s of list) {
    const li = document.createElement('li');
    li.className = 'scenario-item';
    li.innerHTML = `
      <input type="checkbox" class="scenario-check flex-shrink-0" data-name="${s.name}">
      <div class="scenario-meta">
        <strong title="${s.name}">${s.name}</strong>
        <small class="text-muted">${s.tfs_type}, ${s.n_operations} ${window.i18n?.t('scenarios_panel.ops_short', 'оп.')}</small>
      </div>
      <div class="scenario-actions">
        <button class="btn btn-sm btn-outline-primary py-0" data-action="load"   data-name="${s.name}">&#x2193;</button>
        <button class="btn btn-sm btn-outline-danger  py-0" data-action="delete" data-name="${s.name}">&times;</button>
      </div>`;
    ul.appendChild(li);
  }
  ul.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const name = btn.dataset.name;
      if (btn.dataset.action === 'load') {
        await loadScenario(name);
      } else {
        if (confirm(`${window.i18n?.t('scenarios_panel.delete_confirm', 'Видалити сценарій')} "${name}"?`)) {
          await userFetch(`/api/scenarios/${encodeURIComponent(name)}`, { method: 'DELETE' });
          loadScenariosList();
        }
      }
    });
  });
  ul.querySelectorAll('.scenario-check').forEach(cb =>
    cb.addEventListener('change', updateSelectedCount));
  updateSelectedCount();
}

function updateSelectedCount() {
  const checked = document.querySelectorAll('.scenario-check:checked');
  const count = checked.length;
  document.getElementById('selected-count').textContent = count;
  document.getElementById('btn-compare-scenarios').disabled = count < 2;
  if (count >= 5) {
    document.querySelectorAll('.scenario-check:not(:checked)').forEach(cb => cb.disabled = true);
  } else {
    document.querySelectorAll('.scenario-check').forEach(cb => cb.disabled = false);
  }
}

async function saveCurrentScenario() {
  const name = document.getElementById('scenario-name').value.trim();
  if (!name) { alert(window.i18n?.t('errors.empty_scenario_name', "Введіть ім'я сценарію")); return; }
  const res = await userFetch('/api/scenarios', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, input: collectFormData(), result: lastResult || {} }),
  });
  if (res.ok) {
    document.getElementById('scenario-name').value = '';
    loadScenariosList();
  } else {
    const err = await res.json();
    alert(`${window.i18n?.t('errors.generic_prefix', 'Помилка:')} ${err.error}`);
  }
}

async function loadScenario(name) {
  const res  = await userFetch(`/api/scenarios/${encodeURIComponent(name)}`);
  const data = await res.json();
  fillFormFromData(data.input);
  await evaluate();
}

function fillFormFromData(data) {
  fillFormFromJSON(data);
  const sel = document.getElementById('ptimely-method');
  if (sel && data.ptimely_method) sel.value = data.ptimely_method;
}


// ── Порівняння сценаріїв ──────────────────────────────────────────────────────

const CMP_COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'];

async function compareSelected() {
  const checked = [...document.querySelectorAll('.scenario-check:checked')];
  const results = await Promise.all(checked.map(async cb => {
    const res  = await userFetch(`/api/scenarios/${encodeURIComponent(cb.dataset.name)}`);
    const data = await res.json();
    return { name: cb.dataset.name, data };
  }));
  plotComparison(results);
  const modal = document.getElementById('comparison-modal');
  modal.style.display = 'block';
}

function plotComparison(scenarios) {
  const indicators = [
    { key: 'P',    div: 'cmp-plot-P',   title: window.i18n?.t('graph_titles.P',   'P̃ (безпомилкова)'), xk: 'x', yk: 'mu' },
    { key: 'MT',   div: 'cmp-plot-MT',  title: window.i18n?.t('graph_titles.MT',  'M̃(T), с'),          xk: 'x', yk: 'mu' },
    { key: 'DT',   div: 'cmp-plot-DT',  title: window.i18n?.t('graph_titles.DT',  'D̃(T)'),             xk: 'x', yk: 'mu' },
    { key: 'P_sv', div: 'cmp-plot-Psv', title: window.i18n?.t('graph_titles.Psv', 'P̃_св(T_дир)'),      xk: 't_dir', yk: 'p_sv' },
  ];
  indicators.forEach(ind => {
    const traces = scenarios.map((s, i) => {
      const d = s.data.result?.[ind.key];
      if (!d) return null;
      return {
        x: d[ind.xk], y: d[ind.yk],
        type: 'scatter', mode: 'lines',
        line: { color: CMP_COLORS[i % 5], width: 2 },
        name: s.name,
      };
    }).filter(Boolean);
    Plotly.newPlot(ind.div, traces, {
      title: { text: ind.title, font: { size: 13 } },
      xaxis: { title: ind.key === 'P_sv' ? window.i18n?.t('axes.t_dir_label', 'T_дир, с') : '' },
      yaxis: { title: ind.key === 'P_sv' ? window.i18n?.t('axes.psv_label', 'P_св') : window.i18n?.t('axes.mu_label', 'μ'), range: [0, 1.05] },
      showlegend: true,
      legend: { orientation: 'h', y: -0.18 },
      margin: { t: 40, b: 65, l: 50, r: 20 },
    }, { responsive: true, displayModeBar: false });
  });
}

// ── Приклад за замовчуванням ────────────────────────────────────────────────────
async function loadDefaultExample() {
  const data = await fetch('/examples/default_example.json').then(r => r.json());
  fillFormFromJSON(data);
}

function fillFormFromJSON(data) {
  // Встановлюємо тип ТФС; buildForm синхронно будує форму, включно з #par-composer для PAR
  const tfsSel = document.getElementById('tfs-type');
  tfsSel.value = data.tfs_type;
  tfsSel.dispatchEvent(new Event('change'));

  // Для PAR: встановлюємо composer одразу після buildForm (поки #par-composer ще в DOM)
  if (data.tfs_type === 'PAR') {
    const composerVal = data.operations[0]?.composer;
    if (composerVal) {
      const composerSel = document.getElementById('par-composer');
      if (composerSel) composerSel.value = composerVal;
    }
  }

  // Директивний час
  if (data.directive_time !== undefined) {
    document.getElementById('directive-time').value = data.directive_time;
  }

  const container = document.getElementById('operations-container');

  // Для RR і PAR — динамічно нарощуємо R-блоки
  if (data.tfs_type === 'RR' || data.tfs_type === 'PAR') {
    if (data.tfs_type === 'PAR') {
      // Видаляємо лише .op-block; composerDiv із #par-composer залишається
      container.querySelectorAll('.op-block').forEach(b => b.remove());
    } else {
      container.innerHTML = '';
    }
    const rrOps = data.tfs_type === 'PAR' ? data.operations.slice(1) : data.operations;
    rrOps.forEach((op, i) => appendRROperation(container, i + 1));
  }

  // Заповнюємо значення (для PAR пропускаємо operations[0] = {composer})
  const blocks = container.querySelectorAll('.op-block');
  const opsList = data.tfs_type === 'PAR' ? data.operations.slice(1) : data.operations;
  opsList.forEach((op, opIdx) => {
    const block = blocks[opIdx];
    if (!block) return;
    Object.entries(op.params).forEach(([paramName, spec]) => {
      const group = block.querySelector(`.mf-group[data-param="${paramName}"]`);
      if (!group) return;
      const sel = group.querySelector('select');
      if (sel) {
        sel.value = spec.mf_type;
        sel.dispatchEvent(new Event('change'));
      }
      // Невелика затримка щоб поля встигли відрендеритись після change
      setTimeout(() => {
        const inputs = group.querySelectorAll('.mf-params input');
        spec.params.forEach((val, i) => { if (inputs[i]) inputs[i].value = val; });
        // Оновити preview з реальними значеннями
        previewMF(spec.mf_type, spec.params, group.querySelector('.preview-mini'));
      }, 0);
    });
  });
}
