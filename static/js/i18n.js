'use strict';

let translations = {};
let currentLang = 'uk';

async function loadTranslations(lang) {
  try {
    const res = await fetch(`/static/i18n/${lang}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    translations = await res.json();
    currentLang = lang;
    localStorage.setItem('lang', lang);
    applyTranslations();
  } catch (err) {
    console.error('i18n: failed to load', lang, err);
  }
}

// Отримати переклад за точковим ключем: "title_page.btn_start"
function t(key, fallback) {
  const parts = key.split('.');
  let val = translations;
  for (const p of parts) {
    if (val && typeof val === 'object' && p in val) {
      val = val[p];
    } else {
      return fallback !== undefined ? fallback : key;
    }
  }
  return typeof val === 'string' ? val : (fallback !== undefined ? fallback : key);
}

function applyTranslations() {
  // Звичайний текст: data-i18n
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    const text = t(key);
    if (el.tagName === 'INPUT' && el.type !== 'submit' && el.type !== 'button') {
      el.placeholder = text;
    } else {
      el.textContent = text;
    }
  });

  // HTML-вміст: data-i18n-html
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    el.innerHTML = t(el.dataset.i18nHtml);
  });

  // <title> сторінки
  const titleKey = document.documentElement.dataset.i18nTitle;
  if (titleKey) document.title = t(titleKey);

  // data-i18n-title — атрибут title + оновлення Bootstrap tooltip
  document.querySelectorAll('[data-i18n-title]:not(html)').forEach(el => {
    const text = t(el.dataset.i18nTitle);
    el.setAttribute('title', text);
    if (el.dataset.bsToggle === 'tooltip' || el.hasAttribute('data-bs-original-title')) {
      el.setAttribute('data-bs-original-title', text);
      const instance = window.bootstrap?.Tooltip?.getInstance(el);
      if (instance) instance.setContent({ '.tooltip-inner': text });
    }
  });

  // Повідомляємо решту коду (графіки, таблиці)
  document.dispatchEvent(new CustomEvent('languageChanged', { detail: { lang: currentLang } }));
}

document.addEventListener('DOMContentLoaded', async () => {
  const saved = localStorage.getItem('lang') || 'uk';
  await loadTranslations(saved);
});

window.i18n = { loadTranslations, t, applyTranslations, getCurrentLang: () => currentLang };
