(function () {
  const el = (id) => document.getElementById(id);

  const inputText = el('inputText');
  const mergeRange = el('mergeRange');
  const mergeValue = el('mergeValue');
  const chkLower = el('chkLower');
  const chkShowEnds = el('chkShowEnds');
  const chkSampling = el('chkSampling');
  const tokenizerSelect = el('tokenizerSelect');
  const tokenizerInfo = el('tokenizerInfo');
  const tokenizerBadge = el('tokenizerBadge');
  const mergeControl = el('mergeControl');
  const wordEndOption = el('wordEndOption');
  const samplingOption = el('samplingOption');

  const statCount = el('statCount');
  const statVocab = el('statVocab');
  const statUnique = el('statUnique');

  const tokensWrap = el('tokensWrap');
  const vocabWrap = el('vocabWrap');

  const btnCopyTokens = el('btnCopyTokens');
  const btnCopyCount = el('btnCopyCount');
  const btnSample = el('btnSample');
  const btnClear = el('btnClear');

  const END = '</w>';
  
  const API_BASE = window.location.origin;

  const TOKENIZER_INFO = {
    'bpe': {
      name: 'BPE',
      description: 'Iteratively merges the most frequent pairs of bytes/characters',
      showMerges: true,
      showWordEnd: true,
      showSampling: false
    },
    'unigram': {
      name: 'Unigram',
      description: 'Probabilistic segmentation based on likelihood maximization',
      showMerges: false,
      showWordEnd: false,
      showSampling: false
    },
    'subword_regularization': {
      name: 'SubReg',
      description: 'Stochastic segmentation for improved generalization',
      showMerges: false,
      showWordEnd: false,
      showSampling: true
    }
  };

  function debounce(fn, delay = 150) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), delay);
    };
  }

  function displayTokens(tokens) {
    tokensWrap.innerHTML = '';
    
    if (!tokens || tokens.length === 0) {
      tokensWrap.innerHTML = '<div style="color: var(--color-text-tertiary); font-size: 0.875rem; padding: 1rem; text-align: center; width: 100%;">Tokens will appear here...</div>';
      return;
    }

    for (const tok of tokens) {
      const isSpace = /^\s+$/.test(tok);
      
      if (isSpace) {
        const chip = document.createElement('div');
        chip.className = 'token-chip';
        chip.style.cssText = 'color: var(--color-text-tertiary); background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);';
        chip.textContent = tok.replace(/\n/g, '⏎');
        tokensWrap.appendChild(chip);
      } else {
        const chip = document.createElement('div');
        chip.className = 'token-chip';
        const pretty = tok === END ? '∎' : tok;
        const c = colorStyleFor(tok);
        chip.style.backgroundColor = c.bg;
        chip.style.borderColor = c.border;
        chip.style.border = `1px solid ${c.border}`;
        chip.innerHTML = `<span style="color:${c.text}">${escapeHtml(pretty)}</span>`;
        tokensWrap.appendChild(chip);
      }
    }

    btnCopyTokens.onclick = async () => {
      const nonWhitespace = tokens.filter(t => !/^\s+$/.test(t));
      const text = nonWhitespace.map((t) => (t === END ? '</w>' : t)).join(' ');
      await copy(text);
      flash(btnCopyTokens);
    };
    btnCopyCount.onclick = async () => {
      await copy(String(window.currentStats?.token_count || 0));
      flash(btnCopyCount);
    };
  }

  function displayVocab(vocab) {
    vocabWrap.innerHTML = '';
    
    if (!vocab || vocab.length === 0) {
      vocabWrap.innerHTML = '<div style="color: var(--color-text-tertiary); font-size: 0.875rem; padding: 1rem; text-align: center; width: 100%;">Vocabulary will appear here...</div>';
      return;
    }

    for (const v of vocab) {
      const chip = document.createElement('div');
      chip.className = 'token-chip';
      chip.style.fontSize = '0.8125rem';
      const c = colorStyleFor(v);
      chip.style.backgroundColor = c.bg;
      chip.style.borderColor = c.border;
      chip.style.border = `1px solid ${c.border}`;
      chip.innerHTML = `<span style="color:${c.text}">${escapeHtml(v === END ? '∎' : v)}</span>`;
      vocabWrap.appendChild(chip);
    }
  }
  
  function displayStats(stats) {
    if (!stats) return;
    
    statCount.textContent = String(stats.token_count || 0);
    statVocab.textContent = String(stats.vocab_size || 0);
    statUnique.textContent = String(stats.unique_tokens || 0);
  }

  function updateTokenizerUI() {
    const tokenizerType = tokenizerSelect.value;
    const info = TOKENIZER_INFO[tokenizerType];
    
    tokenizerInfo.textContent = info.description;
    
    tokenizerBadge.textContent = info.name;
    
    if (info.showMerges) {
      mergeControl.style.display = 'block';
    } else {
      mergeControl.style.display = 'none';
    }
    
    if (info.showWordEnd) {
      wordEndOption.style.display = 'flex';
    } else {
      wordEndOption.style.display = 'none';
    }
    
    if (info.showSampling) {
      samplingOption.style.display = 'flex';
    } else {
      samplingOption.style.display = 'none';
    }
  }

  async function encodeText(text, options) {
    try {
      const tokenizerType = tokenizerSelect.value;
      
      const requestBody = {
        text: text,
        tokenizer_type: tokenizerType,
        max_merges: parseInt(mergeRange.value, 10),
        lowercase: options.lower,
        show_word_end: options.showEnds,
        use_sampling: options.useSampling || false
      };

      const response = await fetch(`${API_BASE}/api/encode`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Unknown error');
      }
      
      return data;
    } catch (error) {
      console.error('Error encoding text:', error);
      showError('Failed to encode text. Please check your connection.');
      return null;
    }
  }
  
  function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
      position: fixed;
      top: 1rem;
      right: 1rem;
      left: 1rem;
      max-width: 400px;
      margin-left: auto;
      background: rgba(239, 68, 68, 0.95);
      color: white;
      padding: 1rem 1.25rem;
      border-radius: 0.75rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
      z-index: 9999;
      font-size: 0.875rem;
      backdrop-filter: blur(10px);
    `;
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
      errorDiv.remove();
    }, 3000);
  }

  function preprocess(text, options) {
    let t = text || '';
    if (options.lower) t = t.toLowerCase();
    return t;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function hueFromString(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return h % 360;
  }

  function mapHueToBand(h, start = 280, end = 330) {
    const span = (end - start + 360) % 360 || 360;
    const normalized = h % 360;
    const frac = normalized / 360;
    return Math.round(start + frac * span) % 360;
  }

  function colorStyleFor(token) {
    const base = token === END ? 'END' : token;
    const h = mapHueToBand(hueFromString(base), 280, 330);
    return {
      bg: `hsla(${h}, 72%, 56%, 0.18)`,
      border: `hsla(${h}, 86%, 62%, 0.45)`,
      text: `hsl(${h}, 88%, 82%)`,
    };
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
  }

  function flash(btn) {
    const originalBg = btn.style.background;
    btn.style.background = 'rgba(217, 70, 239, 0.4)';
    btn.style.transform = 'scale(0.98)';
    setTimeout(() => {
      btn.style.background = originalBg;
      btn.style.transform = '';
    }, 200);
  }

  const SAMPLE = `Hello from Vivian, Adarsh and Anuj. This is our TokenizerLab project for DAA Lab!

As machine learning algorithms process numbers rather than text, the text must be converted to numbers. In the first step, a vocabulary is decided upon, then integer indices are arbitrarily but uniquely assigned to each vocabulary entry, and finally, an embedding is associated to the integer index. Algorithms include byte-pair encoding (BPE) and WordPiece. There are also special tokens serving as control characters, such as [MASK] for masked-out token (as used in BERT), and [UNK] ("unknown") for characters not appearing in the vocabulary. Also, some special symbols are used to denote special text formatting. For example, "Ġ" denotes a preceding whitespace in RoBERTa and GPT and "##" denotes continuation of a preceding word in BERT.[39]`;

  function currentOptions() {
    return {
      lower: chkLower.checked,
      showEnds: chkShowEnds.checked,
      useSampling: chkSampling ? chkSampling.checked : false,
    };
  }

  const update = debounce(async () => {
    const options = currentOptions();
    const text = inputText.value;
    
    if (!text.trim()) {
      tokensWrap.innerHTML = '';
      vocabWrap.innerHTML = '';
      statCount.textContent = '0';
      statVocab.textContent = '0';
      statUnique.textContent = '0';
      return;
    }
    
    const result = await encodeText(text, options);
    
    if (!result) {
      return;
    }
    
    displayTokens(result.tokens);
    displayVocab(result.vocab);
    displayStats(result.stats);
    
    window.currentTokens = result.tokens;
    window.currentStats = result.stats;
  }, 150);

  tokenizerSelect.addEventListener('change', () => {
    updateTokenizerUI();
    update();
  });

  mergeRange.addEventListener('input', () => {
    mergeValue.textContent = mergeRange.value;
    update();
  });
  
  chkLower.addEventListener('change', update);
  chkShowEnds.addEventListener('change', update);
  if (chkSampling) {
    chkSampling.addEventListener('change', update);
  }
  
  inputText.addEventListener('input', update);

  btnSample.addEventListener('click', () => {
    inputText.value = SAMPLE;
    update();
  });
  
  btnClear.addEventListener('click', () => {
    inputText.value = '';
    update();
  });

  updateTokenizerUI();
  mergeValue.textContent = mergeRange.value;
  inputText.value = SAMPLE;
  update();
})();
