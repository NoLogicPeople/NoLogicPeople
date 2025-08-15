let sessionId = null;
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('chat-input');
const formEl = document.getElementById('chat-form');
const newSessionBtn = document.getElementById('new-session');
const voiceBtn = document.getElementById('voice-btn');
const sessionLabel = document.getElementById('session-label');
const ttsToggleBtn = document.getElementById('tts-toggle');
const historyEl = document.getElementById('history');

function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerText = content;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  // If assistant message and TTS enabled, speak it
  if (role === 'assistant') {
    maybeSpeak(content);
  }
}

async function createSession() {
  const res = await fetch('/api/session/new');
  const data = await res.json();
  sessionId = data.session_id;
  sessionLabel.innerText = `Oturum: ${sessionId.slice(0, 8)}`;
  chatEl.innerHTML = '';
  (data.messages || []).forEach(m => appendMessage(m.role, m.content));
  refreshHistory();
}

async function loadSession(id) {
  const res = await fetch(`/api/session/${id}`);
  const data = await res.json();
  sessionId = data.session_id;
  sessionLabel.innerText = `Oturum: ${sessionId.slice(0, 8)}`;
  chatEl.innerHTML = '';
  (data.messages || []).forEach(m => appendMessage(m.role, m.content));
}

async function sendMessage(text) {
  if (!sessionId) await createSession();
  appendMessage('user', text);
  // If TTS is speaking, stop to avoid overlap when user talks
  stopSpeaking();
  const res = await fetch(`/api/session/${sessionId}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  const reply = data.reply || '';
  // Server already updated full transcript; render the last assistant msg
  appendMessage('assistant', reply);
}

async function refreshHistory() {
  try {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    const sessions = data.sessions || [];
    historyEl.innerHTML = '';
    sessions.forEach(s => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.className = 'history-item';
      btn.innerHTML = `
        <div class="title">${escapeHtml(s.title || s.id)}</div>
        <div class="meta">${escapeHtml(s.updated_at || '')}</div>
        <div class="preview">${escapeHtml(s.last_message || '')}</div>
      `;
      btn.addEventListener('click', () => loadSession(s.id));
      li.appendChild(btn);
      historyEl.appendChild(li);
    });
  } catch (e) {
    console.error('Failed to load history', e);
  }
}

function escapeHtml(str) {
  return (str || '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[c]);
}

formEl.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  sendMessage(text);
});

newSessionBtn.addEventListener('click', () => {
  createSession();
});

// --- Voice (Web Speech API) ---
let recognition = null;
let isRecording = false;
let interimTranscript = '';
let collectedTranscript = '';

function getSpeechRecognizer() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return null;
  const rec = new SpeechRecognition();
  rec.lang = 'tr-TR';
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  rec.continuous = true; // keep listening and chunk results
  return rec;
}

function startRecording() {
  if (isRecording) return;
  recognition = getSpeechRecognizer();
  if (!recognition) {
    alert('Tarayıcı bu özelliği desteklemiyor. Lütfen Chrome kullanın.');
    return;
  }
  isRecording = true;
  interimTranscript = '';
  collectedTranscript = '';
  voiceBtn.setAttribute('aria-pressed', 'true');
  voiceBtn.classList.add('recording');
  voiceBtn.textContent = '⏺️';
  recognition.onresult = (event) => {
    let finalText = '';
    interimTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalText += transcript;
      } else {
        interimTranscript += transcript;
      }
    }
    if (interimTranscript) {
      inputEl.placeholder = `Dinleniyor… ${interimTranscript}`;
    }
    if (finalText) {
      collectedTranscript += (collectedTranscript ? ' ' : '') + finalText.trim();
      inputEl.value = collectedTranscript;
    }
  };
  recognition.onerror = (event) => {
    const err = event && event.error;
    if (err === 'no-speech' || err === 'aborted') {
      // Restart if still recording
      if (isRecording) {
        try { recognition.start(); } catch {}
      }
      return;
    }
    if (err === 'audio-capture') {
      alert('Mikrofon bulunamadı. Sistem ayarlarından doğru mikrofonu seçin.');
    } else if (err === 'not-allowed') {
      alert('Mikrofon izni reddedildi. Tarayıcı izinlerini kontrol edin.');
    }
    stopRecording();
  };
  recognition.onend = () => {
    // If still recording, auto-restart to keep listening.
    if (isRecording) {
      try { recognition.start(); } catch {}
      return;
    }
    // If user stopped, send whatever we collected
    const text = (collectedTranscript || interimTranscript || '').trim();
    interimTranscript = '';
    collectedTranscript = '';
    if (text) {
      inputEl.value = '';
      sendMessage(text);
    }
    stopRecording(true);
  };
  // Prompt mic permission first to avoid immediate end in some browsers
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(() => {
      try { recognition.start(); } catch {}
    }).catch(() => {
      alert('Mikrofon erişimi sağlanamadı. Tarayıcı izinlerini kontrol edin.');
      stopRecording();
    });
  } else {
    try { recognition.start(); } catch {}
  }
}

function stopRecording(fromOnEnd = false) {
  if (!isRecording) return;
  isRecording = false;
  voiceBtn.setAttribute('aria-pressed', 'false');
  voiceBtn.classList.remove('recording');
  voiceBtn.textContent = '🎤';
  inputEl.placeholder = 'Mesajınızı yazın...';
  try {
    if (recognition && !fromOnEnd) recognition.stop();
  } catch {}
  recognition = null;
}

if (voiceBtn) {
  voiceBtn.addEventListener('click', () => {
    // Interrupt TTS when starting or stopping mic
    stopSpeaking();
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
}

// --- TTS (Speech Synthesis) ---
let ttsEnabled = true;
let currentUtterance = null;

function stopSpeaking() {
  try {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  } catch {}
  currentUtterance = null;
}

function maybeSpeak(text) {
  if (!ttsEnabled) return;
  const synth = window.speechSynthesis;
  if (!synth) return;
  stopSpeaking();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = 'tr-TR';
  // Try to pick a Turkish voice if available
  const voices = synth.getVoices ? synth.getVoices() : [];
  const trVoice = voices.find(v => (v.lang || '').toLowerCase().startsWith('tr'))
                 || voices.find(v => (v.name || '').toLowerCase().includes('turk'))
                 || null;
  if (trVoice) utter.voice = trVoice;
  utter.rate = 1.0;
  utter.pitch = 1.0;
  utter.onstart = () => { currentUtterance = utter; };
  utter.onend = () => { currentUtterance = null; };
  utter.onerror = () => { currentUtterance = null; };
  synth.speak(utter);
}

if (ttsToggleBtn) {
  ttsToggleBtn.addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    ttsToggleBtn.setAttribute('aria-pressed', String(ttsEnabled));
    ttsToggleBtn.textContent = ttsEnabled ? '🔊' : '🔇';
    if (!ttsEnabled) stopSpeaking();
  });
}

// boot
createSession();
refreshHistory();
