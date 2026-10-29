const messagesContainer = document.getElementById('messagesContainer');
const chatForm = document.getElementById('chatForm');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const statusDot = document.getElementById('statusDot');

// Set backend URL (defaults to localhost:8000 for local development)
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : 'https://your-backend.onrender.com';

// Auto-grow textarea
queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = `${queryInput.scrollHeight}px`;
});

// Submit on Enter
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
});

// Clear messages
clearBtn.addEventListener('click', () => {
  messagesContainer.innerHTML = `
    <div class="message assistant-message">
      <div class="avatar">🤖</div>
      <div class="message-content">
        <div class="formatted-text">
          <p>Chat cleared! Ready for new questions.</p>
        </div>
      </div>
    </div>
  `;
});

// Check backend status silently
async function checkStatus() {
  try {
    const res = await fetch(`${API_URL}/api/health`);
    if (res.ok) {
      statusDot.className = 'status-dot online';
    } else {
      statusDot.className = 'status-dot';
    }
  } catch (err) {
    statusDot.className = 'status-dot';
  }
}
checkStatus();
setInterval(checkStatus, 15000);

// Quick question buttons
function sendPrompt(text) {
  queryInput.value = text;
  queryInput.dispatchEvent(new Event('input'));
  chatForm.dispatchEvent(new Event('submit'));
}

// Convert citations like [Document Name, page 1] into clean inline tags
function formatCitations(text) {
  return text.replace(/\[([^\]]+,\s*page\s*\d+)\]/gi, (match, p1) => {
    return `<span class="citation-tag">${p1}</span>`;
  });
}

// Render formatted message
function appendMessage(role, content) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}-message`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '👤' : '🤖';

  const msgContent = document.createElement('div');
  msgContent.className = 'message-content';

  const textDiv = document.createElement('div');
  textDiv.className = 'formatted-text';

  if (role === 'assistant') {
    // 1. Process citations
    const withCitations = formatCitations(content);
    // 2. Parse markdown into HTML cleanly
    if (window.marked) {
      textDiv.innerHTML = marked.parse(withCitations);
    } else {
      textDiv.innerHTML = withCitations.replace(/\n/g, '<br>');
    }
  } else {
    const p = document.createElement('p');
    p.textContent = content;
    textDiv.appendChild(p);
  }

  msgContent.appendChild(textDiv);
  msgDiv.appendChild(avatar);
  msgDiv.appendChild(msgContent);

  messagesContainer.appendChild(msgDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return msgDiv;
}

// Typing dots
function showTyping() {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message assistant-message';
  msgDiv.id = 'typingIndicator';
  msgDiv.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="message-content">
      <div class="typing-dots">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
      </div>
    </div>
  `;
  messagesContainer.appendChild(msgDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

// Submit handler
chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = queryInput.value.trim();
  if (!question) return;

  appendMessage('user', question);
  queryInput.value = '';
  queryInput.style.height = 'auto';
  queryInput.disabled = true;
  sendBtn.disabled = true;

  showTyping();

  try {
    const response = await fetch(`${API_URL}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    removeTyping();

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      appendMessage('assistant', `⚠️ ${errorData.detail || 'Could not fetch response from server.'}`);
      return;
    }

    const data = await response.json();
    appendMessage('assistant', data.answer);
  } catch (err) {
    removeTyping();
    appendMessage('assistant', `❌ Unable to reach the server. Please ensure the backend is running.`);
  } finally {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
});
