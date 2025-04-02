// static/js/chat.js
document.addEventListener('DOMContentLoaded', () => {
    const messagesDiv = document.getElementById('messages');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');

    function addMessage(text, isUser) {
        const message = document.createElement('div');
        message.classList.add('message', isUser ? 'user' : 'ai');
        if (isUser) {
            message.textContent = text;  // User messages are plain text
        } else {
            message.innerHTML = text;  // AI messages are HTML
        }
        messagesDiv.appendChild(message);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function showLoading() {
        const loading = document.createElement('div');
        loading.classList.add('message', 'ai');
        loading.textContent = 'Thinking...';
        loading.id = 'loading';
        messagesDiv.appendChild(loading);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function hideLoading() {
        const loading = document.getElementById('loading');
        if (loading) loading.remove();
    }

    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = messageInput.value.trim();
        if (!text) return;

        addMessage(text, true);
        messageInput.value = '';
        showLoading();

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        hideLoading();
        addMessage(data.response, false);
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        showLoading();
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        hideLoading();
        addMessage(data.message, false);
        fileInput.value = '';
    });
});