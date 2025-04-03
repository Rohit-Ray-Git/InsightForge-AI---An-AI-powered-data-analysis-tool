// static/js/chat.js
document.addEventListener('DOMContentLoaded', () => {
    const messagesDiv = document.getElementById('messages');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const reportBtn = document.getElementById('generate-report-btn');

    function addMessage(text, isUser) {
        const message = document.createElement('div');
        message.classList.add('message', isUser ? 'user' : 'ai');
        message.innerHTML = text;  // Use innerHTML to render links
        messagesDiv.appendChild(message);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    messageForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = messageInput.value.trim();
        if (!text) return;

        addMessage(text, true);
        messageInput.value = '';

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        addMessage(data.response, false);
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        addMessage(data.message, false);
        fileInput.value = '';
    });

    reportBtn.addEventListener('click', async () => {
        addMessage('Generating report...', true);
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        addMessage(data.response, false);
    });
});