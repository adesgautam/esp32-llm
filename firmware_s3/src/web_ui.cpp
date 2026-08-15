#include "web_ui.h"

WebServer server(80);

// State variables for generation
static bool s_generation_requested = false;
static String s_prompt = "";
static String s_token_buffer = "";
static bool s_client_connected_sse = false;
static WiFiClient s_sse_client;

const char* html_page = R"=====(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Micro-LM ESP32-S3</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --chat-bg: #1e293b;
            --text-color: #f8fafc;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --user-msg: #3b82f6;
            --bot-msg: #334155;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            background-color: var(--chat-bg);
            padding: 1rem;
            text-align: center;
            border-bottom: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; letter-spacing: -0.025em; }
        .header p { margin: 0.5rem 0 0 0; font-size: 0.875rem; color: #94a3b8; }
        
        .chat-container {
            flex-grow: 1;
            padding: 1.5rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .message {
            max-width: 85%;
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            line-height: 1.5;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .user-message {
            align-self: flex-end;
            background-color: var(--user-msg);
            border-bottom-right-radius: 0.25rem;
        }
        .bot-message {
            align-self: flex-start;
            background-color: var(--bot-msg);
            border-bottom-left-radius: 0.25rem;
            border: 1px solid #475569;
            white-space: pre-wrap;
        }
        
        .input-area {
            padding: 1.5rem;
            background-color: var(--chat-bg);
            border-top: 1px solid #334155;
            display: flex;
            gap: 0.75rem;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 1rem 1.25rem;
            border-radius: 9999px;
            border: 1px solid #475569;
            background-color: #0f172a;
            color: white;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus { border-color: var(--accent); }
        button {
            padding: 0 1.5rem;
            border-radius: 9999px;
            border: none;
            background-color: var(--accent);
            color: white;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover { background-color: var(--accent-hover); }
        button:disabled { background-color: #475569; cursor: not-allowed; }
        
        .typing-indicator {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin-left: 0.5rem;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Micro-LM S3</h1>
        <p>28M Parameter Ternary Engine</p>
    </div>
    
    <div class="chat-container" id="chat">
        <div class="message bot-message">Hello! I am running entirely locally on this ESP32-S3. What would you like to ask me?</div>
    </div>
    
    <div class="input-area">
        <input type="text" id="promptInput" placeholder="Type a prompt..." autocomplete="off">
        <button id="sendBtn">Send</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('promptInput');
        const btn = document.getElementById('sendBtn');
        let currentBotMessage = null;

        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        async function sendPrompt() {
            const text = input.value.trim();
            if (!text) return;
            
            input.value = '';
            btn.disabled = true;
            addMessage(text, true);
            
            currentBotMessage = addMessage("", false);
            currentBotMessage.innerHTML = '<span class="typing-indicator"></span> Thinking...';
            
            try {
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: 'prompt=' + encodeURIComponent(text)
                });
                
                if (res.ok) {
                    currentBotMessage.textContent = "";
                    startSSE();
                } else {
                    currentBotMessage.textContent = "Error: Model busy or failed.";
                    btn.disabled = false;
                }
            } catch (e) {
                currentBotMessage.textContent = "Error connecting to device.";
                btn.disabled = false;
            }
        }

        function startSSE() {
            const evtSource = new EventSource("/stream");
            
            evtSource.onmessage = function(e) {
                if (e.data === "[DONE]") {
                    evtSource.close();
                    btn.disabled = false;
                    input.focus();
                } else {
                    // Replace special characters to handle whitespace correctly
                    let token = e.data.replace(/_SPACE_/g, ' ').replace(/_NEWLINE_/g, '\n');
                    currentBotMessage.textContent += token;
                    chat.scrollTop = chat.scrollHeight;
                }
            };
            
            evtSource.onerror = function() {
                evtSource.close();
                btn.disabled = false;
            };
        }

        btn.addEventListener('click', sendPrompt);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !btn.disabled) sendPrompt();
        });
    </script>
</body>
</html>
)=====";


void handleRoot() {
    server.send(200, "text/html", html_page);
}

void handleGenerate() {
    if (server.hasArg("prompt")) {
        s_prompt = server.arg("prompt");
        s_generation_requested = true;
        s_token_buffer = "";
        
        // Wait briefly for the client to reconnect to /stream
        server.send(200, "text/plain", "OK");
    } else {
        server.send(400, "text/plain", "Missing prompt");
    }
}

void handleStream() {
    // SSE endpoint
    s_sse_client = server.client();
    s_client_connected_sse = true;
    
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/event-stream", "");
    s_sse_client.println("Cache-Control: no-cache");
    s_sse_client.println("Connection: keep-alive");
    s_sse_client.println();
    s_sse_client.flush();
}

void init_web_ui() {
    WiFi.softAP("MicroLM-AP", "12345678");
    IPAddress IP = WiFi.softAPIP();
    Serial.print("AP IP address: ");
    Serial.println(IP);

    server.on("/", handleRoot);
    server.on("/generate", HTTP_POST, handleGenerate);
    server.on("/stream", HTTP_GET, handleStream);

    server.begin();
    Serial.println("Web server started");
}

void handle_web_ui_client() {
    server.handleClient();
}

bool is_generating_for_web() {
    return s_generation_requested;
}

const char* get_web_prompt() {
    return s_prompt.c_str();
}

void clear_web_prompt() {
    s_generation_requested = false;
    s_prompt = "";
}

void send_token_to_web_clients(const char* token) {
    if (!s_client_connected_sse || !s_sse_client.connected()) {
        s_client_connected_sse = false;
        return;
    }
    
    // Check if token indicates completion
    if (token == nullptr || token[0] == '\0') {
        s_sse_client.println("data: [DONE]\n");
        s_sse_client.flush();
        s_client_connected_sse = false;
        return;
    }

    // Sanitize whitespace for SSE data payload
    String t = String(token);
    t.replace("\n", "_NEWLINE_");
    t.replace("\r", "");
    t.replace(" ", "_SPACE_");
    
    if (t.length() > 0) {
        s_sse_client.print("data: ");
        s_sse_client.println(t);
        s_sse_client.println();
        s_sse_client.flush();
    }
}
