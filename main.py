import requests
import json
import hashlib
import uuid
import os
import sys
from flask import Flask, request, jsonify, Response
from datetime import datetime
import logging

# ============================================
# CONFIGURAÇÃO DE LOGS
# ============================================

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/server_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

# ============================================
# CONFIGURAÇÃO DO APP
# ============================================

app = Flask(__name__)
app.logger.disabled = True

# ============================================
# CARREGA CONFIGURAÇÕES
# ============================================


def carregar_headers():
    try:
        with open("hc.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def carregar_payload():
    try:
        with open("payload.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


BASE_HEADERS = carregar_headers()
PAYLOAD_TEMPLATE = carregar_payload()


# ============================================
# CLASSE LUZIA CHAT
# ============================================


class LuziaChat:
    def __init__(self):
        self.url = "https://chat.luzia.com/api/poc/stream"
        self.headers = BASE_HEADERS.copy() if BASE_HEADERS else {}
        self.base_payload = PAYLOAD_TEMPLATE.copy() if PAYLOAD_TEMPLATE else {}
        self.thread_id = str(uuid.uuid4())
        self.thread_message_id = None

    def send_message(self, content):
        """Envia mensagem e retorna a resposta"""
        try:
            # Gera novo X-Correlation-Id
            if "X-Correlation-Id" in self.headers:
                self.headers["X-Correlation-Id"] = str(uuid.uuid4())

            # Monta payload
            payload = self.base_payload.copy()
            payload["content"] = content

            # Gerencia threadId
            if "threadId" not in payload:
                payload["threadId"] = self.thread_id
            else:
                self.thread_id = payload["threadId"]

            # Adiciona threadMessageId se existir
            if self.thread_message_id:
                payload["threadMessageId"] = self.thread_message_id

            # Faz a requisição
            response = requests.post(
                self.url, headers=self.headers, json=payload, stream=True, timeout=60
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "content": "",
                    "thread_message_id": None,
                    "thread_title": None,
                }

            # Processa a resposta
            full_content = ""
            events = []
            request_id = None
            new_thread_message_id = None
            followup_questions = []
            thread_title = None
            model_used = None
            tokens_used = None
            current_event = None

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    events.append(line)

                    if line.startswith("event: "):
                        current_event = line[7:]
                    elif line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])

                            if current_event == "stream_start":
                                request_id = data.get("requestId")
                                new_thread_message_id = data.get("threadMessageId")

                            elif current_event == "content_delta":
                                delta = data.get("delta", "")
                                full_content += delta

                            elif current_event == "enrichment":
                                if data.get("kind") == "follow_up_questions":
                                    followup_questions = data.get("data", [])

                            elif current_event == "done":
                                full_content = data.get("content", full_content)
                                thread_title = data.get("threadTitle")

                                if "metadata" in data and "usage" in data["metadata"]:
                                    usage = data["metadata"]["usage"]
                                    model_used = usage.get("model", "desconhecido")
                                    tokens_used = usage.get("totalTokens", 0)

                                if not followup_questions:
                                    followup_questions = data.get(
                                        "followupQuestions", []
                                    )

                        except json.JSONDecodeError:
                            pass

            # Atualiza thread_message_id
            if new_thread_message_id:
                self.thread_message_id = new_thread_message_id

            return {
                "success": True,
                "content": full_content,
                "thread_message_id": new_thread_message_id,
                "thread_title": thread_title,
                "followup_questions": followup_questions,
                "model_used": model_used,
                "tokens_used": tokens_used,
                "request_id": request_id,
                "raw_events": events,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": "",
                "thread_message_id": None,
                "thread_title": None,
            }


# ============================================
# ROTAS
# ============================================


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "LuzIA Chat API",
            "version": "1.0.0",
            "endpoints": {
                "/chat": "POST - Envia uma mensagem",
                "/health": "GET - Verifica se o servidor está online",
                "/reset": "POST - Reseta a conversa (novo threadId)",
            },
            "example": {
                "POST /chat": {
                    "body": {"message": "Olá, como você está?"},
                    "response": {
                        "success": True,
                        "content": "Olá! Tudo bem sim, e você?",
                        "thread_message_id": "abc123...",
                        "thread_title": "Saudações",
                    },
                }
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "headers_loaded": bool(BASE_HEADERS),
            "payload_loaded": bool(PAYLOAD_TEMPLATE),
        }
    )


@app.route("/reset", methods=["POST"])
def reset_conversation():
    """Reseta a conversa gerando um novo threadId"""
    chat = LuziaChat()
    return jsonify(
        {
            "success": True,
            "message": "Conversa resetada",
            "thread_id": chat.thread_id,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint principal para enviar mensagens"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Body JSON é obrigatório"}), 400

        message = data.get("message", "").strip()
        if not message:
            return (
                jsonify({"success": False, "error": "Campo 'message' é obrigatório"}),
                400,
            )

        # Cria instância do chat
        chat = LuziaChat()

        # Envia a mensagem
        result = chat.send_message(message)

        if result["success"]:
            return jsonify(
                {
                    "success": True,
                    "content": result["content"],
                    "thread_message_id": result["thread_message_id"],
                    "thread_title": result["thread_title"],
                    "followup_questions": result["followup_questions"],
                    "model_used": result["model_used"],
                    "tokens_used": result["tokens_used"],
                    "request_id": result["request_id"],
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": result.get("error", "Erro desconhecido"),
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                500,
            )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            500,
        )


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Endpoint com streaming para o frontend"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Body JSON é obrigatório"}), 400

        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Campo 'message' é obrigatório"}), 400

        chat = LuziaChat()

        def generate():
            result = chat.send_message(message)

            if result["success"]:
                # Envia o conteúdo em partes (simulando streaming)
                content = result["content"]
                chunk_size = 20

                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]
                    yield f"data: {json.dumps({'delta': chunk})}\n\n"

                # Envia metadados finais
                metadata = {
                    "done": True,
                    "isFinish": True,
                    "thread_message_id": result["thread_message_id"],
                    "thread_title": result["thread_title"],
                    "followup_questions": result["followup_questions"],
                    "full_response": content,
                    "model_used": result["model_used"],
                    "tokens_used": result["tokens_used"],
                }
                yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            else:
                yield f"data: {json.dumps({'error': result.get('error', 'Erro desconhecido')})}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# INICIAR SERVIDOR
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("=" * 70)
    print("🚀 LUZIA CHAT API - SERVIDOR")
    print("=" * 70)
    print(f"✅ Headers: {len(BASE_HEADERS)} itens")
    print(f"✅ Payload: {'OK' if PAYLOAD_TEMPLATE else 'FALHA'}")
    print("=" * 70)
    print(f"📌 Servidor: http://0.0.0.0:{port}")
    print("📌 Rotas:")
    print("   GET  /            - Informações do serviço")
    print("   GET  /health      - Health check")
    print("   POST /reset       - Resetar conversa")
    print("   POST /chat        - Enviar mensagem (JSON)")
    print("   POST /chat/stream - Enviar mensagem (SSE)")
    print("=" * 70)

    app.run(host="0.0.0.0", port=port, debug=False)
