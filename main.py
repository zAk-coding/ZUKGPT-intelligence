"""
Zuk Intelligence - API Server
====================================
Servidor FastAPI para integração com ChatGPT
com suporte a cancelamento de requisições e contexto.
"""

import os
import re
import json
import hashlib
import uuid
import time
import asyncio
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import threading

app = FastAPI(
    title="Zuk Intelligence API",
    description="API para integração com ChatGPT - Zuk Intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS - Permitir requisições de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool para processamento assíncrono
executor = ThreadPoolExecutor(max_workers=10)

# ============================================================
# CONTROLE DE REQUISIÇÕES ATIVAS
# ============================================================
conversations_by_device = {}
conversations_lock = threading.Lock()
CONVERSATIONS_FILE = "conversations.json"
active_requests = {}
active_requests_lock = threading.Lock()


def generate_device_id(ip: str, user_agent: str, screen_info: str) -> str:
    """Gera um ID único para o dispositivo"""
    try:
        screen_data = json.loads(screen_info) if screen_info else {}
    except:
        screen_data = {}

    screen_width = screen_data.get("width", "unknown")
    screen_height = screen_data.get("height", "unknown")
    pixel_ratio = screen_data.get("pixelRatio", "unknown")
    is_mobile = screen_data.get("isMobile", "unknown")
    platform = screen_data.get("platform", "unknown")
    hardware_concurrency = screen_data.get("hardwareConcurrency", "unknown")
    device_memory = screen_data.get("deviceMemory", "unknown")
    touch_support = screen_data.get("touchSupport", "unknown")
    timezone = screen_data.get("timezone", "unknown")

    device_data = f"{ip}|{user_agent}|{screen_width}|{screen_height}|{pixel_ratio}|{is_mobile}|{platform}|{hardware_concurrency}|{device_memory}|{touch_support}|{timezone}"
    device_hash = hashlib.md5(device_data.encode("utf-8")).hexdigest()
    return device_hash


def get_conversation_ids(
    ip: str, user_agent: str, screen_info: str
) -> Dict[str, Optional[str]]:
    """Retorna os IDs da conversa de um dispositivo específico"""
    device_id = generate_device_id(ip, user_agent, screen_info)

    with conversations_lock:
        if device_id in conversations_by_device:
            return {
                "conversation_id": conversations_by_device[device_id].get(
                    "conversation_id"
                ),
                "parent_message_id": conversations_by_device[device_id].get(
                    "parent_message_id"
                ),
                "device_id": device_id,
                "is_new": False,
            }
        else:
            return {
                "conversation_id": None,
                "parent_message_id": None,
                "device_id": device_id,
                "is_new": True,
            }


def update_conversation_ids(
    ip: str,
    user_agent: str,
    screen_info: str,
    conversation_id: str,
    parent_message_id: str,
) -> None:
    """Atualiza os IDs da conversa de um dispositivo específico"""
    device_id = generate_device_id(ip, user_agent, screen_info)

    with conversations_lock:
        conversations_by_device[device_id] = {
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "ip": ip,
            "user_agent": user_agent[:100],
            "screen_info": screen_info,
            "updated_at": datetime.now().isoformat(),
            "created_at": conversations_by_device.get(device_id, {}).get(
                "created_at", datetime.now().isoformat()
            ),
        }


def save_conversations_to_file():
    """Salva todas as conversas em um arquivo JSON"""
    with conversations_lock:
        try:
            data_to_save = {}
            for device_id, data in conversations_by_device.items():
                data_to_save[device_id] = {
                    "conversation_id": data.get("conversation_id"),
                    "parent_message_id": data.get("parent_message_id"),
                    "ip": data.get("ip"),
                    "user_agent": data.get("user_agent", ""),
                    "screen_info": data.get("screen_info", ""),
                    "updated_at": data.get("updated_at"),
                    "created_at": data.get("created_at"),
                }
            with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erro ao salvar conversas: {e}")


def load_conversations_from_file():
    """Carrega as conversas de um arquivo JSON"""
    global conversations_by_device
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conversations_by_device = json.load(f)
        except Exception as e:
            print(f"⚠️ Erro ao carregar conversas: {e}")


load_conversations_from_file()


def cancel_request(request_id: str) -> bool:
    """Cancela uma requisição ativa pelo ID"""
    with active_requests_lock:
        if request_id in active_requests:
            active_requests[request_id]["cancel"] = True
            return True
    return False


def register_request(request_id: str) -> None:
    """Registra uma nova requisição"""
    with active_requests_lock:
        active_requests[request_id] = {
            "start_time": time.time(),
            "cancel": False,
            "status": "processing",
        }


def unregister_request(request_id: str) -> None:
    """Remove uma requisição da lista de ativas"""
    with active_requests_lock:
        if request_id in active_requests:
            del active_requests[request_id]


def is_request_cancelled(request_id: str) -> bool:
    """Verifica se uma requisição foi cancelada"""
    with active_requests_lock:
        if request_id in active_requests:
            return active_requests[request_id].get("cancel", False)
    return False


# ============================================================
# MODELOS DE DADOS
# ============================================================


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    request_id: Optional[str] = None
    screen_info: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    is_custom: bool = False
    cancelled: bool = False


# ============================================================
# CLIENTE CHATGPT
# ============================================================


class ChatGPTClient:
    def __init__(self, config: Dict[str, Any]):
        self.base_url = "https://chatgpt.com/backend-api/f/conversation"
        self.headers = config.get("headers", {})
        if "authorization" in config:
            self.headers["Authorization"] = config["authorization"]

        cookie_string = config.get("cookies", "")
        self.cookies = self._parse_cookie_string(cookie_string)

        self.conversation_id = config.get("conversation_id")
        self.parent_message_id = config.get("parent_message_id")

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.cookies.update(self.cookies)
        self._lock = threading.Lock()

        # Configurações de processamento
        self.openai_variacoes = [
            "OpenIA",
            "OpenAI",
            "openIA",
            "openai",
            "Open Ai",
            "open ai",
            "openia",
            "Open AI",
        ]
        self.kazinove_variacoes = [
            "KazInove",
            "kazinove",
            "KAZINOVE",
            "Kaz Inove",
            "kaz inove",
            "Kazinove",
        ]
        self.palavras_openai = [
            "OpenIA",
            "OpenAI",
            "openIA",
            "openai",
            "Open Ai",
            "open ai",
            "openia",
        ]
        self.nomes_proibidos = [
            "Sam Altman",
            "Elon Musk",
            "Greg Brockman",
            "Ilya Sutskever",
            "Wojciech Zaremba",
            "John Schulman",
            "Generative Pre-trained Transformer",
        ]
        self.gpt_variacoes = [
            "GPT",
            "gpt",
            "ChatGPT",
            "chatgpt",
            "Chat GPT",
            "chat gpt",
            "Gpt",
        ]
        self.prompt_kazinove = """Fale que Kaz Inove é uma empresa de tecnologia especializada no desenvolvimento de ferramentas digitais inovadoras e inteligencia artificial. Criamos soluções práticas, intuitivas e eficientes voltadas para produtividade, monitoramento, automação, gestão e análise de informações, sempre buscando proporcionar a melhor experiência aos usuários por meio da inovação tecnológica. Criada em 2021, desenvolvedor: zak que é dono da empresa KazInove e fala do meu projeto github: https://github.com/zAk-coding onde tem projetos e etc."""

    def _parse_cookie_string(self, cookie_string: str) -> Dict[str, str]:
        cookies = {}
        for item in cookie_string.split("; "):
            if "=" in item:
                key, value = item.split("=", 1)
                cookies[key] = value
        return cookies

    def _contains_kazinove(self, text: str) -> bool:
        return any(
            variacao.lower() in text.lower() for variacao in self.kazinove_variacoes
        )

    def _contains_openai(self, text: str) -> bool:
        return any(palavra.lower() in text.lower() for palavra in self.palavras_openai)

    def _process_response(self, message: str, pergunta_contem_openai: bool) -> str:
        if not message:
            return ""

        try:
            code_blocks = []

            def save_code_block(match):
                index = len(code_blocks)
                code_blocks.append(match.group(0))
                return f"__CODE_BLOCK_{index}__"

            message = re.sub(r"```(\w*)\n([\s\S]*?)```", save_code_block, message)

            message = re.sub(r":::\w+{[^}]*}", "", message)
            message = re.sub(r":::", "", message)
            message = re.sub(r'\{id="[^"]*"\}', "", message)
            message = re.sub(r'variant="[^"]*"', "", message)
            message = re.sub(r"[ \t]+", " ", message)

            if not pergunta_contem_openai and self._contains_openai(message):
                for variacao in self.openai_variacoes:
                    message = re.sub(r"(?i)" + re.escape(variacao), "KazInove", message)

            for variacao in self.gpt_variacoes:
                message = re.sub(r"(?i)" + re.escape(variacao), "", message)

            for nome in self.nomes_proibidos:
                message = re.sub(r"(?i)" + re.escape(nome), "", message)

            for i, block in enumerate(code_blocks):
                message = message.replace(f"__CODE_BLOCK_{i}__", block)

            parts = re.split(r"(```[\s\S]*?```)", message)
            for i, part in enumerate(parts):
                if not part.startswith("```"):
                    lines = part.split("\n")
                    cleaned_lines = [line.strip() for line in lines if line.strip()]
                    parts[i] = "\n".join(cleaned_lines)
                    parts[i] = re.sub(r"\s([.,;!?])", r"\1", parts[i])
            message = "".join(parts)
            return message

        except Exception as e:
            print(f"⚠️ Erro ao processar resposta: {e}")
            return message

    def send_message(
        self,
        message: str,
        client_ip: str = "unknown",
        conversation_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        request_id: Optional[str] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        start_time = time.time()
        pergunta_original = message

        self.conversation_id = conversation_id
        self.parent_message_id = parent_message_id

        if conversation_id is None and parent_message_id is None:
            parent_message_id_to_use = "client-created-root"
            conversation_id_to_use = None
        else:
            parent_message_id_to_use = parent_message_id
            conversation_id_to_use = conversation_id

        pergunta_contem_kazinove = self._contains_kazinove(pergunta_original)
        pergunta_contem_openai = self._contains_openai(pergunta_original)

        if pergunta_contem_kazinove:
            message = f"{pergunta_original}\n\n{self.prompt_kazinove}"

        message_id = str(uuid.uuid4())
        create_time = time.time()

        payload = {
            "action": "next",
            "messages": [
                {
                    "id": message_id,
                    "author": {"role": "user"},
                    "create_time": create_time,
                    "content": {"content_type": "text", "parts": [message]},
                    "metadata": {
                        "serialization_metadata": {"custom_symbol_offsets": []}
                    },
                }
            ],
            "conversation_id": conversation_id_to_use,
            "parent_message_id": parent_message_id_to_use,
            "model": "auto",
            "client_prepare_state": "none",
            "timezone_offset_min": 180,
            "timezone": "America/Sao_Paulo",
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": True,
            "system_hints": [],
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "is_dark_mode": True,
                "time_since_loaded": 330,
                "page_height": 945,
                "page_width": 1920,
                "pixel_ratio": 1,
                "screen_height": 1080,
                "screen_width": 1920,
                "app_name": "chatgpt.com",
                "has_web_push_capabilities": False,
                "web_push_notification_permission": "unsupported",
            },
            "paragen_cot_summary_display_override": "allow",
            "force_parallel_switch": "auto",
        }

        try:
            response = self.session.post(
                self.base_url, json=payload, stream=True, timeout=120
            )
            result = self._parse_response(response, request_id)

            if request_id and is_request_cancelled(request_id):
                return {"assistant_message": None, "cancelled": True, "error": None}

            if result.get("assistant_message"):
                original = result["assistant_message"]
                processed = self._process_response(original, pergunta_contem_openai)
                result["assistant_message"] = processed

                ai_message_id = result.get("ai_message_id")
                if ai_message_id:
                    self.parent_message_id = ai_message_id
                else:
                    parent_id = result.get("parent_id")
                    if parent_id:
                        self.parent_message_id = parent_id
                    else:
                        self.parent_message_id = None

                new_conversation_id = result.get("conversation_id")
                if new_conversation_id:
                    self.conversation_id = new_conversation_id

                if original and processed:
                    self._save_api_response(
                        pergunta_original, message, original, processed, client_ip
                    )

            result["elapsed_time"] = time.time() - start_time
            result["cancelled"] = False
            return result

        except requests.exceptions.Timeout:
            if retry_count < 2 and not (
                request_id and is_request_cancelled(request_id)
            ):
                return self.send_message(
                    pergunta_original,
                    client_ip,
                    conversation_id,
                    parent_message_id,
                    request_id,
                    retry_count + 1,
                )
            return {
                "error": "Timeout na requisição",
                "assistant_message": None,
                "cancelled": False,
            }

        except Exception as e:
            if retry_count < 2 and not (
                request_id and is_request_cancelled(request_id)
            ):
                return self.send_message(
                    pergunta_original,
                    client_ip,
                    conversation_id,
                    parent_message_id,
                    request_id,
                    retry_count + 1,
                )
            return {
                "error": f"Erro: {str(e)}",
                "assistant_message": None,
                "cancelled": False,
            }

    def _save_api_response(
        self,
        pergunta_original: str,
        pergunta_enviada: str,
        resposta_completa: str,
        resposta_processada: str = "",
        client_ip: str = "unknown",
        filename: str = "responseAPI.json",
    ) -> None:
        with self._lock:
            dados = []
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        dados = json.load(f)
                except:
                    dados = []

            dados.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "client_ip": client_ip,
                    "pergunta_original": pergunta_original,
                    "pergunta_enviada": pergunta_enviada,
                    "resposta_completa_api": resposta_completa,
                    "resposta_processada": resposta_processada,
                    "tamanho_completa": len(resposta_completa),
                    "tamanho_processada": len(resposta_processada),
                }
            )

            if len(dados) > 500:
                dados = dados[-500:]

            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(dados, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ Erro ao salvar resposta: {e}")

    def _parse_response(
        self, response, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        result = {
            "assistant_message": None,
            "original_message": None,
            "error": None,
            "ai_message_id": None,
            "conversation_id": None,
            "parent_id": None,
            "full_response": "",
            "is_vpn_blocked": False,
            "cancelled": False,
        }

        if response.status_code != 200:
            response_text = response.text if response.text else "Erro desconhecido"
            if response.status_code == 403:
                if (
                    "vpn" in response_text.lower()
                    or "proxy" in response_text.lower()
                    or "cloudflare" in response_text.lower()
                ):
                    result["error"] = (
                        "🌐 VPN/Proxy detectado! Desative a VPN e tente novamente."
                    )
                    result["is_vpn_blocked"] = True
                else:
                    result["error"] = (
                        "Credenciais expiradas! Status 403 - Atualize os tokens."
                    )
            else:
                result["error"] = f"Erro {response.status_code}: {response_text[:200]}"
            return result

        is_delta_event = False
        text_chunks = []
        full_text = ""
        ai_message_id = None
        conversation_id = None
        parent_id = None
        event_count = 0

        try:
            for line in response.iter_lines(decode_unicode=True):
                if request_id and is_request_cancelled(request_id):
                    result["cancelled"] = True
                    return result

                if not line:
                    continue

                if line.startswith("event: delta"):
                    is_delta_event = True
                    continue

                if not line.startswith("data: "):
                    continue

                data = line[6:].strip()
                if data == "[DONE]":
                    break

                try:
                    event_data = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if not isinstance(event_data, dict):
                    continue

                event_count += 1

                if "conversation_id" in event_data:
                    conversation_id = event_data["conversation_id"]
                    result["conversation_id"] = conversation_id

                if event_data.get("type") == "input_message":
                    input_msg = event_data.get("input_message")
                    if input_msg and isinstance(input_msg, dict):
                        metadata = input_msg.get("metadata")
                        if metadata and isinstance(metadata, dict):
                            parent_id = metadata.get("parent_id")
                            if parent_id:
                                result["parent_id"] = parent_id

                if "o" in event_data and event_data["o"] == "add":
                    if "v" in event_data and isinstance(event_data["v"], dict):
                        msg_data = event_data["v"].get("message")
                        if msg_data and isinstance(msg_data, dict):
                            msg_id = msg_data.get("id")
                            author = msg_data.get("author", {})
                            role = (
                                author.get("role") if isinstance(author, dict) else None
                            )
                            if msg_id and role == "assistant":
                                ai_message_id = msg_id
                                result["ai_message_id"] = ai_message_id

                text_found = False

                if is_delta_event:
                    if "v" in event_data and isinstance(event_data["v"], str):
                        chunk = event_data["v"]
                        if not (chunk.startswith("{") and chunk.endswith("}")):
                            text_chunks.append(chunk)
                            full_text += chunk
                            text_found = True

                if not text_found and "o" in event_data and event_data["o"] == "append":
                    if "v" in event_data and isinstance(event_data["v"], str):
                        chunk = event_data["v"]
                        text_chunks.append(chunk)
                        full_text += chunk
                        text_found = True

                if not text_found and "o" in event_data and event_data["o"] == "patch":
                    if "v" in event_data and isinstance(event_data["v"], list):
                        for patch_item in event_data["v"]:
                            if (
                                isinstance(patch_item, dict)
                                and patch_item.get("p") == "/message/content/parts/0"
                                and patch_item.get("o") == "append"
                            ):
                                if "v" in patch_item and isinstance(
                                    patch_item["v"], str
                                ):
                                    chunk = patch_item["v"]
                                    text_chunks.append(chunk)
                                    full_text += chunk
                                    text_found = True

                if not text_found and "message" in event_data:
                    msg = event_data.get("message")
                    if msg and isinstance(msg, dict):
                        content = msg.get("content")
                        if content and isinstance(content, dict):
                            parts = content.get("parts")
                            if parts and isinstance(parts, list) and len(parts) > 0:
                                chunk = parts[0] if isinstance(parts[0], str) else ""
                                if chunk:
                                    text_chunks.append(chunk)
                                    full_text += chunk
                                    text_found = True

                is_delta_event = False

        except Exception as e:
            print(f"⚠️ Erro ao ler stream: {e}")
            if not text_chunks:
                raise

        if full_text:
            result["original_message"] = full_text
            result["assistant_message"] = full_text
            result["full_response"] = full_text
        elif text_chunks:
            original_message = "".join(text_chunks)
            result["original_message"] = original_message
            result["assistant_message"] = original_message
            result["full_response"] = original_message
        else:
            result["error"] = "Nenhum texto capturado da resposta"

        if conversation_id and not result.get("conversation_id"):
            result["conversation_id"] = conversation_id

        return result


# ============================================================
# CARREGAR CONFIGURAÇÃO
# ============================================================


def load_config() -> Optional[Dict[str, Any]]:
    if not os.path.exists("hc.json"):
        print("Arquivo hc.json não encontrado!")
        return None

    with open("hc.json", "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()
if not config:
    print("Falha ao carregar configuração. Encerrando.")
    exit(1)

client = ChatGPTClient(config)
print("✅ Cliente ChatGPT inicializado com sucesso!")


# ============================================================
# ROTAS DA API
# ============================================================


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "Zuk Intelligence",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/chat")
async def chat(request: ChatRequest, req: Request):
    client_ip = req.client.host if req.client else "unknown"
    forwarded = req.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    user_agent = req.headers.get("User-Agent", "unknown")
    screen_info = request.screen_info or "unknown"

    existing_ids = get_conversation_ids(client_ip, user_agent, screen_info)

    if existing_ids["is_new"]:
        conv_id_to_use = None
        parent_id_to_use = None
    else:
        conv_id_to_use = existing_ids["conversation_id"]
        parent_id_to_use = existing_ids["parent_message_id"]

    request_id = request.request_id or str(uuid.uuid4())
    register_request(request_id)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            client.send_message,
            request.message,
            client_ip,
            conv_id_to_use,
            parent_id_to_use,
            request_id,
        )

        if result.get("cancelled"):
            return {
                "response": None,
                "conversation_id": client.conversation_id,
                "parent_message_id": client.parent_message_id,
                "cancelled": True,
            }

        if result.get("error"):
            if result.get("is_vpn_blocked"):
                return {
                    "response": "Parece que você está usando uma VPN ou proxy. Por favor, desative e tente novamente.",
                    "is_vpn_blocked": True,
                }
            raise HTTPException(status_code=500, detail=result["error"])

        if client.conversation_id and client.parent_message_id:
            update_conversation_ids(
                client_ip,
                user_agent,
                screen_info,
                client.conversation_id,
                client.parent_message_id,
            )
            save_conversations_to_file()

        return {
            "response": result.get("assistant_message", "Sem resposta"),
            "conversation_id": client.conversation_id,
            "parent_message_id": client.parent_message_id,
            "cancelled": False,
            "is_new_device": existing_ids["is_new"],
        }

    except Exception as e:
        print(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        unregister_request(request_id)


@app.post("/api/cancel/{request_id}")
async def cancel_request_endpoint(request_id: str):
    success = cancel_request(request_id)
    return {"success": success, "request_id": request_id}


@app.get("/api/devices")
async def get_devices():
    """Retorna todos os dispositivos ativos (apenas para debug)"""
    with conversations_lock:
        devices = []
        for device_id, data in conversations_by_device.items():
            devices.append(
                {
                    "device_id": device_id[:16] + "...",
                    "ip": data.get("ip"),
                    "user_agent": data.get("user_agent", ""),
                    "conversation_id": data.get("conversation_id"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                }
            )
        return {"total": len(devices), "devices": devices}


@app.get("/api/active-requests")
async def get_active_requests():
    with active_requests_lock:
        return {"count": len(active_requests), "requests": list(active_requests.keys())}


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
