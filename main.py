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
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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


# 🔥 Verifica se a pasta static existe; se não, cria
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS
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
    """
    Gera um ID único para o dispositivo baseado em:
    - IP
    - User-Agent (navegador/device)
    - Screen Info (resolução, pixel ratio, etc)
    """
    try:
        screen_data = json.loads(screen_info) if screen_info else {}
    except:
        screen_data = {}

    # 🔥 EXTRAI INFORMAÇÕES CHAVE PARA IDENTIFICAÇÃO
    screen_width = screen_data.get("width", "unknown")
    screen_height = screen_data.get("height", "unknown")
    pixel_ratio = screen_data.get("pixelRatio", "unknown")
    is_mobile = screen_data.get("isMobile", "unknown")
    platform = screen_data.get("platform", "unknown")
    hardware_concurrency = screen_data.get("hardwareConcurrency", "unknown")
    device_memory = screen_data.get("deviceMemory", "unknown")
    touch_support = screen_data.get("touchSupport", "unknown")
    timezone = screen_data.get("timezone", "unknown")

    # 🔥 COMBINA TUDO PARA GERAR UM ID ÚNICO
    device_data = f"{ip}|{user_agent}|{screen_width}|{screen_height}|{pixel_ratio}|{is_mobile}|{platform}|{hardware_concurrency}|{device_memory}|{touch_support}|{timezone}"

    # Gera um hash único
    device_hash = hashlib.md5(device_data.encode("utf-8")).hexdigest()

    return device_hash


# ============================================================
# ARMAZENAMENTO DE CONVERSAS POR DISPOSITIVO
# ============================================================


def get_conversation_ids(
    ip: str, user_agent: str, screen_info: str
) -> Dict[str, Optional[str]]:
    """Retorna os IDs da conversa de um dispositivo específico"""
    device_id = generate_device_id(ip, user_agent, screen_info)

    with conversations_lock:
        if device_id in conversations_by_device:
            print(f"✅ [DeviceID] Dispositivo ENCONTRADO: {device_id[:16]}...")
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
            print(f"🆕 [DeviceID] NOVO DISPOSITIVO DETECTADO! {device_id[:16]}...")
            print(f"   📱 User-Agent: {user_agent[:50]}...")
            print(f"   📱 Screen: {screen_info}")
            return {
                "conversation_id": None,
                "parent_message_id": None,
                "device_id": device_id,
                "is_new": True,  # 🔥 MARCA COMO NOVO!
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
        print(f"💾 [Zuk] Conversa salva para dispositivo {device_id[:8]}... (IP: {ip})")


def save_conversations_to_file():
    """Salva todas as conversas em um arquivo JSON"""
    with conversations_lock:
        try:
            # Remove dados sensíveis para salvar
            data_to_save = {}
            for device_id, data in conversations_by_device.items():
                data_to_save[device_id] = {
                    "conversation_id": data.get("conversation_id"),
                    "parent_message_id": data.get("parent_message_id"),
                    "ip": data.get("ip"),
                    "user_agent": data.get("user_agent", ""),
                    "screen_info": data.get("screen_info", ""),
                    "updated_at": data.get("updated_at"),
                    "created_at": data.get("created_at")
                }
            
            with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            print(f"💾 Conversas salvas em {CONVERSATIONS_FILE} ({len(data_to_save)} dispositivos)")
        except Exception as e:
            print(f"⚠️ Erro ao salvar conversas: {e}")

def load_conversations_from_file():
    """Carrega as conversas de um arquivo JSON"""
    global conversations_by_device
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            conversations_by_device = loaded_data
            print(f"📂 Conversas carregadas de {CONVERSATIONS_FILE}")
            print(f"   Total: {len(conversations_by_device)} dispositivos")
            return True
        except Exception as e:
            print(f"⚠️ Erro ao carregar conversas: {e}")
            return False
    return False

# 🔥 CARREGA CONVERSAS SALVAS AO INICIAR
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
    conversation_id: str
    parent_message_id: str
    is_custom: bool = False
    cancelled: bool = False


# ============================================================
# CLIENTE CHATGPT (Thread-safe com cancelamento)
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

    def _parse_cookie_string(self, cookie_string: str) -> Dict[str, str]:
        cookies = {}
        for item in cookie_string.split("; "):
            if "=" in item:
                key, value = item.split("=", 1)
                cookies[key] = value
        return cookies

    def _contains_kazinove(self, text: str) -> bool:
        text_lower = text.lower()
        for variacao in self.kazinove_variacoes:
            if variacao.lower() in text_lower:
                return True
        return False

    def _contains_openai(self, text: str) -> bool:
        text_lower = text.lower()
        for palavra in self.palavras_openai:
            if palavra.lower() in text_lower:
                return True
        return False

    def _replace_openai(self, text: str) -> str:
        code_blocks = []

        def save_code(match):
            idx = len(code_blocks)
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{idx}__"

        text = re.sub(r"```(\w*)\n([\s\S]*?)```", save_code, text)

        for variacao in self.openai_variacoes:
            text = re.sub(r"(?i)" + re.escape(variacao), "KazInove", text)

        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)

        return text

    def _remove_gpt(self, text: str) -> str:
        """Remove GPT da resposta (preserva blocos de código)"""
        code_blocks = []

        def save_code(match):
            idx = len(code_blocks)
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{idx}__"

        text = re.sub(r"```(\w*)\n([\s\S]*?)```", save_code, text)

        for variacao in self.gpt_variacoes:
            text = re.sub(r"(?i)" + re.escape(variacao), "", text)

        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)

        parts = re.split(r"(```[\s\S]*?```)", text)
        for i, part in enumerate(parts):
            if not part.startswith("```"):
                parts[i] = re.sub(r"\s+", " ", part).strip()
                parts[i] = re.sub(r"\s([.,;!?])", r"\1", parts[i])
        text = "".join(parts)

        return text

    def _remove_nomes_proibidos(self, text: str) -> str:
        """Remove nomes proibidos (preserva blocos de código)"""
        code_blocks = []

        def save_code(match):
            idx = len(code_blocks)
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{idx}__"

        text = re.sub(r"```(\w*)\n([\s\S]*?)```", save_code, text)

        for nome in self.nomes_proibidos:
            text = re.sub(r"(?i)" + re.escape(nome), "", text)

        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)

        parts = re.split(r"(```[\s\S]*?```)", text)
        for i, part in enumerate(parts):
            if not part.startswith("```"):
                parts[i] = re.sub(r"\s+", " ", part).strip()
                parts[i] = re.sub(r"\s([.,;!?])", r"\1", parts[i])
        text = "".join(parts)

        return text

    def _clean_response(self, text: str) -> str:
        """Limpa marcadores indesejados da resposta"""
        text = re.sub(r":::\w+{[^}]*}", "", text)
        text = re.sub(r":::", "", text)
        text = re.sub(r'\{id="[^"]*"\}', "", text)
        text = re.sub(r'variant="[^"]*"', "", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text

    def _process_response(self, message: str, pergunta_contem_openai: bool) -> str:
        """
        Processa a resposta - VERSÃO SIMPLIFICADA QUE PRESERVA BACKTICKS
        """
        if not message or message is None:
            print(f"⚠️ [Zuk] Mensagem vazia recebida para processamento")
            return ""

        try:
            # 🔥 EXTRAI BLOCOS DE CÓDIGO
            code_blocks = []

            def save_code_block(match):
                index = len(code_blocks)
                code_blocks.append(match.group(0))
                return f"__CODE_BLOCK_{index}__"

            # Guarda os blocos de código
            message = re.sub(r"```(\w*)\n([\s\S]*?)```", save_code_block, message)

            # Limpa marcadores indesejados
            message = re.sub(r":::\w+{[^}]*}", "", message)
            message = re.sub(r":::", "", message)
            message = re.sub(r'\{id="[^"]*"\}', "", message)
            message = re.sub(r'variant="[^"]*"', "", message)
            message = re.sub(r"[ \t]+", " ", message)

            # Substitui OpenAI (se necessário)
            if not pergunta_contem_openai:
                if self._contains_openai(message):
                    for variacao in self.openai_variacoes:
                        message = re.sub(
                            r"(?i)" + re.escape(variacao), "KazInove", message
                        )

            # Remove GPT
            for variacao in self.gpt_variacoes:
                message = re.sub(r"(?i)" + re.escape(variacao), "", message)

            # Remove nomes proibidos
            for nome in self.nomes_proibidos:
                message = re.sub(r"(?i)" + re.escape(nome), "", message)

            # 🔥 RESTAURA OS BLOCOS DE CÓDIGO
            for i, block in enumerate(code_blocks):
                message = message.replace(f"__CODE_BLOCK_{i}__", block)

            # Limpa espaços extras (FORA dos blocos)
            parts = re.split(r"(```[\s\S]*?```)", message)
            for i, part in enumerate(parts):
                if not part.startswith("```"):
                    lines = part.split("\n")
                    cleaned_lines = []
                    for line in lines:
                        cleaned_line = line.strip()
                        if cleaned_line:
                            cleaned_lines.append(cleaned_line)
                        else:
                            cleaned_lines.append("")
                    parts[i] = "\n".join(cleaned_lines)
                    parts[i] = re.sub(r"\s([.,;!?])", r"\1", parts[i])
            message = "".join(parts)

            return message

        except Exception as e:
            print(f"⚠️ [Zuk] Erro ao processar resposta: {e}")
            return message if message else ""

    def send_message(
        self,
        message: str,
        client_ip: str = "unknown",
        conversation_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        request_id: Optional[str] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Envia mensagem com suporte a cancelamento e contexto"""

        start_time = time.time()
        pergunta_original = message
        cancelled = False

        # 🔥 LOGS DOS IDs RECEBIDOS
        print(f"\n{'='*70}")
        print(f"📨 [REQ] IP: {client_ip} | ID: {request_id}")
        print(f"🆔 [REQ] conversation_id recebido: {conversation_id}")
        print(f"🆔 [REQ] parent_message_id recebido: {parent_message_id}")
        print(f"📝 [REQ] Mensagem: {message[:100]}{'...' if len(message) > 100 else ''}")
        print(f"🕐 [REQ] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # 🔥 SEMPRE USA OS IDs RECEBIDOS, MESMO QUE SEJAM NONE!
        self.conversation_id = conversation_id
        self.parent_message_id = parent_message_id

        print(f"📌 [Zuk] IDs recebidos do frontend:")
        print(f"   ├─ conversation_id: {conversation_id}")
        print(f"   └─ parent_message_id: {parent_message_id}")

        # 🔥🔥🔥 NOVA LÓGICA PARA PRIMEIRA MENSAGEM 🔥🔥🔥
        # 🔥 SE FOR A PRIMEIRA MENSAGEM (IDs None), USA "client-created-root"
        if conversation_id is None and parent_message_id is None:
            print(f"🆕 [Zuk] PRIMEIRA MENSAGEM DETECTADA!")
            print(f"   💡 Usando 'client-created-root' como parent_message_id")
            # 🔥 USA O ID ESPECIAL PARA PRIMEIRA MENSAGEM
            parent_message_id_to_use = "client-created-root"
            conversation_id_to_use = None

            # 🔥 VERIFICA SE É PERGUNTA SOBRE NOME NA PRIMEIRA MENSAGEM
            if "qual meu nome" in pergunta_original.lower() or "qual é meu nome" in pergunta_original.lower():
                print(f"🔍 [Zuk] Usuário perguntou nome na primeira mensagem!")
                print(f"   💡 O ChatGPT vai responder que não sabe (sem histórico)")
        else:
            # 🔥 USA OS IDs NORMAlS
            parent_message_id_to_use = parent_message_id
            conversation_id_to_use = conversation_id

        # 🔥 VERIFICA SE É PERGUNTA SOBRE KAZINOVE
        pergunta_contem_kazinove = self._contains_kazinove(pergunta_original)
        pergunta_contem_openai = self._contains_openai(pergunta_original)

        if pergunta_contem_kazinove:
            print(f"🔍 [Zuk] Pergunta sobre KazInove detectada!")
            print(f"📝 [Zuk] Adicionando prompt à pergunta...")
            message = f"{pergunta_original}\n\n{self.prompt_kazinove}"
            print(f"📤 [Zuk] Pergunta modificada: {message[:150]}...")

        # 🔥 GERA UM ID ÚNICO PARA A MENSAGEM
        message_id = str(uuid.uuid4())
        create_time = time.time()
        print(f"📝 [Zuk] ID da mensagem gerado: {message_id}")

        # 🔥 PREPARA O PAYLOAD
        payload = {
            "action": "next",
            "messages": [
                {
                    "id": message_id,
                    "author": {"role": "user"},
                    "create_time": create_time,
                    "content": {"content_type": "text", "parts": [message]},
                    "metadata": {"serialization_metadata": {"custom_symbol_offsets": []}},
                }
            ],
            "conversation_id": conversation_id_to_use,  # ← USA O ID MODIFICADO
            "parent_message_id": parent_message_id_to_use,  # ← USA O ID MODIFICADO
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

        print(f"\n📤 [Zuk] IDs que serão enviados na REQUISIÇÃO:")
        print(f"   ├─ conversation_id: {payload['conversation_id']}")
        print(f"   └─ parent_message_id: {payload['parent_message_id']}")

        print(f"\n🚀 [Zuk] Enviando requisição para ChatGPT...")

        try:
            response = self.session.post(
                self.base_url, json=payload, stream=True, timeout=120
            )

            print(f"📡 [Zuk] Status HTTP: {response.status_code}")

            result = self._parse_response(response, request_id) 

            if request_id and is_request_cancelled(request_id):
                print(f"⛔ [Zuk] Requisição cancelada durante processamento")
                cancelled = True
                return {"assistant_message": None, "cancelled": True, "error": None}

            # Depois de processar a resposta, onde você atualiza os IDs:
            if result.get("assistant_message"):
                original = result["assistant_message"]
                processed = self._process_response(original, pergunta_contem_openai)
                result["assistant_message"] = processed

                # 🔥 EXTRAI E ATUALIZA OS IDs DA RESPOSTA
                # PRIORIDADE: ai_message_id > parent_id > None
                ai_message_id = result.get("ai_message_id")
                if ai_message_id:
                    self.parent_message_id = ai_message_id
                    print(f"🔄 [Zuk] parent_message_id ATUALIZADO para (ID da IA): {self.parent_message_id}")
                else:
                    # 🔥 FALLBACK: usa o parent_id da mensagem do usuário
                    parent_id = result.get("parent_id")
                    if parent_id:
                        self.parent_message_id = parent_id
                        print(f"🔄 [Zuk] parent_message_id ATUALIZADO para (parent_id): {self.parent_message_id}")
                    else:
                        print(f"⚠️ [Zuk] Nenhum parent_id encontrado! Usando None.")
                        self.parent_message_id = None

                new_conversation_id = result.get("conversation_id")
                if new_conversation_id:
                    self.conversation_id = new_conversation_id
                    print(f"🔄 [Zuk] conversation_id ATUALIZADO para: {self.conversation_id}")

            if original and processed:
                self._save_api_response(
                pergunta_original=pergunta_original,
                pergunta_enviada=message,
                resposta_completa=original,   # ✅ CORRETO: 'resposta_completa'
                resposta_processada=processed,
                client_ip=client_ip,
            )

            elapsed = time.time() - start_time
            result["elapsed_time"] = elapsed
            result["cancelled"] = False

            print(f"\n💾 [Zuk] IDs ATUALIZADOS (para próxima mensagem):")
            print(f"   ├─ conversation_id: {self.conversation_id}")
            print(f"   └─ parent_message_id: {self.parent_message_id}")
            print(f"{'='*70}\n")

            return result

        except requests.exceptions.Timeout:
            error_msg = "Timeout na requisição para ChatGPT"
            print(f"❌ [Zuk] {error_msg}")

            if retry_count < 2 and not (request_id and is_request_cancelled(request_id)):
                print(f"🔄 [Zuk] Tentando novamente... (tentativa {retry_count + 2})")
                return self.send_message(
                    message=pergunta_original,
                    client_ip=client_ip,
                    conversation_id=conversation_id,
                    parent_message_id=parent_message_id,
                    request_id=request_id,
                    retry_count=retry_count + 1,
                )

            return {"error": error_msg, "assistant_message": None, "cancelled": False}

        except Exception as e:
            error_msg = f"Erro na requisição: {str(e)}"
            print(f"❌ [Zuk] {error_msg}")

            if retry_count < 2 and not (request_id and is_request_cancelled(request_id)):
                print(f"🔄 [Zuk] Tentando novamente... (tentativa {retry_count + 2})")
                return self.send_message(
                    message=pergunta_original,
                    client_ip=client_ip,
                    conversation_id=conversation_id,
                    parent_message_id=parent_message_id,
                    request_id=request_id,
                    retry_count=retry_count + 1,
                )

            return {"error": error_msg, "assistant_message": None, "cancelled": False}

    def _save_stream_events(
        self, events: List[Dict], filename: str = "stream_events_complete.json"
    ) -> None:
        """Salva todos os eventos do stream em um arquivo JSON para debug"""
        try:
            import os
            import json
            from datetime import datetime

            dados = {
                "timestamp": datetime.now().isoformat(),
                "total_events": len(events),
                "events": events,
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(dados, f, indent=2, ensure_ascii=False)
            print(f"💾 [Zuk] {len(events)} eventos salvos em {filename}")
        except Exception as e:
            print(f"⚠️ [Zuk] Erro ao salvar stream events: {e}")

    def save_stream_debug(chunk_data, filename="stream_debug.json"):
        """Salva todos os chunks recebidos da API para debug"""
        try:
            dados = []
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    dados = json.load(f)

            dados.append({"timestamp": datetime.now().isoformat(), "chunk": chunk_data})

            # Mantém últimos 500
            if len(dados) > 500:
                dados = dados[-500:]

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(dados, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erro ao salvar stream debug: {e}")

    def _save_api_response(
        self,
        pergunta_original: str,
        pergunta_enviada: str,
        resposta_completa: str,
        resposta_processada: str = "",  # 🔥 OPCIONAL
        client_ip: str = "unknown",
        filename: str = "responseAPI.json",
    ) -> None:
        """
        Salva a resposta COMPLETA da API em um arquivo JSON
        """
        import os
        import json
        from datetime import datetime

        with self._lock:
            dados = []
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        dados = json.load(f)
                except:
                    dados = []

            dados.append({
                "timestamp": datetime.now().isoformat(),
                "client_ip": client_ip,
                "pergunta_original": pergunta_original or "",
                "pergunta_enviada": pergunta_enviada or "",
                "resposta_completa_api": resposta_completa or "",
                "resposta_processada": resposta_processada or "",
                "tamanho_completa": len(resposta_completa or ""),
                "tamanho_processada": len(resposta_processada or ""),
            })

            if len(dados) > 500:
                dados = dados[-500:]

            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(dados, f, ensure_ascii=False, indent=2)
                print(f"💾 [Zuk] Resposta COMPLETA salva em {filename}")
                print(f"   📝 Tamanho completo: {len(resposta_completa)} caracteres")
                print(f"   📝 Tamanho processado: {len(resposta_processada)} caracteres")
            except Exception as e:
                print(f"⚠️ [Zuk] Erro ao salvar resposta: {e}")

    def _parse_response(self, response, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa a resposta SSE (Server-Sent Events) do ChatGPT.
        
        Args:
            response: Objeto Response da requisição
            request_id: ID da requisição para verificar cancelamento
        
        Returns:
            Dict com:
                - assistant_message: Mensagem completa do assistente
                - original_message: Mensagem original (sem processamento)
                - conversation_id: ID da conversação
                - parent_id: ID do pai
                - ai_message_id: ID da mensagem da IA
                - error: Mensagem de erro (se houver)
                - is_vpn_blocked: Flag para bloqueio de VPN
                - cancelled: Flag de cancelamento
        """
        # 🔥 INICIALIZA O RESULTADO
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

        # 🔥 VERIFICA STATUS HTTP
        if response.status_code != 200:
            response_text = ""
            try:
                response_text = response.text if response.text else ""
            except:
                response_text = "Erro desconhecido"

            if response.status_code == 403:
                if "vpn" in response_text.lower() or "proxy" in response_text.lower() or "cloudflare" in response_text.lower():
                    error_msg = "🌐 VPN/Proxy detectado! Desative a VPN e tente novamente."
                    print(f"🔐 [Zuk] {error_msg}")
                    result["error"] = error_msg
                    result["is_vpn_blocked"] = True
                else:
                    error_msg = "❌ Credenciais expiradas! Status 403 - Atualize os tokens no hc.json"
                    print(f"🔐 [Zuk] {error_msg}")
                    result["error"] = error_msg
            else:
                result["error"] = f"Erro {response.status_code}: {response_text[:200]}"

            return result

        # 🔥 VARIÁVEIS DE CONTROLE
        is_delta_event = False
        text_chunks: List[str] = []
        full_text = ""
        ai_message_id = None
        conversation_id = None
        parent_id = None
        event_count = 0
        all_events = []
# 🔥 ADICIONA O EVENTO À LISTA
        try:
            # 🔥 ITERA SOBRE AS LINHAS DO STREAM
            for line in response.iter_lines(decode_unicode=True):
                # 🔥 VERIFICA CANCELAMENTO
                if request_id and is_request_cancelled(request_id):
                    print(f"⛔ [Zuk] Streaming cancelado pelo usuário")
                    result["cancelled"] = True
                    return result

                if not line or line is None:
                    continue

                # 🔥 DETECTA EVENTO DELTA
                if line.startswith("event: delta"):
                    is_delta_event = True
                    continue

                # 🔥 PROCESSA LINHAS COM DADOS
                if not line.startswith("data: "):
                    continue

                data = line[6:].strip()

                # 🔥 VERIFICA FIM DO STREAM
                if data == "[DONE]":
                    break

                # 🔥 TENTA PARSEAR O JSON
                try:
                    event_data = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if not isinstance(event_data, dict):
                    continue

                event_count += 1
                all_events.append({
                                "event_count": event_count,
                                "line": line,
                                "type": "data",
                                "parsed": event_data
                            })
                # ============================================================
                # 🔥 EXTRAÇÃO DE METADADOS (ID, CONVERSATION, PARENT)
                # ============================================================

                # 🔥 CONVERSATION_ID
                if "conversation_id" in event_data:
                    conversation_id = event_data["conversation_id"]
                    result["conversation_id"] = conversation_id

                # 🔥 PARENT_ID (mensagem do usuário)
                if event_data.get("type") == "input_message":
                    input_msg = event_data.get("input_message")
                    if input_msg and isinstance(input_msg, dict):
                        metadata = input_msg.get("metadata")
                        if metadata and isinstance(metadata, dict):
                            parent_id = metadata.get("parent_id")
                            if parent_id:
                                result["parent_id"] = parent_id

                # 🔥 AI MESSAGE ID
                if "o" in event_data and event_data["o"] == "add":
                    if "v" in event_data and isinstance(event_data["v"], dict):
                        msg_data = event_data["v"].get("message")
                        if msg_data and isinstance(msg_data, dict):
                            msg_id = msg_data.get("id")
                            author = msg_data.get("author", {})
                            role = author.get("role") if isinstance(author, dict) else None
                            if msg_id and role == "assistant":
                                ai_message_id = msg_id
                                result["ai_message_id"] = ai_message_id

                # ============================================================
                # 🔥 EXTRAÇÃO DE TEXTO (TODOS OS CHUNKS)
                # ============================================================

                text_found = False

                # 🔥 CASO 1: Evento delta com texto
                if is_delta_event:
                    if "v" in event_data and isinstance(event_data["v"], str):
                        chunk = event_data["v"]
                        if not (chunk.startswith("{") and chunk.endswith("}")):
                            text_chunks.append(chunk)
                            full_text += chunk
                            text_found = True
                            print(f"📝 [Delta] +{len(chunk)} chars (total: {len(full_text)})")

                # 🔥 CASO 2: Evento "append" direto
                if not text_found and "o" in event_data and event_data["o"] == "append":
                    if "v" in event_data and isinstance(event_data["v"], str):
                        chunk = event_data["v"]
                        text_chunks.append(chunk)
                        full_text += chunk
                        text_found = True
                        print(f"📝 [Append] +{len(chunk)} chars (total: {len(full_text)})")

                # 🔥 🔥 🔥 CASO 3: Evento "patch" com lista de alterações (MAIS COMUM!)
                if not text_found and "o" in event_data and event_data["o"] == "patch":
                    if "v" in event_data and isinstance(event_data["v"], list):
                        for patch_item in event_data["v"]:
                            if not isinstance(patch_item, dict):
                                continue

                            # 🔥 VERIFICA SE É APPEND NO CONTEÚDO DA MENSAGEM
                            if patch_item.get("p") == "/message/content/parts/0" and patch_item.get("o") == "append":
                                if "v" in patch_item and isinstance(patch_item["v"], str):
                                    chunk = patch_item["v"]
                                    text_chunks.append(chunk)
                                    full_text += chunk
                                    text_found = True
                                    print(f"📝 [Patch] +{len(chunk)} chars (total: {len(full_text)})")

                # 🔥 CASO 4: Mensagem completa (fallback)
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
                                    print(f"📝 [Message] +{len(chunk)} chars (total: {len(full_text)})")

                # 🔥 RESETA O FLAG DELTA
                is_delta_event = False

        except Exception as e:
            print(f"⚠️ [Zuk] Erro ao ler stream: {e}")
            if not text_chunks:
                raise

        # ============================================================
        # 🔥 FINALIZAÇÃO
        # ============================================================

        # 🔥 USA O TEXTO COMPLETO ACUMULADO
        if full_text:
            result["original_message"] = full_text
            result["assistant_message"] = full_text
            result["full_response"] = full_text
            print(f"✅ [Zuk] Resposta COMPLETA capturada! ({len(full_text)} caracteres)")
            print(f"   📝 Preview: {full_text[:150]}...")
        elif text_chunks:
            # 🔥 FALLBACK: junta os chunks se full_text estiver vazio
            original_message = "".join(text_chunks)
            result["original_message"] = original_message
            result["assistant_message"] = original_message
            result["full_response"] = original_message
            print(f"⚠️ [Zuk] Resposta via fallback: {len(original_message)} caracteres")
        else:
            print(f"⚠️ [Zuk] NENHUM TEXTO CAPTURADO!")
            result["error"] = "Nenhum texto capturado da resposta"

        # 🔥 GARANTE QUE O CONVERSATION_ID FOI CAPTURADO
        if conversation_id and not result.get("conversation_id"):
            result["conversation_id"] = conversation_id

        # 🔥 LOG DO RESULTADO
        print(f"📊 [Zuk] Resumo da resposta:")
        print(f"   ├─ conversation_id: {result.get('conversation_id')}")
        print(f"   ├─ parent_id: {result.get('parent_id')}")
        print(f"   ├─ ai_message_id: {result.get('ai_message_id')}")
        print(f"   └─ texto: {len(result.get('assistant_message', ''))} caracteres")
        # 🔥 SALVA TODOS OS EVENTOS PARA DEBUG
        try:
            self._save_stream_events(all_events)
        except Exception as e:
            print(f"⚠️ [Zuk] Erro ao salvar eventos: {e}")
        return result


# ============================================================
# CARREGAR CONFIGURAÇÃO
# ============================================================


def load_config() -> Optional[Dict[str, Any]]:
    if not os.path.exists("hc.json"):
        print("❌ [Zuk] Arquivo hc.json não encontrado!")
        print("📝 [Zuk] Crie um arquivo hc.json com headers, cookies e authorization")
        return None

    with open("hc.json", "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()
if not config:
    print("❌ [Zuk] Falha ao carregar configuração. Encerrando.")
    exit(1)

client = ChatGPTClient(config)
print("✅ [Zuk] Cliente ChatGPT inicializado com sucesso!")


# ============================================================
# ROTAS DA API
# ============================================================

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
                    "screen_info": data.get("screen_info", ""),
                    "conversation_id": data.get("conversation_id"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                }
            )
        return {"total": len(devices), "devices": devices}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("""
        <h1>Zuk Intelligence</h1>
        <p>index.html não encontrado. Verifique a pasta static/</p>
    """)


@app.post("/api/chat")
async def chat(request: ChatRequest, req: Request):
    # 🔥 CAPTURA INFORMAÇÕES DO DISPOSITIVO
    client_ip = req.client.host if req.client else "unknown"
    forwarded = req.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    user_agent = req.headers.get("User-Agent", "unknown")
    screen_info = request.screen_info or "unknown"

    # 🔥 GERA O ID DO DISPOSITIVO
    device_id = generate_device_id(client_ip, user_agent, screen_info)

    print(f"\n{'='*70}")
    print(f"🌐 [API] Requisição de dispositivo:")
    print(f"   📱 Device ID: {device_id[:16]}...")
    print(f"   🌍 IP: {client_ip}")
    print(f"   📱 User-Agent: {user_agent[:50]}...")
    print(f"   🖥️  Screen Info: {screen_info}")
    print(f"{'='*70}")

    # 🔥 VERIFICA SE É UM DISPOSITIVO NOVO
    existing_ids = get_conversation_ids(client_ip, user_agent, screen_info)

    # 🔥 SE FOR DISPOSITIVO NOVO, USA NULL PARA CRIAR NOVA CONVERSA
    if existing_ids["is_new"]:
        print(f"🆕 [API] NOVO DISPOSITIVO DETECTADO! ({device_id[:16]}...)")
        print(f"   💡 Será criada uma NOVA conversa para este dispositivo.")
        conv_id_to_use = None
        parent_id_to_use = None
    else:
        print(f"🔄 [API] DISPOSITIVO JÁ EXISTE!")
        print(f"   📌 conversation_id: {existing_ids['conversation_id']}")
        print(f"   📌 parent_message_id: {existing_ids['parent_message_id']}")
        conv_id_to_use = existing_ids["conversation_id"]
        parent_id_to_use = existing_ids["parent_message_id"]

    request_id = request.request_id or str(uuid.uuid4())
    register_request(request_id)

    try:
        print(f"\n📤 [API] Enviando para ChatGPT com:")
        print(f"   ├─ conversation_id: {conv_id_to_use}")
        print(f"   └─ parent_message_id: {parent_id_to_use}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            client.send_message,
            request.message,
            client_ip,
            conv_id_to_use,  # ← USA NULL SE FOR NOVO
            parent_id_to_use,  # ← USA NULL SE FOR NOVO
            request_id,
        )

        if result.get("cancelled"):
            print(f"⛔ [API] Requisição cancelada: {request_id}")
            return {
                "response": None,
                "conversation_id": client.conversation_id,
                "parent_message_id": client.parent_message_id,
                "is_custom": False,
                "cancelled": True,
            }

        if result.get("error"):
            # 🔥 VERIFICA SE É BLOQUEIO DE VPN
            if result.get("is_vpn_blocked"):
                print(f"🌐 [API] VPN detectada para IP {client_ip}")
                # 🔥 RETORNA UMA MENSAGEM AMIGÁVEL PARA O USUÁRIO
                return {
                            "response": "Parece que você está usando uma VPN ou proxy. O ChatGPT bloqueia esse tipo de conexão para evitar abusos. Por favor, desative a VPN e tente novamente.",
                            "conversation_id": None,
                            "parent_message_id": None,
                            "is_custom": True,
                            "is_vpn_blocked": True
                        }
            else:
                print(f"❌ [API] Erro para IP {client_ip}: {result['error']}")
                raise HTTPException(status_code=500, detail=result["error"])

        # 🔥 ATUALIZA OS IDs DA CONVERSA PARA ESTE DISPOSITIVO
        if client.conversation_id and client.parent_message_id:
            update_conversation_ids(
                client_ip,
                user_agent,
                screen_info,
                client.conversation_id,
                client.parent_message_id
            )
            save_conversations_to_file()
            print(f"💾 [API] IDs salvos para dispositivo {device_id[:8]}...")

        response_data = {
            "response": result.get("assistant_message", "Sem resposta"),
            "conversation_id": client.conversation_id,
            "parent_message_id": client.parent_message_id,
            "is_custom": False,
            "cancelled": False,
            "device_id": device_id,
            "is_new_device": existing_ids["is_new"]  # 🔥 INFORMA SE É NOVO
        }

        print(f"\n📤 [API] Retornando resposta:")
        print(f"   ├─ response: {response_data['response'][:50]}...")
        print(f"   ├─ conversation_id: {response_data['conversation_id']}")
        print(f"   └─ parent_message_id: {response_data['parent_message_id']}")

        return response_data

    except Exception as e:
        print(f"❌ [API] Erro interno para IP {client_ip}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        unregister_request(request_id)


@app.post("/api/cancel/{request_id}")
async def cancel_request_endpoint(request_id: str, req: Request):
    """Cancela uma requisição em andamento"""
    client_ip = req.client.host if req.client else "unknown"
    forwarded = req.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    user_agent = req.headers.get("User-Agent", "unknown")

    # 🔥 LOG DE CANCELAMENTO
    print(f"\n{'='*70}")
    print(f"🛑 [CANCEL] Requisição cancelada!")
    print(f"   📱 Request ID: {request_id}")
    print(f"   🌍 IP: {client_ip}")
    print(f"   📱 User-Agent: {user_agent[:50]}...")
    print(f"   🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    success = cancel_request(request_id)

    # 🔥 SALVA O LOG DE CANCELAMENTO
    try:
        log_file = "cancel_logs.json"
        logs = []
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)

        logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "ip": client_ip,
                "user_agent": user_agent,
                "success": success,
            }
        )

        # Mantém apenas os últimos 1000 logs
        if len(logs) > 1000:
            logs = logs[-1000:]

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Erro ao salvar log de cancelamento: {e}")

    return {"success": success, "request_id": request_id}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "Zuk Intelligence",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_requests": len(active_requests),
    }


@app.get("/api/active-requests")
async def get_active_requests():
    with active_requests_lock:
        return {"count": len(active_requests), "requests": list(active_requests.keys())}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 ZUK INTELLIGENCE - API SERVER")
    print("=" * 60)
    print(f"📂 Configuração: hc.json")
    print(f"🌐 Servidor: http://localhost:8000")
    print(f"📚 Documentação: http://localhost:8000/docs")
    print("=" * 60)
    print("\n🚀 Servidor iniciado. Pressione Ctrl+C para parar.\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
