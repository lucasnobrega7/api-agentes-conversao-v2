"""
Agentes de Conversão API v2.0 - Railway Template
Fast, modern web framework for building APIs with Python 3.7+
"""

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager
import httpx
import asyncio
import json
import os
import uuid
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, AsyncGenerator
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with Railway defaults
PORT = int(os.environ.get("PORT", 8000))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Agentes de Conversão API v2.0 on Railway")
    logger.info(f"📡 Port: {PORT}")
    logger.info(f"🔗 Redis: {REDIS_URL}")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down API v2.0")
    logger.info("✅ Shutdown complete")

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title="Agentes de Conversação API v2.0",
    description="API avançada para criação e gerenciamento de agentes conversacionais inteligentes - Railway Deploy",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "system", "description": "Sistema e health check"},
        {"name": "agents", "description": "Operações com agentes"},
        {"name": "chat", "description": "Chat e streaming"},
        {"name": "tools", "description": "Ferramentas de agentes"},
        {"name": "analytics", "description": "Analytics e métricas"},
        {"name": "knowledge", "description": "Base de conhecimento"},
        {"name": "webhooks", "description": "Webhooks e notificações"},
        {"name": "events", "description": "Eventos em tempo real"},
    ]
)

# Production security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*.agentesdeconversao.com.br", "*.railway.app", "*.up.railway.app", "localhost"]
)

# Enhanced CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Initialize Redis client (Railway Redis addon)
try:
    redis_client = redis.from_url(
        REDIS_URL,
        max_connections=20,
        retry_on_timeout=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        health_check_interval=30
    )
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

security = HTTPBearer()

# Store for SSE connections
sse_connections: Dict[str, List[Any]] = {}

# ============================================
# PYDANTIC MODELS v2.0
# ============================================

class AgentTool(BaseModel):
    id: Optional[str] = None
    type: str = Field(..., description="Tipo da ferramenta: http, search, calculator, lead_capture, form, webhook")
    name: str = Field(..., description="Nome da ferramenta")
    description: Optional[str] = None
    config: Dict[str, Any] = Field(..., description="Configuração específica da ferramenta")
    is_active: bool = True

class CreateAgentRequest(BaseModel):
    name: str = Field(..., description="Nome do agente")
    description: str = Field(..., description="Descrição do agente")
    system_prompt: str = Field(..., description="Prompt do sistema")
    model_id: str = Field(default="gpt-4", description="ID do modelo")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Temperatura do modelo")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Máximo de tokens")
    tools: List[AgentTool] = Field(default=[], description="Ferramentas do agente")
    visibility: str = Field(default="private", description="Visibilidade: public, private")
    knowledge_base_id: Optional[str] = None

class StreamChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="Histórico de mensagens")
    conversation_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_data: Optional[Dict[str, Any]] = None
    system_prompt_override: Optional[str] = None
    streaming: bool = True

class StreamEvent(BaseModel):
    event: str = Field(..., description="Tipo do evento: progress, answer, source, metadata, done, error")
    data: Any = Field(..., description="Dados do evento")
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class WebhookConfig(BaseModel):
    name: str = Field(..., description="Nome do webhook")
    url: str = Field(..., description="URL do webhook")
    events: List[str] = Field(..., description="Eventos para escutar")
    secret: Optional[str] = None
    is_active: bool = True

class AnalyticsRequest(BaseModel):
    period: str = Field(default="week", description="Período: day, week, month")
    agent_id: Optional[str] = None
    event_type: Optional[str] = None

# ============================================
# AUTHENTICATION & MIDDLEWARE
# ============================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificação de autenticação simplificada para demo"""
    token = credentials.credentials
    
    try:
        # For demo purposes, accept any token
        if not token or len(token) < 10:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )
        
        # Mock user data
        return {
            "id": "user_123",
            "email": "user@example.com",
            "organization_id": "org_456"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Erro de autenticação: {str(e)}"
        )

async def get_organization_id(user_data: dict) -> str:
    """Extrair organization_id do usuário"""
    return user_data.get("organization_id", "default_org")

# ============================================
# CORE ENDPOINTS - SYSTEM
# ============================================

@app.get("/", tags=["system"])
async def root():
    """Endpoint raiz da API - Railway Deploy"""
    return {
        "message": "Agentes de Conversação API v2.0 - Railway Deploy",
        "version": "2.0.0",
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "deployment": "railway",
        "features": [
            "streaming_chat",
            "agent_tools",
            "real_time_analytics",
            "sse_events",
            "advanced_webhooks",
            "knowledge_search"
        ],
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", tags=["system"])
async def health_check():
    """Health check completo para Railway"""
    try:
        # Test Redis connection
        redis_status = "ok"
        try:
            if redis_client:
                await redis_client.ping()
            else:
                redis_status = "not_configured"
        except:
            redis_status = "error"
        
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "deployment": "railway",
            "services": {
                "redis": redis_status,
                "database": "not_configured" if not DATABASE_URL else "ok"
            },
            "features": {
                "streaming": True,
                "sse": True,
                "tools": True,
                "analytics": True
            },
            "environment": {
                "port": PORT,
                "cors_origins": len(CORS_ORIGINS)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================
# CORE ENDPOINTS - AGENTS v2.0
# ============================================

@app.post("/api/v2/agents", tags=["agents"])
async def create_agent_v2(
    agent_data: CreateAgentRequest,
    user_data: dict = Depends(get_current_user)
):
    """Criar agente com ferramentas avançadas"""
    organization_id = await get_organization_id(user_data)
    agent_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Criar agente principal
    agent = {
        "id": agent_id,
        "organization_id": organization_id,
        "user_id": user_data["id"],
        "name": agent_data.name,
        "description": agent_data.description,
        "system_prompt": agent_data.system_prompt,
        "model_id": agent_data.model_id,
        "temperature": agent_data.temperature,
        "max_tokens": agent_data.max_tokens,
        "visibility": agent_data.visibility,
        "knowledge_base_id": agent_data.knowledge_base_id,
        "is_active": True,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "tools": [tool.dict() for tool in agent_data.tools]
    }
    
    try:
        # Cache no Redis se disponível
        if redis_client:
            await redis_client.setex(f"agent:{agent_id}", 3600, json.dumps(agent))
        
        return {
            "success": True,
            "data": agent,
            "message": "Agente criado com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar agente: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/api/v2/agents", tags=["agents"])
async def list_agents_v2(
    user_data: dict = Depends(get_current_user)
):
    """Listar agentes com ferramentas - Demo data"""
    try:
        # Demo agents data
        demo_agents = [
            {
                "id": "agent_demo_1",
                "name": "Agente de Vendas",
                "description": "Especializado em converter leads em clientes",
                "system_prompt": "Você é um especialista em vendas...",
                "model_id": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2048,
                "visibility": "private",
                "tools": [
                    {
                        "id": "tool_1",
                        "type": "lead_capture",
                        "name": "Captura de Leads",
                        "description": "Coleta informações de contato",
                        "config": {"required_fields": ["name", "email"]},
                        "is_active": True
                    }
                ],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            },
            {
                "id": "agent_demo_2", 
                "name": "Agente de Suporte",
                "description": "Fornece suporte técnico e resolve problemas",
                "system_prompt": "Você é um especialista em suporte...",
                "model_id": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 1024,
                "visibility": "private",
                "tools": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        ]
        
        return {
            "success": True,
            "data": demo_agents,
            "count": len(demo_agents)
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar agentes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/api/v2/agents/{agent_id}", tags=["agents"])
async def get_agent_v2(
    agent_id: str,
    user_data: dict = Depends(get_current_user)
):
    """Buscar agente específico com ferramentas"""
    try:
        # Try cache first if Redis available
        if redis_client:
            cached = await redis_client.get(f"agent:{agent_id}")
            if cached:
                agent = json.loads(cached)
                return {"success": True, "data": agent}
        
        # Demo agent data
        demo_agent = {
            "id": agent_id,
            "name": "Agente Demo",
            "description": "Agente de demonstração",
            "system_prompt": "Você é um assistente útil...",
            "model_id": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2048,
            "visibility": "private",
            "tools": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return {"success": True, "data": demo_agent}
        
    except Exception as e:
        logger.error(f"Erro ao buscar agente: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# ============================================
# STREAMING CHAT v2.0
# ============================================

@app.post("/api/v2/agents/{agent_id}/chat/stream", tags=["chat"])
async def stream_chat_v2(
    agent_id: str,
    chat_request: StreamChatRequest,
    user_data: dict = Depends(get_current_user)
):
    """Chat com streaming avançado e eventos estruturados"""
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            message_id = str(uuid.uuid4())
            conversation_id = chat_request.conversation_id or str(uuid.uuid4())
            user_message = chat_request.messages[-1]["content"] if chat_request.messages else ""
            
            # Evento de início
            yield f"data: {json.dumps({'event': 'progress', 'data': {'status': 'started', 'progress': 0}, 'conversation_id': conversation_id, 'message_id': message_id})}\\n\\n"
            
            # Simular processamento
            yield f"data: {json.dumps({'event': 'progress', 'data': {'status': 'processing', 'progress': 50}})}\\n\\n"
            
            # Gerar resposta demo
            full_response = f"Esta é uma resposta de demonstração para: '{user_message}'. A API v2.0 está funcionando corretamente no Railway com streaming em tempo real e eventos estruturados."
            
            # Streaming de tokens
            words = full_response.split(' ')
            current_response = ""
            
            for i, word in enumerate(words):
                current_response += (word + " " if i < len(words) - 1 else word)
                
                yield f"data: {json.dumps({'event': 'answer', 'data': word + (' ' if i < len(words) - 1 else ''), 'conversation_id': conversation_id, 'message_id': message_id})}\\n\\n"
                
                # Progress update
                progress = 50 + int((i + 1) / len(words) * 40)
                if progress % 10 == 0:
                    yield f"data: {json.dumps({'event': 'progress', 'data': {'status': 'generating', 'progress': progress}})}\\n\\n"
                
                await asyncio.sleep(0.1)  # Simular delay
            
            # Metadata final
            metadata = {
                "model": "gpt-4",
                "temperature": 0.7,
                "tokens_used": len(words),
                "sources_count": 0,
                "processing_time": 2000,
                "deployment": "railway"
            }
            yield f"data: {json.dumps({'event': 'metadata', 'data': metadata})}\\n\\n"
            
            # Evento final
            yield f"data: {json.dumps({'event': 'done', 'data': {'conversation_id': conversation_id, 'message_id': message_id, 'complete': True}})}\\n\\n"
            yield "data: [DONE]\\n\\n"
            
        except Exception as e:
            logger.error(f"Erro no streaming: {str(e)}")
            yield f"data: {json.dumps({'event': 'error', 'data': f'Erro: {str(e)}'})}\\n\\n"
    
    return EventSourceResponse(generate_stream())

# ============================================
# ANALYTICS v2.0
# ============================================

@app.get("/api/v2/analytics", tags=["analytics"])
async def get_analytics_v2(
    period: str = "week",
    agent_id: Optional[str] = None,
    user_data: dict = Depends(get_current_user)
):
    """Analytics com dados demo"""
    try:
        # Generate demo time series data
        days = {"day": 1, "week": 7, "month": 30}[period]
        time_series = []
        
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            time_series.append({
                "date": date,
                "conversations": max(0, 150 - i * 5),
                "messages": max(0, 1200 - i * 40)
            })
        
        time_series.reverse()
        
        return {
            "success": True,
            "data": {
                "overview": {
                    "total_agents": 3,
                    "total_conversations": 150,
                    "total_messages": 1200,
                    "avg_response_time": 2.5,
                    "active_conversations": 23,
                    "user_satisfaction": 87.5
                },
                "time_series": time_series,
                "top_agents": [
                    {
                        "id": "agent_demo_1",
                        "name": "Agente de Vendas",
                        "conversations": 45,
                        "messages": 380,
                        "avg_response_time": 2.1,
                        "satisfaction": 89.2
                    }
                ],
                "period": period,
                "deployment": "railway"
            }
        }
        
    except Exception as e:
        logger.error(f"Erro ao gerar analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# ============================================
# STARTUP & SHUTDOWN
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )