# 🚀 Agentes de Conversação API v2.0

Fast, modern web framework for building APIs with Python 3.7+ based on standard Python type hints.

## ✨ Features

- **High Performance**: On par with NodeJS and Go
- **Fast Development**: Intuitive and easy to use
- **Fewer Bugs**: Reduce about 40% of human (developer) induced errors
- **Automatic Documentation**: Interactive API docs with OpenAPI and JSON Schema
- **Streaming Chat**: Real-time chat with Server-Sent Events
- **Agent Tools**: HTTP, search, calculator, lead capture, webhooks
- **Real-time Analytics**: Dashboard with live metrics
- **Production Ready**: CORS, security middleware, health checks

## 📚 API Documentation

Once deployed, visit:
- **Interactive Docs**: `https://your-railway-app.up.railway.app/docs`
- **ReDoc**: `https://your-railway-app.up.railway.app/redoc`
- **Health Check**: `https://your-railway-app.up.railway.app/health`

## 🔧 Environment Variables

Railway will automatically provide:
- `PORT` - Server port (default: 8000)
- `REDIS_URL` - Redis connection string (if Redis addon is added)
- `DATABASE_URL` - Database connection string (if database addon is added)

Optional:
- `CORS_ORIGINS` - Comma-separated list of allowed origins (default: "*")

## 🚀 Deploy to Railway

1. **One-Click Deploy**: Use Railway template
2. **From GitHub**: Connect your repository
3. **CLI**: Use Railway CLI for custom deployment

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000

# Visit http://localhost:8000/docs
```

## 📊 API Endpoints

### System
- `GET /` - API information
- `GET /health` - Health check

### Agents v2.0
- `POST /api/v2/agents` - Create agent
- `GET /api/v2/agents` - List agents
- `GET /api/v2/agents/{id}` - Get agent

### Chat
- `POST /api/v2/agents/{id}/chat/stream` - Streaming chat

### Analytics
- `GET /api/v2/analytics` - Get analytics data

## 🔐 Authentication

All endpoints (except health and root) require Bearer token authentication:

```bash
curl -H "Authorization: Bearer your-token-here" \
     https://your-api.railway.app/api/v2/agents
```

## 📈 Monitoring

Railway provides built-in monitoring:
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, network usage
- **Health Checks**: Automatic `/health` endpoint monitoring

## 🌐 Custom Domain

To use custom domain (api.agentesdeconversao.com.br):

1. Get Railway app URL after deployment
2. Update DNS CNAME record:
   ```
   api.agentesdeconversao.com.br → your-app.up.railway.app
   ```

## 🔄 Updates

Railway automatically redeploys when you push to connected GitHub repository.

## 📞 Support

- Railway Docs: https://docs.railway.com/
- FastAPI Docs: https://fastapi.tiangolo.com/
