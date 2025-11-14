# RingRift Quick Start Guide

Get the AI service running in under 5 minutes!

## Choose Your Setup Method

### 🐳 Option 1: Docker (Recommended - Once Docker is Installed)

**Prerequisites:** Docker Desktop, OrbStack, or Colima installed
- See `DOCKER_SETUP.md` for installation instructions

**Quick Start:**
```bash
# Start AI service only
docker compose up ai-service

# Or start everything
docker compose up
```

**That's it!** The service will be running at http://localhost:8001

---

### 🐍 Option 2: Python Virtual Environment (No Docker Required)

**Prerequisites:** Python 3.11+ installed

**Quick Start:**
```bash
cd ai-service
./setup.sh
./run.sh
```

**Done!** The service will be running at http://localhost:8001

---

## Testing the AI Service

Once running, visit:
- **API Documentation:** http://localhost:8001/docs
- **Health Check:** http://localhost:8001/health
- **Service Info:** http://localhost:8001/

### Quick API Test

```bash
# Health check
curl http://localhost:8001/health

# Should return: {"status":"healthy"}
```

### Test AI Move Generation

Visit http://localhost:8001/docs and try the interactive API documentation.

---

## Next Steps

### 1. Install Docker (If Not Installed)

**Quick Install via Homebrew:**
```bash
# OrbStack (recommended - lightweight and fast)
brew install orbstack
open -a OrbStack

# OR Colima (command-line only)
brew install colima docker docker-compose
colima start

# OR download Docker Desktop from docker.com
```

See `DOCKER_SETUP.md` for detailed instructions.

### 2. Run the Full Stack

```bash
# Start all services
docker compose up

# Services will be available at:
# - Main app: http://localhost:3000
# - AI service: http://localhost:8001
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001
```

### 3. Development Workflow

**With Docker:**
```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f ai-service

# Stop
docker compose down
```

**With Python venv:**
```bash
cd ai-service
./run.sh
# Make changes - hot reload enabled!
```

---

## Troubleshooting

### Python Setup

**"command not found: python3"**
```bash
brew install python@3.11
```

**"zsh: permission denied: ./setup.sh"**
```bash
chmod +x ai-service/setup.sh ai-service/run.sh
```

**"pip install fails"**
```bash
cd ai-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Docker Setup

**"command not found: docker"**
- Install Docker Desktop, OrbStack, or Colima (see DOCKER_SETUP.md)

**"Cannot connect to Docker daemon"**
```bash
# Check if Docker is running
docker info

# Start Docker Desktop or Colima
colima start
```

**"Port 8001 already in use"**
```bash
# Find what's using the port
lsof -i :8001

# Kill it
kill -9 <PID>
```

---

## File Structure

```
RingRift/
├── ai-service/              # Python AI microservice
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── models.py       # Data models
│   │   └── ai/            # AI implementations
│   ├── setup.sh           # Setup virtual environment
│   ├── run.sh             # Start service
│   ├── Dockerfile         # Docker image
│   └── requirements.txt   # Python dependencies
├── docker-compose.yml      # All services configuration
├── DOCKER_SETUP.md        # Docker installation guide
└── QUICKSTART.md          # This file
```

---

## What's Next?

1. ✅ Get the AI service running (you're here!)
2. 📝 Explore the API at http://localhost:8001/docs
3. 🧪 Run tests: `npm test`
4. 🎮 Start the main app: `npm run dev`
5. 🚀 Deploy with Docker: `docker compose up`

---

## Links

- **AI Service README:** `ai-service/README.md`
- **Docker Setup:** `DOCKER_SETUP.md`
- **Main README:** `README.md`
- **Project TODO:** `TODO.md`

---

## Need Help?

- Check logs: `docker compose logs ai-service` or check terminal output
- View API docs: http://localhost:8001/docs
- Read `DOCKER_SETUP.md` for detailed Docker troubleshooting
- Read `ai-service/README.md` for AI service details
