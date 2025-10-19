#!/bin/bash
# 🚀 POISE TRADER DEPLOYMENT SCRIPT
# Automated deployment with health checks and rollback capabilities

set -e

# Configuration
APP_NAME="poise-trader"
DOCKER_COMPOSE_FILE="docker-compose.yml"
BACKUP_DIR="./backups/deployments"
LOG_FILE="./logs/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Create necessary directories
mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

# Pre-deployment checks
pre_deployment_checks() {
    log "🔍 Running pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if docker-compose file exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error "Docker compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        warning ".env file not found. Using default environment variables."
    fi
    
    # Validate configuration
    if ! python -c "from core.config_manager import config_manager; config_manager.load_config()" 2>/dev/null; then
        warning "Configuration validation failed. Proceeding with caution."
    fi
    
    success "Pre-deployment checks completed"
}

# Backup current deployment
backup_deployment() {
    log "💾 Creating deployment backup..."
    
    local backup_timestamp=$(date +'%Y%m%d_%H%M%S')
    local backup_path="$BACKUP_DIR/backup_$backup_timestamp"
    
    mkdir -p "$backup_path"
    
    # Backup configuration and data
    if [ -d "./config" ]; then
        cp -r ./config "$backup_path/"
    fi
    
    if [ -d "./data" ]; then
        cp -r ./data "$backup_path/"
    fi
    
    if [ -f ".env" ]; then
        cp .env "$backup_path/"
    fi
    
    # Backup Docker images
    docker save $(docker images --format "table {{.Repository}}:{{.Tag}}" | grep poise-trader | head -1) > "$backup_path/poise-trader-image.tar" 2>/dev/null || true
    
    echo "$backup_timestamp" > "$backup_path/backup_info.txt"
    
    success "Backup created: $backup_path"
}

# Build and deploy
deploy() {
    log "🚀 Starting deployment..."
    
    # Pull latest images
    log "📥 Pulling base images..."
    docker-compose pull
    
    # Build application
    log "🔨 Building application..."
    docker-compose build --no-cache
    
    # Stop existing containers gracefully
    log "⏹️ Stopping existing containers..."
    docker-compose down --timeout 30
    
    # Start new deployment
    log "▶️ Starting new deployment..."
    docker-compose up -d
    
    success "Deployment started"
}

# Health checks
health_checks() {
    log "🏥 Running health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts"
        
        # Check if containers are running
        if docker-compose ps | grep -q "Up"; then
            log "✅ Containers are running"
            
            # Check application health endpoint
            if curl -f http://localhost:8080/health > /dev/null 2>&1; then
                success "🎉 Application is healthy!"
                return 0
            fi
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "❌ Health checks failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Rollback function
rollback() {
    error "🔄 Rolling back deployment..."
    
    # Stop current deployment
    docker-compose down --timeout 30
    
    # Find latest backup
    local latest_backup=$(ls -t "$BACKUP_DIR" | head -1)
    
    if [ -n "$latest_backup" ]; then
        log "📁 Restoring from backup: $latest_backup"
        
        # Restore configuration
        if [ -d "$BACKUP_DIR/$latest_backup/config" ]; then
            rm -rf ./config
            cp -r "$BACKUP_DIR/$latest_backup/config" ./
        fi
        
        # Restore data
        if [ -d "$BACKUP_DIR/$latest_backup/data" ]; then
            rm -rf ./data
            cp -r "$BACKUP_DIR/$latest_backup/data" ./
        fi
        
        # Restore environment
        if [ -f "$BACKUP_DIR/$latest_backup/.env" ]; then
            cp "$BACKUP_DIR/$latest_backup/.env" ./
        fi
        
        # Load backup image if available
        if [ -f "$BACKUP_DIR/$latest_backup/poise-trader-image.tar" ]; then
            docker load < "$BACKUP_DIR/$latest_backup/poise-trader-image.tar"
        fi
        
        # Start backup deployment
        docker-compose up -d
        
        success "🔄 Rollback completed"
    else
        error "No backup found for rollback"
        exit 1
    fi
}

# Main deployment flow
main() {
    log "🏆 POISE TRADER DEPLOYMENT STARTED"
    
    # Run pre-deployment checks
    pre_deployment_checks
    
    # Create backup
    backup_deployment
    
    # Deploy
    deploy
    
    # Run health checks
    if health_checks; then
        success "🎉 DEPLOYMENT SUCCESSFUL!"
        
        # Clean up old backups (keep last 5)
        log "🧹 Cleaning up old backups..."
        ls -t "$BACKUP_DIR" | tail -n +6 | xargs -I {} rm -rf "$BACKUP_DIR/{}"
        
        # Display status
        echo ""
        log "📊 Deployment Status:"
        docker-compose ps
        
        echo ""
        log "🌐 Access URLs:"
        log "  • Trading Bot Dashboard: http://localhost:8080"
        log "  • Monitoring (Grafana): http://localhost:3000 (admin/admin123)"
        log "  • Metrics (Prometheus): http://localhost:9090"
        
    else
        error "🚨 DEPLOYMENT FAILED - INITIATING ROLLBACK"
        rollback
        exit 1
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "health")
        health_checks
        ;;
    "backup")
        backup_deployment
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "stop")
        log "⏹️ Stopping Poise Trader..."
        docker-compose down
        success "Stopped"
        ;;
    "restart")
        log "🔄 Restarting Poise Trader..."
        docker-compose restart
        success "Restarted"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health|backup|logs|stop|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment with health checks"
        echo "  rollback - Rollback to previous version"
        echo "  health   - Run health checks only"
        echo "  backup   - Create backup only"
        echo "  logs     - Show container logs"
        echo "  stop     - Stop all containers"
        echo "  restart  - Restart all containers"
        exit 1
        ;;
esac
