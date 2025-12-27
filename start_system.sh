#!/bin/bash
# =============================================================
# 🚀 FACT-CHECK AI SYSTEM - FULL AUTOMATION STARTUP
# =============================================================
# 
# Script này khởi động toàn bộ hệ thống:
# 1. Infrastructure (Postgres, Kafka, Zookeeper)
# 2. Application Services (Backend, Dashboard, Consumer)
# 3. Workflow Orchestration (Airflow)
#
# Usage:
#   ./start_system.sh          # Start tất cả
#   ./start_system.sh --build  # Build lại images trước khi start
#   ./start_system.sh --stop   # Dừng tất cả
#   ./start_system.sh --logs   # Xem logs
# =============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "============================================================="
    echo "🛡️  FACT-CHECK AI AUTOMATION SYSTEM"
    echo "============================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

# Check requirements
check_requirements() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker chưa được cài đặt!"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose chưa được cài đặt!"
        exit 1
    fi
    
    print_step "Docker và Docker Compose đã sẵn sàng"
}

# Stop all services
stop_services() {
    print_warn "Đang dừng tất cả services..."
    docker compose down --remove-orphans || true
    print_step "Đã dừng tất cả services"
}

# Start infrastructure
start_infrastructure() {
    echo ""
    echo -e "${BLUE}[1/4] Khởi động Infrastructure (DB, Kafka)...${NC}"
    
    # Start database and message queue first
    docker compose up -d db zookeeper kafka
    
    # Wait for DB to be healthy
    echo "   Đang chờ PostgreSQL sẵn sàng..."
    sleep 5
    until docker compose exec -T db pg_isready -U "$POSTGRES_USER" > /dev/null 2>&1; do
        sleep 2
        echo "   ..."
    done
    print_step "PostgreSQL đã sẵn sàng"
    
    # Wait for Kafka
    echo "   Đang chờ Kafka sẵn sàng..."
    sleep 5
    print_step "Kafka đã sẵn sàng"
}

# Initialize database
init_database() {
    echo ""
    echo -e "${BLUE}[2/4] Khởi tạo Database Schema...${NC}"
    
    # Run init script
    docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;" || true
    
    # Run full init if needed
    if docker compose run --rm backend python init_db_full.py 2>/dev/null; then
        print_step "Database schema đã được tạo"
    else
        print_warn "Có thể schema đã tồn tại"
    fi
}

# Start application services
start_applications() {
    echo ""
    echo -e "${BLUE}[3/4] Khởi động Application Services...${NC}"
    
    docker compose up -d backend consumer dashboard pgadmin kafka-ui
    
    sleep 3
    print_step "Backend API đang chạy tại http://localhost:8000"
    print_step "Dashboard đang chạy tại http://localhost:8501"
    print_step "PgAdmin đang chạy tại http://localhost:5050"
    print_step "Kafka UI đang chạy tại http://localhost:8888"
    print_step "Consumer đang chạy (xử lý Kafka messages)"
}

# Start Airflow
start_airflow() {
    echo ""
    echo -e "${BLUE}[4/4] Khởi động Airflow (Workflow Orchestration)...${NC}"
    
    # Create logs directory with proper permissions
    mkdir -p logs/scheduler
    chmod -R 777 logs
    
    # Initialize Airflow DB
    docker compose up airflow-init
    
    # Start webserver and scheduler
    docker compose up -d airflow-webserver airflow-scheduler
    
    sleep 5
    print_step "Airflow đang chạy tại http://localhost:8080"
    print_step "Login: admin / admin"
}

# Show status
show_status() {
    echo ""
    echo -e "${BLUE}=== TRẠNG THÁI HỆ THỐNG ===${NC}"
    docker compose ps
}

# Show logs
show_logs() {
    docker compose logs -f --tail=100
}

# Main
print_header

# Load env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

case "${1:-}" in
    --stop)
        stop_services
        ;;
    --logs)
        show_logs
        ;;
    --build)
        check_requirements
        print_warn "Building images..."
        docker compose build
        start_infrastructure
        init_database
        start_applications
        start_airflow
        show_status
        ;;
    *)
        check_requirements
        start_infrastructure
        init_database
        start_applications
        start_airflow
        show_status
        
        echo ""
        echo -e "${GREEN}============================================================="
        echo "✅ HỆ THỐNG ĐÃ SẴN SÀNG!"
        echo "============================================================="
        echo ""
        echo "📌 SERVICES:"
        echo "   • API:       http://localhost:8000/docs"
        echo "   • Dashboard: http://localhost:8501"
        echo "   • Airflow:   http://localhost:8080  (admin/admin)"
        echo "   • PgAdmin:   http://localhost:5050  ($PGADMIN_EMAIL)"
        echo "   • Kafka UI:  http://localhost:8888"
        echo ""
        echo "📌 COMMANDS:"
        echo "   • Xem logs:  ./start_system.sh --logs"
        echo "   • Dừng:      ./start_system.sh --stop"
        echo "   • Rebuild:   ./start_system.sh --build"
        echo -e "=============================================================${NC}"
        ;;
esac
