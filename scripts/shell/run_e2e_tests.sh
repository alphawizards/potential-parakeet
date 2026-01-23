#!/bin/bash

################################################################################
# E2E Test Execution Script
# ==========================
# Comprehensive end-to-end testing for Potential Parakeet Trading Platform
#
# This script:
# 1. Starts the FastAPI backend server
# 2. Serves the production React frontend
# 3. Runs Playwright E2E tests
# 4. Generates HTML test report
# 5. Cleans up processes on exit
#
# Usage: ./run_e2e_tests.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/ubuntu/potential-parakeet"
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_LOG="$PROJECT_ROOT/logs/backend-e2e.log"
FRONTEND_LOG="$PROJECT_ROOT/logs/frontend-e2e.log"
BACKEND_PID_FILE="$PROJECT_ROOT/logs/backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/logs/frontend.pid"
TEST_REPORT_DIR="$PROJECT_ROOT/playwright-report"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${CYAN}"
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Cleanup function - called on exit
cleanup() {
    print_header "Cleaning Up"
    
    # Kill backend if running
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if ps -p $BACKEND_PID > /dev/null 2>&1; then
            print_info "Stopping backend server (PID: $BACKEND_PID)..."
            kill $BACKEND_PID 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if ps -p $BACKEND_PID > /dev/null 2>&1; then
                kill -9 $BACKEND_PID 2>/dev/null || true
            fi
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Kill frontend if running
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if ps -p $FRONTEND_PID > /dev/null 2>&1; then
            print_info "Stopping frontend server (PID: $FRONTEND_PID)..."
            kill $FRONTEND_PID 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if ps -p $FRONTEND_PID > /dev/null 2>&1; then
                kill -9 $FRONTEND_PID 2>/dev/null || true
            fi
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    # Kill any remaining processes on the ports
    print_info "Cleaning up any remaining processes on ports $BACKEND_PORT and $FRONTEND_PORT..."
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    
    print_success "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    print_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 1
    done
    
    echo ""
    print_error "$service_name failed to start within $max_attempts seconds"
    return 1
}

################################################################################
# Main Execution
################################################################################

print_header "Potential Parakeet E2E Test Suite"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

################################################################################
# Step 1: Pre-flight Checks
################################################################################

print_header "Step 1: Pre-flight Checks"

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Creating..."
    python3.11 -m venv venv
fi

# Check if frontend build exists
if [ ! -d "dashboard/dist" ]; then
    print_error "Frontend build not found. Please run 'npm run build' first."
    exit 1
fi

# Check if database exists
if [ ! -f "data/trades.db" ]; then
    print_warning "Database not found. Running seed script..."
    source venv/bin/activate
    python backend/seed_data.py
    deactivate
fi

# Check if Playwright is installed
if ! command -v npx &> /dev/null; then
    print_error "npx not found. Please install Node.js."
    exit 1
fi

print_success "Pre-flight checks complete"
echo ""

################################################################################
# Step 2: Start Backend Server
################################################################################

print_header "Step 2: Starting Backend Server"

# Kill any existing process on backend port
if check_port $BACKEND_PORT; then
    print_warning "Port $BACKEND_PORT is already in use. Killing existing process..."
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start backend
print_info "Starting FastAPI backend on port $BACKEND_PORT..."
source venv/bin/activate
cd "$PROJECT_ROOT"
nohup python -m backend.main > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$BACKEND_PID_FILE"
deactivate

print_info "Backend PID: $BACKEND_PID"
print_info "Backend logs: $BACKEND_LOG"

# Wait for backend to be ready
if ! wait_for_service "http://localhost:$BACKEND_PORT/health" "Backend API"; then
    print_error "Backend failed to start. Check logs at $BACKEND_LOG"
    tail -20 "$BACKEND_LOG"
    exit 1
fi

# Verify backend health
HEALTH_STATUS=$(curl -s http://localhost:$BACKEND_PORT/health | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
if [ "$HEALTH_STATUS" = "healthy" ]; then
    print_success "Backend health check: $HEALTH_STATUS"
else
    print_error "Backend health check failed: $HEALTH_STATUS"
    exit 1
fi

echo ""

################################################################################
# Step 3: Start Frontend Server
################################################################################

print_header "Step 3: Starting Frontend Server"

# Kill any existing process on frontend port
if check_port $FRONTEND_PORT; then
    print_warning "Port $FRONTEND_PORT is already in use. Killing existing process..."
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Check if serve is installed
if ! command -v npx serve &> /dev/null; then
    print_info "Installing 'serve' package..."
    cd "$PROJECT_ROOT/dashboard"
    npm install -g serve
fi

# Start frontend
print_info "Starting frontend server on port $FRONTEND_PORT..."
cd "$PROJECT_ROOT/dashboard"
nohup npx serve dist -p $FRONTEND_PORT > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$FRONTEND_PID_FILE"

print_info "Frontend PID: $FRONTEND_PID"
print_info "Frontend logs: $FRONTEND_LOG"

# Wait for frontend to be ready
if ! wait_for_service "http://localhost:$FRONTEND_PORT" "Frontend Server"; then
    print_error "Frontend failed to start. Check logs at $FRONTEND_LOG"
    tail -20 "$FRONTEND_LOG"
    exit 1
fi

print_success "Frontend is serving production build"
echo ""

################################################################################
# Step 4: Verify Services
################################################################################

print_header "Step 4: Verifying Services"

# Test backend endpoints
print_info "Testing backend endpoints..."

# Test health endpoint
if curl -s -f http://localhost:$BACKEND_PORT/health > /dev/null; then
    print_success "✓ /health endpoint responding"
else
    print_error "✗ /health endpoint failed"
fi

# Test trades endpoint
if curl -s -f http://localhost:$BACKEND_PORT/api/trades/ > /dev/null; then
    print_success "✓ /api/trades/ endpoint responding"
else
    print_error "✗ /api/trades/ endpoint failed"
fi

# Test frontend
print_info "Testing frontend..."
if curl -s -f http://localhost:$FRONTEND_PORT > /dev/null; then
    print_success "✓ Frontend serving index.html"
else
    print_error "✗ Frontend not responding"
fi

echo ""

################################################################################
# Step 5: Run Playwright E2E Tests
################################################################################

print_header "Step 5: Running Playwright E2E Tests"

cd "$PROJECT_ROOT"

# Check if Playwright browsers are installed
if [ ! -d "$HOME/.cache/ms-playwright" ]; then
    print_info "Installing Playwright browsers (this may take a few minutes)..."
    npx playwright install
fi

# Set environment variables for tests
export BASE_URL="http://localhost:$FRONTEND_PORT"
export API_URL="http://localhost:$BACKEND_PORT"

print_info "Test Configuration:"
print_info "  Frontend URL: $BASE_URL"
print_info "  Backend URL: $API_URL"
print_info "  Report Directory: $TEST_REPORT_DIR"
echo ""

# Run Playwright tests
print_info "Executing Playwright test suite..."
echo ""

if npx playwright test --reporter=html,list; then
    print_success "All E2E tests passed! ✨"
    TEST_STATUS="PASSED"
    EXIT_CODE=0
else
    print_error "Some E2E tests failed. Check the report for details."
    TEST_STATUS="FAILED"
    EXIT_CODE=1
fi

echo ""

################################################################################
# Step 6: Generate Test Report
################################################################################

print_header "Step 6: Generating Test Report"

# Check if report was generated
if [ -d "$TEST_REPORT_DIR" ]; then
    print_success "Test report generated at: $TEST_REPORT_DIR"
    
    # Count test results
    if [ -f "$TEST_REPORT_DIR/index.html" ]; then
        print_info "Opening test report in browser..."
        
        # Create a summary
        echo ""
        print_info "Test Summary:"
        echo "  Report: file://$TEST_REPORT_DIR/index.html"
        echo "  Status: $TEST_STATUS"
        echo ""
        
        # Try to open report in browser (if running with display)
        if command -v xdg-open &> /dev/null; then
            xdg-open "$TEST_REPORT_DIR/index.html" 2>/dev/null || true
        fi
    fi
else
    print_warning "Test report directory not found"
fi

################################################################################
# Step 7: Collect Logs and Artifacts
################################################################################

print_header "Step 7: Collecting Logs and Artifacts"

ARTIFACTS_DIR="$PROJECT_ROOT/test-artifacts-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ARTIFACTS_DIR"

# Copy logs
print_info "Copying logs to artifacts directory..."
cp "$BACKEND_LOG" "$ARTIFACTS_DIR/" 2>/dev/null || true
cp "$FRONTEND_LOG" "$ARTIFACTS_DIR/" 2>/dev/null || true

# Copy test results
if [ -d "$TEST_REPORT_DIR" ]; then
    cp -r "$TEST_REPORT_DIR" "$ARTIFACTS_DIR/" 2>/dev/null || true
fi

# Create test summary file
cat > "$ARTIFACTS_DIR/test-summary.txt" << EOF
Potential Parakeet E2E Test Summary
===================================

Test Date: $(date '+%Y-%m-%d %H:%M:%S')
Status: $TEST_STATUS

Backend:
  Port: $BACKEND_PORT
  PID: $BACKEND_PID
  Log: $BACKEND_LOG

Frontend:
  Port: $FRONTEND_PORT
  PID: $FRONTEND_PID
  Log: $FRONTEND_LOG

Test Report:
  Directory: $TEST_REPORT_DIR
  HTML Report: file://$TEST_REPORT_DIR/index.html

Artifacts:
  Location: $ARTIFACTS_DIR
EOF

print_success "Artifacts collected at: $ARTIFACTS_DIR"
echo ""

################################################################################
# Final Summary
################################################################################

print_header "E2E Test Execution Complete"

echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Summary:"
echo "  Test Status: $TEST_STATUS"
echo "  Backend: http://localhost:$BACKEND_PORT (PID: $BACKEND_PID)"
echo "  Frontend: http://localhost:$FRONTEND_PORT (PID: $FRONTEND_PID)"
echo "  Test Report: file://$TEST_REPORT_DIR/index.html"
echo "  Artifacts: $ARTIFACTS_DIR"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    print_success "🎉 All tests passed! Platform is ready for production deployment."
else
    print_error "⚠️  Some tests failed. Review the report and fix issues before deploying."
fi

echo ""
print_info "Servers will be stopped automatically on script exit."
print_info "Press Ctrl+C to stop servers and exit, or wait for automatic cleanup..."
echo ""

# Keep servers running for manual inspection if needed
read -t 30 -p "Press Enter to stop servers now, or wait 30s for automatic shutdown..." || true

exit $EXIT_CODE
