#!/bin/bash
# SSH Cluster Setup Script
# Run this script interactively to set up SSH key-based authentication
# for distributed computing across Macs connected via Thunderbolt/WiFi
#
# Configuration is read from ssh-cluster.config (not tracked in git).
# Copy ssh-cluster.config.example to ssh-cluster.config and customize.

set -o pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration file paths
CONFIG_FILE="${SCRIPT_DIR}/ssh-cluster.config"
CONFIG_EXAMPLE="${SCRIPT_DIR}/ssh-cluster.config.example"

# Default configuration (can be overridden by config file or env vars)
SSH_USER="${SSH_USER:-$(whoami)}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_cluster}"
SSH_PUBKEY="${SSH_KEY}.pub"
TIMEOUT=5

# Host configurations - populated from config file
declare -a HOST_CONFIGS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo ""
}

# Load configuration from file
load_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        if [[ -f "$CONFIG_EXAMPLE" ]]; then
            log_error "Configuration file not found: $CONFIG_FILE"
            log_info "Copy the example config to get started:"
            echo ""
            echo "    cp ssh-cluster.config.example ssh-cluster.config"
            echo "    # Edit ssh-cluster.config with your hosts"
            echo ""
        else
            log_error "Configuration file not found: $CONFIG_FILE"
            log_info "Create a config file with the following format:"
            echo ""
            echo "    # ssh-cluster.config"
            echo "    SSH_USER=yourusername"
            echo "    SSH_KEY=~/.ssh/id_cluster"
            echo "    HOSTS=("
            echo "        \"hostname1.local||Description 1\""
            echo "        \"hostname2.local||Description 2\""
            echo "    )"
            echo ""
        fi
        return 1
    fi
    
    # Source the config file
    source "$CONFIG_FILE"
    
    # Copy HOSTS array to HOST_CONFIGS if defined
    if [[ ${#HOSTS[@]} -gt 0 ]]; then
        HOST_CONFIGS=("${HOSTS[@]}")
    fi
    
    # Expand ~ in SSH_KEY path
    SSH_KEY="${SSH_KEY/#\~/$HOME}"
    SSH_PUBKEY="${SSH_KEY}.pub"
    
    return 0
}

# Check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check SSH key exists
    if [[ ! -f "$SSH_PUBKEY" ]]; then
        log_error "SSH public key not found: $SSH_PUBKEY"
        log_info "Generate one with: ssh-keygen -t ed25519 -f $SSH_KEY -C 'cluster-automation'"
        return 1
    fi
    log_success "SSH public key found: $SSH_PUBKEY"
    
    # Check ssh-copy-id is available
    if ! command -v ssh-copy-id &> /dev/null; then
        log_error "ssh-copy-id not found"
        return 1
    fi
    log_success "ssh-copy-id is available"
    
    # Check nc (netcat) is available
    if ! command -v nc &> /dev/null; then
        log_warning "nc (netcat) not found - connectivity tests will be skipped"
    else
        log_success "nc (netcat) is available"
    fi
    
    # Check we have hosts configured
    if [[ ${#HOST_CONFIGS[@]} -eq 0 ]]; then
        log_error "No hosts configured in $CONFIG_FILE"
        return 1
    fi
    log_success "${#HOST_CONFIGS[@]} host(s) configured"
    
    # Display the key to be installed
    echo ""
    log_info "Key to install:"
    echo -e "${YELLOW}$(cat "$SSH_PUBKEY")${NC}"
    echo ""
    
    return 0
}

# Test SSH connectivity to a host
test_ssh_port() {
    local host=$1
    if command -v nc &> /dev/null; then
        nc -z -w "$TIMEOUT" "$host" 22 2>/dev/null
        return $?
    else
        # Fallback: try a quick SSH connection
        ssh -o ConnectTimeout="$TIMEOUT" -o BatchMode=yes -o StrictHostKeyChecking=no "$SSH_USER@$host" "exit" 2>/dev/null
        return $?
    fi
}

# Resolve hostname to IP (for diagnostics)
resolve_host() {
    local host=$1
    local ip
    ip=$(getent hosts "$host" 2>/dev/null | awk '{print $1}' | head -1)
    if [[ -z "$ip" ]]; then
        # macOS fallback
        ip=$(dscacheutil -q host -a name "$host" 2>/dev/null | grep "ip_address:" | awk '{print $2}' | head -1)
    fi
    echo "$ip"
}

# Pick the first reachable endpoint (primary first, then fallback)
choose_reachable_host() {
    local primary=$1
    local fallback=$2

    if [[ -n "$primary" ]] && test_ssh_port "$primary"; then
        echo "$primary"
        return 0
    fi

    if [[ -n "$fallback" ]] && test_ssh_port "$fallback"; then
        echo "$fallback"
        return 0
    fi

    # Default to primary if nothing reachable (caller will treat as failure)
    echo "$primary"
    return 1
}

# Check if key is already installed on remote host
check_key_installed() {
    local host=$1
    ssh -o ConnectTimeout="$TIMEOUT" -o BatchMode=yes -i "$SSH_KEY" "$SSH_USER@$host" "echo 'key_ok'" 2>/dev/null | grep -q "key_ok"
    return $?
}

# Copy SSH key to a host
copy_key_to_host() {
    local host=$1
    local fallback=$2
    local description=$3
    
    echo ""
    echo -e "${BLUE}────────────────────────────────────────${NC}"
    echo -e "${BLUE}  Setting up: $description ($host)${NC}"
    echo -e "${BLUE}────────────────────────────────────────${NC}"
    echo ""
    
    # Test connectivity
    log_info "Resolving hostnames..."
    local resolved_ip primary_ip fallback_ip
    primary_ip=$(resolve_host "$host")
    fallback_ip=$(resolve_host "$fallback")
    if [[ -n "$primary_ip" ]]; then
        log_info "Primary:  $host -> $primary_ip"
    else
        log_warning "Primary:  $host -> (no DNS/mDNS resolution)"
    fi
    if [[ -n "$fallback" ]]; then
        if [[ -n "$fallback_ip" ]]; then
            log_info "Fallback: $fallback -> $fallback_ip"
        else
            log_warning "Fallback: $fallback -> (no DNS/mDNS resolution)"
        fi
    fi
    
    local chosen_host
    if chosen_host=$(choose_reachable_host "$host" "$fallback"); then
        if [[ "$chosen_host" == "$host" ]]; then
            log_success "SSH port reachable on primary ($host)"
        else
            log_success "Using fallback endpoint: $chosen_host"
        fi
    else
        log_error "Cannot reach SSH on $host${fallback:+ (or fallback $fallback)}"
        log_info "Ensure Remote Login is enabled in System Settings > General > Sharing"
        return 1
    fi
    
    # Check if key is already installed
    log_info "Checking if key is already installed..."
    if check_key_installed "$chosen_host"; then
        log_success "Key is already installed on $chosen_host - skipping"
        return 0
    fi
    
    # Copy the key
    echo ""
    log_info "Copying SSH key to $chosen_host..."
    log_warning "Enter password for ${SSH_USER}@${chosen_host} when prompted:"
    echo ""
    
    # Use ssh-copy-id with appropriate options
    if ssh-copy-id -i "$SSH_PUBKEY" \
        -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout="$TIMEOUT" \
        "${SSH_USER}@${chosen_host}"; then
        
        log_success "SSH key copied successfully"
        return 0
    else
        log_error "Failed to copy SSH key"
        return 1
    fi
}

# Test final SSH connection
test_final_connection() {
    local host=$1
    local fallback=$2
    local description=$3
    
    echo -n "  $description ($host): "
    
    local chosen_host
    if ! chosen_host=$(choose_reachable_host "$host" "$fallback"); then
        echo -e "${RED}✗ Failed${NC} (SSH port not reachable on primary${fallback:+ or fallback})"
        return 1
    fi

    local result
    result=$(ssh -o ConnectTimeout="$TIMEOUT" \
        -o BatchMode=yes \
        -i "$SSH_KEY" \
        "${SSH_USER}@${chosen_host}" \
        "hostname && uname -m" 2>/dev/null)
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Connected${NC} via ${chosen_host} - $(echo "$result" | tr '\n' ' ')"
        return 0
    else
        echo -e "${RED}✗ Failed${NC}"
        return 1
    fi
}

# Show suggested SSH config entries
show_ssh_config() {
    local config_file="$HOME/.ssh/config"
    
    log_header "SSH Config Helper"
    
    log_info "You may want to add these entries to $config_file:"
    echo ""
    
    for config in "${HOST_CONFIGS[@]}"; do
        local host fallback description
        IFS='|' read -r host fallback description <<< "$config"
        
        # Generate a clean hostname alias
        local alias
        alias=$(echo "$description" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd '[:alnum:]-')
        
        echo -e "${YELLOW}Host $alias${NC}"
        echo "    HostName $host"
        echo "    User $SSH_USER"
        echo "    IdentityFile $SSH_KEY"
        echo "    IdentitiesOnly yes"
        echo ""
    done
}

# Main
main() {
    log_header "SSH Cluster Setup"
    
    echo "This script will set up SSH key-based authentication"
    echo "to enable passwordless connections to other machines."
    echo ""
    echo "Configuration:"
    echo "  User: $SSH_USER"
    echo "  Key:  $SSH_KEY"
    echo "  Config: $CONFIG_FILE"
    echo ""
    
    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    # Process each host
    local success_count=0
    local fail_count=0
    local skip_count=0
    
    for config in "${HOST_CONFIGS[@]}"; do
        local host fallback description
        IFS='|' read -r host fallback description <<< "$config"
        
        if copy_key_to_host "$host" "$fallback" "$description"; then
            ((success_count++))
        else
            if check_key_installed "$host" 2>/dev/null || { [[ -n "$fallback" ]] && check_key_installed "$fallback" 2>/dev/null; }; then
                ((skip_count++))
            else
                ((fail_count++))
            fi
        fi
    done
    
    # Test all connections
    log_header "Testing SSH Connections"
    
    local test_success=0
    local test_fail=0
    
    for config in "${HOST_CONFIGS[@]}"; do
        local host fallback description
        IFS='|' read -r host fallback description <<< "$config"
        
        if test_final_connection "$host" "$fallback" "$description"; then
            ((test_success++))
        else
            ((test_fail++))
        fi
    done
    
    # Summary
    log_header "Summary"
    
    echo "Key Setup:"
    echo -e "  ${GREEN}✓ Successful:${NC} $success_count"
    echo -e "  ${YELLOW}⊘ Skipped:${NC}    $skip_count (already installed)"
    echo -e "  ${RED}✗ Failed:${NC}     $fail_count"
    echo ""
    echo "Connection Tests:"
    echo -e "  ${GREEN}✓ Working:${NC}    $test_success"
    echo -e "  ${RED}✗ Failed:${NC}     $test_fail"
    echo ""
    
    if [[ $test_success -eq ${#HOST_CONFIGS[@]} ]]; then
        log_success "All hosts configured successfully!"
        echo ""
    elif [[ $test_fail -gt 0 ]]; then
        log_warning "Some hosts could not be configured."
        log_info "Try enabling Remote Login on those machines:"
        log_info "  System Settings > General > Sharing > Remote Login"
    fi
}

# Handle script arguments
case "${1:-}" in
    -h|--help)
        echo "Usage: $0 [options]"
        echo ""
        echo "SSH Cluster Setup - configure passwordless SSH to multiple hosts"
        echo ""
        echo "Options:"
        echo "  -h, --help     Show this help message"
        echo "  -t, --test     Only test existing connections"
        echo "  -c, --config   Show suggested SSH config entries"
        echo "  -i, --init     Create example config file"
        echo ""
        echo "Configuration:"
        echo "  Create ssh-cluster.config in the same directory as this script."
        echo "  See ssh-cluster.config.example for the format."
        echo ""
        echo "Environment variables:"
        echo "  SSH_USER       Username for SSH (default: current user)"
        echo "  SSH_KEY        Path to SSH private key (default: ~/.ssh/id_cluster)"
        exit 0
        ;;
    -t|--test)
        if ! load_config; then
            exit 1
        fi
        log_header "Testing SSH Connections"
        for config in "${HOST_CONFIGS[@]}"; do
            host=$(echo "$config" | cut -d'|' -f1)
            fallback=$(echo "$config" | cut -d'|' -f2)
            description=$(echo "$config" | cut -d'|' -f3)
            test_final_connection "$host" "$fallback" "$description"
        done
        exit 0
        ;;
    -c|--config)
        if ! load_config; then
            exit 1
        fi
        show_ssh_config
        exit 0
        ;;
    -i|--init)
        if [[ -f "$CONFIG_EXAMPLE" ]]; then
            log_info "Example config already exists: $CONFIG_EXAMPLE"
        else
            cat > "$CONFIG_EXAMPLE" << 'EOF'
# SSH Cluster Configuration
# Copy this file to ssh-cluster.config and customize for your network
#
# This file is sourced by setup-ssh-cluster.sh

# SSH username (defaults to current user if not set)
SSH_USER="$(whoami)"

# Path to SSH private key (will be created if it doesn't exist)
SSH_KEY="~/.ssh/id_cluster"

# Target hosts to configure
# Format: "hostname|fallback_hostname|description"
# The fallback hostname is optional (for IPv6 or alternate addresses)
HOSTS=(
    "example-host-1.local||Host 1 Description"
    "example-host-2.local||Host 2 Description"
    # Add more hosts as needed:
    # "192.168.1.100||Server at fixed IP"
    # "hostname.local||Another Mac"
)
EOF
            log_success "Created example config: $CONFIG_EXAMPLE"
            log_info "Copy and customize:"
            echo ""
            echo "    cp ssh-cluster.config.example ssh-cluster.config"
            echo "    # Edit ssh-cluster.config with your hosts"
            echo ""
        fi
        exit 0
        ;;
    *)
        if ! load_config; then
            exit 1
        fi
        main
        ;;
esac
