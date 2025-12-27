#!/bin/bash
# Overnight Orchestrator Host Configuration
# Copy this file to orchestrator_hosts.sh and fill in your host details
# This file contains private network information and should NOT be committed

# ============================================
# Lambda Labs Instances
# ============================================

# Primary GPU (H100) - selfplay + NNUE + CMA-ES
LAMBDA_H100_HOST="xxx.xxx.xxx.xxx"
LAMBDA_H100_USER="ubuntu"
LAMBDA_H100_PATH="/home/ubuntu/ringrift/ai-service"
LAMBDA_H100_MIN_JOBS=3
LAMBDA_H100_ROLE="primary"  # primary, selfplay, or disabled

# Secondary GPU (A10) - selfplay only
LAMBDA_A10_HOST="xxx.xxx.xxx.xxx"
LAMBDA_A10_USER="ubuntu"
LAMBDA_A10_PATH="/home/ubuntu/ringrift/ai-service"
LAMBDA_A10_MIN_JOBS=2
LAMBDA_A10_ROLE="selfplay"

# ============================================
# AWS Instances (require SSH key)
# ============================================

AWS_STAGING_HOST="xxx.xxx.xxx.xxx"
AWS_STAGING_USER="ubuntu"
AWS_STAGING_PATH="/home/ubuntu/ringrift/ai-service"
AWS_STAGING_KEY="$HOME/.ssh/your-aws-key.pem"
AWS_STAGING_MIN_JOBS=2
AWS_STAGING_ROLE="selfplay"

# ============================================
# Vast.ai Instances (custom SSH ports)
# ============================================

VAST_3090_HOST="xxx.xxx.xxx.xxx"
VAST_3090_USER="root"
VAST_3090_PORT=12345  # Custom SSH port from Vast.ai dashboard
VAST_3090_PATH="/root/ringrift/ai-service"
VAST_3090_MIN_JOBS=2
VAST_3090_ROLE="selfplay"

VAST_5090_DUAL_HOST="xxx.xxx.xxx.xxx"
VAST_5090_DUAL_USER="root"
VAST_5090_DUAL_PORT=12345
VAST_5090_DUAL_PATH="/root/ringrift/ai-service"
VAST_5090_DUAL_MIN_JOBS=3
VAST_5090_DUAL_ROLE="selfplay"

VAST_5090_QUAD_HOST="xxx.xxx.xxx.xxx"
VAST_5090_QUAD_USER="root"
VAST_5090_QUAD_PORT=12345
VAST_5090_QUAD_PATH="/root/ringrift/ai-service"
VAST_5090_QUAD_MIN_JOBS=4
VAST_5090_QUAD_ROLE="selfplay"

# ============================================
# Mac Cluster (via Tailscale)
# ============================================

MAC_STUDIO_HOST="100.x.x.x"  # Tailscale IP
MAC_STUDIO_USER="yourusername"
MAC_STUDIO_PATH="$HOME/Development/RingRift/ai-service"
MAC_STUDIO_MIN_JOBS=2
MAC_STUDIO_ROLE="selfplay"

MBP_16GB_HOST="100.x.x.x"
MBP_16GB_USER="yourusername"
MBP_16GB_PATH="$HOME/Development/RingRift/ai-service"
MBP_16GB_MIN_JOBS=1
MBP_16GB_ROLE="selfplay"

MBP_64GB_HOST="100.x.x.x"
MBP_64GB_USER="yourusername"
MBP_64GB_PATH="$HOME/Development/RingRift/ai-service"
MBP_64GB_MIN_JOBS=2
MBP_64GB_ROLE="selfplay"

# ============================================
# Configuration Notes
# ============================================
#
# Roles:
#   - primary: Runs selfplay + NNUE training + CMA-ES + improvement loop
#   - selfplay: Only runs selfplay games
#   - disabled: Host is skipped by orchestrator
#
# SSH Setup:
#   1. Ensure passwordless SSH is configured
#   2. Add SSH public key to remote hosts
#   3. Test with: ssh -o BatchMode=yes user@host "echo ok"
#
# For Vast.ai:
#   - Get the custom SSH port from your Vast.ai dashboard
#   - Format: ssh -p PORT root@IP
#
# For AWS:
#   - Specify the path to your .pem key file
#   - Ensure the key has correct permissions (chmod 400)
#
# For Tailscale:
#   - Use Tailscale IPs (100.x.x.x) for reliable connectivity
#   - Ensure Tailscale is running on both machines
