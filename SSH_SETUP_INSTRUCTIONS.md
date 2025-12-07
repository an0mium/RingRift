# SSH Setup Instructions for Your Mac Cluster

## Current Status (Dec 6, 2025)

| Machine          | Hostname            | SSH Status         | Key Installed |
| ---------------- | ------------------- | ------------------ | ------------- |
| Mac Studio       | Mac-Studio.local    | ✅ **CONNECTED**   | ✅ Yes        |
| MacBook Pro (M1) | MacBook-Pro.local   | ⛔ **NO L3 ROUTE** | ❌ Not yet    |
| MacBook Pro 3    | MacBook-Pro-3.local | ⛔ **NO L3 ROUTE** | ❌ Not yet    |

### Network Diagnostic Summary

**What works:**

- Bridge0 interface is UP with IP `169.254.167.99`
- Thunderbolt ports en1, en2, en3 are active
- L2 (Layer 2/MAC) discovery works - both MacBooks appear in ARP table
- Mac-Studio.local is fully reachable via en7

**What doesn't work:**

- MacBook-Pro.local and MacBook-Pro-3.local don't respond to ping or SSH
- Even direct IP connections (169.254.56.178, 169.254.42.153) fail
- Both machines visible in ARP but not responding at IP layer

**Likely causes (investigate on each MacBook):**

1. Machine is in deep sleep (wake it physically)
2. Thunderbolt Bridge network service not configured
3. Firewall/Remote Login issue persists

---

## Completed: Mac-Studio.local ✅

The SSH key `id_cluster` is already installed on Mac-Studio.local. No further action needed for this machine.

Test connection:

```bash
ssh -i ~/.ssh/id_cluster armand@Mac-Studio.local
```

---

## NEEDS ACTION: MacBook Pros Configuration

### Step 1: Wake & Verify Physical Connection

On **MacBook-Pro.local** and **MacBook-Pro-3.local**:

1. **Wake the machine** - ensure display is on and not sleeping
2. **Verify Thunderbolt cable** is connected (should show in System Settings)
3. **Check for any connection dialogs** that need approval

### Step 2: Configure Thunderbolt Bridge Network Service

On each MacBook:

1. **System Settings → Network**
2. Find **"Thunderbolt Bridge"** in the left sidebar
3. If not there, click **"+"** to add a new service:
   - Interface: Thunderbolt Bridge
   - Service Name: Thunderbolt Bridge
4. Set **Configure IPv4: Using DHCP** OR **Link-Local Only**
5. Click **Apply**

Alternatively via Terminal:

```bash
# List network services
networksetup -listallnetworkservices

# If Thunderbolt Bridge exists, ensure DHCP or link-local
networksetup -setdhcp "Thunderbolt Bridge"
```

### Step 3: Verify Remote Login is Enabled

On each MacBook:

1. **System Settings → General → Sharing**
2. Ensure **Remote Login** is **ON**
3. Ensure **"Allow access for: All users"** is selected (or your user is in the list)

Via Terminal:

```bash
# Check status
sudo systemsetup -getremotelogin

# Enable if needed
sudo systemsetup -setremotelogin on
```

### Step 4: Disable Firewall (or Allow SSH)

On each MacBook:

**Option A - Disable firewall entirely:**

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
```

**Option B - Allow SSH through firewall:**

1. System Settings → Network → Firewall → Options
2. Uncheck "Block all incoming connections"
3. Uncheck "Enable stealth mode"
4. Add `/usr/sbin/sshd` to allowed apps if not present

Via Terminal:

```bash
# Disable stealth mode
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode off

# Add SSH explicitly
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/sbin/sshd
```

### Step 5: Test Local SSH

On each MacBook, verify SSH works locally:

```bash
ssh localhost
```

If this fails, Remote Login is not properly enabled.

### Step 6: Verify Thunderbolt IP

On each MacBook, check it has a link-local IP on Thunderbolt:

```bash
ifconfig | grep -A5 "bridge0\|en[1-3]:" | grep "inet "
```

Should show something like `inet 169.254.x.x`

---

## After MacBooks are Configured

From this machine (Armands-MacBook-Pro), test connectivity:

```bash
# Quick reachability test
for host in Mac-Studio.local MacBook-Pro.local MacBook-Pro-3.local; do
  echo -n "$host: "
  nc -z -w 3 "$host" 22 2>/dev/null && echo "✓ SSH reachable" || echo "✗ unreachable"
done
```

Once reachable, install SSH keys:

```bash
# For MacBook-Pro.local (when reachable)
ssh-copy-id -i ~/.ssh/id_cluster armand@MacBook-Pro.local

# For MacBook-Pro-3.local (when reachable)
ssh-copy-id -i ~/.ssh/id_cluster armand@MacBook-Pro-3.local
```

---

## Final Verification Script

After all machines are configured:

```bash
for host in Mac-Studio.local MacBook-Pro.local MacBook-Pro-3.local; do
  echo -n "$host: "
  ssh -o ConnectTimeout=3 -o BatchMode=yes -i ~/.ssh/id_cluster armand@"$host" \
    "hostname && uname -m && sysctl -n machdep.cpu.brand_string 2>/dev/null | head -1" \
    2>/dev/null && echo "" || echo "✗ failed"
done
```

---

## SSH Config Helper

Add these to `~/.ssh/config` for easier connections:

```
Host mac-studio
    HostName Mac-Studio.local
    User armand
    IdentityFile ~/.ssh/id_cluster
    IdentitiesOnly yes

Host macbook-pro
    HostName MacBook-Pro.local
    User armand
    IdentityFile ~/.ssh/id_cluster
    IdentitiesOnly yes

Host macbook-pro-3
    HostName MacBook-Pro-3.local
    User armand
    IdentityFile ~/.ssh/id_cluster
    IdentitiesOnly yes
```

Then connect simply with: `ssh mac-studio`

---

## SSH Public Key (for manual installation)

If ssh-copy-id fails, manually add this to `~/.ssh/authorized_keys` on the target machine:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGIkECBKFgaDCGQDqqBC5LomwDL5p215assJaXPG4pg9 cluster-automation
```

---

## Quick Diagnosis Reference

From this machine, current network state:

```bash
# Check bridge0 and Thunderbolt interfaces
ifconfig bridge0; ifconfig en7

# See what's visible at L2
arp -a | grep bridge

# Test all hosts
for h in Mac-Studio.local MacBook-Pro.local MacBook-Pro-3.local; do
  echo -n "$h: "; ping -c1 -W2 "$h" &>/dev/null && echo "✓" || echo "✗"
done
```
