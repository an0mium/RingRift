# Transfer Method Comparison for RingRift Cluster

**Date:** December 28, 2025
**Test Environment:** 6 providers, 10+ nodes tested

## Executive Summary

| Method              | 1MB Success | 10MB Success    | Best Use Case                |
| ------------------- | ----------- | --------------- | ---------------------------- |
| **SCP**             | 80%         | 33%             | Small files (<5MB)           |
| **Base64 pipe**     | 80%         | 33%             | No advantage over SCP        |
| **Rsync --partial** | 100%        | 67%             | Medium files, resume support |
| **Chunked (2MB)**   | **100%**    | **100%**        | ✅ **Large files (>5MB)**    |
| **aria2**           | Works       | Firewall issues | Public URLs only             |
| **BitTorrent**      | N/A         | DHT blocked     | Not viable                   |

**Recommendation:** Use **chunked transfers (2MB chunks)** for files >5MB.

## Provider-Specific Results

### Nebius (backbone-1, h100-3)

| Test         | Result         | Notes                           |
| ------------ | -------------- | ------------------------------- |
| SSH Commands | ✅ OK          | Stable connection               |
| SCP 1MB      | ✅ 4/5         | Occasionally drops              |
| SCP 10MB     | ❌ 1/3         | Frequent connection resets      |
| Rsync 10MB   | ⚠️ 2/3         | Better with --partial           |
| Chunked 2MB  | ✅ 5/5         | **Most reliable**               |
| aria2c       | ✅ Available   | Can't reach other cluster nodes |
| BitTorrent   | ❌ DHT blocked | Port 6881 filtered              |

### RunPod (a100-1, l40s-2, h100)

| Test         | Result         | Notes                        |
| ------------ | -------------- | ---------------------------- |
| SSH Commands | ✅ OK          | NAT traversal required       |
| SCP 1MB      | ✅ OK          | Works well                   |
| SCP 10MB     | ❌ FAIL        | Connection resets frequently |
| Rsync 10MB   | ❌ FAIL        | Same as SCP                  |
| aria2c       | ✅ Available   | Works for public URLs        |
| BitTorrent   | ❌ DHT blocked | Container restrictions       |

### Vast.ai (29128352, 29129529)

| Test         | Result           | Notes                        |
| ------------ | ---------------- | ---------------------------- |
| SSH Commands | ✅ OK            | Via ssh\*.vast.ai proxy      |
| SCP 1MB      | ✅ OK            | Works well                   |
| SCP 10MB     | ❌ FAIL          | Connection drops after ~4-8s |
| Rsync 10MB   | ❌ FAIL          | Same as SCP                  |
| aria2c       | ❌ Not installed | Not in default image         |
| BitTorrent   | ❌ Not available | No torrent tools             |

### Vultr (a100-20gb)

| Test         | Result         | Notes                 |
| ------------ | -------------- | --------------------- |
| SSH Commands | ✅ OK          | Direct connection     |
| SCP 1MB      | ⚠️ Flaky       | 80% success           |
| SCP 10MB     | ⚠️ 1/3         | Often drops           |
| Rsync 10MB   | ⚠️ 2/3         | Better than SCP       |
| aria2c       | ✅ Available   | Works for public URLs |
| BitTorrent   | ❌ DHT blocked | Firewall rules        |

### Hetzner (cpu1, cpu2, cpu3)

| Test         | Result           | Notes                    |
| ------------ | ---------------- | ------------------------ |
| SSH Commands | ✅ OK            | Very stable              |
| SCP 1MB      | ✅ OK            | Reliable                 |
| SCP 10MB     | ✅ OK            | **Most stable provider** |
| Rsync 10MB   | ✅ OK            | Works well               |
| aria2c       | ❌ Not installed | CPU-only nodes           |
| BitTorrent   | ❌ Not available | No tools                 |

## Tool Availability

| Node              | aria2c | transmission | python3 |
| ----------------- | ------ | ------------ | ------- |
| nebius-backbone-1 | ✅     | ❌           | ✅      |
| nebius-h100-3     | ✅     | ❌           | ✅      |
| runpod-a100-1     | ✅     | ❌           | ✅      |
| vultr-a100-20gb   | ✅     | ❌           | ✅      |
| hetzner-cpu1      | ❌     | ❌           | ✅      |
| vast-29128352     | ❌     | ❌           | ✅      |
| vast-29129529     | ❌     | ❌           | ✅      |

## Detailed Test Results

### Base64 vs SCP Comparison

Base64 encoding does **NOT** improve transfer reliability:

- Both methods have identical failure patterns
- Connection resets occur at the TCP level, not data encoding
- Base64 adds 33% overhead without benefit

```
1MB file, 5 iterations to nebius-backbone-1:
SCP:    4/5 (80%)
Base64: 4/5 (80%)
Rsync:  5/5 (100%)
```

### Chunked Transfer Success

Breaking large files into 2MB chunks achieves 100% success:

```bash
# 10MB file split into 2MB chunks
split -b 2m source.npz /tmp/chunk_

# Transfer results:
chunk_aa: ✓
chunk_ab: ✓
chunk_ac: ✓
chunk_ad: ✓
chunk_ae: ✓

# Reassembly verified: 10485760 bytes
```

### aria2 Test Results

aria2 works for **public URLs** but fails for intra-cluster transfers:

```
Public HTTP (speedtest.tele2.net):
- nebius: OK - 1MB in ~0.5s
- vultr: OK - 1MB in ~0.5s

Intra-cluster HTTP (hetzner:8888 -> others):
- nebius: FAILED - connection refused
- vultr: FAILED - connection refused

Root cause: Firewall rules block non-standard ports between nodes
```

### BitTorrent Test Results

BitTorrent is **not viable** for cluster distribution:

```
DHT connectivity test (port 6881):
- nebius: BLOCKED
- vultr: BLOCKED
- runpod: BLOCKED (container isolation)

aria2 --bt-* options: Available but unusable
```

## Recommended Transfer Implementation

### For files < 5MB

```bash
rsync -az --partial -e "ssh -i ~/.ssh/id_cluster" \
    source.npz user@host:/destination/
```

### For files >= 5MB (RECOMMENDED)

```bash
# Split into 2MB chunks
split -b 2m source.npz /tmp/chunk_

# Transfer each chunk with retry
for chunk in /tmp/chunk_*; do
    for attempt in 1 2 3; do
        rsync -az --partial -e "ssh -i ~/.ssh/key" \
            "$chunk" user@host:/tmp/chunks/ && break
        sleep 5
    done
done

# Reassemble on remote
ssh user@host "cat /tmp/chunks/chunk_* > /destination/source.npz"

# Cleanup
rm -f /tmp/chunk_*
ssh user@host "rm -rf /tmp/chunks"
```

### Implementation in scripts/lib/transfer.py

```python
CHUNK_SIZE = 2 * 1024 * 1024  # 2MB

def chunked_transfer(source: Path, host: str, dest: Path,
                     config: TransferConfig) -> TransferResult:
    """Transfer large files in chunks for reliability."""
    if source.stat().st_size < 5 * 1024 * 1024:
        return rsync_transfer(source, host, dest, config)

    # Split file
    chunks = split_file(source, CHUNK_SIZE)

    # Transfer each chunk with retries
    for chunk in chunks:
        for attempt in range(3):
            if rsync_transfer(chunk, host, f"/tmp/chunks/{chunk.name}", config):
                break
            time.sleep(5 * (attempt + 1))

    # Reassemble on remote
    ssh_command(host, f"cat /tmp/chunks/* > {dest}")
    ssh_command(host, "rm -rf /tmp/chunks")

    return TransferResult(success=True)
```

## Network Topology Notes

### Why transfers fail

1. **Connection resets after 4-8 seconds**
   - Likely intermediate firewall/NAT timeout
   - Affects all providers except Hetzner

2. **Port restrictions**
   - Non-standard ports (8888, 6881, etc.) blocked between nodes
   - Only SSH (22) and established provider ports work

3. **Provider-specific issues**
   - Vast.ai: SSH proxy adds latency
   - RunPod: Container network isolation
   - Nebius: Strict security groups

### Recommendations for infrastructure

1. **Open port 8780** between nodes for P2P HTTP data server
2. **Consider Tailscale mesh** for direct node-to-node transfers
3. **Use Hetzner as distribution hub** - most stable connections
4. **Pre-install aria2c** on Vast.ai images for public URL fallback

## See Also

- `scripts/lib/transfer.py` - Transfer implementation
- `app/coordination/sync_bandwidth.py` - Bandwidth-coordinated rsync
- `app/coordination/unified_distribution_daemon.py` - Model distribution
- `docs/architecture/SYNC_INFRASTRUCTURE_ARCHITECTURE.md` - Sync layer design
