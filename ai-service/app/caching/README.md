# Caching Module

Unified caching abstraction for the RingRift AI service. Provides thread-safe, TTL-aware caching with both in-memory and persistent backends.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Abstractions](#core-abstractions)
   - [Cache Interface](#cache-interface)
   - [CacheEntry](#cacheentry)
   - [CacheStats](#cachestats)
   - [CacheConfig](#cacheconfig)
4. [In-Memory Caches](#in-memory-caches)
   - [MemoryCache](#memorycache)
   - [LRUCache](#lrucache)
   - [TTLCache](#ttlcache)
5. [File-Based Caches](#file-based-caches)
   - [FileCache](#filecache)
   - [ValidatedFileCache](#validatedfilecache)
6. [Decorators](#decorators)
   - [@cached](#cached)
   - [@async_cached](#async_cached)
   - [invalidate_cache](#invalidate_cache)
7. [Usage Examples](#usage-examples)
8. [Integration Guidelines](#integration-guidelines)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Caching Module                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────┐      ┌────────────────┐                    │
│   │  In-Memory     │      │  Persistent    │                    │
│   │                │      │                │                    │
│   │ • MemoryCache  │      │ • FileCache    │                    │
│   │ • LRUCache     │      │ • Validated    │                    │
│   │ • TTLCache     │      │   FileCache    │                    │
│   └───────┬────────┘      └───────┬────────┘                    │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│              ┌────────────────┐                                  │
│              │  Cache[K, V]   │  ◄── Abstract base               │
│              │  (Protocol)    │                                  │
│              └────────────────┘                                  │
│                       │                                          │
│           ┌───────────┴───────────┐                              │
│           ▼                       ▼                              │
│   ┌──────────────┐       ┌──────────────┐                       │
│   │  @cached     │       │  CacheStats  │                       │
│   │  @async_cached│       │  CacheEntry  │                       │
│   └──────────────┘       └──────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

| File            | Lines | Description                                         |
| --------------- | ----- | --------------------------------------------------- |
| `base.py`       | ~237  | Cache protocol, CacheEntry, CacheStats, CacheConfig |
| `memory.py`     | ~258  | MemoryCache, LRUCache, TTLCache implementations     |
| `file.py`       | ~389  | FileCache and ValidatedFileCache implementations    |
| `decorators.py` | ~243  | @cached, @async_cached decorators                   |
| `__init__.py`   | ~78   | Public API exports                                  |

---

## Quick Start

```python
from app.caching import (
    MemoryCache,
    FileCache,
    TTLCache,
    cached,
    async_cached,
)

# In-memory cache with TTL and size limit
cache = MemoryCache(max_size=100, ttl_seconds=3600)
cache.set("key", value)
value = cache.get("key")

# File-backed persistent cache
cache = FileCache("/path/to/cache.json", ttl_seconds=86400)

# Decorator for function memoization
@cached(ttl_seconds=300, max_size=100)
def expensive_function(arg):
    ...
```

---

## Core Abstractions

### Cache Interface

All cache implementations inherit from the abstract `Cache[K, V]` class:

```python
from app.caching import Cache

class Cache(ABC, Generic[K, V]):
    def get(self, key: K, default: V | None = None) -> V | None:
        """Get a value from the cache."""

    def set(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache."""

    def delete(self, key: K) -> bool:
        """Delete a key from the cache."""

    def clear(self) -> None:
        """Clear all entries from the cache."""

    def has(self, key: K) -> bool:
        """Check if a key exists and is not expired."""

    def get_or_set(self, key: K, factory: Callable[[], V], ttl_seconds: float | None = None) -> V:
        """Get a value or compute and cache it if missing."""

    async def get_or_set_async(self, key: K, factory: Callable[[], Awaitable[V]], ttl_seconds: float | None = None) -> V:
        """Async version of get_or_set."""
```

Supports `in` operator and `len()`:

```python
if "key" in cache:
    print(f"Cache has {len(cache)} entries")
```

### CacheEntry

Container for cached values with metadata:

```python
from app.caching import CacheEntry

entry = CacheEntry(
    value=my_data,
    created_at=time.time(),
    last_accessed=time.time(),
    access_count=0,
    ttl_seconds=3600,
)

# Check expiration
if entry.is_expired():
    ...

# Update access time
entry.touch()

# Get age
age = entry.age_seconds
```

### CacheStats

Statistics tracking for cache performance:

```python
from app.caching import CacheStats

stats = cache.stats

print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Hit Rate: {stats.hit_rate:.2%}")
print(f"Evictions: {stats.evictions}")
print(f"Expirations: {stats.expirations}")
print(f"Size: {stats.size}")

# Reset stats
stats.reset()
```

### CacheConfig

Configuration for cache instances:

```python
from app.caching import CacheConfig

config = CacheConfig(
    max_size=1000,               # None = unlimited
    ttl_seconds=3600,            # None = no expiration
    cleanup_interval_seconds=60, # How often to run cleanup
    eviction_policy="lru",       # "lru", "lfu", "fifo"
)
```

---

## In-Memory Caches

### MemoryCache

General-purpose thread-safe cache with TTL and LRU eviction:

```python
from app.caching import MemoryCache

# Create cache with limits
cache = MemoryCache(max_size=1000, ttl_seconds=3600)

# Basic operations
cache.set("user:123", user_data)
cache.set("temp:data", temp_data, ttl_seconds=60)  # Override TTL

user = cache.get("user:123")
user = cache.get("user:456", default=None)

# Delete
if cache.delete("user:123"):
    print("Deleted")

# Check existence
if cache.has("user:123"):
    ...

# Clear all
cache.clear()

# Iterate
for key in cache.keys():
    print(key)

for key, value in cache.items():
    print(f"{key}: {value}")

# Cleanup expired entries
expired_count = cache.cleanup_expired()
```

### LRUCache

Explicit LRU semantics (alias for MemoryCache):

```python
from app.caching import LRUCache

# LRUCache requires max_size
cache = LRUCache(max_size=100, ttl_seconds=3600)
```

### TTLCache

Time-based expiration with automatic cleanup:

```python
from app.caching import TTLCache

# TTLCache requires ttl_seconds
cache = TTLCache(
    ttl_seconds=300,              # Required: 5 minute TTL
    max_size=1000,                # Optional size limit
    cleanup_interval_seconds=60,  # Auto-cleanup every 60 seconds
)

# Automatic cleanup runs on get/set
cache.set("key", value)
result = cache.get("key")  # Triggers cleanup if interval passed
```

---

## File-Based Caches

### FileCache

Persistent JSON-backed cache:

```python
from app.caching import FileCache

cache = FileCache(
    cache_path="/path/to/cache.json",
    ttl_seconds=86400,      # 24 hours
    max_entries=10000,
    auto_save=True,         # Persist after each write
    cleanup_on_load=True,   # Evict expired on load
)

# Same interface as MemoryCache
cache.set("model:hash", model_metadata)
metadata = cache.get("model:hash")

# Explicit save (if auto_save=False)
cache.save()

# Keys must be strings (JSON requirement)
# Values must be JSON-serializable
```

Features:

- Lazy loading on first access
- Atomic writes (temp file + rename)
- LRU eviction when over max_entries
- Automatic cleanup of expired entries on load

### ValidatedFileCache

Adds custom validation hooks:

```python
from app.caching import ValidatedFileCache

def validate_model_cache(entry: dict) -> bool:
    """Check if cached model data is still valid."""
    source_path = entry.get("source_path")
    cached_hash = entry.get("source_hash")

    if not source_path or not Path(source_path).exists():
        return False

    current_hash = compute_file_hash(source_path)
    return current_hash == cached_hash

cache = ValidatedFileCache(
    cache_path="/path/to/model_cache.json",
    validator=validate_model_cache,
    ttl_seconds=None,  # Rely on validation, not TTL
)

# Entries failing validation are automatically evicted
model_info = cache.get("model:v2")
```

---

## Decorators

### @cached

Memoize synchronous functions:

```python
from app.caching import cached

@cached(ttl_seconds=300, max_size=100)
def expensive_computation(x: int, y: int) -> int:
    # Complex calculation...
    return result

# First call computes
result = expensive_computation(10, 20)

# Second call returns cached value
result = expensive_computation(10, 20)
```

Options:

```python
@cached(
    ttl_seconds=300,     # Time-to-live (None = no expiration)
    max_size=1000,       # Max cache size (default 1000)
    key_func=None,       # Custom key generation function
    cache_name="users",  # Name for invalidation
)
def get_user(user_id: int) -> User:
    ...
```

Custom key function:

```python
@cached(key_func=lambda self, id: f"user:{id}")
def get_user(self, user_id: int) -> User:
    ...
```

### @async_cached

Memoize async functions:

```python
from app.caching import async_cached

@async_cached(ttl_seconds=60)
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Usage
data = await fetch_data("https://api.example.com/data")
```

### invalidate_cache

Clear cached results:

```python
from app.caching import invalidate_cache

# By cache name
@cached(cache_name="users")
def get_user(user_id): ...

invalidate_cache("users")

# By function reference
invalidate_cache(func=get_user)

# Clear all caches
invalidate_cache()
```

Access underlying cache:

```python
@cached(ttl_seconds=300)
def my_func(x):
    ...

# Access cache stats
print(my_func._cache.stats.hit_rate)

# Manual cache operations
my_func._cache.clear()
```

Get global cache stats:

```python
from app.caching.decorators import get_cache_stats

stats = get_cache_stats()
for name, data in stats.items():
    print(f"{name}: hits={data['hits']}, hit_rate={data['hit_rate']:.2%}")
```

---

## Usage Examples

### Caching AI Model Evaluations

```python
from app.caching import MemoryCache

# Cache position evaluations
eval_cache = MemoryCache(max_size=100000, ttl_seconds=None)

def evaluate_position(state_hash: str, model) -> float:
    cached = eval_cache.get(state_hash)
    if cached is not None:
        return cached

    value = model.evaluate(state_hash)
    eval_cache.set(state_hash, value)
    return value
```

### Caching Database Queries

```python
from app.caching import cached

@cached(ttl_seconds=60, cache_name="game_queries")
def get_game_by_id(game_id: str) -> dict | None:
    return db.execute("SELECT * FROM games WHERE id = ?", game_id)

# After updating game
def update_game(game_id: str, data: dict):
    db.execute("UPDATE games SET ... WHERE id = ?", ...)
    invalidate_cache("game_queries")
```

### Persistent Model Metadata Cache

```python
from app.caching import ValidatedFileCache
import hashlib

def validate_checkpoint(entry: dict) -> bool:
    path = entry.get("checkpoint_path")
    if not path or not Path(path).exists():
        return False

    with open(path, "rb") as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    return current_hash == entry.get("checkpoint_hash")

model_cache = ValidatedFileCache(
    "/path/to/model_metadata_cache.json",
    validator=validate_checkpoint,
    ttl_seconds=86400 * 7,  # 7 days max
)

def get_model_metadata(checkpoint_path: str) -> dict:
    return model_cache.get_or_set(
        checkpoint_path,
        factory=lambda: extract_metadata(checkpoint_path),
    )
```

### Async API Caching

```python
from app.caching import async_cached

@async_cached(ttl_seconds=300)
async def fetch_elo_rating(model_id: str) -> float | None:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/api/models/{model_id}/elo")
        if response.status_code == 200:
            return response.json()["elo"]
        return None
```

---

## Integration Guidelines

### When to Use Each Cache Type

| Cache Type           | Use Case                                  |
| -------------------- | ----------------------------------------- |
| `MemoryCache`        | General-purpose, frequently accessed data |
| `LRUCache`           | Fixed-size cache with strict eviction     |
| `TTLCache`           | Time-sensitive data with auto-cleanup     |
| `FileCache`          | Persist cache across restarts             |
| `ValidatedFileCache` | Cache with external validity checks       |
| `@cached`            | Function memoization                      |
| `@async_cached`      | Async function memoization                |

### Thread Safety

All cache implementations are thread-safe using `threading.RLock`.

### Memory Considerations

- Set appropriate `max_size` to prevent unbounded growth
- Use TTL for data that becomes stale
- Call `cleanup_expired()` periodically for long-running processes
- Monitor `stats.evictions` to tune cache size

### Best Practices

1. **Choose appropriate TTL**: Match data freshness requirements
2. **Set max_size**: Prevent memory exhaustion
3. **Use cache names**: Enable targeted invalidation
4. **Monitor hit rates**: Low rates indicate cache inefficiency
5. **Handle None values**: `get()` returns `None` for misses AND for cached `None`

```python
# Handle cached None explicitly
MISSING = object()
result = cache.get("key", default=MISSING)
if result is MISSING:
    # Key not in cache
    ...
```

---

## See Also

- `app/ai/README.md` - AI model caching patterns
- `app/db/README.md` - Database query caching
- `app/models/loader.py` - Model loading with caching

---

_Last updated: December 2025_
