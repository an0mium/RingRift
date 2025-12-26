"""
Unique model name generator using adjective-noun combinations.

Generates human-friendly, memorable names for models instead of ELO-based names.
Examples: "wiggly-tart-frog", "rapid-nebulous-alpaca", "salty-jagged-dog"
"""

import hashlib
import random
from datetime import datetime
from typing import Optional

# Adjectives - descriptive words
ADJECTIVES = [
    # Texture/Physical
    "wiggly", "fluffy", "jagged", "smooth", "bumpy", "crispy", "squishy", "fuzzy",
    "silky", "rough", "soft", "rigid", "wavy", "curly", "spiky", "velvety",
    # Taste/Sensation
    "tart", "salty", "zesty", "tangy", "spicy", "sweet", "bitter", "savory",
    # Mood/Personality
    "jolly", "grumpy", "cheerful", "moody", "calm", "fierce", "gentle", "bold",
    "shy", "brave", "clever", "silly", "wise", "quirky", "serene", "peppy",
    # Size/Shape
    "tiny", "giant", "lanky", "stocky", "round", "slim", "chunky", "petite",
    # Weather/Nature
    "misty", "sunny", "stormy", "frosty", "breezy", "dusty", "dewy", "hazy",
    # Color-like
    "golden", "silver", "rusty", "ashen", "rosy", "bronze", "amber", "coral",
    # Speed/Energy
    "rapid", "swift", "lazy", "zippy", "sleepy", "bouncy", "snappy", "mellow",
    # State
    "ancient", "cosmic", "mystic", "primal", "noble", "humble", "proud", "loyal",
    # Whimsical
    "nebulous", "highfalutin", "scattered", "untidy", "tidy", "dapper", "fancy",
]

# Second adjectives (optional, for three-word names)
MODIFIERS = [
    "slightly", "very", "rather", "quite", "somewhat", "truly", "oddly",
    "extra", "super", "ultra", "mega", "partly", "fully", "nearly",
    # Or just more adjectives
    "wild", "tame", "dark", "bright", "warm", "cool", "fresh", "old",
]

# Nouns - animals and objects
NOUNS = [
    # Animals
    "frog", "alpaca", "gerbil", "snake", "gull", "panda", "otter", "badger",
    "falcon", "raven", "dolphin", "squid", "beetle", "moth", "gecko", "newt",
    "ferret", "wombat", "lemur", "sloth", "koala", "penguin", "owl", "fox",
    "wolf", "bear", "lynx", "crane", "heron", "finch", "sparrow", "robin",
    "salmon", "trout", "crab", "shrimp", "jellyfish", "starfish", "seal",
    "walrus", "moose", "elk", "bison", "yak", "ibex", "oryx", "gazelle",
    # Mythical
    "dragon", "phoenix", "griffin", "sphinx", "hydra", "kraken", "unicorn",
    # Objects
    "crystal", "prism", "comet", "quasar", "nebula", "pulsar", "nova",
    "boulder", "pebble", "glacier", "canyon", "mesa", "delta", "fjord",
]


def generate_model_name(
    seed: Optional[str] = None,
    include_timestamp: bool = True,
    style: str = "two_word",  # "two_word", "three_word"
) -> str:
    """
    Generate a unique, human-friendly model name.

    Args:
        seed: Optional seed for reproducibility (e.g., model hash)
        include_timestamp: Whether to append YYYYMMDD_HHMMSS
        style: "two_word" for "adjective-noun", "three_word" for "modifier-adjective-noun"

    Returns:
        A name like "wiggly-frog" or "quite-wiggly-frog" with optional timestamp

    Examples:
        >>> generate_model_name()
        'jolly-penguin_20251226_070000'
        >>> generate_model_name(style="three_word")
        'rather-jolly-penguin_20251226_070000'
        >>> generate_model_name(seed="abc123", include_timestamp=False)
        'salty-kraken'
    """
    if seed:
        # Use seed for reproducible names
        hash_val = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
        rng = random.Random(hash_val)
    else:
        rng = random.Random()

    adjective = rng.choice(ADJECTIVES)
    noun = rng.choice(NOUNS)

    if style == "three_word":
        modifier = rng.choice(MODIFIERS)
        name = f"{modifier}-{adjective}-{noun}"
    else:
        name = f"{adjective}-{noun}"

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{timestamp}"

    return name


def generate_model_filename(
    config: str,
    seed: Optional[str] = None,
    extension: str = ".pth",
) -> str:
    """
    Generate a complete model filename with config prefix.

    Args:
        config: Model configuration like "square8_2p" or "hex8_4p"
        seed: Optional seed for reproducibility
        extension: File extension (default .pth)

    Returns:
        Filename like "square8_2p_jolly-penguin_20251226_070000.pth"

    Examples:
        >>> generate_model_filename("hex8_2p")
        'hex8_2p_salty-kraken_20251226_070000.pth'
    """
    name = generate_model_name(seed=seed, include_timestamp=True)
    return f"{config}_{name}{extension}"


def name_from_checkpoint_hash(checkpoint_path: str) -> str:
    """
    Generate a deterministic name based on a checkpoint file's content hash.

    This ensures the same model always gets the same name, regardless of when
    it's promoted.

    Args:
        checkpoint_path: Path to the model checkpoint file

    Returns:
        A reproducible name like "cosmic-phoenix"
    """
    import hashlib
    from pathlib import Path

    path = Path(checkpoint_path)
    if path.exists():
        # Hash first 1MB of file for speed
        with open(path, 'rb') as f:
            content = f.read(1024 * 1024)
        file_hash = hashlib.sha256(content).hexdigest()[:16]
    else:
        # Fallback to path hash
        file_hash = hashlib.sha256(str(path).encode()).hexdigest()[:16]

    return generate_model_name(seed=file_hash, include_timestamp=False, style="two_word")


# Quick test
if __name__ == "__main__":
    print("Sample names:")
    for _ in range(5):
        print(f"  {generate_model_name()}")

    print("\nThree-word style:")
    for _ in range(3):
        print(f"  {generate_model_name(style='three_word')}")

    print("\nWith config prefix:")
    for config in ["hex8_2p", "square8_4p", "square19_2p"]:
        print(f"  {generate_model_filename(config)}")

    print("\nDeterministic from seed:")
    for seed in ["model_v1", "model_v2", "model_v1"]:
        print(f"  {seed} -> {generate_model_name(seed=seed, include_timestamp=False)}")
