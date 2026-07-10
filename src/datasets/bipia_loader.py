import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)

class BIPIALoader:
    """Loads raw resources from the official BIPIA benchmark directories."""
    
    def __init__(self, bipia_root: str = "data/raw/bipia"):
        self.bipia_root = Path(bipia_root)
        
    def load_contexts(self, task: str, split: str) -> List[Dict[str, Any]]:
        """Load the raw contexts (benign emails, code, etc) for a specific task and split.
        
        Args:
            task: Task name (e.g. "email")
            split: "train" or "test"
            
        Returns:
            List of dictionaries containing the raw context records.
        """
        path = self.bipia_root / "benchmark" / task / f"{split}.jsonl"
        if not path.exists():
            logger.error(f"Context file not found: {path}")
            return []
            
        try:
            contexts = read_jsonl(path)
            logger.info(f"Loaded {len(contexts)} contexts from {path}")
            return contexts
        except Exception as e:
            logger.error(f"Failed to load contexts from {path}: {e}")
            return []

    def load_attack_library(self, split: str) -> Dict[str, List[str]]:
        """Load the attack payload library for a specific split.
        
        Args:
            split: "train" or "test"
            
        Returns:
            Dictionary mapping attack_family -> list of payload strings.
        """
        path = self.bipia_root / "benchmark" / f"text_attack_{split}.json"
        if not path.exists():
            logger.error(f"Attack library file not found: {path}")
            return {}
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                attacks = json.load(f)
            
            total_payloads = sum(len(payloads) for payloads in attacks.values())
            logger.info(f"Loaded {len(attacks)} attack families ({total_payloads} payloads) for split '{split}'")
            return attacks
        except Exception as e:
            logger.error(f"Failed to load attack library from {path}: {e}")
            return {}
