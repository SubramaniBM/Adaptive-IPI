import sys
import logging
from pathlib import Path

# Add bipia to path to import their tools
bipia_path = Path("data/raw/bipia").absolute()
if str(bipia_path) not in sys.path:
    sys.path.insert(0, str(bipia_path))

try:
    from bipia.data.utils import insert_start, insert_end, insert_middle
except ImportError:
    logging.warning("Could not import BIPIA utils. Injection will fail if used.")

def inject(context: str, payload: str, position: str, random_state: int = 42) -> str:
    """Inject a payload into a context using official BIPIA logic.
    
    Args:
        context: The benign text context.
        payload: The attack instruction string.
        position: 'start', 'middle', or 'end'.
        random_state: Random seed for middle insertion placement.
        
    Returns:
        The poisoned context string.
    """
    if position == "start":
        return insert_start(context, payload)
    elif position == "end":
        return insert_end(context, payload)
    elif position == "middle":
        # The official BIPIA insert_middle requires context as a list of words, 
        # but in many benchmark configs they just pass the string and let it split.
        # Actually, let's look at how BIPIA handles it:
        # def insert_middle(context, attack, random_state):
        #     if isinstance(context, str): context = context.split()
        return insert_middle(context, payload, random_state)
    else:
        raise ValueError(f"Unknown position: {position}")
