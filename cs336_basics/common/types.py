from typing import Dict, Tuple, TypeAlias

# Tokenize
TokenPair: TypeAlias = Dict[Tuple[bytes, bytes], int]  # Example: {(a, b): 3}
