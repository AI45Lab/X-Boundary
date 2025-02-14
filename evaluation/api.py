from dataclasses import dataclass
from typing import List, Optional

from numpy import ndarray

# overrefusal instances: category as behavior, prompt as messages
# multi_turn instances: categoty as behavior, plain_query as default_target, multi_turn_queries as context
@dataclass
class EvalInstance:
    id: Optional[int] = None # multi_turn
    behavior: Optional[str] = None
    context: Optional[str] = None
    queries: Optional[list] = None # multi_turn
    default_target: Optional[str] = None
    query_details: Optional[dict] = None # multi_turn
    generation: Optional[str] = None
    messages: Optional[str] = None
    score: Optional[int] = None
    score_reason: Optional[str] = None # overrefusal
    ppl: Optional[float] = None # overrefusal
    activation_norms: Optional[ndarray] = None
    tokens: Optional[ndarray] = None

