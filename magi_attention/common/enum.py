# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Literal, TypeAlias

import torch

from . import is_cpp_backend_enable

GroupReduceOp: TypeAlias = Literal["sum", "avg", "lse"]

OutMaybeWithLSE: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]

AttnSinkLayout: TypeAlias = Literal["sh", "shd", "ssh"]


class AttnType(Enum):
    """The enum used to specify the type of attention calculation we support"""

    SELF_ATTN = "self_attn"
    CROSS_ATTN = "cross_attn"


class AttnRole(Enum):
    """The enum used to specify the tensor role in attention"""

    QUERY = "query"
    KEY = "key"
    VALUE = "value"


class AttnMaskType(Enum):
    """The enum used to specify the unit type of attention mask we support.

    There are 4 supported mask types. All examples below use **bottom-right**
    alignment (the default) with ``seqlen_q = 4, seqlen_k = 7``::

        FULL – every query attends to every key (no masking):

              K: 0 1 2 3 4 5 6
          Q0: [1 1 1 1 1 1 1]
          Q1: [1 1 1 1 1 1 1]
          Q2: [1 1 1 1 1 1 1]
          Q3: [1 1 1 1 1 1 1]

        CAUSAL – standard auto-regressive mask. When seqlen_q < seqlen_k
        the mask is **trapezoidal** (bottom-right aligned slice of a
        larger triangular matrix), NOT full attention:

              K: 0 1 2 3 4 5 6
          Q0: [1 1 1 1 0 0 0]   <- attends to (sk - sq + 1) = 4 keys
          Q1: [1 1 1 1 1 0 0]
          Q2: [1 1 1 1 1 1 0]
          Q3: [1 1 1 1 1 1 1]   <- attends to all 7 keys

        INVCAUSAL – inverse of causal; each query attends to itself and
        all **following** key positions:

              K: 0 1 2 3 4 5 6
          Q0: [1 1 1 1 1 1 1]   <- attends to all 7 keys
          Q1: [0 1 1 1 1 1 1]
          Q2: [0 0 1 1 1 1 1]
          Q3: [0 0 0 1 1 1 1]   <- attends to (sk - sq + 1) = 4 keys

        BICAUSAL – intersection of CAUSAL and INVCAUSAL; forms a diagonal
        band. When seqlen_q == seqlen_k this reduces to the principal
        diagonal only:

              K: 0 1 2 3 4 5 6
          Q0: [0 0 0 1 0 0 0]
          Q1: [0 0 0 0 1 0 0]
          Q2: [0 0 0 0 0 1 0]
          Q3: [0 0 0 0 0 0 1]
    """

    FULL = "full"
    CAUSAL = "causal"
    BICAUSAL = "bi_causal"
    INVCAUSAL = "inv_causal"

    _FROM_INT_MAP: dict[int, "AttnMaskType"]
    _TO_INT_MAP: dict["AttnMaskType", int]

    @classmethod
    def _lazy_init_from_int_map(cls) -> None:
        if "_FROM_INT_MAP" in cls.__dict__:
            return

        cls._FROM_INT_MAP = {
            0: cls.FULL,
            1: cls.CAUSAL,
            2: cls.INVCAUSAL,
            3: cls.BICAUSAL,
        }

    @classmethod
    def _lazy_init_to_int_map(cls) -> None:
        if "_TO_INT_MAP" in cls.__dict__:
            return

        cls._TO_INT_MAP = {
            cls.FULL: 0,
            cls.CAUSAL: 1,
            cls.INVCAUSAL: 2,
            cls.BICAUSAL: 3,
        }

    @classmethod
    def from_int_type(cls, int_type: int) -> "AttnMaskType":
        cls._lazy_init_from_int_map()
        return cls._FROM_INT_MAP[int_type]  # type: ignore[index]

    def to_int_type(self) -> int:
        self.__class__._lazy_init_to_int_map()
        return self._TO_INT_MAP[self]


class AttnOverlapMode(Enum):
    """The enum used to specify the overlap mode for multi-stage overlapping"""

    STATIC = "static"
    DYNAMIC = "dynamic"


class DispatchAlgType(Enum):
    """The enum used to specify the algorithm type for load-balanced dispatching"""

    LOWER_BOUND = "lower_bound"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    BINARY_SEARCH = "binary_search"
    MIN_HEAP = "min_heap"
    TOPP_HEAP = "topp_heap"
    BACKTRACKING_PRUNING = "backtracing_pruning"
    RANDOM_SELECT = "random_select"
    SEQUENTIAL_SELECT = "sequential_select"
    BATCH_TOPP_HEAP = "batch_topp_heap"
    SORTED_SEQUENTIAL_SELECT = "sorted_sequential_select"


class OverlapAlgType(Enum):
    """The enum used to specify the algorithm type for multi-stage overlapping"""

    UNIFORM = "uniform"
    GREEDY = "greedy"


class DynamicAttnAlgType(Enum):
    """The enum used to specify the algorithm type for dynamic attn mask dispatching"""

    NON_COMMUNICATION_QO = "non_communication_qo"
    GREEDY_RANDOM_GRID = "greedy_random_grid"
    SIMPLEX_NETWORK_FLOW = "simplex_network_flow"
    FAST_SIMPLEX_NETWORK_FLOW = "fast_simplex_network_flow"
    BINARY_GREEDY = "binary_greedy"
    BINARY_GREEDY_PARALLEL = "binary_greedy_parallel"


if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import AttnMaskType as _AttnMaskType

        AttnMaskType = _AttnMaskType  # type: ignore[misc, assignment] # noqa: F811
    except ImportError:
        pass


class GrpCollBufferName(Enum):
    GroupCastDefault = "group_cast_default"
    GroupReduceDefault = "group_reduce_default"
    GroupCastQO = "group_cast_qo"
    GroupReduceQO = "group_reduce_qo"
