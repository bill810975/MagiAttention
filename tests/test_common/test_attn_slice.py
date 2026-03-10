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

import unittest
from unittest import TestCase

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.container.slice import AttnSlice, MultiKAttnSlice


class TestAttnSliceArea(TestCase):
    def test_attn_slice_causal_area_sq_eq_sk(self):
        """Test area computation for square causal mask (sq == sk)."""
        s = AttnSlice(
            q_range=AttnRange(0, 4),
            k_range=AttnRange(0, 4),
            mask_type=AttnMaskType.CAUSAL,
        )
        # Triangle: 1 + 2 + 3 + 4 = 10
        self.assertEqual(s.area, 10)

    def test_attn_slice_causal_area_sq_lt_sk(self):
        """Test area computation for trapezoidal causal mask (sq < sk)."""
        s = AttnSlice(
            q_range=AttnRange(0, 3),
            k_range=AttnRange(0, 5),
            mask_type=AttnMaskType.CAUSAL,
        )
        # Trapezoid (bottom-right): rows have 3, 4, 5 ones → area = 12
        self.assertEqual(s.area, 12)

    def test_attn_slice_causal_area_sq_gt_sk(self):
        """Test area computation for tall causal mask (sq > sk)."""
        s = AttnSlice(
            q_range=AttnRange(0, 7),
            k_range=AttnRange(0, 4),
            mask_type=AttnMaskType.CAUSAL,
        )
        # Triangle: first 3 rows empty, then 1+2+3+4 = 10
        self.assertEqual(s.area, 10)


class TestMultiKAttnSliceArea(TestCase):
    def test_multi_k_attn_slice_bicausal_area(self):
        """Test that BICAUSAL area is accumulated (+=), not overwritten (=)."""
        q_range = AttnRange(0, 3)
        k_ranges = AttnRanges.from_ranges([(0, 5), (10, 16)])
        mask_types = [AttnMaskType.FULL, AttnMaskType.BICAUSAL]

        s = MultiKAttnSlice(
            q_range=q_range,
            k_ranges=k_ranges,
            mask_types=mask_types,
        )

        # FULL area: 3 * 5 = 15
        # BICAUSAL area (parallelogram): (6 - 3 + 1) * 3 = 12
        # Total: 15 + 12 = 27
        self.assertEqual(s.area, 27)

    def test_multi_k_attn_slice_mixed_area(self):
        """Test area with multiple k_ranges including BICAUSAL."""
        q_range = AttnRange(0, 4)
        k_ranges = AttnRanges.from_ranges([(0, 6), (10, 18)])
        mask_types = [AttnMaskType.CAUSAL, AttnMaskType.BICAUSAL]

        s = MultiKAttnSlice(
            q_range=q_range,
            k_ranges=k_ranges,
            mask_types=mask_types,
        )

        # CAUSAL area (trapezoid, sk=6 > sq=4): (2*6 - 4 + 1) * 4 // 2 = 9 * 4 // 2 = 18
        # BICAUSAL area (parallelogram): (8 - 4 + 1) * 4 = 20
        # Total: 18 + 20 = 38
        self.assertEqual(s.area, 38)

    def test_multi_k_attn_slice_single_bicausal(self):
        """Test area with a single BICAUSAL k_range."""
        q_range = AttnRange(0, 3)
        k_ranges = AttnRanges.from_ranges([(0, 7)])
        mask_types = [AttnMaskType.BICAUSAL]

        s = MultiKAttnSlice(
            q_range=q_range,
            k_ranges=k_ranges,
            mask_types=mask_types,
        )

        # BICAUSAL area (parallelogram): (7 - 3 + 1) * 3 = 15
        self.assertEqual(s.area, 15)


if __name__ == "__main__":
    unittest.main()
