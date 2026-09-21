import torch

from lev.model import PointerHead, branch_mask
from lev.train import scheduler_pct_start


def test_pointer_head_returns_one_logit_per_option():
    head = PointerHead(hidden_size=8, pointer_size=4)
    logits = head(torch.randn(8), torch.randn(77, 8))

    assert logits.shape == (77,)


def test_branch_mask_hides_sibling_questions():
    mask = branch_mask([0, 0, 1, 1, 2, 2], "cpu")[0, 0]

    assert mask[3, 1] == 0
    assert mask[3, 2] == 0
    assert mask[3, 4] < -1e30
    assert mask[5, 1] == 0
    assert mask[5, 2] < -1e30


def test_tiny_runs_get_a_nonzero_one_cycle_warmup():
    assert scheduler_pct_start(10) == 0.3
    assert scheduler_pct_start(11) == 0.1
