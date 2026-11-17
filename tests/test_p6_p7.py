"""CPU checks for p6 (depth SD) and p7 (soft/relational SDP). Run: python -m pytest tests/test_p6_p7.py -q"""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model import GraphSmile
from module import build_match_sen_shift_label, build_soft_shift_target


def make_args(**kw):
    a = dict(textf_mode='textf0', no_cuda=True, win=[17, 17], modals='tva', shift_win=19,
             hidden_dim=32, drop=0.2, heter_n_layers=[7, 7, 7], no_res=False, no_fcres=False,
             branch_sd=0.0, branch_mkl=0.0, sd_tau=2.0, depth_sd=0.0, depth_ce=0.0, depth_exits=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


def batch(lens=(5, 9), seed=0):
    g = torch.Generator().manual_seed(seed)
    T, B = max(lens), len(lens)
    t = [torch.randn(T, B, 16, generator=g) for _ in range(4)]
    v, a = torch.randn(T, B, 8, generator=g), torch.randn(T, B, 12, generator=g)
    return (*t, v, a, None, None, list(lens))


def build(**kw):
    torch.manual_seed(0)
    return GraphSmile(make_args(**kw), [16, 8, 12], 6)


def test_flags_off_unchanged_and_four_outputs():
    m0, m1 = build(), build(depth_sd=0.0)
    out0 = m0.eval()(*batch())
    assert len(out0) == 4
    for x, y in zip(out0, m1.eval()(*batch())):
        assert torch.equal(x, y)


def test_depth_flags_do_not_shift_base_init():
    base, d = build(), build(depth_sd=1.0)
    sd = d.state_dict()
    for k, v in base.state_dict().items():
        if 'edge_weight' not in k:
            assert torch.equal(v, sd[k]), k


def test_exit_at_full_depth_equals_H():
    m = build(depth_sd=1.0, depth_exits=[1, 3, 7]).eval()
    out = m(*batch())
    H = out[3]
    parts = [c.partials[-1] for c in (m.hetergconv_tv, m.hetergconv_ta, m.hetergconv_va)]
    assert torch.equal(m._fuse(*parts), H)
    assert out[4]['exit_logits'].shape == (3, 14, 6)
    assert torch.equal(out[4]['exit_logits'][2], m.exit_heads[2](H))


def test_depth_loss_reaches_exit_heads_only_through_kd():
    m = build(depth_sd=1.0).train()
    out = m(*batch())
    out[4]['sd_loss'].backward()
    assert m.exit_heads[0].weight.grad.abs().sum() > 0
    assert m.emo_output.weight.grad is None  # teacher logits are detached


def test_soft_target_onehot_equals_hard_labels():
    g = torch.Generator().manual_seed(1)
    for win, lens in [(19, [5, 40, 19, 38]), (-1, [5, 9]), (4, [7, 8, 3])]:
        lab = torch.randint(0, 3, (sum(lens),), generator=g)
        hard = build_match_sen_shift_label(win, lens, lab)
        soft = build_soft_shift_target(win, lens, torch.nn.functional.one_hot(lab, 3).float(), 'differ')
        assert torch.equal(soft, hard.float())
        ordn = build_soft_shift_target(win, lens, torch.nn.functional.one_hot(lab, 3).float(), 'ordinal')
        assert torch.equal(ordn > 0, hard.bool())  # same zero pattern, magnitudes 0.5 or 1
        assert set(ordn.unique().tolist()) <= {0.0, 0.5, 1.0}


def test_soft_target_in_unit_interval():
    p = torch.softmax(torch.randn(30, 3), -1)
    for mode in ('differ', 'ordinal'):
        t = build_soft_shift_target(19, [30], p, mode)
        assert t.shape == (19 * 19 + 11 * 11,) and t.min() >= 0 and t.max() <= 1
