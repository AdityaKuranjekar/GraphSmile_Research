"""Loss functions and helpers for the gap-plan flags (Phase 2 and Phase 3).

Everything here is used only when its flag is set, so the default training path is unchanged.
"""
import torch
import torch.nn.functional as F


def inverse_frequency_weights(counts):
    """N / (K * n_c) from class counts, the same formula as --class_weight."""
    counts = torch.as_tensor(counts, dtype=torch.float32)
    k = counts.numel()
    n = counts.sum()
    return n / (k * counts.clamp(min=1.0))


def focal_loss(logp, labels, gamma, alpha=None):
    """Focal loss on log-probabilities. gamma=0 and alpha=None gives NLLLoss (mean reduction)."""
    logp_t = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
    if gamma > 0:
        loss = -((1.0 - logp_t.exp()) ** gamma) * logp_t
    else:
        loss = -logp_t
    if alpha is not None:
        loss = loss * alpha[labels]
    return loss.mean()


def ordinal_targets(labels, n_classes, s):
    """Soft targets: 1 - s on the true class; s is split between its adjacent classes.

    An edge class (first or last) sends all of s to its one neighbour. Each row sums to 1.
    """
    n = labels.size(0)
    rows = torch.arange(n, device=labels.device)
    t = torch.zeros(n, n_classes, device=labels.device)
    t[rows, labels] = 1.0 - s
    if s > 0:
        has_left = labels > 0
        has_right = labels < n_classes - 1
        n_nb = has_left.float() + has_right.float()
        share = s / n_nb.clamp(min=1.0)
        left = (labels - 1).clamp(min=0)
        right = (labels + 1).clamp(max=n_classes - 1)
        t = t.index_put((rows[has_left], left[has_left]), share[has_left], accumulate=True)
        t = t.index_put((rows[has_right], right[has_right]), share[has_right], accumulate=True)
    return t


def ordinal_loss(logp, labels, s):
    """Cross-entropy against ordinal soft targets (equal to KL divergence up to a constant)."""
    t = ordinal_targets(labels, logp.size(1), s)
    return -(t * logp).sum(dim=1).mean()


def aux_reg_loss(pred, target, kind='l1'):
    if kind == 'mse':
        return F.mse_loss(pred, target)
    return F.l1_loss(pred, target)


def ogm_coefficient(s_self, s_other, alpha):
    """On-the-fly gradient modulation coefficient (Peng et al., CVPR 2022), without the GE noise term.

    s_* are batch means of the ground-truth softmax probability for each modality.
    rho = s_self / s_other. A modality that dominates (rho > 1) has its gradients scaled by
    1 - tanh(alpha * rho); otherwise the coefficient is 1.
    NOTE: implemented from a summary of the method; check the coefficient against the paper.
    """
    rho = s_self / max(s_other, 1e-8)
    if rho <= 1.0:
        return 1.0
    return float(1.0 - torch.tanh(torch.tensor(alpha * rho)))
