import numpy as np, random
import torch
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from module import build_match_sen_shift_label
from utils import AutomaticWeightedLoss
from losses import focal_loss, ordinal_loss, aux_reg_loss, ogm_coefficient

# Phase 2 and Phase 3 options, set by run.py. Defaults keep the original training path.
OPTS = dict(aux_reg=0.0, aux_reg_loss='l1', ordinal_smooth=0.0, focal=False,
            focal_gamma=0.0, focal_alpha_w=None, uni_aux=0.0, ogm=False, ogm_alpha=1.0)


def set_options(**kw):
    OPTS.update(kw)


# Output of the most recent eval call: emotion softmax probabilities, utterance order as labels.
LAST = {}


def apply_ogm(model, uni, labels, alpha):
    """Scale encoder gradients by the OGM coefficient of each modality (after backward)."""
    with torch.no_grad():
        s = {m: F.softmax(uni[m], dim=1).gather(1, labels.unsqueeze(1)).mean().item()
             for m in 'tva'}
    base = model.module if hasattr(model, 'module') else model
    groups = {
        't': list(base.batchnorms_t.parameters()) + list(base.dim_layer_t.parameters()),
        'v': list(base.dim_layer_v.parameters()),
        'a': list(base.dim_layer_a.parameters()),
    }
    for m in 'tva':
        others = [s[o] for o in 'tva' if o != m]
        k = ogm_coefficient(s[m], sum(others) / len(others), alpha)
        if k != 1.0:
            for p in groups[m]:
                if p.grad is not None:
                    p.grad.mul_(k)

seed = 2024


def set_seed(new_seed):
    # --seed in run.py; the default stays 2024, so unchanged runs are unchanged.
    global seed
    seed = new_seed


def seed_everything(seed=None):
    seed = globals()['seed'] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync_optimizer(model, optimizer):
    """Add parameters that were created after the optimizer was built.

    module.py creates edge_weight lazily inside forward, so it is not in the
    optimizer that run.py builds before training. Without this, edge_weight
    stays at its initial value of 1. Replaced parameters are left in the
    optimizer; AdamW skips parameters whose grad is None.
    Returns the number of parameters added.
    """
    in_opt = {id(p) for g in optimizer.param_groups for p in g['params']}
    new = [p for p in model.parameters() if id(p) not in in_opt]
    if new:
        optimizer.add_param_group({'params': new})
    return len(new)


def train_or_eval_model(
    model,
    loss_function_emo,
    loss_function_sen,
    loss_function_shift,
    dataloader,
    epoch,
    cuda,
    modals,
    optimizer=None,
    train=False,
    dataset='IEMOCAP',
    loss_type='',
    lambd=[1.0, 1.0, 1.0],
    epochs=100,
    classify='',
    shift_win=5,
):
    losses, preds_emo, labels_emo = [], [], []
    preds_sft, labels_sft = [], []
    preds_sen, labels_sen = [], []
    vids = []
    initial_feats, extracted_feats = [], []
    probs_list = []
    keys_list = []  # "<dialogue id>_<utterance index>", same order as probs_list

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for iter, data in enumerate(dataloader):

        if train:
            optimizer.zero_grad()

        textf0, textf1, textf2, textf3, visuf, acouf, qmask, umask, label_emotion, label_sentiment = (
            [d.cuda() for d in data[:-1]] if cuda else data[:-1])

        if 't' not in modals:
            textf0, textf1, textf2, textf3 = [
                torch.zeros_like(f) for f in (textf0, textf1, textf2, textf3)
            ]
        if 'v' not in modals:
            visuf = torch.zeros_like(visuf)
        if 'a' not in modals:
            acouf = torch.zeros_like(acouf)

        dia_lengths, label_emotions, label_sentiments = [], [], []
        for j in range(umask.size(1)):
            dia_lengths.append((umask[:, j] == 1).nonzero().tolist()[-1][0] +
                               1)
            label_emotions.append(label_emotion[:dia_lengths[j], j])
            label_sentiments.append(label_sentiment[:dia_lengths[j], j])
        label_emo = torch.cat(label_emotions)
        label_sen = torch.cat(label_sentiments)

        logit_emo, logit_sen, logit_sft, extracted_feature, aux = model(
            textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask,
            dia_lengths)

        if train:
            # Parameters created during this forward (edge_weight) join the optimizer.
            sync_optimizer(model, optimizer)
            if iter == 0:
                n_opt = sum(len(g['params']) for g in optimizer.param_groups)
                n_model = sum(1 for _ in model.parameters())
                print('optimizer params: {}, model params: {}'.format(n_opt, n_model))

        prob_emo = F.log_softmax(logit_emo, -1)
        if not train:
            probs_list.append(prob_emo.detach().exp().cpu().numpy())
            for j, vid in enumerate(data[-1]):
                keys_list.extend('%s_%d' % (vid, k) for k in range(dia_lengths[j]))
        if OPTS['ordinal_smooth'] > 0:
            loss_emo = ordinal_loss(prob_emo, label_emo, OPTS['ordinal_smooth'])
        elif OPTS['focal']:
            loss_emo = focal_loss(prob_emo, label_emo, OPTS['focal_gamma'],
                                  OPTS['focal_alpha_w'])
        else:
            loss_emo = loss_function_emo(prob_emo, label_emo)
        prob_sen = F.log_softmax(logit_sen, -1)
        loss_sen = loss_function_sen(prob_sen, label_sen)
        prob_sft = F.log_softmax(logit_sft, -1)
        label_sft = build_match_sen_shift_label(shift_win, dia_lengths,
                                                label_sen)
        loss_sft = loss_function_shift(prob_sft, label_sft)

        if loss_type == 'auto':
            awl = AutomaticWeightedLoss(3)
            loss = awl(loss_emo, loss_sen, loss_sft)
        elif loss_type == 'epoch':
            loss = (epoch / epochs) * (lambd[0] * loss_emo) + (
                1 - epoch / epochs) * (lambd[1] * loss_sen +
                                       lambd[2] * loss_sft)
        elif loss_type == 'emo_sen_sft':
            loss = lambd[0] * loss_emo + lambd[1] * loss_sen + lambd[
                2] * loss_sft
        elif loss_type == 'emo_sen':
            loss = lambd[0] * loss_emo + lambd[1] * loss_sen
        elif loss_type == 'emo_sft':
            loss = lambd[0] * loss_emo + lambd[2] * loss_sft
        elif loss_type == 'emo':
            loss = loss_emo
        elif loss_type == 'sen_sft':
            loss = lambd[1] * loss_sen + lambd[2] * loss_sft
        elif loss_type == 'sen':
            loss = loss_sen
        else:
            NotImplementedError

        if OPTS['aux_reg'] > 0 and 'reg' in aux:
            # Raw continuous sentiment score for the utterances in this batch, dialogue-major.
            ds = dataloader.dataset
            raw = np.concatenate([np.asarray(ds.videoLabels[v][:L], dtype=np.float32)
                                  for v, L in zip(data[-1], dia_lengths)])
            raw = torch.as_tensor(raw, device=label_emo.device)
            loss = loss + OPTS['aux_reg'] * aux_reg_loss(aux['reg'], raw, OPTS['aux_reg_loss'])
        if OPTS['uni_aux'] > 0 and 'uni' in aux:
            loss = loss + OPTS['uni_aux'] * sum(
                F.cross_entropy(aux['uni'][m], label_emo) for m in 'tva')

        preds_emo.append(torch.argmax(prob_emo, 1).cpu().numpy())
        labels_emo.append(label_emo.cpu().numpy())
        preds_sen.append(torch.argmax(prob_sen, 1).cpu().numpy())
        labels_sen.append(label_sen.cpu().numpy())
        preds_sft.append(torch.argmax(prob_sft, 1).cpu().numpy())
        labels_sft.append(label_sft.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            if OPTS['ogm'] and 'uni' in aux:
                apply_ogm(model, {m: aux['uni'][m].detach() for m in 'tva'}, label_emo,
                          OPTS['ogm_alpha'])
            optimizer.step()

        extracted_feats.append(extracted_feature.cpu().detach().numpy())

    if preds_emo != []:
        preds_emo = np.concatenate(preds_emo)
        labels_emo = np.concatenate(labels_emo)
        preds_sen = np.concatenate(preds_sen)
        labels_sen = np.concatenate(labels_sen)
        preds_sft = np.concatenate(preds_sft)
        labels_sft = np.concatenate(labels_sft)

        extracted_feats = np.concatenate(extracted_feats)

    vids += data[-1]
    labels_emo = np.array(labels_emo)
    preds_emo = np.array(preds_emo)
    labels_sen = np.array(labels_sen)
    preds_sen = np.array(preds_sen)
    labels_sft = np.array(labels_sft)
    preds_sft = np.array(preds_sft)
    vids = np.array(vids)

    extracted_feats = np.array(extracted_feats)

    LAST['emo_probs'] = np.concatenate(probs_list) if probs_list else None
    LAST['emo_keys'] = np.array(keys_list) if keys_list else None
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_acc_emo = round(accuracy_score(labels_emo, preds_emo) * 100, 2)
    avg_f1_emo = round(
        f1_score(labels_emo, preds_emo, average='weighted') * 100, 2)
    avg_acc_sen = round(accuracy_score(labels_sen, preds_sen) * 100, 2)
    avg_f1_sen = round(
        f1_score(labels_sen, preds_sen, average='weighted') * 100, 2)
    avg_acc_sft = round(accuracy_score(labels_sft, preds_sft) * 100, 2)
    avg_f1_sft = round(
        f1_score(labels_sft, preds_sft, average='weighted') * 100, 2)

    return avg_loss, labels_emo, preds_emo, avg_acc_emo, avg_f1_emo, labels_sen, preds_sen, avg_acc_sen, avg_f1_sen, avg_acc_sft, avg_f1_sft, vids, initial_feats, extracted_feats
