import numpy as np, random
import torch
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from module import build_match_sen_shift_label, build_soft_shift_target
from utils import AutomaticWeightedLoss

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
    kd=None,
    stats=None,
):
    # kd (p6/p7, all optional): depth_ce, teacher {vid: {'sen','emo','H'}}, sft_soft,
    # sft_tau, sft_target, rel_kd. stats: dict filled with per-exit eval metrics.
    kd = kd or {}
    teacher = kd.get('teacher') if train else None
    exit_preds = []
    losses, preds_emo, labels_emo = [], [], []
    preds_sft, labels_sft = [], []
    preds_sen, labels_sen = [], []
    vids = []
    initial_feats, extracted_feats = [], []

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

        _out = model(
            textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask,
            dia_lengths)
        extras = _out[4] if len(_out) == 5 else {}
        sd_loss = extras.get('sd_loss')
        logit_emo, logit_sen, logit_sft, extracted_feature = _out[:4]
        if teacher is not None:
            vids_b = data[-1]
            t_sen = torch.cat([torch.as_tensor(teacher[v]['sen'][:n]) for v, n in zip(vids_b, dia_lengths)]).float().to(logit_emo.device)
            t_H = torch.cat([torch.as_tensor(teacher[v]['H'][:n]) for v, n in zip(vids_b, dia_lengths)]).float().to(logit_emo.device)

        prob_emo = F.log_softmax(logit_emo, -1)
        loss_emo = loss_function_emo(prob_emo, label_emo)
        prob_sen = F.log_softmax(logit_sen, -1)
        loss_sen = loss_function_sen(prob_sen, label_sen)
        prob_sft = F.log_softmax(logit_sft, -1)
        label_sft = build_match_sen_shift_label(shift_win, dia_lengths,
                                                label_sen)
        loss_sft = loss_function_shift(prob_sft, label_sft)
        beta = kd.get('sft_soft', 0.0)
        if teacher is not None and beta > 0:
            # p7: blend the hard 0/1 shift label with the teacher's soft "sentiments differ" target.
            p_t = F.softmax(t_sen / kd.get('sft_tau', 1.0), -1)
            soft = build_soft_shift_target(shift_win, dia_lengths, p_t, kd.get('sft_target', 'differ'))
            y = (1 - beta) * label_sft.float() + beta * soft
            loss_sft = -(y * prob_sft[:, 1] + (1 - y) * prob_sft[:, 0]).mean()
        rel_loss = None
        if teacher is not None and kd.get('rel_kd', 0.0) > 0:
            # p7: similarity-preserving KD on H, per dialogue: match cos-sim matrices to the teacher's.
            hs, ht = F.normalize(extracted_feature, dim=-1), F.normalize(t_H, dim=-1)
            terms, a = [], 0
            for n in dia_lengths:
                terms.append(((hs[a:a + n] @ hs[a:a + n].t() - ht[a:a + n] @ ht[a:a + n].t()) ** 2).mean())
                a += n
            rel_loss = kd['rel_kd'] * torch.stack(terms).mean()

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

        preds_emo.append(torch.argmax(prob_emo, 1).cpu().numpy())
        labels_emo.append(label_emo.cpu().numpy())
        preds_sen.append(torch.argmax(prob_sen, 1).cpu().numpy())
        labels_sen.append(label_sen.cpu().numpy())
        preds_sft.append(torch.argmax(prob_sft, 1).cpu().numpy())
        labels_sft.append(label_sft.cpu().numpy())
        if train and sd_loss is not None:
            loss = loss + sd_loss
        if train and rel_loss is not None:
            loss = loss + rel_loss
        if 'exit_logits' in extras:
            if train and kd.get('depth_ce', 0.0) > 0:
                ce = sum(loss_function_emo(F.log_softmax(z, -1), label_emo)
                         for z in extras['exit_logits']) / len(extras['exit_logits'])
                loss = loss + kd['depth_ce'] * ce
            if not train:
                exit_preds.append(extras['exit_logits'].argmax(-1).cpu().numpy())  # (K, N)
        losses.append(loss.item())

        if train:
            loss.backward()
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
        if stats is not None and exit_preds:
            ep = np.concatenate(exit_preds, axis=1)
            stats['exit_f1'] = [round(f1_score(labels_emo, e, average='weighted') * 100, 2) for e in ep]
            stats['exit_acc'] = [round(accuracy_score(labels_emo, e) * 100, 2) for e in ep]

    vids += data[-1]
    labels_emo = np.array(labels_emo)
    preds_emo = np.array(preds_emo)
    labels_sen = np.array(labels_sen)
    preds_sen = np.array(preds_sen)
    labels_sft = np.array(labels_sft)
    preds_sft = np.array(preds_sft)
    vids = np.array(vids)

    extracted_feats = np.array(extracted_feats)

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


def dump_teacher_outputs(model, dataset, batch_size, cuda):
    """p7: eval-mode sentiment/emotion logits and H for every dialogue in `dataset`, keyed by vid.

    Called by the teacher (baseline) run at its valid-selected epoch. Edge-weight parameters are
    restored afterwards, so a forward over unseen batch shapes cannot change the training run.
    """
    from torch.utils.data import DataLoader
    net = model.module if hasattr(model, 'module') else model
    convs = (net.hetergconv_tv, net.hetergconv_ta, net.hetergconv_va)
    saved = [c.edge_weight for c in convs]
    was_training = model.training
    model.eval()
    out = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    with torch.no_grad():
        for data in loader:
            textf0, textf1, textf2, textf3, visuf, acouf, qmask, umask, _, _ = (
                [d.cuda() for d in data[:-1]] if cuda else data[:-1])
            lens = [(umask[:, j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(umask.size(1))]
            o = model(textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask, lens)
            a = 0
            for v, n in zip(data[-1], lens):
                out[v] = {'emo': o[0][a:a + n].cpu().numpy(), 'sen': o[1][a:a + n].cpu().numpy(),
                          'H': o[3][a:a + n].cpu().numpy().astype(np.float16)}
                a += n
    for c, w in zip(convs, saved):
        c.edge_weight = w
    model.train(was_training)
    return out
