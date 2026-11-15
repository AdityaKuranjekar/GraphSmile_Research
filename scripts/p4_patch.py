"""Apply the p4 KD flags to trainer.py and run.py (all default off). Run once in the repo root."""
import re
t = open('trainer.py').read()
t = t.replace("    shift_win=5,\n):\n    losses,", "    shift_win=5,\n    collect=None,\n    teacher=None,\n    kd_alpha=0.0,\n    kd_tau=2.0,\n):\n    losses,", 1)
t = t.replace("""        prob_emo = F.log_softmax(logit_emo, -1)
""", """        if collect is not None:  # p4: store per-dialogue emotion logits keyed by dialogue id
            off = 0
            for vid_, n_ in zip(data[-1], dia_lengths):
                collect[vid_] = (logit_emo[off:off + n_].detach().cpu(),
                                 label_emo[off:off + n_].detach().cpu())
                off += n_

        prob_emo = F.log_softmax(logit_emo, -1)
""", 1)
t = t.replace("""        preds_emo.append(torch.argmax(prob_emo, 1).cpu().numpy())
""", """        if teacher is not None and train and kd_alpha > 0:  # p4: KD from a stored teacher
            tl, tlab = [], []
            for vid_, n_ in zip(data[-1], dia_lengths):
                a_, b_ = teacher[vid_]
                tl.append(a_[:n_]); tlab.append(b_[:n_])
            tl = torch.cat(tl).to(logit_emo.device)
            assert (torch.cat(tlab).to(label_emo.device) == label_emo).all(), 'teacher/label misalignment'
            kd = F.kl_div(F.log_softmax(logit_emo / kd_tau, -1),
                          F.softmax(tl / kd_tau, -1), reduction='batchmean') * kd_tau * kd_tau
            loss = loss + kd_alpha * kd

        preds_emo.append(torch.argmax(prob_emo, 1).cpu().numpy())
""", 1)
open('trainer.py', 'w').write(t)

r = open('run.py').read()
r = r.replace("args = parser.parse_args()", """parser.add_argument('--save_teacher_logits', default='',
                    help='p4: write train-dialogue emotion logits at the best-valid epoch to this file')
parser.add_argument('--teacher_logits', default='', help='p4: teacher logits file for KD')
parser.add_argument('--kd_alpha', type=float, default=0.0, help='p4: KD weight (0 = off)')
parser.add_argument('--kd_tau', type=float, default=2.0, help='p4: KD temperature')

args = parser.parse_args()""", 1)
# teacher load + snapshot
r = r.replace("    for epoch in range(n_epochs):\n        setup_samplers", """    teacher = torch.load(args.teacher_logits) if args.teacher_logits else None
    snap = None
    for epoch in range(n_epochs):
        setup_samplers""", 1)
# pass teacher to the train call: first call has 'True,' after optimizer
r = r.replace("""            args.classify,
            args.shift_win,
        )

        valid_loss,""", """            args.classify,
            args.shift_win,
            teacher=teacher,
            kd_alpha=args.kd_alpha,
            kd_tau=args.kd_tau,
        )

        valid_loss,""", 1)
r = r.replace("""                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen
            elif""", """                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen
                    if args.save_teacher_logits:
                        snap = {}
                        train_or_eval_model(model, loss_function_emo, loss_function_sen, loss_function_shift,
                                            train_loader, epoch, cuda, args.modals, None, False, args.dataset,
                                            args.loss_type, args.lambd, args.epochs, args.classify,
                                            args.shift_win, collect=snap)
            elif""", 1)
r = r.replace("""    if local_rank == 0:
        print("Test performance..")""", """    if local_rank == 0 and args.save_teacher_logits and snap is not None:
        torch.save(snap, args.save_teacher_logits)
        print('saved teacher logits for {} dialogues to {}'.format(len(snap), args.save_teacher_logits))
    if local_rank == 0:
        print("Test performance..")""", 1)
open('run.py', 'w').write(r)
print('ok')
