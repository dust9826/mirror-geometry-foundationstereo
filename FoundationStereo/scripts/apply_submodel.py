"""
apply_submodel.py — 범용 거울검출 head 를 "한 번 학습해 저장" + "새 데이터에 불러와 적용".

submodel_head.py 는 매번 재학습(단일/LOSO)하는 구조라, 여기서는 제안서 접근 #3 을
실제 배포 흐름처럼 둘로 분리한다:
  (1) train : 여러 scene 으로 head 하나를 학습해 model.pt 로 저장 (held-out 없음)
  (2) apply : 저장된 model.pt 를 불러와 임의의 새 scene 에 적용
              (GT mask 가 있으면 평가지표도 출력)

사용:
  # 1) stereo_dataset 12개로 범용 모델 학습 -> experiments/_model/general_head.pt
  python scripts/apply_submodel.py train --train_scenes a,b,c,... --model_out experiments/_model/general_head.pt

  # 2) 학습에 안 쓴 새 데이터(minigym_view04)에 적용/평가
  python scripts/apply_submodel.py apply --load experiments/_model/general_head.pt --scene minigym_view04
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import submodel_head as SM


def cmd_train(args, device):
    import torch
    scenes = [s.strip() for s in args.train_scenes.split(",") if s.strip()]
    Xs, ys, F, names = [], [], None, None
    used = []
    for s in scenes:
        d = SM.build_features(s, out_root=args.out_root)
        if d is None:
            continue
        if F is None:
            F, names = d["X"].shape[1], d["feature_names"]
        if d["X"].shape[1] != F:
            print(f"[skip] {s}: feature 차원 {d['X'].shape[1]} != {F}")
            continue
        Xs.append(d["X"]); ys.append(d["y"]); used.append(s)
    if not Xs:
        raise SystemExit("학습 가능한 scene 이 없음")
    X = np.concatenate(Xs); y = np.concatenate(ys)
    mean, std = SM.standardize_fit(X)
    Xn = SM.standardize_apply(X, mean, std)
    print(f"[train] {len(used)} scene, pixels={X.shape[0]}, feat_dim={F}, pos={int(y.sum())}")
    head = SM.train_head(Xn, y, F, linear=args.linear, hidden=args.hidden,
                         epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    out = args.model_out
    C.ensure_dir(os.path.dirname(out) or ".")
    torch.save(dict(state_dict=head.state_dict(), mean=mean, std=std,
                    linear=bool(args.linear), hidden=int(args.hidden),
                    in_dim=int(F), feature_names=names, train_scenes=used),
               out)
    print(f"[save] 범용 모델 -> {out}  (학습 scene {len(used)}개: {used})")


def cmd_apply(args, device):
    import torch
    ck = torch.load(args.load, map_location=device, weights_only=False)
    head = SM.make_head(ck["in_dim"], linear=ck["linear"], hidden=ck["hidden"]).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()
    mean, std = ck["mean"], ck["std"]
    print(f"[load] {args.load}  (train_scenes={ck.get('train_scenes')})")

    d = SM.build_features(args.scene, out_root=args.out_root)
    if d is None:
        raise SystemExit(f"{args.scene}: features.npz/mask 없음 (먼저 run_foundation_stereo)")
    if d["X"].shape[1] != ck["in_dim"]:
        raise SystemExit(f"feature 차원 불일치: 모델 {ck['in_dim']} vs scene {d['X'].shape[1]} "
                         "(feat_left 유무 다름). 동일 설정으로 run_foundation_stereo 필요.")
    Xn = SM.standardize_apply(d["X"], mean, std)
    p = SM.predict_proba(head, Xn, device=device)

    out_dir = args.out or os.path.join(args.out_root, args.scene, "submodel_applied")
    C.ensure_dir(out_dir)
    hw = d["hw"]
    full_hw = SM._full_hw_for(args.scene, hw, args.out_root)
    left = os.path.join(args.out_root, args.scene, "input", "left.png")
    SM.save_pred_mask(os.path.join(out_dir, "pred_mask.png"), p, hw, full_hw)
    SM.save_panel(os.path.join(out_dir, "panel.png"), args.scene, p, d["y"], hw, left)

    has_gt = bool(np.asarray(d["y"]).sum() > 0)
    metrics = SM._eval_block(d["y"], p) if has_gt else {}
    if has_gt:
        print(f"\n[APPLY] {args.scene}  (저장모델로 새 데이터 평가)")
        SM._print_metrics(metrics)
    else:
        print(f"\n[APPLY] {args.scene}: GT mask 없음 -> 예측만 저장")
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(metrics=metrics, model=os.path.abspath(args.load),
                       train_scenes=ck.get("train_scenes"), scene=args.scene,
                       has_gt=has_gt), f, indent=2, ensure_ascii=False)
    print(f"[save] {out_dir}  (pred_mask.png, panel.png, metrics.json)")


def main():
    ap = argparse.ArgumentParser(description="범용 거울검출 head 학습/적용 (제안서 #3 배포형)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="여러 scene 으로 head 하나 학습+저장")
    t.add_argument("--train_scenes", required=True, help="쉼표목록")
    t.add_argument("--model_out", default=os.path.join(C.EXPERIMENTS_DIR, "_model", "general_head.pt"))

    a = sub.add_parser("apply", help="저장 모델을 새 scene 에 적용/평가")
    a.add_argument("--load", required=True)
    a.add_argument("--scene", required=True)
    a.add_argument("--out", default=None)

    for p in (t, a):
        p.add_argument("--linear", action="store_true")
        p.add_argument("--hidden", type=int, default=64)
        p.add_argument("--epochs", type=int, default=300)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
        p.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[device] {device}")
    (cmd_train if args.cmd == "train" else cmd_apply)(args, device)


if __name__ == "__main__":
    main()
