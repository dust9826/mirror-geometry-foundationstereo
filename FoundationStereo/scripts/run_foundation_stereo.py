"""
run_foundation_stereo.py — FoundationStereo 추론 + prior 텐서 추출 (헤드리스).

repo/scripts/run_demo.py 를 로컬 실험용으로 감싼 버전.
  - Open3D GUI 시각화 제거(헤드리스에서 죽는 부분) -> ply/npy 만 저장
  - disparity/depth/points/valid 저장
  - forward hook 으로 FoundationStereo "prior" 중간 텐서를 추출:
      * features_left[0]      : side-tuning feature (prior 의 핵심 2D 특징)
      * corr_stem 출력        : AHCF '이전' cost volume
      * cost_agg 출력         : AHCF '이후' cost volume
    두 cost volume 에 동일한 model.classifier 를 적용해
    disparity 분포의 confidence/entropy/argmax 를 비교 가능하게 만든다
    (제안서 RQ: "AHCF 단계 전후 특징 비교로 거울 마스크 신호가 생기는가").
  - cost-volume softmax confidence/entropy 는 거울 영역 신호 후보(거울은
    photometric consistency 를 깨므로 matching 이 불안정 -> 낮은 conf/높은 entropy 기대).

대량 처리용으로 load_model() / infer_scene() 으로 분리 — run_fs_server.py 가
모델을 한 번 로드해 여러 scene 을 연속 추론한다(모델 상주).

사용(단일):
  python scripts/run_foundation_stereo.py --scene cozy_living_room_baseline080 \
      --ckpt pretrained_models/23-51-11/model_best_bp2.pth
"""

import argparse
import os
import sys

import numpy as np

import common as C

sys.path.insert(0, C.REPO_DIR)


def _to_rgb(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3]


def softmax_stats(logits_bdhw):
    """(B,D,h,w) logits -> (conf,entropy,argdisp) 각 (h,w) numpy (B=1 가정)."""
    import torch
    import torch.nn.functional as F
    p = F.softmax(logits_bdhw.float(), dim=1)
    conf = p.max(dim=1).values
    entropy = -(p * (p + 1e-9).log()).sum(dim=1)
    argdisp = p.argmax(dim=1).float()
    return (conf[0].cpu().numpy(), entropy[0].cpu().numpy(), argdisp[0].cpu().numpy())


def resolve_ckpt(ckpt=None):
    if ckpt is not None:
        return ckpt
    import glob
    cands = glob.glob(os.path.join(C.PRETRAINED_DIR, "**", "*.pth"), recursive=True)
    cands = [c for c in cands if "best" in os.path.basename(c).lower()] or cands
    if not cands:
        raise SystemExit("체크포인트(.pth)를 찾지 못함. setup_local.ps1 로 모델 다운로드.")
    return cands[0]


def load_model(ckpt=None, valid_iters=32, hiera=0):
    """모델 1회 로드. 반환 (model, cfg, ckpt_path) — infer_scene 에 넘김."""
    import torch
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo

    ckpt_path = resolve_ckpt(ckpt)
    print("ckpt:", ckpt_path)
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"cfg.yaml 없음: {cfg_path} (체크포인트 폴더째로 받아야 함)")
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = valid_iters
    cfg["hiera"] = hiera
    cfg = OmegaConf.create(cfg)

    torch.autograd.set_grad_enabled(False)
    model = FoundationStereo(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"ckpt global_step:{ck.get('global_step')}, epoch:{ck.get('epoch')}")
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    return model, cfg, ckpt_path


def infer_scene(model, scene, out_root=C.EXPERIMENTS_DIR,
                valid_iters=32, scale=1.0, z_far=10.0, save_features=1):
    """input/ 이 준비된 scene 하나를 추론해 fs/ 산출물 저장. 끝나면 fs/.done 기록."""
    import imageio.v2 as imageio
    import cv2
    import torch
    from core.utils.utils import InputPadder

    run_dir = os.path.join(out_root, scene)
    in_dir = os.path.join(run_dir, "input")
    fs_dir = C.ensure_dir(os.path.join(run_dir, "fs"))
    if not os.path.isdir(in_dir):
        raise SystemExit(f"input 없음. 먼저: python scripts/prepare_data.py --scene {scene}")

    img0 = _to_rgb(imageio.imread(os.path.join(in_dir, "left.png")))
    img1 = _to_rgb(imageio.imread(os.path.join(in_dir, "right.png")))
    if scale != 1.0:
        img0 = cv2.resize(img0, None, fx=scale, fy=scale)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()

    t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)

    # --- forward hooks: prior 중간 텐서 캡처 ---
    cap = {}
    handles = []
    if save_features:
        def mk(name):
            def hook(mod, inp, out):
                cap[name] = out.detach() if torch.is_tensor(out) else out
            return hook
        handles.append(model.feature.register_forward_hook(
            lambda m, i, o: cap.__setitem__("feature_out", o)))
        handles.append(model.corr_stem.register_forward_hook(mk("cost_pre")))
        handles.append(model.cost_agg.register_forward_hook(mk("cost_post")))

    with torch.cuda.amp.autocast(True):
        disp = model.forward(t0, t1, iters=valid_iters, test_mode=True)
    for h in handles:
        h.remove()
    disp = padder.unpad(disp.float()).cpu().numpy().reshape(H, W)

    # disparity 시각화
    from Utils import vis_disparity
    vis = np.concatenate([img0_ori, vis_disparity(disp)], axis=1)
    imageio.imwrite(os.path.join(fs_dir, "vis_disp.png"), vis.astype(np.uint8))
    np.save(os.path.join(fs_dir, "disp.npy"), disp.astype(np.float32))

    # depth / points (K.txt)
    K, baseline = C.read_K_txt(os.path.join(in_dir, "K.txt"))
    K = K.copy(); K[:2] *= scale
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    us_right = xx - disp
    valid = (disp > 0) & (us_right >= 0)
    disp_safe = np.where(disp > 1e-6, disp, 1e-6)
    depth = K[0, 0] * baseline / disp_safe
    depth[~valid] = 0
    depth[depth < 0] = 0
    # 주의: z_far 클리핑은 point cloud(ply) 시각화에만 적용한다.
    # 거울 반사 영역은 반사된 씬 깊이가 z_far 보다 클 수 있는데, 그 점들이야말로
    # mirror geometry 분석 대상이므로 저장 depth/points 에서는 클리핑하지 않는다.
    points = C.depth_to_points(depth, K)
    np.save(os.path.join(fs_dir, "depth_meter.npy"), depth.astype(np.float32))
    np.save(os.path.join(fs_dir, "points.npy"), points.astype(np.float32))
    np.save(os.path.join(fs_dir, "valid.npy"), (depth > 0))

    # point cloud ply
    try:
        from Utils import toOpen3dCloud
        import open3d as o3d
        m = (depth > 0) & (depth <= z_far)   # ply 는 z_far 로 클리핑(시각화용)
        pcd = toOpen3dCloud(points[m].reshape(-1, 3), img0_ori[m].reshape(-1, 3))
        o3d.io.write_point_cloud(os.path.join(fs_dir, "cloud.ply"), pcd)
    except Exception as e:
        print("ply 저장 건너뜀:", e)

    # --- prior 텐서: AHCF 전/후 + cost confidence ---
    if save_features and "cost_pre" in cap and "cost_post" in cap:
        with torch.cuda.amp.autocast(False):
            pre = model.classifier(cap["cost_pre"].float()).squeeze(1)    # (B,D,h,w)
            post = model.classifier(cap["cost_post"].float()).squeeze(1)
        pre_conf, pre_ent, pre_arg = softmax_stats(pre)
        post_conf, post_ent, post_arg = softmax_stats(post)

        feat = cap.get("feature_out")
        feat_left = None
        if isinstance(feat, (list, tuple)) and len(feat) >= 1:
            out = feat[0]
            fl = (out[0] if isinstance(out, (list, tuple)) else out)  # features list[0]
            # feature() returns (out_list, vit_feat); out_list[0] = (2B,C,h,w)
            try:
                feat_left = fl[:1].float()[0].cpu().numpy().astype(np.float16)
            except Exception:
                feat_left = None

        npz = dict(
            pre_conf=pre_conf.astype(np.float32), pre_entropy=pre_ent.astype(np.float32),
            pre_argdisp=pre_arg.astype(np.float32),
            post_conf=post_conf.astype(np.float32), post_entropy=post_ent.astype(np.float32),
            post_argdisp=post_arg.astype(np.float32),
            scale_to_full=np.float32(H / pre_conf.shape[0]),
        )
        if feat_left is not None:
            npz["feat_left"] = feat_left
        np.savez_compressed(os.path.join(fs_dir, "features.npz"), **npz)

        # 전체 해상도 conf/entropy (post = 모델 실제 init 분포) 저장
        full_conf = cv2.resize(post_conf, (W, H), interpolation=cv2.INTER_LINEAR)
        full_ent = cv2.resize(post_ent, (W, H), interpolation=cv2.INTER_LINEAR)
        np.save(os.path.join(fs_dir, "conf.npy"), full_conf.astype(np.float32))
        np.save(os.path.join(fs_dir, "entropy.npy"), full_ent.astype(np.float32))
        print("features.npz 저장:", {k: getattr(v, "shape", v) for k, v in npz.items()})

    cap.clear()
    # 모든 산출물 완료 표시(후처리 워커의 준비완료 마커 — 부분쓰기 레이스 방지)
    with open(os.path.join(fs_dir, ".done"), "w") as f:
        f.write("ok")
    print("DONE:", fs_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="experiments/<scene> (prepare_data 먼저)")
    ap.add_argument("--ckpt", default=None, help="model_best_bp2.pth 경로")
    ap.add_argument("--valid_iters", type=int, default=32)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--z_far", type=float, default=10.0)
    ap.add_argument("--hiera", type=int, default=0)
    ap.add_argument("--save_features", type=int, default=1)
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    args = ap.parse_args()

    model, _cfg, _ = load_model(args.ckpt, valid_iters=args.valid_iters, hiera=args.hiera)
    infer_scene(model, args.scene, out_root=args.out_root,
                valid_iters=args.valid_iters, scale=args.scale,
                z_far=args.z_far, save_features=args.save_features)


if __name__ == "__main__":
    main()
