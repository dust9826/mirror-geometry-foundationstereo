"""run_pipeline_full.py — 입력(input/)이 준비된 scene 하나에 전체 파이프라인 실행.

  run_foundation_stereo -> estimate_mirror_plane -> viz_ransac_inliers
  -> analyze_features -> submodel_head -> apply_submodel(apply, _model/general_head.pt)

→ minigym_view04 와 동일한 구성 생성:
   fs/ , mirror_geometry/(+ransac_inliers.png) , features/ , submodel/ , submodel_applied/

입력은 미리 준비:
  - Archive(ds_v1):  python scripts/prepare_views_archive.py --scene blue_bathroom --view v00 --name archive_views/blue_bathroom_v00
  - 기존 dataset:    python scripts/prepare_data.py --scene <name>

실행:
  python scripts/run_pipeline_full.py --scene archive_views/blue_bathroom_v00
  python scripts/run_pipeline_full.py --scene legacy/computer_room_baseline080 --no_apply
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PY = sys.executable
S = C.SCRIPT_DIR


class FileLock:
    """다중 워커 병렬 실행 시 VRAM 큰 단계(FS 추론)를 직렬화하는 파일락.

    O_CREAT|O_EXCL 스핀락. 워커가 죽어 락이 남으면(stale) mtime 기준으로 회수.
    """

    def __init__(self, path, stale_sec=300):
        self.path, self.stale_sec, self.fd = path, stale_sec, None

    def __enter__(self):
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale_sec:
                        os.unlink(self.path)   # stale 락 회수
                        continue
                except OSError:
                    pass
                time.sleep(1.0)

    def __exit__(self, *a):
        try:
            os.close(self.fd)
            os.unlink(self.path)
        except OSError:
            pass


def step(name, argv, lock_path=None):
    print("\n" + "=" * 60 + f"\nSTEP: {name}\n" + "=" * 60)
    cmd = [PY, os.path.join(S, argv[0])] + argv[1:]
    if lock_path:
        t0 = time.time()
        with FileLock(lock_path):
            wait = time.time() - t0
            if wait > 1:
                print(f"[gpu_lock] {wait:.0f}s 대기 후 획득")
            rc = subprocess.call(cmd)
    else:
        rc = subprocess.call(cmd)
    if rc != 0:
        print(f"!! step '{name}' 실패 (rc={rc})")
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="experiments/<scene> (하위경로 가능)")
    ap.add_argument("--ckpt", default=None, help="FS 체크포인트(.pth) 경로(미지정 시 자동탐색)")
    ap.add_argument("--general_head",
                    default=os.path.join(C.EXPERIMENTS_DIR, "_model", "general_head.pt"),
                    help="submodel_applied 에 쓸 범용 헤드")
    ap.add_argument("--no_apply", action="store_true", help="submodel_applied 단계 생략")
    ap.add_argument("--no_submodel", action="store_true", help="단일 scene submodel 학습 생략")
    ap.add_argument("--gpu_lock", default=None,
                    help="FS 추론을 이 파일락으로 직렬화(다중 워커 병렬용)")
    ap.add_argument("--no_fs", action="store_true",
                    help="FS 추론 생략(run_fs_server 가 미리 fs/ 생성한 경우)")
    args = ap.parse_args()
    sc = args.scene

    if args.no_fs:
        ok = True
    else:
        fs = ["run_foundation_stereo.py", "--scene", sc]
        if args.ckpt:
            fs += ["--ckpt", args.ckpt]
        ok = step("run_foundation_stereo", fs, lock_path=args.gpu_lock)
    ok &= step("estimate_mirror_plane", ["estimate_mirror_plane.py", "--scene", sc])
    ok &= step("viz_ransac_inliers", ["viz_ransac_inliers.py", "--scene", sc])
    # GT 거울평면(있으면): ds_v1 처럼 input/depth_gt.npy 가 있으면 GT depth 역투영으로 생성.
    step("make_mirror_gt_from_depth", ["make_mirror_gt_from_depth.py", "--scene", sc])
    ok &= step("analyze_features", ["analyze_features.py", "--scene", sc])
    if not args.no_submodel:
        ok &= step("submodel_head", ["submodel_head.py", "--scene", sc])

    if args.no_apply:
        pass
    elif os.path.isfile(args.general_head):
        step("apply_submodel", ["apply_submodel.py", "apply",
                                "--load", args.general_head, "--scene", sc])
    else:
        print(f"[warn] general_head 없음 → submodel_applied 생략: {args.general_head}")

    print("\nDONE:", os.path.join(C.EXPERIMENTS_DIR, sc), "(ok)" if ok else "(일부 실패)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
