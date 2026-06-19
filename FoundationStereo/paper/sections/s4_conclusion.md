# 6. 결론

본 논문은 사전학습 스테레오 모델 FoundationStereo를 재학습하지 않고, 그 내부 prior만으로 스테레오 한 쌍에서 거울 기하의 세 요소(경계, 법선, 깊이)를 복원할 수 있음을 정량적으로 입증하였다. 먼저 거울이 스테레오 정합을 체계적으로 깨뜨림을 GT 대비로 확인하였다: 거울 영역의 disparity MAE는 비거울 대비 약 41배(0.21 px → 8.78 px), depth MAE는 약 17배(0.151 m → 2.59 m)이며, bad@1px는 2.6%에서 98.4%로 치솟는다(표 1). 이 불안정성에서 비롯되는 단일 cost-volume 신호는 그 자체로는 약하지만(AHCF 이후 entropy AUC 중앙값 0.587), AHCF 통과 후 일관되게 상승하며(0.545 → 0.587), 강한 분별력은 동결 특징과 cost 통계를 묶은 136차원 prior의 학습된 조합이 담당한다. 그 위에 얹은 경량 MLP 헤드(136→64→1)는 한 번도 본 적 없는 장면에서 AUC 0.928(IoU 0.492, F1 0.642)을 달성하고, 장면당 기준뷰 1장(총 12장)만으로 학습해도 1,073뷰 학습과 동급(AUC 0.955 vs. 0.958)으로 일반화되어, 거울 단서가 장면·시점에 거의 불변인 prior 수준에서 형성됨을 보였다(표 2). 마지막으로, 거울 안 점군의 단순 평면 적합이 거울면이 아닌 반사 장면의 면을 잡는 원리적 한계(법선각 중앙값 87.5°, 거리오차 중앙값 2.52 m, 성공률 0.1%)를 학습이 전혀 필요 없는 반사 대칭 RANSAC으로 교정하여, 1,536뷰에서 법선각 중앙값 2.3°(<10°: 92%), 평면 수직거리 오차 중앙값 0.137 m, 동시 충족(<10° & <0.5 m) 성공률 71.7%를 달성하였다(표 3, 그림 3). 예측 마스크만으로 동작하는 완전 end-to-end 파이프라인도 예비 평가(n=15)에서 법선각 중앙값 1.50°, 거리오차 중앙값 0.243 m, 성공률 67%를 보여, 무거운 분할 네트워크 없이도 FoundationStereo prior 안에 거울 기하가 충분히 들어 있다는 본 논문의 가설을 뒷받침한다. 본 결과는 거울 반사를 포함한 3D 재구성과 로봇 주행에서 추가 학습 비용 없이 활용 가능한 경량 거울 prior를 제공한다.

**한계.** 첫째, 평가가 Reflect3r 기반 합성 실내 장면(평면 거울 중심)에 한정되어 실영상 검증은 향후 과제다. 둘째, 평면 복원은 거울이 비추는 영역과 카메라 관측 영역의 겹침이 작을 때 거리 추정이 미정이 되는 퇴화($d$-슬라이딩)에 취약하여, 12씬 중 2씬(scandinavian 2%, green_house 40%)에서 성공률이 무너진다 — 마스크 경계 링 제약, 가시성 인지 점수, 멀티뷰 합의가 직접적 개선 후보다. 셋째, end-to-end(예측 마스크) 평가는 3씬 × 5뷰의 파일럿 규모(n=15)로, 전수 평가가 필요하다. 넷째, 검출 IoU는 임계 0.5 고정 기준이며, 실패 사례 다수가 AUC는 높은데(>0.92) 임계에서 무너지는 캘리브레이션 문제이므로 적응 임계로 개선 여지가 있다.

# References

[1] X. Yang, H. Mei, K. Xu, X. Wei, B. Yin, and R. W. H. Lau, "Where is my mirror?," in *IEEE International Conference on Computer Vision (ICCV)*, 2019.

[2] J. Lin, G. Wang, and R. W. H. Lau, "Progressive mirror detection," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

[3] J. Liu, X. Tang, F. Cheng, R. Yang, Z. Li, J. Liu, Y. Huang, J. Lin, S. Liu, X. Wu, S. Xu, and C. Yuan, "MirrorGaussian: Reflecting 3D gaussians for reconstructing mirror reflections," in *European Conference on Computer Vision (ECCV)*, 2024.

[4] J. Wu et al., "Reflect3r: Single-view 3D stereo reconstruction aided by mirror reflections," in *International Conference on 3D Vision (3DV)*, 2026. arXiv:2509.20607.

[5] B. Wen, M. Trepte, J. Aribido, J. Kautz, O. Gallo, and S. Birchfield, "FoundationStereo: Zero-shot stereo matching," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025.

[6] L. Lipson, Z. Teed, and J. Deng, "RAFT-Stereo: Multilevel recurrent field transforms for stereo matching," in *International Conference on 3D Vision (3DV)*, 2021.

[7] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, "Depth anything V2," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.
