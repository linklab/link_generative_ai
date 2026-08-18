"""
VAE(a_vae), DDPM(b_ddpm), EBM(a_basic_score_based_model) 예시에 공통으로 쓰이는
가상의 실제 데이터(p_data) 생성 코드

세 모델이 완전히 동일한 two-moons 데이터를 사용하도록 이 모듈을 공유합니다
(VAE, DDPM, EBM 세 모델의 결과를 공정하게 비교하기 위함).
"""

import numpy as np
import torch
from sklearn.datasets import make_moons

N_DATA = 3000  # 가상의 실제 데이터(two-moons) 샘플 수


def load_data(n_data=N_DATA, device="cpu"):
    """가상의 실제 데이터: two-moons (p_data)"""
    x0_data, _ = make_moons(n_samples=n_data, noise=0.05, random_state=0)
    x0_data = x0_data.astype(np.float32)
    # 학습을 안정시키기 위해 평균 0, 표준편차 1 근처로 정규화
    x0_data = (x0_data - x0_data.mean(axis=0)) / x0_data.std(axis=0)
    return torch.tensor(x0_data, device=device)
