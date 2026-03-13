import numpy as np
import librosa


def compute_period_stability_index(
    audio_path: str,
    sr: int = 16000,
    fmin: float = 80.0,
    fmax: float = 400.0,
    frame_length: int = 2048,
    hop_length: int = 256,
    natural_center: float = 0.015,
    natural_sigma: float = 0.008,
):
    """
    计算一个简单的周期稳定度指数 PSI
    返回:
        {
            "mean_delta": ...,
            "std_delta": ...,
            "smooth_score": ...,
            "natural_score": ...,
            "valid_f0_count": ...,
        }
    """

    # 1) 读音频
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # 2) 提取基频 (YIN)
    f0 = librosa.yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # 3) 去掉无效值
    f0 = np.asarray(f0, dtype=np.float64)
    valid_mask = np.isfinite(f0) & (f0 > 0)
    f0_valid = f0[valid_mask]

    if len(f0_valid) < 3:
        return {
            "mean_delta": None,
            "std_delta": None,
            "smooth_score": 0.0,
            "natural_score": 0.0,
            "valid_f0_count": int(len(f0_valid)),
        }

    # 4) 转周期
    period = 1.0 / f0_valid

    # 5) 相邻周期变化率
    eps = 1e-8
    delta = np.abs(np.diff(period)) / (period[:-1] + eps)

    mean_delta = float(np.mean(delta))
    std_delta = float(np.std(delta))

    # 6) 一个“越平滑越高”的分数
    # mean_delta 越小，smooth_score 越高
    smooth_score = float(100.0 * np.exp(-25.0 * mean_delta))
    smooth_score = max(0.0, min(100.0, smooth_score))

    # 7) 一个“接近自然波动区间越高”的分数
    natural_score = float(
        100.0 * np.exp(-((mean_delta - natural_center) ** 2) / (2.0 * natural_sigma ** 2))
    )
    natural_score = max(0.0, min(100.0, natural_score))

    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "smooth_score": smooth_score,
        "natural_score": natural_score,
        "valid_f0_count": int(len(f0_valid)),
    }


if __name__ == "__main__":
    result = compute_period_stability_index("test.wav")
    print(result)
