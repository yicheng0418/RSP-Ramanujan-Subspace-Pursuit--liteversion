import math
import numpy as np
import soundfile as sf
from math import gcd


def ramanujan_sum_sequence(q: int) -> np.ndarray:
    """
    生成一个长度为 q 的 Ramanujan sum 基序列 c_q(n), n=0..q-1
    使用定义:
        c_q(n) = sum_{1<=a<=q, gcd(a,q)=1} exp(2πi a n / q)
    结果理论上应为整数，这里取实部即可。
    """
    n = np.arange(q)
    seq = np.zeros(q, dtype=np.complex128)
    for a in range(1, q + 1):
        if gcd(a, q) == 1:
            seq += np.exp(2j * np.pi * a * n / q)
    return np.real(seq)


def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(x) < frame_length:
        pad = frame_length - len(x)
        x = np.pad(x, (0, pad))
    num_frames = 1 + (len(x) - frame_length) // hop_length
    frames = np.stack([
        x[i * hop_length:i * hop_length + frame_length]
        for i in range(num_frames)
    ])
    return frames


def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.mean(x)
    maxv = np.max(np.abs(x)) + 1e-12
    return x / maxv


def ramanujan_periodogram_for_frame(
    frame: np.ndarray,
    q_min: int,
    q_max: int,
    cache: dict
):
    """
    对单帧计算候选周期 q 的响应分数 R(q)
    """
    scores = []
    N = len(frame)
    for q in range(q_min, q_max + 1):
        if q not in cache:
            cache[q] = ramanujan_sum_sequence(q)
        cq = cache[q]

        # 扩展到帧长度
        tiled = np.resize(cq, N)

        # 简单投影分数
        score = np.abs(np.dot(frame, tiled)) ** 2
        scores.append(score)

    scores = np.array(scores, dtype=np.float64)
    q_values = np.arange(q_min, q_max + 1)
    return q_values, scores


def extract_ramanujan_track(
    audio_path: str,
    target_sr: int = 16000,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    fmin: float = 80.0,
    fmax: float = 400.0
):
    x, sr = sf.read(audio_path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    # 这里先假设输入已经是 16k；如果不是，可加 librosa/resampy 重采样
    if sr != target_sr:
        raise ValueError(f"Please resample to {target_sr} Hz first. Current sr={sr}")

    x = normalize_audio(x)

    frame_length = int(target_sr * frame_ms / 1000)
    hop_length = int(target_sr * hop_ms / 1000)
    frames = frame_signal(x, frame_length, hop_length)

    q_min = int(target_sr / fmax)
    q_max = int(target_sr / fmin)

    cache = {}
    dominant_q = []
    dominant_score = []
    second_score = []
    all_scores = []

    # 可加窗，减少边缘影响
    window = np.hanning(frame_length)

    for frame in frames:
        frame = frame * window
        q_values, scores = ramanujan_periodogram_for_frame(frame, q_min, q_max, cache)

        idx_sorted = np.argsort(scores)[::-1]
        idx1 = idx_sorted[0]
        idx2 = idx_sorted[1] if len(idx_sorted) > 1 else idx_sorted[0]

        dominant_q.append(q_values[idx1])
        dominant_score.append(scores[idx1])
        second_score.append(scores[idx2])
        all_scores.append(scores)

    dominant_q = np.array(dominant_q, dtype=np.float64)
    dominant_score = np.array(dominant_score, dtype=np.float64)
    second_score = np.array(second_score, dtype=np.float64)
    all_scores = np.vstack(all_scores)

    dominant_f0 = target_sr / dominant_q

    return {
        "dominant_q": dominant_q,
        "dominant_f0": dominant_f0,
        "dominant_score": dominant_score,
        "second_score": second_score,
        "q_min": q_min,
        "q_max": q_max,
        "all_scores": all_scores,
    }


def compute_ramanujan_psi(track: dict):
    q = track["dominant_q"]
    s1 = track["dominant_score"]
    s2 = track["second_score"]

    if len(q) < 3:
        return {
            "psi_ram": 0.0,
            "mean_delta_q": None,
            "mean_peak_ratio": None,
            "entropy_q": None,
        }

    eps = 1e-12

    # A. 主周期变化率
    delta_q = np.abs(np.diff(q)) / (q[:-1] + eps)
    mean_delta_q = float(np.mean(delta_q))

    # B. 峰值比
    peak_ratio = s1 / (s2 + eps)
    mean_peak_ratio = float(np.mean(peak_ratio))

    # C. 主周期分布熵
    q_int = q.astype(int)
    values, counts = np.unique(q_int, return_counts=True)
    p = counts / counts.sum()
    entropy_q = float(-np.sum(p * np.log(p + eps)))

    # 把三个量压到 0-100
    # 这些参数先经验设定，后续必须靠真人/AI样本校准
    score_delta = 100.0 * np.exp(-((mean_delta_q - 0.015) ** 2) / (2 * 0.01 ** 2))
    score_peak = 100.0 * np.exp(-((mean_peak_ratio - 2.5) ** 2) / (2 * 1.2 ** 2))
    score_entropy = 100.0 * np.exp(-((entropy_q - 1.2) ** 2) / (2 * 0.8 ** 2))

    psi_ram = 0.4 * score_delta + 0.3 * score_peak + 0.3 * score_entropy

    return {
        "psi_ram": float(np.clip(psi_ram, 0, 100)),
        "mean_delta_q": mean_delta_q,
        "mean_peak_ratio": mean_peak_ratio,
        "entropy_q": entropy_q,
        "score_delta": float(np.clip(score_delta, 0, 100)),
        "score_peak": float(np.clip(score_peak, 0, 100)),
        "score_entropy": float(np.clip(score_entropy, 0, 100)),
    }


if __name__ == "__main__":
    track = extract_ramanujan_track("test_16k.wav")
    result = compute_ramanujan_psi(track)
    print(result)
