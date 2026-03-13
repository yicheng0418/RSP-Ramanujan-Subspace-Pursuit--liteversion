# RSP-Ramanujan-Subspace-Pursuit--liteversion
这是一个基于拉马努金周期结构检测算法的声纹检测小工具

这版程序跑出来后，该怎么看结果

不要一开始就问“准不准”。
先问三个更重要的问题：

1. Ramanujan periodogram 的主峰是不是稳定可解释,也就是 dominant_q 有没有跟语音听感一致。
如果看见：
清音、噪声段乱跳、有声音段比较集中
那是正常的。

3. 真人和 AI 的 dominant_q 轨迹有没有形态差异
重点看：
真人是否呈现“稳定 + 轻微漂移”
AI 是否更容易“过平滑”或“模板化台阶变化”

4. mean_peak_ratio 和 entropy_q 是否能拉开分布，这个很关键。
只要真/假样本在这两个维度上开始有统计分离，这条路就值得继续。

要彻底验证它有没有用，实验必须这么做，这是最重要的部分。
至少要做一个小实验矩阵：
数据分三组
1、真人原始录音
2、AI TTS 合成音
3、AI voice clone 音
每组至少 30–50 段短音频，每段 5–15 秒就够。

对每段提特征，至少提这些：
psi_ram
mean_delta_q
mean_peak_ratio
entropy_q

然后和传统方法对比，最少对比：
自相关法 pitch stability
YIN 得到的 F0 stability
MFCC-based 简单分类
要看的不是 Ramanujan 单独神不神，而是它有没有增量信息。
研究里也正是把它当成“periodicity analysis / decomposition”的一种不同工具，而不是万能替代品。
