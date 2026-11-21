# Debug Log — Chạy 3 thuật toán trên cùng kịch bản (17–20/11/2026)

**Mục tiêu**: chạy được flatDRL-MAPPO, Hierarchical (SAC+DQN) multi-BWP và Hierarchical single-BWP trên cùng một kịch bản.
**Kết quả**: cả 3 đều chạy đủ 2000 episode, update được, lưu checkpoint, resume được và vẽ biểu đồ.

```bash
python main.py --algo ppo                          # flatDRL-MAPPO
python main.py --algo hierarchical                 # SAC + DQN, multi-BWP
python main.py --algo hierarchical --single-bwp    # SAC + DQN, single-BWP
```

---

## Tổng quan

```
main.py --algo ppo           → train_ppo.py          → MAPPOTrainer            → PPOISACEnv  + MAPPOAgent
main.py --algo hierarchical  → train_hierarchical.py → HierarchicalISACTrainer → HierISACEnv + SACAgent + DQNAgent
              [--single-bwp]                                                     (bwp_mode=False)

Lỗi theo vị trí:
  MAPPO   (AgentComponents.py, train_ppo.py, mappo_trainer.py) .. #1 #21
  Hier    (hier_trainer.py) ...................................... #2 #3 #4 #5 #6 #7 #8
  Agent   (SACagent.py / DQNagent.py) ............................ #9 #10 #11 #12 #15 #16 #17 #18 #19
  Config  (mappo.yaml / hierarchical.yaml) #13 ; (system.yaml) #14
  Test    (docs/test_agents.py) ................................... #20

Vấn đề của bài toán chưa sửa: B0 (C2S), nhóm B (reward), nhóm C (phát hiện từ kết quả 2000 episode)
```

---

## MAPPO (flatDRL)

### #1 — Trainer chặn training vì Actor còn head công suất
- **File**: `ppo/agents/AgentComponents.py` (class `Actor`)
- **Triệu chứng**: `RuntimeError: The Env uses fixed pEach power. Migrate the PPO Actor/log-probability...` (`mappo_trainer.py`).
- **Nguyên nhân**: env dùng công suất cố định `pEach`, nhưng Actor vẫn sample công suất và cộng log-prob của nó vào ratio PPO → gradient cho một hành động env không hề dùng.
- **Sửa**: bỏ `power_mean`, `power_logstd`. `action_power` giờ là placeholder toàn 1 (giữ nguyên interface với buffer/RuntimeContract); `log_prob` và `entropy` chỉ tính trên PRB.
- **Ảnh hưởng**: checkpoint MAPPO cũ (có `power_mean`) không load được nữa.

### #21 — MAPPO không chạy tiếp được quá `total_episodes` khi resume (20/11/2026)
- **File**: `ppo/train_ppo.py`, `ppo/utils/mappo_trainer.py`
- **Triệu chứng**: CheckpointManager chỉ cho resume khi config giống hệt lúc lưu; MAPPO chỉ đọc số episode từ YAML → muốn chạy thêm phải sửa YAML → resume bị từ chối.
- **Sửa**: thêm `--episodes N` (giống hierarchical), truyền vào `MAPPOTrainer.run(total_episodes=...)`. Kiểm tra: chạy 2 ep rồi resume tới ep 3 → return ep 3 trùng khớp lần chạy liền (278.4136).

---

## Hierarchical — `ppo/utils/hier_trainer.py`

### #2 — Gọi constructor SAC sai API
- **Triệu chứng**: trainer gọi `SACAgent(state_dim=..., num_rus=..., num_bwp_ru=..., train_cons=...)`, agent nhận `SACAgent(config, device)`.
- **Sửa**: trainer lấy `hierarchical_parameters.sac` rồi chèn các chiều đọc từ env (`state_dim`, `num_bs`, `num_bwp`, `num_targets`, `num_slices`).

### #3 — Import sai tên file DQN + gọi constructor sai API
- **Triệu chứng**: `from ppo.agents.DQNAgent import DQNAgent` (file thật là `DQNagent.py`); constructor dùng `state_dim, n_prb, epsilon_start...`.
- **Sửa**: import `ppo.agents.DQNagent`; lấy `hierarchical_parameters.dqn` rồi chèn chiều từ env (`local_obs_dim`, `global_state_dim`, `total_prb`, `num_destinations = 1+M+K`).

### #4 — DQN không dùng state theo spec
- **Triệu chứng**: trainer đưa `get_dqn_state()` (env không có hàm này → rơi về `global_state`) vào DQN; spec yêu cầu `concat(local_obs[b], global_state)`.
- **Sửa**: gọi `dqn_agent.select_action(local_obs, global_state, ...)`; chiều state DQN = 55 + 135 = 190 (sau #16 thêm 18 chiều ngân sách).

### #5 — Action DQN vượt ngân sách PRB do SAC đặt → env từ chối
- **Triệu chứng**: `HierISACEnv.action_mapping` raise `allocation exceeds the current integer PRB budget` khi số PRB gán cho một target/slice vượt quota; env cố ý không tự cắt.
- **Sửa (tạm)**: thêm `_apply_budget()` trong trainer: PRB vượt quota → idle (0). Đã thay bằng cách sửa đúng ở #16.

### #6 — Push replay buffer DQN sai số trường
- **Triệu chứng**: trainer push 5 trường `(s, a, r, s', done)` cho từng BS; buffer DQN cần 7 trường `(local_obs, global_state, action, reward, next_local_obs, next_global_state, done)`.
- **Sửa**: dùng `dqn_agent.store_transition(...)` với đủ trường, 1 transition / subframe (action `[B, 60]`). Sau #16 thêm trường thứ 8 là ngân sách.

### #7 — Gọi `update()` của SAC/DQN sai API
- **Triệu chứng**: trainer gọi `sac_agent.update(step=, policy_delay=, last_actor_loss=, batch_size=)` và nhận về tuple 3 giá trị; gọi `dqn_agent.update(batch_size=)`.
- **Sửa**: gọi `update()` không tham số, đọc dict `sac/actor_loss`, `sac/critic_loss`, `dqn/q_loss`. Bỏ `_enforce_single_bwp` của trainer (trùng với cờ `single_bwp` trong `SACAgent.select_action`).

### #8 — Epsilon DQN không bao giờ giảm
- **Triệu chứng**: không chỗ nào gọi `update_epsilon()` → DQN chọn ngẫu nhiên 100% suốt quá trình train.
- **Sửa**: gọi `dqn_agent.update_epsilon()` cuối mỗi episode (0.995^episode → chạm 0.05 sau khoảng 600 episode).

---

## Agent

### #9 — SAC thiếu thuộc tính mà RuntimeContract/CheckpointManager yêu cầu
- **File**: `ppo/agents/SACagent.py`
- **Triệu chứng**: contract đòi `critic_1_optimizer`, `critic_2_optimizer`; agent chỉ có `critic_optimizer` chung.
- **Sửa**: thêm 2 alias trỏ về `critic_optimizer`.

### #10 — `SACAgent.alpha` chỉ đọc → crash khi resume
- **Triệu chứng**: `CheckpointManager.load` gán `sac_agent.alpha = ...`; `alpha` là property không setter → `AttributeError`.
- **Sửa**: thêm setter rỗng (alpha luôn suy ra từ `log_alpha`, mà `log_alpha` đã được restore).

### #11 — SAC lưu action ở không gian khác với không gian critic học
- **Triệu chứng (sai thầm lặng, không crash)**: `store_transition` lưu ratio ∈ [0,1], nhưng trong update critic đánh giá action của actor ∈ [-1,1] (tanh) → Q(s,a) học trên hai thang đo khác nhau.
- **Sửa**: lưu action ở thang tanh. Ban đầu đổi `a = 2·ratio − 1`; sau #18 lưu thẳng output tanh chưa lọc.

### #12 — DQN: alias + lỗi shape trong `update()`
- **File**: `ppo/agents/DQNagent.py`
- **Triệu chứng**:
  - contract/checkpoint đòi `q_net` và `epsilon`; agent dùng `policy_net`, `eps`.
  - `update()`: reward/done có `batch` hàng nhưng Q-value có `batch × B` hàng → `expand_as` crash khi B = 5.
- **Sửa**: alias `q_net = policy_net`; property `epsilon` (đọc/ghi `eps`); `repeat_interleave(num_bs)` cho reward và done.

### #15 — SAC/DQN crash khi `warmup_steps < batch_size` (20/11/2026)
- **File**: `ppo/agents/SACagent.py`, `ppo/agents/DQNagent.py` (`update()`), `ppo/utils/hier_trainer.py` (điều kiện gọi SAC update)
- **Triệu chứng**: `ValueError: Sample larger than population or is negative` trong `replay_buffer.sample()` khi `warmup_steps` nhỏ hơn `batch_size` (ví dụ SAC `warmup_steps: 5`, `batch_size: 32`).
- **Nguyên nhân**: chỉ kiểm tra `len(buffer) >= warmup_steps`, không kiểm tra đủ `batch_size` mẫu. Config mặc định (warmup 1000/500 > batch 32/64) không gặp lỗi này.
- **Sửa**: điều kiện update thành `len(buffer) >= max(warmup_steps, batch_size)`. Hành vi với config mặc định không đổi.

### #16 — DQN không biết ngân sách SAC → phần lớn PRB bị cắt (20/11/2026)
- **File**: `ppo/agents/DQNagent.py`, `ppo/utils/hier_trainer.py`
- **Triệu chứng (sai thầm lặng, không crash)**: DQN chọn đích cho 60 PRB không biết ngân sách `env.budget [B, W, M+S]`; `_apply_budget()` (#5) cắt phần vượt quota về idle. Đo 5 frame: DQN chọn dùng ~95% PRB nhưng chỉ ~58% (multi-BWP) và ~29% (single-BWP) được dùng thật.
- **Nguyên nhân**: DQN rải đều cho 18 đích (12 UE thuộc 4 slice → mỗi slice ~3/18 PRB), SAC chia ~1/9 mỗi cột → slice luôn vượt quota. State DQN không có ngân sách nên DQN không thể học để khớp, trong khi ngân sách đổi mỗi frame → yêu cầu "DQN dựa trên action của SAC" không được đáp ứng.
- **Sửa**:
  - DQN chỉ chọn đích còn quota (`_assign_within_budget`): greedy — PRB có max-Q lớn nhất chọn trước; khám phá — chọn đều trong các đích hợp lệ. Bỏ `_apply_budget()`; env vẫn kiểm tra lại.
  - State DQN thêm `budget[b] / nPrb[w]` (18 chiều): input 195 → 213.
  - Replay DQN lưu ngân sách của transition; TD target chỉ lấy max trên các đích có quota > 0, dùng cùng ngân sách cho s và s' (quy ước goal-conditioned như h-DQN).
  - `_bwp_of_prb` dựng từ `n_prb_per_bwp` (trước đó giả định mọi BWP cùng số PRB).
- **Kết quả**: PRB thực dùng 87–88% (multi-BWP), 81–85% (single-BWP); phần còn lại là idle do DQN chọn. Env không báo vi phạm.

### #17 — Single-BWP bị lọc hai lần theo hai tiêu chí khác nhau (20/11/2026)
- **File**: `ppo/agents/DQNagent.py` (`select_action`)
- **Triệu chứng**: DQN tự giữ BWP có nhiều PRB nhất cho mỗi đích, rồi ngân sách SAC chỉ cho BWP mà SAC chọn; khi hai lựa chọn khác nhau, mọi PRB của dịch vụ đó bị cắt. Bộ lọc này làm mất thêm ~11 điểm % PRB (29% → 40% khi bỏ nó).
- **Sửa**: bỏ cờ `single_bwp` khỏi DQN. Ngân sách single-BWP bằng 0 ở BWP bị loại nên mask của #16 đã đảm bảo ràng buộc.

### #18 — SAC single-BWP: critic học action đã lọc, actor hỏi action chưa lọc (20/11/2026)
- **File**: `ppo/agents/SACagent.py`, `ppo/utils/hier_trainer.py`
- **Triệu chứng (sai thầm lặng)**: replay lưu ratio đã lọc single-BWP (BWP bị loại → −1 trong thang tanh), nên critic chỉ học trên action đã lọc; khi cập nhật actor, critic được hỏi `Q(s, a_new)` với `a_new` chưa lọc — vùng critic chưa thấy → gradient actor không đáng tin. Multi-BWP không bị (không có bước lọc).
- **Sửa**: `select_action` trả `(ratios, raw_action)`; replay lưu `raw_action` (output tanh chưa lọc), bước lọc coi là một phần của môi trường. Với multi-BWP giá trị lưu không đổi (raw = 2·ratio − 1).

### #19 — Resume hierarchical làm SAC warmup lại 100 episode (20/11/2026)
- **File**: `ppo/agents/SACagent.py` (`SACReplayBuffer.state_dict/load_state_dict`), `ppo/utils/hier_trainer.py`
- **Triệu chứng**: checkpoint không lưu replay; sau `--resume` buffer rỗng, SAC đợi lại 1000 frame (100 episode) mới update, DQN 500 subframe.
- **Sửa**: mỗi lần lưu checkpoint, ghi replay SAC mới nhất vào `checkpoints/<run>/sac_replay.pt` (~5 MB, ghi đè, kèm số episode); khi resume nạp lại nếu khớp episode của checkpoint. Replay DQN (~300 MB) không lưu. Sau resume đặt `warmup_steps = 0` cho cả hai agent (chỉ cần đủ `batch_size`).
- **Lưu ý**:
  - Checkpoint hierarchical trước #16 không nạp được nữa vì input DQN đổi 195 → 213.
  - File metrics được ghi nối (append) và không lọc trùng. Khi resume từ checkpoint episode N mà metrics đã ghi quá N, phải cắt metrics về đúng N dòng trước, nếu không biểu đồ có điểm trùng.
  - Trạng thái ngẫu nhiên (RNG) không được lưu → sau resume kết quả không trùng từng số với lần chạy liền.

---

## Config

### #13 — Hai bộ siêu tham số SAC/DQN trùng nhau
- **Triệu chứng**: có `sac`/`dqn` ở cấp ngoài cùng của `mappo.yaml` (không được `ProjectConfig` kiểm tra, có cả key chiều mạng cứng như `state_dim: 135`) và `hierarchical_parameters.sac/dqn` trong `hierarchical.yaml` (có kiểm tra) — giá trị khác nhau, tên key khác nhau.
- **Sửa**:
  - Xóa `sac`/`dqn` khỏi `mappo.yaml`. Agent đọc `hierarchical_parameters.sac/dqn`, dùng đúng tên key đã được kiểm tra (`init_alpha`, `replay_capacity`, `epsilon_start`, `epsilon_min`, `epsilon_decay`, `target_update_freq`).
  - Giữ nguyên các giá trị đang có trong `hierarchical.yaml`; bổ sung 3 key agent cần: `hidden_dims`, `grad_clip`, `warmup_steps`. Bỏ `action_scale`, `action_bias` (không có code nào dùng).
  - Chiều mạng không ghi trong YAML; trainer lấy từ env.
- **Lưu ý**: các giá trị cũ ở `mappo.yaml` khác với `hierarchical.yaml`, ví dụ SAC lr 1e-4 vs 3e-4, batch 256 vs 32; DQN lr 1e-4 vs 1e-3, gamma 0.99 vs 0.95, target update 2 vs 50. Hiện đang dùng bộ trong `hierarchical.yaml`.

### #14 — Tạp âm nhiệt trong `system.yaml` bị nhân đôi (+3 dB) (18/11/2026)
- **File**: `config/system.yaml` (`noise_power_per_prb_dbm`, `noise_power_per_prb_watt`)
- **Triệu chứng (sai thầm lặng, không crash)**: yaml ghi −111.4 dBm / 7.24e-15 W, tính cho PRB 360 kHz (BWP μ=1). Nhưng `channel.py` (dòng 298, 332) và `communication.py` (dòng 30-31) nhân thêm `2^μ` cho từng BWP → tạp âm cao gấp 2 (+3 dB) ở cả hai BWP so với `B_w·N_0` trong `ISAC_probability.md` (dòng 659, 664, 681).
- **Nguyên nhân**: yaml hiểu giá trị là tạp âm của PRB μ=1, code hiểu là tạp âm PRB cơ sở μ=0 (`prbBW` = 180 kHz). `scenario.py:153-156` (phương án dự phòng) cũng dùng quy ước cơ sở 180 kHz.
- **Sửa**: đặt `noise_power_per_prb_dbm: -114.45`, `noise_power_per_prb_watt: 3.59e-15` (cơ sở 180 kHz) và sửa chú thích. Sau sửa: BWP 0 = −111.44 dBm, BWP 1 = −108.43 dBm, đúng bằng N0 + NF + 10·log10(B_w). Code không đổi.
- **Ảnh hưởng**: SINR tăng 3 dB ở mọi PRB so với trước khi sửa.
- **Lưu ý**: `noise_psd_dbm_hz` và `noise_figure_db` không được code dùng để tính lại tạp âm (chỉ là phương án dự phòng ở `scenario.py`, mà biến `sigma2_w` ở đó cũng không có chỗ nào đọc). Muốn đổi NF phải sửa tay hai giá trị `noise_power_per_prb_*`.

---

## Test — `docs/test_agents.py`

### #20 — `docs/test_agents.py` không chạy được (20/11/2026)
- **Triệu chứng**: đọc `sac`/`dqn` từ `mappo.yaml` (đã xóa ở #13) và gọi API cũ.
- **Sửa**: viết lại theo API mới, đọc `hierarchical.yaml`, thêm test ngân sách DQN, single-BWP, replay SAC lưu raw action và round-trip `state_dict`. 6/6 qua.

---

## Quyết định thiết kế

- **Số episode = 2000 cho cả 3 kịch bản**: `mappo.yaml` và `hierarchical.yaml` đều `total_episodes: 2000` (100 slot/episode → cùng 200.000 bước môi trường). SAC warmup 1000 frame = 5% số frame.
- **C2S có rò rỉ qua các BWP**: nhiễu C2S tính cả rò rỉ giữa hai BWP. Code (`channel.py:67-69`, ma trận `c2s_matrix[W,R,W,R]`) đúng ý đồ; `ISAC_probability.md` dòng 613 (chỉ cộng r′ ∈ R_{b,w}, cùng BWP) là bản cũ. Hệ quả: `fStart` ảnh hưởng trực tiếp tới C2S (xem B0).
- **SAC state 22 chiều**: SAC dùng `env.get_sac_state()` (tỷ lệ phục vụ eMBB/URLLC, 3 chi phí, AoI, U_pos, U_vel trung bình frame), theo Approach_Benchmark.md, thay vì `global_state` 135 chiều như spec cũ. spec.md đã cập nhật.
- **Reward so sánh được giữa 3 thuật toán**: cả 3 đều tính reward slot bằng `mappo_parameters.reward_weights` (`ppo/envs/env.py:67`).

---

## Kiểm tra đã chạy

| Test | Kết quả |
|---|---|
| `python docs/test_agents.py` | 6/6 qua |
| Smoke test 3 thuật toán (MAPPO 3 ep; hierarchical 4 ep, SAC `warmup_steps: 5`, DQN `warmup_steps: 64`) | chạy hết, update, vẽ biểu đồ; DQN loss giảm (multi 1.21 → 0.42, single 1.30 → 0.49); env không báo vi phạm ngân sách / single-BWP |
| Resume hierarchical (chạy 4 ep rồi resume tới ep 6, SAC `warmup_steps: 1000`) | nạp lại replay SAC (40 frame), SAC update ngay ep 5 thay vì đợi 100 episode; epsilon tiếp tục 0.980 → 0.975 |
| Resume MAPPO với `--episodes` | return ep 3 trùng khớp lần chạy liền |
| PRB thực dùng hierarchical (10 frame, ε = 1 và 0) | multi 87–88% (trước #16: 58%), single 81–85% (trước: ~29%) |

Với config mặc định, SAC chỉ bắt đầu update sau 100 episode (`warmup_steps: 1000` frame, 10 frame/episode), DQN sau 5 episode.

---

## Kết quả chạy đầy đủ 2000 episode (20/11/2026)

Config hiện tại (η0 = 1e-5, `fStart` theo kHz, không có hạng phạt). Metrics: `results/metrics/<run>.jsonl`, biểu đồ: `results/figures/<run>/`. Single-BWP được chạy tiếp từ checkpoint episode 1580 (`--resume`).

Trung bình theo cửa sổ 100 episode (ep 1–100 → ep 1901–2000):

| Chỉ số | flatDRL-MAPPO | Hier multi-BWP | Hier single-BWP |
|---|---|---|---|
| Reward / slot | 2.77 → 3.38 | 3.39 → 3.68 (đỉnh 3.80) | 3.45 → 3.72 |
| Độ lệch chuẩn reward 200 ep cuối | 0.076 | 0.130 | 0.086 |
| P_suc sensing | 0.013 → 0.161 | 0.080 → 0.212 (đỉnh 0.312) | 0.117 → 0.253 |
| AoI trung bình (slot) | 40.3 → 17.4 | 14.0 → 6.0 | 12.2 → 5.4 |
| Sai số vị trí (m) | 2.57 → 0.44 | 0.98 → 0.46 | 0.84 → 0.40 |
| Sai số vận tốc (m/s) | 3.5 → 3.3 | 3.4 → 3.2 | 3.5 → 3.0 |
| UE đạt QoS (/12) | 12 → 11.86 | 12 → 11.87 | 12 → 11.96 |
| Tốc độ comm TB (Mbps) | 49.6 → 47.4 | 33.9 → 29.1 | 33.9 → 30.2 |
| Switch cost | 15.4 → 6.4 | 3.4 → 2.8 | 4.8 → 4.6 |
| Mất target | 0 | 0 | 0 |

Diễn biến của Hier multi-BWP (reward / slot, P_suc trung bình mỗi 100 ep):
`3.39 3.49 3.62 3.76 3.74 3.68 3.31 3.33 3.42 3.47 3.64 3.72 3.64 3.44 3.51 3.51 3.64 3.75 3.80 3.68`
`0.08 0.12 0.17 0.26 0.25 0.22 0.07 0.07 0.09 0.13 0.22 0.25 0.19 0.11 0.12 0.13 0.20 0.28 0.31 0.21`

Nhận xét:
- Cả 3 đều học được theo cùng một hướng: dành thêm tài nguyên cho sensing và giảm tốc độ comm, trong khi QoS comm vẫn gần như giữ nguyên.
- Hierarchical cao hơn MAPPO ngay từ episode đầu, vì ngân sách của SAC buộc phải có PRB cho target. MAPPO mất khoảng 500 episode mới thoát khỏi vùng P_suc ≈ 1%.
- Hier multi-BWP dao động mạnh: reward lên đỉnh rồi sụt, khoảng 750 episode một chu kỳ. Single-BWP tăng đều và ổn định hơn.
- Kết luận so sánh hiện chưa có giá trị, vì vẫn còn B0 và nhóm B/C bên dưới. Sensing tối đa chỉ khoảng 25–30%.

---

## Còn tồn tại (chưa sửa)

### B0 — C2S quá mạnh, sensing gần như không thành công khi có truyền thông (chờ nhóm chốt)
- `c2s_leakage_eta0: 1.0e-5` (−50 dB) nhưng chú thích ghi −110 dB (= 1e-11). Giá trị ban đầu là 1e-11 (16/10), đổi thành 1e-5 ở commit b6eca84 (26/10) mà không sửa chú thích; mô hình C2S Gauss viết lại 11/11.
- `fStart: [0, 11]` được code hiểu là kHz (`channel.py:62` nhân 1e3) → BWP1 bắt đầu 11 kHz, chồng lên BWP0 (0–10.8 MHz). Có thể ý là MHz.
- Đo P_suc với chính sách cố định (100 slot, seed 42):

  | Phân bổ | Hiện tại | η0 = 1e-11 |
  |---|---|---|
  | 1 PRB/target + 1 PRB/UE | 0.195 | 0.64 |
  | BWP0 sensing / BWP1 comm | 0.002 | 0.92 |
  | Xen kẽ nửa/nửa | 0.002 | 0.91 |

  Chỉ đổi `fStart` sang MHz cứu được trường hợp tách BWP (0.90), không cứu trường hợp xen kẽ.
- Kết quả 2000 episode xác nhận: P_suc cao nhất chỉ khoảng 0.16 (MAPPO) và 0.25–0.31 (hierarchical).

### Nhóm B — reward (chờ nhóm quyết)
- **B1 — Hạng phạt bị xóa** ở commit 244688b (11/11): `r_penalty` luôn bằng 0. `mappo.yaml` mô tả phạt hinge một phía; Approach_Benchmark.md dùng (U/Umax − 1)², công thức này phạt cả khi bám tốt, cần `max(0, ·)`.
- **B2 — Reward chung hay reward theo BS**: env tính `local_rewards` theo từng BS nhưng không dùng; các BS chỉ khác nhau ở hạng chi phí.
- **B3 — Định nghĩa π khác nhau**: reward slot dùng "được cấp PRB" (`served_ues`), reward frame của SAC dùng "đạt QoS" (`qos_served_ues`). R_min = 1 Mbps nên 1 PRB/UE đã đủ đạt QoS, hai định nghĩa gần như trùng nhau.
- **B4 — `hierarchical_parameters.reward_weights` không được đọc**: trọng số SAC/DQN trong `hierarchical.yaml` không có code nào dùng.

### Nhóm C — phát hiện từ kết quả 2000 episode
- **C1 — Hạng comm bão hòa**: `r_comm` gần như hằng số 0.632 = 1 − e^{−1} (π = 1, đủ 12/12 UE) ở cả 3 thuật toán. Agent không nhận được tín hiệu nào để giữ tốc độ comm, nên tốc độ eMBB giảm dần (hierarchical 34 → 29 Mbps) mà reward không bị trừ. Liên quan B1 và B3.
- **C2 — Guardband cost không mang thông tin**: `guardband_cost` = 10 gần như suốt quá trình (= `cost_gb` 2 × 5 BS), tức mọi BS luôn bật cả hai BWP. Single-BWP cũng vậy, vì các dịch vụ khác nhau vẫn nằm ở hai BWP khác nhau.
- **C3 — PPO cập nhật quá mạnh**: `clip_fraction` của MAPPO là 0.67–0.74 suốt quá trình (thường khoảng 0.1–0.3); ở episode cuối `ratio_std` ≈ 0.99. Nguyên nhân có thể là log-prob là tổng của 60 PRB (`AgentComponents.py:88`), nên một thay đổi nhỏ ở từng PRB làm ratio chung lệch rất xa. Gợi ý: giảm `lr_actor` hoặc `ppo_epochs`, hoặc tính ratio/clip theo từng PRB.
- **C4 — Hier multi-BWP dao động theo chu kỳ** (khoảng 750 episode, xem bảng diễn biến). Cần kiểm tra thêm: SAC alpha giảm 0.2 → 0.013 nên SAC gần như ngừng khám phá; ngân sách thay đổi làm phân phối dữ liệu trong replay DQN bị lệch.
- **C5 — Bất định vận tốc gần như không giảm**: sai số vận tốc ≈ 3 m/s, `u_vel` ≈ 10 suốt quá trình ở cả 3 thuật toán, trong khi sai số vị trí giảm tốt. Có thể do P_suc thấp (B0) nên EKF hiếm khi có phép đo liên tiếp để ước lượng vận tốc.
