# Results tables

760 configurations, 20 seeds each, 24 semantic features.

## Success fraction by residual width

`computes` = the program returned its exact expected value. `recovers` = the blind analyst scored 12/12 on the ISA.

### pinv readout

| d | countdown_5 (max 5) | rot_jz_nop (max 99) | sum_1_to_15 (max 120) | sum_1_to_100 (max 5050) | recovers (ideal) | recovers (self) |
|--:|--:|--:|--:|--:|--:|--:|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 12 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 14 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 0.35 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 |
| 18 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 20 | 0.45 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 21 | 0.15 | 0.05 | 0.00 | 0.00 | 0.00 | 0.05 |
| 22 | 0.35 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| 23 | 0.05 | 0.00 | 0.00 | 0.00 | 0.10 | 0.05 |
| 24 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 26 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 48 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### dot readout

| d | countdown_5 (max 5) | rot_jz_nop (max 99) | sum_1_to_15 (max 120) | sum_1_to_100 (max 5050) | recovers (ideal) | recovers (self) |
|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 24 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| 32 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| 64 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| 256 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| 1024 | 0.20 | 0.00 | 0.00 | 0.00 | 0.05 | 0.05 |
| 2048 | 0.30 | 0.00 | 0.00 | 0.00 | 0.05 | 0.20 |
| 4096 | 0.40 | 0.10 | 0.00 | 0.00 | 0.20 | 0.05 |
| 8192 | 0.40 | 0.00 | 0.00 | 0.00 | 0.40 | 0.00 |
| 16384 | 0.40 | 0.05 | 0.00 | 0.00 | 0.55 | 0.05 |

### scaled readout  (post-hoc arm)

| d | countdown_5 (max 5) | rot_jz_nop (max 99) | sum_1_to_15 (max 120) | sum_1_to_100 (max 5050) | recovers (ideal) | recovers (self) |
|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 24 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 32 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 64 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 256 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1024 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2048 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4096 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8192 | 0.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16384 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Where the machine first parted company with the reference

Counted over failed runs. `tiebreak:<head>` means the argmax moved to another row at the *same* address, so only the `1e-6` write-order tiebreak separated them; `argmax:<head>` means it moved to a different address.

| arm | argmax:prog_op | opcode_decode | tiebreak:stack_* | value_drift | argmax:stack_* |
|---|--:|--:|--:|--:|--:|
| pinv | 37% | 40% | 2% | 19% | 2% |
| dot | 39% | 31% | 18% | 8% | 4% |
| scaled | 35% | 18% | 0% | 48% | 0% |

## Failure mode by program, dot arm at d >= 1024

The regime where interference is small enough that addressing mostly survives.

| program | max value | dominant first divergence | median step |
|---|--:|---|--:|
| countdown_5 | 5 | tiebreak:stack_a (68%) | 3 |
| rot_jz_nop | 99 | argmax:prog_op (34%) | 4 |
| sum_1_to_15 | 120 | tiebreak:stack_a (34%) | 4 |
| sum_1_to_100 | 5050 | opcode_decode (54%) | 0 |

## Per-feature scales used by the post-hoc scaled arm

| feature | typical magnitude |
|---|--:|
| `is_prog` | 1.00 |
| `is_stack` | 1.00 |
| `is_state` | 1.00 |
| `prog_k0` | 11.76 |
| `prog_k1` | 46.90 |
| `stack_k0` | 4.42 |
| `stack_k1` | 5.70 |
| `opcode` | 6.43 |
| `value` | 2147.20 |
| `ip` | 5.91 |
| `sp` | 2.41 |
| `one` | 1.00 |
| `op_PUSH` | 1.00 |
| `op_POP` | 1.00 |
| `op_ADD` | 1.00 |
| `op_DUP` | 1.00 |
| `op_HALT` | 1.00 |
| `op_SUB` | 1.00 |
| `op_JZ` | 1.00 |
| `op_JNZ` | 1.00 |
| `op_NOP` | 1.00 |
| `op_SWAP` | 1.00 |
| `op_OVER` | 1.00 |
| `op_ROT` | 1.00 |
