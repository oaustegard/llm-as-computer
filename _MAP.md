# llm-as-computer/
*Python files: 15 | Other: requirements.txt, 5 JSON results, 4 markdown docs*

## Inheritance Chain (active executor path)
```
phase4_stack_machine.py  → Instruction, Trace, TraceStep, ReferenceExecutor, ParabolicMemory
  └─ phase11_compile_executor.py  → CompiledExecutorNumpy, ExtendedExecutor (SUB/JZ/JNZ)
       └─ phase12_percepta_model.py  → PerceptaModel, CompiledAttentionHead, embedding dims
            └─ phase13_isa_completeness.py  → Phase13Model/Executor (SWAP/OVER/ROT)
                 └─ phase14_extended_isa.py  → Phase14Model/Executor (42-opcode ISA)
```

### phase10_digit_decomposition.py
*803 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `from torch.utils.data`, `random`, `time`, `json`, `os`, `from typing`, `from dataclasses`, `sys`...
- **num_to_digits** (f) `(n: int, n_digits: int) → List[int]` :65
- **digits_to_num** (f) `(digits: List[int]) → int` :75
- **encode_digit** (f) `(d: int) → int` :83
- **decode_digit** (f) `(idx: int) → int` :88
- **encode_opcode** (f) `(op: int) → int` :95
- **decode_opcode** (f) `(idx: int) → int` :100
- **encode_special** (f) `(raw: int) → int` :107
- **encode_num_field** (f) `(val: int) → List[int]` :118
- **decode_num_field** (f) `(tokens: List[int]) → int` :123
- **microop_trace_to_digit_tokens** (f) `(trace: MicroOpTrace) → List[int]` :133
- **generate_digit_data** (f) `(allowed_ops: Set[int], n_samples: int, min_len: int, max_len: int, max_push_val: int) → List[List[int]]` :159
- **DigitTraceDataset** (Dataset) (C) :188
  - **__init__** (m) `(self, sequences: List[List[int]], max_len: int)` :191
  - **__len__** (m) `(self)` :206
  - **__getitem__** (m) `(self, idx)` :209
- **DigitTransformerBlock** (nn.Module) (C) :215
  - **__init__** (m) `(self, d_model, n_heads, d_ff, dropout)` :216
  - **forward** (m) `(self, x, mask)` :229
- **DigitTransformer** (nn.Module) (C) :237
  - **__init__** (m) `(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, max_len: int, dropout: float)` :240
  - **forward** (m) `(self, x: torch.Tensor) → torch.Tensor` :268
- **evaluate_digit_execution** (f) `(model: DigitTransformer, test_progs: List[List[Instruction]], verbose: bool) → Dict` :285
- **run_add_diagnostic_digit** (f) `(model: DigitTransformer, n_tests: int, verbose: bool) → Dict` :377
- **train_digit_stage** (f) `(model: DigitTransformer, train_data: DigitTraceDataset, val_data: DigitTraceDataset, stage: int, max_epochs: int, lr: float, batch_size: int, patience: int, max_wall_time: float, checkpoint_prefix: str, checkpoint_dir: str, resume: bool, verbose: bool) → CheckpointMeta` :439
- **run_stage** (f) `(stage: int, model: DigitTransformer, n_train: int, n_val: int, n_test: int, checkpoint_prefix: str, checkpoint_dir: str, verbose: bool) → Dict` :592
- **run_digit_experiment** (f) `(checkpoint_dir: str) → Dict` :691

### phase11_compile_executor.py
*937 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `time`, `sys`, `os`, `from phase4_stack_machine`, `from phase1_hull_cache`
- **encode_token** (f) `(raw)` :52
- **decode_token** (f) `(idx)` :59
- **CompiledExecutorNumpy** (C) :73
  - **__init__** (m) `(self)` :80
  - **execute** (m) `(self, prog, max_steps)` :83
- **HardMaxAttention** (nn.Module) (C) :172
  - **__init__** (m) `(self, d_model, head_dim)` :179
  - **forward** (m) `(self, x, causal_mask)` :187
- **CompiledTransformer** (nn.Module) (C) :210
  - **__init__** (m) `(self)` :229
  - **describe** (m) `(self)` :245
- **HullKVCache** (C) :254
  - **__init__** (m) `(self, eps)` :264
  - **write** (m) `(self, addr, value)` :270
  - **read** (m) `(self, addr)` :277
  - **read_fast** (m) `(self, addr)` :288
  - **__len__** (m) `(self)` :313
- **CompiledExecutorWithHull** (CompiledExecutorNumpy) (C) :317
  - **execute** (m) `(self, prog, max_steps)` :324
- `OP_SUB` = `6` :387
- `OP_JZ` = `7` :388
- `OP_JNZ` = `8` :389
- `OP_NOP` = `9` :390
- `OP_NAMES_EXT` = `{**OP_NAMES, OP_SUB: 'SUB', OP_JZ: 'JZ', OP_JNZ: 'JNZ', O...` :392
- **ExtendedExecutor** (CompiledExecutorNumpy) (C) :401
  - **execute** (m) `(self, prog, max_steps)` :411
- **FastExecutor** (C) :515
  - **__init__** (m) `(self)` :522
  - **execute** (m) `(self, prog, max_steps)` :525
- **test_compiled_executor** (f) `()` :595
- **test_hull_executor** (f) `()` :637
- **test_extended_executor** (f) `()` :673
- **test_fast_executor** (f) `()` :775
- **benchmark_scaling** (f) `()` :811
- **main** (f) `()` :867

### phase12_percepta_model.py
*1390 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `time`, `sys`, `os`, `from phase4_stack_machine`
- `OP_SUB` = `6` :49
- `OP_JZ` = `7` :50
- `OP_JNZ` = `8` :51
- `OP_NOP` = `9` :52
- `OP_NAMES_EXT` = `{**OP_NAMES, OP_SUB: 'SUB', OP_JZ: 'JZ', OP_JNZ: 'JNZ', O...` :54
- `D_MODEL` = `36` :68
- `DIM_IS_PROG` = `0` :71
- `DIM_IS_STACK` = `1` :72
- `DIM_IS_STATE` = `2` :73
- `DIM_PROG_KEY_0` = `3` :74
- `DIM_PROG_KEY_1` = `4` :75
- `DIM_STACK_KEY_0` = `5` :76
- `DIM_STACK_KEY_1` = `6` :77
- `DIM_OPCODE` = `7` :78
- `DIM_VALUE` = `8` :79
- `DIM_IP` = `9` :80
- `DIM_SP` = `10` :81
- `DIM_ONE` = `11` :82
- `DIM_IS_PUSH` = `12` :83
- `DIM_IS_POP` = `13` :84
- `DIM_IS_ADD` = `14` :85
- `DIM_IS_DUP` = `15` :86
- `DIM_IS_HALT` = `16` :87
- `DIM_IS_SUB` = `17` :88
- `DIM_IS_JZ` = `18` :89
- `DIM_IS_JNZ` = `19` :90
- `DIM_IS_NOP` = `20` :91
- `N_OPCODES` = `9` :107
- `EPS` = `1e-06` :112
- `DTYPE` = `torch.float64` :118
- **CompiledAttentionHead** (nn.Module) (C) :123
  - **__init__** (m) `(self, d_model, head_dim, v_dim, use_bias_q)` :137
  - **forward** (m) `(self, query_emb, memory_embs)` :145
- **PerceptaModel** (nn.Module) (C) :173
  - **__init__** (m) `(self, d_model)` :190
  - **_compile_weights** (m) `(self)` :220
  - **forward** (m) `(self, query_emb, prog_embs, stack_embs)` :319
- **embed_program_token** (f) `(pos, instr)` :404
- **embed_stack_entry** (f) `(addr, value, write_order)` :425
- **embed_state** (f) `(ip, sp)` :441
- **PerceptaExecutor** (C) :453
  - **__init__** (m) `(self, model)` :460
  - **execute** (m) `(self, prog, max_steps)` :464
  - **_read_stack_top** (m) `(self, stack_embs_list, addr)` :553
- **PerceptaExtendedExecutor** (PerceptaExecutor) (C) :575
  - **execute** (m) `(self, prog, max_steps)` :582
- **PerceptaFullSequenceModel** (nn.Module) (C) :653
  - **__init__** (m) `(self, d_model)` :669
  - **_compile_weights** (m) `(self)` :689
  - **forward** (m) `(self, embeddings)` :721
- **inspect_weights** (f) `(model)` :760
- **_dim_name** (f) `(d)` :833
- **test_attention_primitives** (f) `()` :839
- **test_ff_dispatch** (f) `()` :948
- **test_compiled_executor** (f) `()` :994
- **test_extended_isa** (f) `()` :1034
- **test_full_sequence_attention** (f) `()` :1110
- **test_model_parameter_count** (f) `()` :1175
- **benchmark_vs_phase11** (f) `()` :1212
- **demo_trace** (f) `()` :1265
- **main** (f) `()` :1294

### phase13_isa_completeness.py
*1154 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `time`, `sys`, `os`, `from phase4_stack_machine`, `from phase12_percepta_model`, `from phase11_compile_executor`
- `OP_SWAP` = `10` :57
- `OP_OVER` = `11` :58
- `OP_ROT` = `12` :59
- `OP_NAMES_P13` = `{**OP_NAMES, OP_SUB: 'SUB', OP_JZ: 'JZ', OP_JNZ: 'JNZ', O...` :61
- `DIM_IS_SWAP` = `21` :68
- `DIM_IS_OVER` = `22` :69
- `DIM_IS_ROT` = `23` :70
- `N_OPCODES` = `12` :87
- **embed_program_token_ext** (f) `(pos, instr)` :92
- **Phase13Executor** (ExtendedExecutor) (C) :109
  - **execute** (m) `(self, prog, max_steps)` :115
- **Phase13Model** (PerceptaModel) (C) :229
  - **__init__** (m) `(self, d_model)` :236
  - **_compile_weights** (m) `(self)` :257
  - **forward** (m) `(self, query_emb, prog_embs, stack_embs)` :362
- **Phase13PyTorchExecutor** (C) :420
  - **__init__** (m) `(self, model)` :423
  - **execute** (m) `(self, prog, max_steps)` :427
- **fib** (f) `(n)` :509
- **make_fibonacci** (f) `(n)` :518
- **make_multiply** (f) `(a, b)` :566
- **make_power_of_2** (f) `(n)` :608
- **make_sum_1_to_n** (f) `(n)` :647
- **make_is_even** (f) `(n)` :698
- **compare_traces** (f) `(trace_a, trace_b)` :735
- **test_new_opcodes** (f) `()` :745
- **test_head_sp2** (f) `()` :823
- **test_algorithm** (f) `(name, prog, expected, np_exec, pt_exec, verbose)` :870
- **test_fibonacci** (f) `()` :899
- **test_multiply** (f) `()` :919
- **test_power_of_2** (f) `()` :939
- **test_sum_1_to_n** (f) `()` :959
- **test_is_even** (f) `()` :979
- **test_regression** (f) `()` :1000
- **test_model_summary** (f) `()` :1058
- **main** (f) `()` :1099

### phase14_extended_isa.py
*2886 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `time`, `sys`, `os`, `from phase4_stack_machine`, `from phase12_percepta_model`, `from phase13_isa_completeness`
- `OP_MUL` = `13` :120
- `OP_DIV_S` = `14` :121
- `OP_DIV_U` = `15` :122
- `OP_REM_S` = `16` :123
- `OP_REM_U` = `17` :124
- `OP_EQZ` = `18` :125
- `OP_EQ` = `19` :126
- `OP_NE` = `20` :127
- `OP_LT_S` = `21` :128
- `OP_LT_U` = `22` :129
- `OP_GT_S` = `23` :130
- `OP_GT_U` = `24` :131
- `OP_LE_S` = `25` :132
- `OP_LE_U` = `26` :133
- `OP_GE_S` = `27` :134
- `OP_GE_U` = `28` :135
- `OP_AND` = `29` :136
- `OP_OR` = `30` :137
- `OP_XOR` = `31` :138
- `OP_SHL` = `32` :139
- `OP_SHR_S` = `33` :140
- `OP_SHR_U` = `34` :141
- `OP_ROTL` = `35` :142
- `OP_ROTR` = `36` :143
- `OP_CLZ` = `37` :144
- `OP_CTZ` = `38` :145
- `OP_POPCNT` = `39` :146
- `OP_ABS` = `40` :147
- `OP_NEG` = `41` :148
- `OP_SELECT` = `42` :149
- `OP_TRAP` = `99` :150
- `OP_NAMES_P14` = `{**OP_NAMES_P13, OP_MUL: 'MUL', OP_DIV_S: 'DIV_S', OP_DIV...` :152
- `DIM_IS_MUL` = `24` :189
- `DIM_IS_DIV_S` = `25` :190
- `DIM_IS_DIV_U` = `26` :191
- `DIM_IS_REM_S` = `27` :192
- `DIM_IS_REM_U` = `28` :193
- `DIM_IS_EQZ` = `29` :196
- `DIM_IS_EQ` = `30` :197
- `DIM_IS_NE` = `31` :198
- `DIM_IS_LT` = `32` :199
- `DIM_IS_GT` = `33` :200
- `DIM_IS_LE` = `34` :201
- `DIM_IS_GE` = `35` :202
- `N_OPCODES` = `42` :258
- **_trunc_div** (f) `(b, a)` :273
- **_trunc_rem** (f) `(b, a)` :280
- `MASK32` = `4294967295` :289
- **_to_i32** (f) `(val)` :291
- **_shr_u** (f) `(b, a)` :295
- **_shr_s** (f) `(b, a)` :301
- **_rotl32** (f) `(b, a)` :314
- **_rotr32** (f) `(b, a)` :320
- **_clz32** (f) `(val)` :327
- **_ctz32** (f) `(val)` :340
- **_popcnt32** (f) `(val)` :353
- **embed_program_token_ext** (f) `(pos, instr)` :357
- **Phase14Executor** (Phase13Executor) (C) :374
  - **execute** (m) `(self, prog, max_steps)` :381
- **Phase14Model** (Phase13Model) (C) :649
  - **__init__** (m) `(self, d_model)` :668
  - **_compile_weights** (m) `(self)` :686
  - **forward** (m) `(self, query_emb, prog_embs, stack_embs)` :803
- **Phase14PyTorchExecutor** (C) :915
  - **__init__** (m) `(self, model)` :918
  - **execute** (m) `(self, prog, max_steps)` :922
- **make_native_multiply** (f) `(a, b)` :1025
- **make_native_divmod** (f) `(a, b)` :1038
- **make_native_remainder** (f) `(a, b)` :1059
- **make_native_is_even** (f) `(n)` :1076
- **make_factorial** (f) `(n)` :1096
- **make_gcd** (f) `(a, b)` :1134
- **make_compare_eqz** (f) `(a)` :1173
- **make_compare_binary** (f) `(op, a, b)` :1182
- **make_native_max** (f) `(a, b)` :1217
- **make_native_abs** (f) `(n)` :1254
- **make_native_clamp** (f) `(val, lo, hi)` :1279
- **make_bitwise_binary** (f) `(op, a, b)` :1317
- **make_popcount_loop** (f) `(n)` :1353
- **make_bit_extract** (f) `(n, bit_pos)` :1471
- **make_native_clz** (f) `(n)` :1492
- **make_native_ctz** (f) `(n)` :1504
- **make_native_popcnt** (f) `(n)` :1512
- **make_native_abs** (f) `(n)` :1524
- **make_native_neg** (f) `(n)` :1536
- **make_select** (f) `(a, b, c)` :1544
- **make_select_max** (f) `(a, b)` :1559
- **make_log2_floor** (f) `(n)` :1586
- **make_is_power_of_2** (f) `(n)` :1605
- **test_trap_algorithm** (f) `(name, prog, np_exec, pt_exec, verbose)` :1624
- **test_arithmetic_unit** (f) `()` :1651
- **test_division_by_zero** (f) `()` :1734
- **test_native_multiply** (f) `()` :1770
- **test_native_division** (f) `()` :1794
- **test_native_is_even** (f) `()` :1824
- **test_factorial** (f) `()` :1846
- **test_gcd** (f) `()` :1866
- **test_regression** (f) `()` :1886
- **test_model_summary** (f) `()` :1965
- **test_step_count_comparison** (f) `()` :1989
- **test_comparison_unit** (f) `()` :2027
- **test_comparison_algorithms** (f) `()` :2121
- **test_bitwise_unit** (f) `()` :2171
- **test_bitwise_algorithms** (f) `()` :2371
- **test_unary_unit** (f) `()` :2413
- **test_select_unit** (f) `()` :2511
- **test_unary_algorithms** (f) `()` :2571
- **test_step_count_chunk4** (f) `()` :2645
- **test_integration_chunk5** (f) `()` :2682
- **main** (f) `()` :2803

### phase1_hull_cache.py
*539 lines*
> Imports: `numpy`, `time`, `json`
- **BruteForceKVCache** (C) :22
  - **__init__** (m) `(self)` :25
  - **add** (m) `(self, key: tuple, value: float)` :32
  - **_sync** (m) `(self)` :37
  - **query** (m) `(self, q: tuple) → float` :43
  - **__len__** (m) `(self)` :52
- **HullKVCache** (C) :58
  - **__init__** (m) `(self)` :69
  - **_key_id** (m) `(self, k)` :78
  - **add** (m) `(self, key: tuple, value: float)` :81
  - **_rebuild** (m) `(self)` :88
  - **query** (m) `(self, q: tuple) → float` :120
  - **hull_size** (m) `(self)` :148
  - **__len__** (m) `(self)` :153
- **ParabolicKVCache** (C) :159
  - **__init__** (m) `(self)` :175
  - **add** (m) `(self, key: tuple, value: float)` :178
  - **query_direct** (m) `(self, index: int) → float` :182
  - **query_ternary** (m) `(self, q: tuple) → float` :186
  - **__len__** (m) `(self)` :219
- **test_correctness** (f) `()` :225
- **benchmark_query_scaling** (f) `()` :311
- **benchmark_execution_trace** (f) `()` :400
- **benchmark_scaling_fit** (f) `()` :445

### phase2_parabolic.py
*217 lines*
> Imports: `numpy`, `json`
- **test_exact_retrieval** (f) `()` :14
- **test_precision_analysis** (f) `()` :75
- **test_overwrites** (f) `()` :110
- **test_noninteger** (f) `()` :173

### phase2b_address_limits.py
*432 lines*
> Imports: `numpy`, `from typing`
- **parabolic_encode** (f) `(j: int, dtype) → Tuple[float, float]` :23
- **parabolic_query** (f) `(i: int, dtype) → Tuple[float, float]` :27
- **find_breakpoint** (f) `(encode_fn, query_fn, max_n, dtype) → int` :31
- **OffsetParabolicSegment** (C) :89
  - **__init__** (m) `(self, center: int, radius: int, dtype)` :92
  - **encode** (m) `(self, j: int) → Tuple[float, float]` :97
  - **query** (m) `(self, i: int) → Tuple[float, float]` :101
  - **covers** (m) `(self, addr: int) → bool` :105
- **SegmentedMemory** (C) :109
  - **__init__** (m) `(self, max_addr: int, segment_size: int, dtype)` :116
  - **write** (m) `(self, addr: int, value: int)` :134
  - **read** (m) `(self, addr: int) → Optional[int]` :145
- **ResidualAddressMemory** (C) :171
  - **__init__** (m) `(self, block_size: int, dtype)` :180
  - **_split** (m) `(self, addr: int) → Tuple[int, int]` :188
  - **write** (m) `(self, addr: int, value: int)` :191
  - **read_via_attention** (m) `(self, addr: int) → Optional[int]` :197
  - **max_addressable** (m) `(self) → int` :226
- **hybrid_encode** (f) `(j: int, modulus: int, scale: float, dtype) → Tuple[float, float]` :239
- **hybrid_query** (f) `(i: int, modulus: int, scale: float, dtype) → Tuple[float, float]` :244
- **test_baseline** (f) `()` :256
- **test_segmented** (f) `(max_addr: int)` :266
- **test_residual** (f) `(max_addr: int)` :291
- **test_stress_residual** (f) `()` :317
- **test_offset_breakpoint** (f) `()` :355
- **main** (f) `()` :383

### phase3_cumsum.py
*241 lines*
> Imports: `numpy`, `json`
- **cumsum_via_attention** (f) `(deltas)` :21
- **cumsum_via_attention_vectorized** (f) `(deltas)` :45
- **test_basic_correctness** (f) `()` :58
- **test_numerical_drift** (f) `()` :86
- **test_realistic_stack** (f) `()` :131
- **test_alternative_cumsum** (f) `()` :171

### phase4_stack_machine.py
*679 lines*
> Imports: `numpy`, `from typing`, `from dataclasses`
- `OP_PUSH` = `1` :34
- `OP_POP` = `2` :35
- `OP_ADD` = `3` :36
- `OP_DUP` = `4` :37
- `OP_HALT` = `5` :38
- `OP_NAMES` = `{OP_PUSH: 'PUSH', OP_POP: 'POP', OP_ADD: 'ADD', OP_DUP: '...` :40
- **Instruction** (C) :58
  - **__repr__** (m) `(self)` :62
- **program** (f) `() → List[Instruction]` :69
- **TraceStep** (C) :87
  - **tokens** (m) `(self) → List[int]` :94
- **Trace** (C) :99
  - **to_token_sequence** (m) `(self) → List[int]` :104
  - **format_trace** (m) `(self) → str` :119
- **ReferenceExecutor** (C) :134
  - **execute** (m) `(self, prog: List[Instruction], max_steps: int) → Trace` :137
- **ParabolicMemory** (C) :180
  - **__init__** (m) `(self, dtype)` :189
  - **write** (m) `(self, addr: int, value: int)` :196
  - **read** (m) `(self, addr: int) → Optional[int]` :204
  - **read_second** (m) `(self, addr: int) → Optional[int]` :227
- **SequentialState** (C) :250
  - **__init__** (m) `(self, initial: int)` :257
  - **update** (m) `(self, delta: int)` :261
  - **current** (m) `(self) → int` :265
  - **at** (m) `(self, step: int) → int` :268
- **AttentionExecutor** (C) :274
  - **execute** (m) `(self, prog: List[Instruction], max_steps: int) → Trace` :287
- **HandWiredTransformer** (C) :397
  - **__init__** (m) `(self)` :415
  - **describe_weight_structure** (m) `(self) → str` :432
- **test_basic** (f) `()` :483
- **test_push_halt** (f) `()` :488
- **test_push_pop** (f) `()` :493
- **test_dup_add** (f) `()` :498
- **test_multi_add** (f) `()` :503
- **test_stack_depth** (f) `()` :508
- **test_overwrite** (f) `()` :513
- **test_complex** (f) `()` :519
- **test_many_pushes** (f) `()` :526
- **test_alternating** (f) `()` :534
- **main** (f) `()` :559

### phase5_training.py
*723 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `from torch.utils.data`, `random`, `time`, `math`, `from typing`, `from dataclasses`, `sys`, `from phase4_stack_machine`
- **encode_token** (f) `(raw: int) → int` :62
- **decode_token** (f) `(idx: int) → int` :78
- **random_program** (f) `(min_len: int, max_len: int, max_push_val: int) → List[Instruction]` :97
- **generate_dataset** (f) `(n_samples: int, max_prog_len: int, max_push_val: int) → List[List[int]]` :147
- **TraceDataset** (Dataset) (C) :170
  - **__init__** (m) `(self, sequences: List[List[int]], max_len: int)` :173
  - **__len__** (m) `(self)` :190
  - **__getitem__** (m) `(self, idx)` :193
- **MicroTransformer** (nn.Module) (C) :199
  - **__init__** (m) `(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, max_len: int, dropout: float)` :206
  - **forward** (m) `(self, x: torch.Tensor) → torch.Tensor` :235
- **TransformerBlock** (nn.Module) (C) :254
  - **__init__** (m) `(self, d_model, n_heads, d_ff, dropout)` :255
  - **forward** (m) `(self, x, mask)` :268
- **train_model** (f) `(model: MicroTransformer, train_data: TraceDataset, val_data: TraceDataset, epochs: int, lr: float, batch_size: int, patience: int, verbose: bool) → Dict` :281
- **evaluate_execution** (f) `(model: MicroTransformer, test_progs: List[List[Instruction]], verbose: bool) → Dict` :383
- **analyze_attention** (f) `(model: MicroTransformer, sample_prog: List[Instruction])` :469
- **main** (f) `()` :494

### phase6_curriculum.py
*514 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `from torch.utils.data`, `random`, `time`, `json`, `os`, `from typing`, `from dataclasses`, `sys`...
- **constrained_random_program** (f) `(allowed_ops: Set[int], min_len: int, max_len: int, max_push_val: int) → List[Instruction]` :49
- **generate_stage_data** (f) `(allowed_ops: Set[int], n_samples: int, min_len: int, max_len: int, max_push_val: int) → List[List[int]]` :98
- **CheckpointMeta** (C) :130
  - **to_dict** (m) `(self)` :139
- **save_checkpoint** (f) `(model: MicroTransformer, optimizer, meta: CheckpointMeta, path: str)` :143
- **load_checkpoint** (f) `(path: str, model: MicroTransformer, optimizer)` :157
- **train_stage** (f) `(model: MicroTransformer, train_data: TraceDataset, val_data: TraceDataset, stage: int, max_epochs: int, lr: float, batch_size: int, patience: int, max_wall_time: float, checkpoint_dir: str, resume: bool, verbose: bool) → CheckpointMeta` :168
- **run_stage** (f) `(stage: int, model: MicroTransformer, n_train: int, n_val: int, n_test: int, checkpoint_dir: str, verbose: bool) → Dict` :359
- **run_all_stages** (f) `(checkpoint_dir: str) → Dict` :450

### phase7_percepta_arch.py
*501 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `from torch.utils.data`, `random`, `time`, `json`, `os`, `from typing`, `from dataclasses`, `sys`...
- **train_stage** (f) `(model: MicroTransformer, train_data: TraceDataset, val_data: TraceDataset, stage: int, max_epochs: int, lr: float, batch_size: int, patience: int, max_wall_time: float, checkpoint_prefix: str, checkpoint_dir: str, resume: bool, verbose: bool) → CheckpointMeta` :76
- **run_add_diagnostic** (f) `(model: MicroTransformer, n_tests: int, verbose: bool) → Dict` :235
- **run_stage** (f) `(stage: int, model: MicroTransformer, n_train: int, n_val: int, n_test: int, checkpoint_prefix: str, checkpoint_dir: str, verbose: bool) → Dict` :298
- **run_percepta_curriculum** (f) `(checkpoint_dir: str) → Dict` :390

### phase8_microop_traces.py
*745 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `from torch.utils.data`, `random`, `time`, `json`, `os`, `from typing`, `from dataclasses`, `sys`...
- **MicroOpStep** (C) :69
  - **tokens** (m) `(self) → List[int]` :78
- **MicroOpTrace** (C) :83
  - **to_token_sequence** (m) `(self) → List[int]` :88
- **MicroOpExecutor** (C) :99
  - **execute** (m) `(self, prog: List[Instruction], max_steps: int) → MicroOpTrace` :102
- **generate_microop_data** (f) `(allowed_ops: Set[int], n_samples: int, min_len: int, max_len: int, max_push_val: int) → List[List[int]]` :167
- **evaluate_microop_execution** (f) `(model: MicroTransformer, test_progs: List[List[Instruction]], verbose: bool) → Dict` :197
- **run_add_diagnostic_microop** (f) `(model: MicroTransformer, n_tests: int, verbose: bool) → Dict` :274
- **train_stage** (f) `(model: MicroTransformer, train_data: TraceDataset, val_data: TraceDataset, stage: int, max_epochs: int, lr: float, batch_size: int, patience: int, max_wall_time: float, checkpoint_prefix: str, checkpoint_dir: str, resume: bool, verbose: bool) → CheckpointMeta` :331
- **run_stage** (f) `(stage: int, model: MicroTransformer, n_train: int, n_val: int, n_test: int, checkpoint_prefix: str, checkpoint_dir: str, verbose: bool) → Dict` :482
- **run_microop_curriculum** (f) `(checkpoint_dir: str) → Dict` :573
- **sanity_check** (f) `()` :702

### phase9_weighted_arithmetic.py
*551 lines*
> Imports: `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, `from torch.utils.data`, `random`, `time`, `json`, `os`, `from typing`, `from dataclasses`, `sys`...
- **compute_arithmetic_weights** (f) `(batch: torch.Tensor, lengths: torch.Tensor, add_top_weight: float, add_fetch_weight: float) → torch.Tensor` :59
- **train_stage_weighted** (f) `(model: MicroTransformer, train_data: TraceDataset, val_data: TraceDataset, stage: int, add_top_weight: float, add_fetch_weight: float, max_epochs: int, lr: float, batch_size: int, patience: int, max_wall_time: float, checkpoint_prefix: str, checkpoint_dir: str, resume: bool, verbose: bool) → CheckpointMeta` :117
- **run_stage** (f) `(stage: int, model: MicroTransformer, add_top_weight: float, add_fetch_weight: float, n_train: int, n_val: int, n_test: int, checkpoint_prefix: str, checkpoint_dir: str, verbose: bool) → Dict` :290
- **run_weight_sweep** (f) `(checkpoint_dir: str) → Dict` :387
