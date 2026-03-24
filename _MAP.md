# oaustegard-llm-as-computer-4954e88/
*Files: 31*

## Files

### CLAUDE.md
- LLM-as-Computer `h1` :1
- Muninn Boot `h2` :5
- Project Context `h2` :33
- Phases `h2` :47
- Development Notes `h2` :173

### FINDINGS.md
- Percepta "Can LLMs Be Computers?" — R&D Findings `h1` :1
- Context `h2` :5
- Phase 1: Convex Hull KV Cache — Does the Geometry Work? `h2` :12
- Phase 2: Parabolic Key Encoding — Numerical Precision `h2` :30
- Phase 2b: Breaking the Float32 Address Limit `h2` :61
- Phase 3: Cumulative Sum via Attention `h2` :97
- Summary: Primitive Viability `h2` :123
- Phase 4: Minimal Stack Machine via Attention `h2` :134
- Updated Summary: All Phases `h2` :203
- Phase 5: Trained Micro-Executor `h2` :215
- Updated Summary: All Phases `h2` :264
- Phase 6: Curriculum Learning `h2` :277
- Phase 7: Percepta Architecture (d=36, h=18, L=7) `h2` :419
- Phase 8: Micro-Op Trace Diagnostics — THE RETRIEVAL/ARITHMETIC SEPARATION `h2` :436
- Phase 9: Weighted Arithmetic Loss `h2` :457
- Inflection Point: Return to Compilation `h2` :481
- Phase 11: Compiled Executor (Numpy) `h2` :489
- Phase 12: Real PyTorch Compiled Transformer `h2` :507
- Phase 13: ISA Completeness `h2` :528
- Final Summary: All Phases `h2` :551
- Key Insights Across All Phases `h2` :569
- Files `h2` :579

### RD-PLAN.md
- R&D Plan: Prototyping 2D Convex Hull Attention for In-Model Execution `h1` :1
- Phase 1: Convex Hull KV Cache — Does the Geometry Work? ✅ `h2` :11
- Phase 2: Parabolic Key Encoding — Does Index Lookup Work? ✅ `h2` :21
- Phase 2b: Extended Addressing ✅ `h2` :29
- Phase 3: Cumulative Sum Attention — Tracking Running State ✅ `h2` :37
- Phase 4: Minimal Stack Machine via Attention ✅ `h2` :45
- Phases 5–9: The Training Detour ✅ `h2` :55
- Phase 11: Compiled Executor (Numpy) ✅ `h2` :101
- Phase 12: Real PyTorch Compiled Transformer ✅ `h2` :111
- Phase 13: ISA Completeness ✅ `h2` :119
- Success Criteria — Final Status `h2` :127
- Overall Conclusion `h2` :140

### README.md
- llm-as-computer `h1` :1
- What is this? `h2` :7
- Status `h2` :18
- Files `h2` :54
- Running `h2` :77
- Key Takeaways `h2` :101

### WRITEUP.md
- Yes, LLMs Can Be Computers. Now What? `h1` :8
- What We Built `h2` :22
- A Useful Wrong Turn `h2` :38
- Where Does Their Idea Lead? `h2` :48
- Where Does Architectural Review by AI Lead? `h2` :58
- The Bigger Question `h2` :72
- What We're Left With `h2` :92

### executor.py
> Imports: `torch, isa`
- **NumPyExecutor** (C) :43
  - **execute** (m) `(self, prog, max_steps=5000)` :49
- **CompiledModel** (C) :313
  - **__init__** (m) `(self, d_model=D_MODEL)` :332
  - **_compile_weights** (m) `(self)` :349
  - **forward** (m) `(self, query_emb, prog_embs, stack_embs)` :459
- **TorchExecutor** (C) :570
  - **__init__** (m) `(self, model=None)` :576
  - **execute** (m) `(self, prog, max_steps=5000)` :580

### isa.py
> Imports: `torch, typing, dataclasses`
- **program** (f) `(*instrs)` :35
- **CompiledAttentionHead** (C) :350
  - **__init__** (m) `(self, d_model=D_MODEL, head_dim=2, v_dim=1, use_bias_q=False)` :364
  - **forward** (m) `(self, query_emb, memory_embs)` :371
- **embed_program_token** (f) `(pos, instr)` :399
- **embed_stack_entry** (f) `(addr, value, write_order)` :414
- **embed_state** (f) `(ip, sp)` :425
- **compare_traces** (f) `(trace_a, trace_b)` :437
- **test_algorithm** (f) `(name, prog, expected, np_exec, pt_exec, verbose=False)` :447
- **test_trap_algorithm** (f) `(name, prog, np_exec, pt_exec, verbose=False)` :476

### phase10_digit_decomposition.py
> Imports: `torch, torch.utils.data, random, time, json`...
- **num_to_digits** (f) `(n: int, n_digits: int = N_DIGITS)` :65
- **digits_to_num** (f) `(digits: List[int])` :75
- **encode_digit** (f) `(d: int)` :83
- **decode_digit** (f) `(idx: int)` :88
- **encode_opcode** (f) `(op: int)` :95
- **decode_opcode** (f) `(idx: int)` :100
- **encode_special** (f) `(raw: int)` :107
- **encode_num_field** (f) `(val: int)` :118
- **decode_num_field** (f) `(tokens: List[int])` :123
- **microop_trace_to_digit_tokens** (f) `(trace: MicroOpTrace)` :133
- **generate_digit_data** (f) `(
    allowed_ops: Set[int],
    n_samples: int,
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
)` :159
- **DigitTraceDataset** (C) :188
  - **__init__** (m) `(self, sequences: List[List[int]], max_len: int = None)` :191
  - **__len__** (m) `(self)` :206
  - **__getitem__** (m) `(self, idx)` :209
- **DigitTransformerBlock** (C) :215
  - **__init__** (m) `(self, d_model, n_heads, d_ff, dropout)` :216
  - **forward** (m) `(self, x, mask)` :229
- **DigitTransformer** (C) :237
  - **__init__** (m) `(self, vocab_size: int = DIGIT_VOCAB_SIZE,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = None,
                 max_len: int = 300, dropout: float = 0.1)` :240
  - **forward** (m) `(self, x: torch.Tensor)` :268
- **evaluate_digit_execution** (f) `(
    model: DigitTransformer,
    test_progs: List[List[Instruction]],
    verbose: bool = False
)` :285
- **run_add_diagnostic_digit** (f) `(
    model: DigitTransformer,
    n_tests: int = 30,
    verbose: bool = True
)` :377
- **train_digit_stage** (f) `(
    model: DigitTransformer,
    train_data: DigitTraceDataset,
    val_data: DigitTraceDataset,
    stage: int,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 25,
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase10",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
)` :439
- **run_stage** (f) `(
    stage: int,
    model: DigitTransformer,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase10",
    checkpoint_dir: str = ".",
    verbose: bool = True,
)` :592
- **run_digit_experiment** (f) `(checkpoint_dir: str = ".")` :691

### phase11_compile_executor.py
> Imports: `torch, time, sys, os, phase4_stack_machine`...
- **encode_token** (f) `(raw)` :52
- **decode_token** (f) `(idx)` :59
- **CompiledExecutorNumpy** (C) :73
  - **__init__** (m) `(self)` :80
  - **execute** (m) `(self, prog, max_steps=1000)` :83
- **HardMaxAttention** (C) :172
  - **__init__** (m) `(self, d_model, head_dim=2)` :179
  - **forward** (m) `(self, x, causal_mask=None)` :187
- **CompiledTransformer** (C) :210
  - **__init__** (m) `(self)` :229
  - **describe** (m) `(self)` :245
- **HullKVCache** (C) :254
  - **__init__** (m) `(self, eps=1e-10)` :264
  - **write** (m) `(self, addr, value)` :270
  - **read** (m) `(self, addr)` :277
  - **read_fast** (m) `(self, addr)` :288
  - **__len__** (m) `(self)` :313
- **CompiledExecutorWithHull** (C) :317
  - **execute** (m) `(self, prog, max_steps=1000)` :324
- **ExtendedExecutor** (C) :401
  - **execute** (m) `(self, prog, max_steps=1000)` :411
- **FastExecutor** (C) :515
  - **__init__** (m) `(self)` :522
  - **execute** (m) `(self, prog, max_steps=1000)` :525
- **test_compiled_executor** (f) `()` :595
- **test_hull_executor** (f) `()` :637
- **test_extended_executor** (f) `()` :673
- **test_fast_executor** (f) `()` :775
- **benchmark_scaling** (f) `()` :811
- **main** (f) `()` :867

### phase12_percepta_model.py
> Imports: `torch, time, sys, os, phase4_stack_machine`
- **CompiledAttentionHead** (C) :123
  - **__init__** (m) `(self, d_model=D_MODEL, head_dim=2, v_dim=1, use_bias_q=False)` :137
  - **forward** (m) `(self, query_emb, memory_embs)` :145
- **PerceptaModel** (C) :173
  - **__init__** (m) `(self, d_model=D_MODEL)` :190
  - **_compile_weights** (m) `(self)` :220
  - **forward** (m) `(self, query_emb, prog_embs, stack_embs)` :319
- **embed_program_token** (f) `(pos, instr)` :404
- **embed_stack_entry** (f) `(addr, value, write_order)` :425
- **embed_state** (f) `(ip, sp)` :441
- **PerceptaExecutor** (C) :453
  - **__init__** (m) `(self, model=None)` :460
  - **execute** (m) `(self, prog, max_steps=1000)` :464
  - **_read_stack_top** (m) `(self, stack_embs_list, addr)` :553
- **PerceptaExtendedExecutor** (C) :575
  - **execute** (m) `(self, prog, max_steps=1000)` :582
- **PerceptaFullSequenceModel** (C) :653
  - **__init__** (m) `(self, d_model=D_MODEL)` :669
  - **_compile_weights** (m) `(self)` :689
  - **forward** (m) `(self, embeddings)` :721
- **inspect_weights** (f) `(model)` :760
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
> Imports: `torch, time, sys, os, phase4_stack_machine`...
- **embed_program_token_ext** (f) `(pos, instr)` :92
- **Phase13Executor** (C) :109
  - **execute** (m) `(self, prog, max_steps=5000)` :115
- **Phase13Model** (C) :229
  - **__init__** (m) `(self, d_model=D_MODEL)` :236
  - **_compile_weights** (m) `(self)` :257
  - **forward** (m) `(self, query_emb, prog_embs, stack_embs)` :362
- **Phase13PyTorchExecutor** (C) :420
  - **__init__** (m) `(self, model=None)` :423
  - **execute** (m) `(self, prog, max_steps=5000)` :427
- **fib** (f) `(n)` :509
- **make_fibonacci** (f) `(n)` :518
- **make_multiply** (f) `(a, b)` :566
- **make_power_of_2** (f) `(n)` :608
- **make_sum_1_to_n** (f) `(n)` :647
- **make_is_even** (f) `(n)` :698
- **compare_traces** (f) `(trace_a, trace_b)` :735
- **test_new_opcodes** (f) `()` :745
- **test_head_sp2** (f) `()` :823
- **test_algorithm** (f) `(name, prog, expected, np_exec, pt_exec, verbose=False)` :870
- **test_fibonacci** (f) `()` :899
- **test_multiply** (f) `()` :919
- **test_power_of_2** (f) `()` :939
- **test_sum_1_to_n** (f) `()` :959
- **test_is_even** (f) `()` :979
- **test_regression** (f) `()` :1000
- **test_model_summary** (f) `()` :1058
- **main** (f) `()` :1099

### phase14_extended_isa.py
> Imports: `torch, time, sys, os, phase4_stack_machine`...
- **embed_program_token_ext** (f) `(pos, instr)` :357
- **Phase14Executor** (C) :374
  - **execute** (m) `(self, prog, max_steps=5000)` :381
- **Phase14Model** (C) :649
  - **__init__** (m) `(self, d_model=D_MODEL)` :668
  - **_compile_weights** (m) `(self)` :686
  - **forward** (m) `(self, query_emb, prog_embs, stack_embs)` :803
- **Phase14PyTorchExecutor** (C) :915
  - **__init__** (m) `(self, model=None)` :918
  - **execute** (m) `(self, prog, max_steps=5000)` :922
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
- **test_trap_algorithm** (f) `(name, prog, np_exec, pt_exec, verbose=False)` :1624
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
> Imports: `time, json`
- **BruteForceKVCache** (C) :22
  - **__init__** (m) `(self)` :25
  - **add** (m) `(self, key: tuple, value: float)` :32
  - **_sync** (m) `(self)` :37
  - **query** (m) `(self, q: tuple)` :43
  - **__len__** (m) `(self)` :52
- **HullKVCache** (C) :58
  - **__init__** (m) `(self)` :69
  - **_key_id** (m) `(self, k)` :78
  - **add** (m) `(self, key: tuple, value: float)` :81
  - **_rebuild** (m) `(self)` :88
  - **query** (m) `(self, q: tuple)` :120
  - **__len__** (m) `(self)` :153
- **ParabolicKVCache** (C) :159
  - **__init__** (m) `(self)` :175
  - **add** (m) `(self, key: tuple, value: float)` :178
  - **query_direct** (m) `(self, index: int)` :182
  - **query_ternary** (m) `(self, q: tuple)` :186
  - **__len__** (m) `(self)` :219
- **test_correctness** (f) `()` :225
- **benchmark_query_scaling** (f) `()` :311
- **benchmark_execution_trace** (f) `()` :400
- **benchmark_scaling_fit** (f) `()` :445

### phase2_parabolic.py
> Imports: `json`
- **test_exact_retrieval** (f) `()` :14
- **test_precision_analysis** (f) `()` :75
- **test_overwrites** (f) `()` :110
- **test_noninteger** (f) `()` :173

### phase2b_address_limits.py
> Imports: `typing`
- **parabolic_encode** (f) `(j: int, dtype=np.float32)` :23
- **parabolic_query** (f) `(i: int, dtype=np.float32)` :27
- **find_breakpoint** (f) `(encode_fn, query_fn, max_n=200_000, dtype=np.float32)` :31
- **OffsetParabolicSegment** (C) :89
  - **__init__** (m) `(self, center: int, radius: int, dtype=np.float32)` :92
  - **encode** (m) `(self, j: int)` :97
  - **query** (m) `(self, i: int)` :101
  - **covers** (m) `(self, addr: int)` :105
- **SegmentedMemory** (C) :109
  - **__init__** (m) `(self, max_addr: int, segment_size: int = 6000, dtype=np.float32)` :116
  - **write** (m) `(self, addr: int, value: int)` :134
  - **read** (m) `(self, addr: int)` :145
- **ResidualAddressMemory** (C) :171
  - **__init__** (m) `(self, block_size: int = 5000, dtype=np.float32)` :180
  - **_split** (m) `(self, addr: int)` :188
  - **write** (m) `(self, addr: int, value: int)` :191
  - **read_via_attention** (m) `(self, addr: int)` :197
  - **max_addressable** (m) `(self)` :226
- **hybrid_encode** (f) `(j: int, modulus: int = 5000, scale: float = 100000.0,
                  dtype=np.float32)` :239
- **hybrid_query** (f) `(i: int, modulus: int = 5000, scale: float = 100000.0,
                 dtype=np.float32)` :244
- **test_baseline** (f) `()` :256
- **test_segmented** (f) `(max_addr: int = 50000)` :266
- **test_residual** (f) `(max_addr: int = 50000)` :291
- **test_stress_residual** (f) `()` :317
- **test_offset_breakpoint** (f) `()` :355
- **main** (f) `()` :383

### phase3_cumsum.py
> Imports: `json`
- **cumsum_via_attention** (f) `(deltas)` :21
- **cumsum_via_attention_vectorized** (f) `(deltas)` :45
- **test_basic_correctness** (f) `()` :58
- **test_numerical_drift** (f) `()` :86
- **test_realistic_stack** (f) `()` :131
- **test_alternative_cumsum** (f) `()` :171

### phase4_stack_machine.py
> Imports: `typing, dataclasses`
- **program** (f) `(*instrs)` :69
- **ReferenceExecutor** (C) :134
  - **execute** (m) `(self, prog: List[Instruction], max_steps: int = 1000)` :137
- **ParabolicMemory** (C) :180
  - **__init__** (m) `(self, dtype=np.float64)` :189
  - **write** (m) `(self, addr: int, value: int)` :196
  - **read** (m) `(self, addr: int)` :204
  - **read_second** (m) `(self, addr: int)` :227
- **SequentialState** (C) :250
  - **__init__** (m) `(self, initial: int = 0)` :257
  - **update** (m) `(self, delta: int)` :261
  - **current** (m) `(self)` :265
  - **at** (m) `(self, step: int)` :268
- **AttentionExecutor** (C) :274
  - **execute** (m) `(self, prog: List[Instruction], max_steps: int = 1000)` :287
- **HandWiredTransformer** (C) :397
  - **__init__** (m) `(self)` :415
  - **describe_weight_structure** (m) `(self)` :432
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
> Imports: `torch, torch.utils.data, random, time, math`...
- **encode_token** (f) `(raw: int)` :62
- **decode_token** (f) `(idx: int)` :78
- **random_program** (f) `(min_len: int = 3, max_len: int = 12, 
                   max_push_val: int = 50)` :97
- **generate_dataset** (f) `(n_samples: int, max_prog_len: int = 12,
                     max_push_val: int = 50)` :147
- **TraceDataset** (C) :170
  - **__init__** (m) `(self, sequences: List[List[int]], max_len: int = None)` :173
  - **__len__** (m) `(self)` :190
  - **__getitem__** (m) `(self, idx)` :193
- **MicroTransformer** (C) :199
  - **__init__** (m) `(self, d_model: int = 32, n_heads: int = 4, 
                 n_layers: int = 2, d_ff: int = None, 
                 max_len: int = 200, dropout: float = 0.1)` :206
  - **forward** (m) `(self, x: torch.Tensor)` :235
- **TransformerBlock** (C) :254
  - **__init__** (m) `(self, d_model, n_heads, d_ff, dropout)` :255
  - **forward** (m) `(self, x, mask)` :268
- **train_model** (f) `(model: MicroTransformer, train_data: TraceDataset,
                val_data: TraceDataset, epochs: int = 100,
                lr: float = 3e-4, batch_size: int = 64,
                patience: int = 15, verbose: bool = True)` :281
- **evaluate_execution** (f) `(model: MicroTransformer, test_progs: List[List[Instruction]],
                      verbose: bool = False)` :383
- **analyze_attention** (f) `(model: MicroTransformer, sample_prog: List[Instruction])` :469
- **main** (f) `()` :494

### phase6_curriculum.py
> Imports: `torch, torch.utils.data, random, time, json`...
- **constrained_random_program** (f) `(
    allowed_ops: Set[int],
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
)` :49
- **generate_stage_data** (f) `(
    allowed_ops: Set[int],
    n_samples: int,
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
)` :98
- **save_checkpoint** (f) `(
    model: MicroTransformer,
    optimizer,
    meta: CheckpointMeta,
    path: str
)` :143
- **load_checkpoint** (f) `(path: str, model: MicroTransformer, optimizer=None)` :157
- **train_stage** (f) `(
    model: MicroTransformer,
    train_data: TraceDataset,
    val_data: TraceDataset,
    stage: int,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 20,
    max_wall_time: float = 500.0,  # stop before 600s bash tool limit
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
)` :168
- **run_stage** (f) `(
    stage: int,
    model: MicroTransformer,
    n_train: int = 1000,
    n_val: int = 150,
    n_test: int = 50,
    checkpoint_dir: str = ".",
    verbose: bool = True,
)` :359
- **run_all_stages** (f) `(checkpoint_dir: str = ".")` :450

### phase7_percepta_arch.py
> Imports: `torch, torch.utils.data, random, time, json`...
- **train_stage** (f) `(
    model: MicroTransformer,
    train_data: TraceDataset,
    val_data: TraceDataset,
    stage: int,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 20,
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase7",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
)` :76
- **run_add_diagnostic** (f) `(model: MicroTransformer, n_tests: int = 30,
                       verbose: bool = True)` :235
- **run_stage** (f) `(
    stage: int,
    model: MicroTransformer,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase7",
    checkpoint_dir: str = ".",
    verbose: bool = True,
)` :298
- **run_percepta_curriculum** (f) `(checkpoint_dir: str = ".")` :390

### phase8_microop_traces.py
> Imports: `torch, torch.utils.data, random, time, json`...
- **MicroOpExecutor** (C) :99
  - **execute** (m) `(self, prog: List[Instruction], max_steps: int = 1000)` :102
- **generate_microop_data** (f) `(
    allowed_ops: Set[int],
    n_samples: int,
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
)` :167
- **evaluate_microop_execution** (f) `(
    model: MicroTransformer,
    test_progs: List[List[Instruction]],
    verbose: bool = False
)` :197
- **run_add_diagnostic_microop** (f) `(
    model: MicroTransformer,
    n_tests: int = 30,
    verbose: bool = True
)` :274
- **train_stage** (f) `(
    model: MicroTransformer,
    train_data: TraceDataset,
    val_data: TraceDataset,
    stage: int,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 20,
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase8",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
)` :331
- **run_stage** (f) `(
    stage: int,
    model: MicroTransformer,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase8",
    checkpoint_dir: str = ".",
    verbose: bool = True,
)` :482
- **run_microop_curriculum** (f) `(checkpoint_dir: str = ".")` :573
- **sanity_check** (f) `()` :702

### phase9_weighted_arithmetic.py
> Imports: `torch, torch.utils.data, random, time, json`...
- **compute_arithmetic_weights** (f) `(
    batch: torch.Tensor,
    lengths: torch.Tensor,
    add_top_weight: float = 20.0,
    add_fetch_weight: float = 5.0,
)` :59
- **train_stage_weighted** (f) `(
    model: MicroTransformer,
    train_data: TraceDataset,
    val_data: TraceDataset,
    stage: int,
    add_top_weight: float = 20.0,
    add_fetch_weight: float = 5.0,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 25,
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase9",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
)` :117
- **run_stage** (f) `(
    stage: int,
    model: MicroTransformer,
    add_top_weight: float = 20.0,
    add_fetch_weight: float = 5.0,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase9",
    checkpoint_dir: str = ".",
    verbose: bool = True,
)` :290
- **run_weight_sweep** (f) `(checkpoint_dir: str = ".")` :387

### programs.py
> Imports: `math, isa`
- **test_basic** (f) `()` :30
- **test_push_halt** (f) `()` :35
- **test_push_pop** (f) `()` :40
- **test_dup_add** (f) `()` :45
- **test_multi_add** (f) `()` :50
- **test_stack_depth** (f) `()` :55
- **test_overwrite** (f) `()` :60
- **test_complex** (f) `()` :65
- **test_many_pushes** (f) `()` :71
- **test_alternating** (f) `()` :79
- **fib** (f) `(n)` :103
- **make_fibonacci** (f) `(n)` :112
- **make_power_of_2** (f) `(n)` :149
- **make_sum_1_to_n** (f) `(n)` :176
- **make_multiply** (f) `(a, b)` :202
- **make_is_even** (f) `(n)` :232
- **make_native_multiply** (f) `(a, b)` :261
- **make_native_divmod** (f) `(a, b)` :271
- **make_native_remainder** (f) `(a, b)` :288
- **make_native_is_even** (f) `(n)` :305
- **make_factorial** (f) `(n)` :320
- **make_gcd** (f) `(a, b)` :349
- **make_compare_eqz** (f) `(a)` :375
- **make_compare_binary** (f) `(op, a, b)` :384
- **make_native_max** (f) `(a, b)` :407
- **make_native_abs** (f) `(n)` :427
- **make_native_clamp** (f) `(val, lo, hi)` :445
- **make_bitwise_binary** (f) `(op, a, b)` :471
- **make_popcount_loop** (f) `(n)` :493
- **make_bit_extract** (f) `(n, bit_pos)` :519
- **make_native_clz** (f) `(n)` :535
- **make_native_ctz** (f) `(n)` :543
- **make_native_popcnt** (f) `(n)` :551
- **make_native_abs_unary** (f) `(n)` :559
- **make_native_neg** (f) `(n)` :567
- **make_select** (f) `(a, b, c)` :575
- **make_select_max** (f) `(a, b)` :586
- **make_log2_floor** (f) `(n)` :600
- **make_is_power_of_2** (f) `(n)` :615

### test_consolidated.py
> Imports: `sys, os, time, isa, executor`...
- **test_numpy_equivalence** (f) `()` :48
- **test_torch_equivalence** (f) `()` :253
- **test_new_np_vs_new_pt** (f) `()` :356
- **main** (f) `()` :431

## Other Files

- phase10_results.json
- phase6_results.json
- phase6b_results.json
- phase7_results.json
- phase8_results.json
- phase9_results.json
- requirements.txt

