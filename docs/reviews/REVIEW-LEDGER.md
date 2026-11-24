================================================================================
taskREVIEW-LEDGER  -  consolidated verdicts for the five completed review items
================================================================================
Compiled      : 2026-11-23 (PST)
Corrected     : 2026-11-23 22:31 PST (CORRECTIONS section below; two entries
                rewritten, one verdict downgraded)
Host          : mi50 (3x AMD MI50 32GB, gfx906)
Lane          : documentation / consolidation ONLY. No GPU work performed.
                No lock taken, no llama-server started or stopped, no builds, no
                git commit, no push. /llama-swap untouched.
Method        : every figure below re-derived from the cited artifact, not copied
                from prior prose. Where a supplied figure disagreed with its
                artifact, THE ARTIFACT WINS and the disagreement is flagged in a
                DISCREPANCY block inside that entry.

================================================================================
CORRECTIONS  (2026-11-23 22:31 PST - two errors, both in this ledger's own text)
================================================================================
This section supersedes the body where they conflict. Two entries were WRONG in
the 22:20 compile and are fixed below. Neither fix changes item 1, 3 or 5.

C-1. ITEM 2 - ACCEPTANCE RATES RESTORED to 33.1160 / 33.6209 percent.
     The 22:20 pass wrote q4_0 = 29.5410 percent (1815/6144) and q8_0 = 30.6152
     percent (1881/6144) and asserted that "the ~33% figure appears NOWHERE". Both
     statements were WRONG. The 6144 denominator is bogus: 6144 = 4 x 1536 is the
     sum of DEPTH counter slots, not drafted-token count, and 1815/1881 are summed
     ACCEPTED TOKENS, so the division mixed an accepted-token numerator with a
     step-count-times-depth denominator. Correct denominator = sum of
     resp_draft_n over the 1536 kind=='tf' rows.
     RE-VERIFIED from the jsonl files directly (python3, read-only):
         q4_0: 3352 accepted / 10122 drafted = 33.1160 percent
         q8_0: 3390 accepted / 10083 drafted = 33.6209 percent
     These MATCH the original brief figures 33.12 / 33.62. The brief was RIGHT
     and this ledger's "CRITICAL DISCREPANCY" claim was the error.
     Gap = 33.6209 - 33.1160 = +0.5049 pp in favour of q8_0 (q8_0 is HIGHER).
     CONCLUSION UNCHANGED AND STRENGTHENED: there is no +25 percent acceptance win
     from q4_0 KV; the claim stays REFUTED as path confounding.
     RECORDED SO IT IS NOT REINTRODUCED: never divide accepted-token sums by
     4 x step_count. Check that the denominator is the pooled resp_draft_n.
     Side effect of the same bad aggregation: the per-depth marginal rates, the
     paired token-diff test and the freegen/repetition numbers in item 2 were all
     computed off the wrong base and are re-derived in the rewritten item 2 below.

C-2. ITEM 4 / 5A - VERDICT DOWNGRADED from MEASURED WIN to NOT PROVEN.
     The 22:20 pass recorded +11.00 percent decode (57.9284 vs 52.1895) as a
     measured win parked only for a missing quiet-box ledger. That ledger NOW
     EXISTS (/root/taskR9-LEDGER.md, landed 22:20-22:30 PST after the compile) and
     its 14-leg ABBA-counterbalanced quiet-box sweep CONTRADICTS the win:
         champ median 53.56 vs lds median 56.61 = +5.70 percent
         Welch t = 1.065 (n.s.); 3 of 7 paired reps NEGATIVE
         lds arm is BIMODAL: fast mode 60.03/60.11/60.24, slow mode
         47.22/51.32/51.52, on IDENTICAL gate witnesses, VRAM and acceptance.
     The +11 percent was reachable only by comparing a derived champion median
     against the LDS FAST MODE cluster - a selection artifact, not an effect.
     NEW VERDICT: NOT PROVEN / NOT ADOPTION-SAFE. Do not adopt on this evidence.

     ALSO RECORDED (new, was unresolved in the 22:20 compile): the 58.20 vs 53.5
     champion-baseline gap is EXPLAINED, not contention. It is the documented
     draft/target WALK REGRESSION in taskG4-PHASE2-LEDGER.md section 3. See
     ITEM 5A section "58.20 DISCREPANCY - EXPLAINED".

--------------------------------------------------------------------------------
VERDICT SUMMARY  (read this first)
--------------------------------------------------------------------------------
 1 SKINNY KERNEL            REFUTED at the pass bar             (item 1)
 2 TEACHER-FORCED ACCEPT    +25% CLAIM REFUTED (path confound)  (item 2)
                            acceptance 33.1160 vs 33.6209 pct;
                            q8_0 HIGHER by +0.5049 pp
 3 PINNED DRAFTER           DETERMINISTIC-BUT-SEED-SENSITIVE    (item 3)
 4 FLOOR-ON + LDS           NOT PROVEN / NOT ADOPTION-SAFE      (item 4/5A)
                            quiet-box sweep: +5.70 pct median,
                            t=1.065 n.s., LDS BIMODAL 47.2-60.2
 5 Q4STACK -16.6%           MARGINAL EFFECT, measured; NOT a   (item 5)
                            q4stack build, NOT kernel-caused

 3 taskD1 CORRECTIONS       v_dot8 FALSE claim struck; i8narrow  (corrections)
                            NOT TESTED (not refuted); canonical
                            probe hash is NOT a numerics gate.

--------------------------------------------------------------------------------
SOURCES  (all paths absolute, as read for this ledger)
--------------------------------------------------------------------------------
  item 1  /root/int4-r1-wt/taskR1-SKINNY-LEDGER.md
          NOTE: the dispatch brief gave this path as /root/taskR1-SKINNY-LEDGER.md,
          which DOES NOT EXIST. The real file is under the int4-r1-wt worktree.
          See DISCREPANCY in item 1.
  item 2  /root/taskR4/q4kv-p-d4.tf.jsonl   (1600 records)
          /root/taskR4/q8kv-p-d4.tf.jsonl   (1600 records)
          /root/taskR4/r4-agg.py            (project aggregator, re-run here)
  item 3  /root/taskR8-LEDGER.md
          /root/r8b/logs/*.acc              (38 files present)
          /root/int4-noquant-wt/src/llama-sampler.cpp
          /root/int4-noquant-wt/common/speculative.cpp
          /root/int4-noquant-wt/src/llama-context.cpp
  item 4  /root/taskR9-LEDGER.md            (quiet box, DECISIVE - section 4/5/7)
          /root/r9/parsed.json              (14 legs, per-leg timings)
          /root/r9/logs/*.resp.json, *.gate.witness, *.samples
          /root/taskR6/*.agg.json           (contended, SUPERSEDED)
          /root/taskR6/probe-*.sha16
          /root/wt-g4-phase2/taskG4-PHASE2-LEDGER.md section 3
                                            (walk regression, for the 58.20 gap)
  item 5  /root/wt-g4-phase2/taskU-LEDGER.md  (section 3.3 especially)
          /root/wt-g4-phase2/taskD1-LEDGER.md (the struck claim)
          /root/int4-noquant-wt/ggml/src/ggml-cuda/common.cuh
          /root/wt-k/taskB2-LEDGER.md        (i8narrow stock head-to-head)


================================================================================
ITEM 1 - SKINNY KERNEL:  REFUTED AT THE PASS BAR
================================================================================
SOURCE: /root/int4-r1-wt/taskR1-SKINNY-LEDGER.md
        Branch int4-r1-skinny from int4-noquant tip 64f522e88b. NOT pushed.

VERDICT: REFUTED. Built to spec, sealed, bit-matched, and still lost the bar.

WHAT WAS BUILT (spec compliance, all 6 reviewer points present)
  - Interleaved repack: one wave step reads one contiguous 1 KB run
    (64 lanes x 16 B); f16 d plane and signed int16 nib-sum plane interleaved.
  - One lane per output row; 4 waves/block, 256 threads.
  - Wave-uniform activation loads held in SGPRs (ISA: s_load_dwordx4).
  - v_dot8_i32_i4 on int4 activations via ggml_cuda_dp8a (i8split identity
    S = 16*D_hi + D_lo + 8*SUMW, sealed on CPU).
  - Two-level split-K: grid.z KSPLIT x per-wave sub-segment; ks in {1,2,4,8}.
  - Per-wave K start rotation: ROT=7 arm vs ROT=0 control.

NUMERIC GATES (passed before any perf claim)
  - CPU seal 7/7 PASS (r1-cpu-seal.cpp, g++ -O2 -std=c++17 -ffp-contract=off).
    S1 dot8 identity 20000 cases 0 bad; S2 sw plane == nibble-sum-256;
    S3 repack permutation bijection; S4 fp32 mirror vs fp64 oracle;
    S5 rot0 vs rot7 deterministic; S6 two runs bit-identical;
    S7 KSPLIT/ROT grouping coverage.
  - GPU: fp64 stock-oracle band, CPU-mirror BIT-MATCH, two-run bit determinism.
  - Cross-process hash identity: 162/162 cells IDENTICAL (rep1 pre-pause binary
    vs repA/B/C committed rebuild); rep2/rep3 also 162/162 IDENTICAL.
  - => layout/repack is sound. The shortfall is not a correctness or packing bug.

PASS BAR: beat stock at T=5 AND T=8, AND reach >= 70 pct of the read-bound
          roofline.  RESULT: FAIL on all 6 shapes.

MEASURED TABLE  (absolute values; us and GB/s are the arm's own; vs-stock and
pct_roof re-derived from these columns and reconciled to the ledger - no
discrepancies, all 18 rows):
  shape        T  best r1 arm            r1 us   GB/s    vs stock  pct roof  stock us/GB-s     roof GB/s
  ffn_gate/up  1  r1-ks4-rot7             84.7  591.8    1.23x      93.4%   103.9 / 482.5     633.7
  ffn_gate/up  5  r1-ks4-rot7            172.6  290.4    1.27x      46.4%   218.9 / 229.0     626.3
  ffn_gate/up  8  r1-ks2-rot7-nosgpr     260.6  192.4    1.17x      30.4%   305.9 / 163.9     632.6
  ffn_down     1  r1-ks4-rot7             78.5  638.6    1.62x      92.9%   127.0 / 394.7     687.2
  ffn_down     5  r1-ks8-rot7            198.3  252.9    1.15x      36.8%   228.6 / 219.3     687.4
  ffn_down     8  r1-ks2-rot7-nosgpr     251.0  199.8    1.32x      29.1%   330.8 / 151.5     685.9
  attn_qkv     1  r1-ks2-rot7-nosgpr      49.8  592.0    1.34x      91.5%    66.7 / 442.4     646.7
  attn_qkv     5  r1-ks8-rot7            105.9  278.5    1.24x      43.1%   131.1 / 224.9     646.8
  attn_qkv     8  r1-ks2-rot7-nosgpr     143.5  205.6    1.34x      31.8%   192.9 / 152.9     646.7
  attn_gate    1  r1-ks4-rot7             31.9  555.3    1.34x      86.2%    42.8 / 413.4     644.2
  attn_gate    5  r1-ks8-rot7             68.1  259.8    1.18x      40.3%    80.5 / 219.8     645.2
  attn_gate    8  r1-ks8-rot7            106.6  166.1    1.12x      25.7%   118.9 / 148.8     645.2
  attn_output  1  r1-ks4-rot7             31.8  555.7    1.36x      85.3%    43.1 / 410.5     651.8
  attn_output  5  r1-ks8-rot7             73.4  241.0    1.10x      37.0%    81.1 / 218.1     651.5
  attn_output  8  r1-ks2-rot7-nosgpr      88.5  200.1    1.45x      30.8%   128.7 / 137.5     650.2
  attn_k/v     1  r1-ks8-rot7             12.0  245.6    0.93x      43.0%    11.2 / 263.9     571.2
  attn_k/v     5  r1-ks8-rot7             23.7  124.7    0.85x      21.8%    20.1 / 146.9     571.5
  attn_k/v     8  r1-ks8-rot7             30.8   95.8    0.84x      17.1%    26.0 / 113.5     560.4

WHY IT FAILED (two independent sub-conditions)
  (a) Beat stock at T=5 AND T=8: PASS on 5 of 6 shapes.
      T=5 on the 5 large shapes: 1.10x - 1.27x.
      T=8 on the 5 large shapes: 1.12x - 1.45x.
      FAIL on attn_k/v (K=5120, N=1024): LOST to stock at all three widths --
        T=1 12.0 us / 245.6 GB-s  vs stock 11.2 us / 263.9 GB-s  = 0.93x
        T=5 23.7 us / 124.7 GB-s  vs stock 20.1 us / 146.9 GB-s  = 0.85x
        T=8 30.8 us /  95.8 GB-s  vs stock 26.0 us / 113.5 GB-s  = 0.84x
  (b) >= 70 pct of the read-bound roofline: FAIL on EVERY shape.
      Best pct_roof at T=5 = 46.4% (ffn_gate/up, 290.4 of 626.3 GB/s).
      Best pct_roof at T=8 = 31.8% (attn_qkv, 205.6 of 646.7 GB/s).
      The same arms reach 85.3% - 93.4% of roof at T=1 on those 5 shapes
      (e.g. ffn_down T=1: 638.6 of 687.2 GB/s = 92.9%).
      Roof arms read the IDENTICAL interleaved bytes with zero act/dot work,
      and measured 560.4 - 687.4 GB/s, inside the box's 574-743 GB/s
      weights-only band.
  => Shortfall is NOT layout/repack (T=1 hits 85-93% of the same roof on the
     same bytes). It is the T-scaling activation traffic + dot ALU + fp32
     epilogue cost. This matches the established D-wave finding that the mmvq
     family is load-latency + activation-re-read bound; SGPR acts + v_dot8 did
     not remove the T-scaling term.

SALVAGE LEAD (a LEAD, not an adoption claim)
  - T=1 decode-width: 1.23x - 1.62x vs stock mmvq at 85.3% - 93.4% of roof on
    5 of 6 shapes (all except attn_k/v, which is 0.93x).
  - This is decode-width, NOT verify-width. Any adoption needs the production
    matrix per box policy; out of scope for this lane. No e2e leg was run and
    no floor-ON wiring was attempted (bar failed at the microbench stage).
  - If ever wired: small-N shapes (attn_k/v class) must stay on stock mmvq.

LIMITS (stated, not laundered)
  - No e2e run, no KLD, no floor-ON wiring. Microbench verdict only.
  - 9 attn_q tensors [5120,12288] NOT benched separately; their verdict is
    INFERRED from attn_qkv (same K/nkb geometry, N-scaled grid). Inference, not
    measurement.
  - Run on ROCm1 only; ROCm0 excluded (degraded), ROCm3 dead.
  - Repeat spread: median 0.4 pct, mean 1.8 pct over 234 spec-arm cells. Roof
    diagnostic arms vary up to 27 pct, but best-roof GB/s per shape is stable,
    so the verdict is spread-insensitive.
  - The PREF (software-pipelined) arm was an OUT-OF-SPEC working-tree
    experiment; left uncommitted; does not change the verdict.

DISCREPANCY (item 1)
  - The dispatch brief cited /root/taskR1-SKINNY-LEDGER.md. That path does not
    exist. The artifact lives at /root/int4-r1-wt/taskR1-SKINNY-LEDGER.md.
    No figure conflict, path correction only.
  - The brief said "best 46.4% T=5, 32.3% T=8". The artifact's table gives best
    T=8 = 30.8% (attn_output, 200.1/650.2). Its own section-6 verdict text says
    "best 32.3 pct" at T=8. The two numbers in the SAME artifact disagree with
    each other. Recomputed from the table: 205.6/646.7 = 31.8% (attn_qkv) and
    200.1/650.2 = 30.8% (attn_output). NO cell yields 32.3%. The 32.3 figure
    does not reconcile to any row and is treated as ERRONEOUS. Correct best
    T=8 pct_roof = 31.8% (attn_qkv). Either way it is a FAIL vs the 70% bar, so
    the verdict is unaffected.


================================================================================
ITEM 2 - TEACHER-FORCED ACCEPTANCE:  '+25% FROM q4_0 KV' IS REFUTED
================================================================================
SOURCE: /root/taskR4/q4kv-p-d4.tf.jsonl, /root/taskR4/q8kv-p-d4.tf.jsonl
        Re-aggregated with the project's own /root/taskR4/r4-agg.py.

VERDICT: REFUTED AS PATH CONFOUNDING. q8_0 KV is HIGHER than q4_0 KV. There is
no +25% acceptance win from q4_0 KV on this path; the two arms are within
0.51 pp of each other, in the OPPOSITE direction to the claim.

CORRECTION NOTICE (item 2) - see also CORRECTIONS C-1 at the top of this file
  The 22:20 compile of this entry was WRONG and is corrected here.
  - It reported q4_0 29.5410% (1815/6144) and q8_0 30.6152% (1881/6144) and
    asserted "the ~33% figure appears NOWHERE". Both were errors.
  - Bad denominator: 6144 = 4 x 1536 (depth slots per step), not drafted tokens.
    Numerators 1815/1881 are summed ACCEPTED TOKENS, so that division was
    dimensionally invalid.
  - Re-verified from the jsonl files (python3, kind=='tf', 1536 rows each,
    denominator = sum of resp_draft_n):
        q4_0 KV : 3352 / 10122 = 33.1160%
        q8_0 KV : 3390 / 10083 = 33.6209%
    The original brief figures 33.12 / 33.62 were CORRECT.
  - Do not reintroduce the 6144 denominator. 6144 is a step/depth count, not a
    token count.
  - The earlier prose also claimed "only 1536 rows are teacher-forced" as if it
    undercut the campaign; that part VERIFIES (see below) and is not an error.

ARM DEFINITIONS AND SCALE (verified)
  - Each file: 1600 records = 1536 `tf` step rows + 32 `freegen` rows + 32
    `ref` rows. Verified by counting kind values in each file.
    tf=1536, freegen=32, ref=32 in BOTH arms.
  - So 1600 records/arm is correct at the file level and 1536 of 1600 (96.0%)
    are TF step rows. The other 64 rows are 32 free-run generations and 32
    shared reference continuations, which are NOT teacher-forced acceptance
    measurements and are excluded from the acceptance rate.
  - 32 prompts: wiki-01 .. wiki-32. 3 reps; 512 tf step rows per rep.
  - Draft depth M = 4 on all 1536 rows (n_max 4).
  - Campaign: q4kv-p-d4 started 2026-11-19T09:47:56, ended 10:45:06;
    q8kv-p-d4 started 10:45:36, ended 11:38:33. GPU0 placement verified.
    f16-KV arm STARTED then TERMINATED (rc=143); no f16-KV acceptance data.

MEASURED ACCEPTANCE (absolute) - 1536 tf rows per arm
  arm          accepted / drafted   accept rate   acc tokens per step
  q4kv-p-d4      3352 / 10122        0.331160       2.182292
  q8kv-p-d4      3390 / 10083        0.336209       2.207031
  Difference: +38 accepted tokens for q8_0 on 39 FEWER drafted tokens, i.e.
  q8_0 is +1.523% RELATIVE / +0.5049 pp ABSOLUTE above q4_0. The claim's
  DIRECTION (+25% from q4_0) is not attenuated - it is inverted.

MEAN ACCEPTED LENGTH COMPATIBILITY CHECK
  Drafted per step: q4_0 10122/1536 = 6.5897; q8_0 10083/1536 = 6.5645.
  Accepted per step: 2.1823 / 2.2070. Accepted length is below drafted length
  in both arms, as required. The two arms draft nearly the same number of
  tokens (0.4% apart), so the +0.5049 pp acceptance gap is not a length artifact.

REPETITION METRICS (re-derived after the C-1 correction; field = reps_word)
  `ref` rows = SHARED reference continuations, IDENTICAL in both arms by
  construction:
    longest_rep_gram mean 2.28, max 9, n=32
    distinct-2 0.962959 ; distinct-3 0.988325
  `freegen` rows (arm-specific free generation) DO differ:
    q4_0: longest_rep_gram mean 11.97, max 86 ; distinct-2 0.822347
    q8_0: longest_rep_gram mean 16.94, max 115 ; distinct-2 0.811631
  freegen acceptance (32 rows): q4_0 5412/10886 = 49.72% ;
                                q8_0 5405/10902 = 49.58%
  => The "longest repeated n-gram 2-3" figure describes ONLY the shared `ref`
     rows (mean 2.28). For ARM-GENERATED text it is much longer: mean 11.97
     (q4_0) and 16.94 (q8_0), up to 86 and 115 words respectively. The
     "2-3" characterization MUST NOT be applied to generated output.
     (The 22:20 compile carried different n-gram and distinct-n values from a
     bad aggregation; the values above are read straight from the jsonl.)

PER-DEPTH ACCEPTANCE (re-derived from the `n_accept_lines` field, n=1536 each)
  P(draft at depth j matches) = count(n_accept_lines >= j) / 1536.
  Note the depth indexing used by the artifact: n_accept_lines counts accepted
  draft lines INCLUDING the depth-1 line, which is accepted in every step
  (1536/1536 = 1.000000), so depth 1 carries no discriminating information.
    depth 1: q4_0 1536/1536 = 1.000000   q8_0 1536/1536 = 1.000000   diff 0.0000 pp
    depth 2: q4_0 1284/1536 = 0.835938   q8_0 1260/1536 = 0.820312   diff -1.5625 pp
    depth 3: q4_0  624/1536 = 0.406250   q8_0  636/1536 = 0.414062   diff +0.7812 pp
    depth 4: q4_0  144/1536 = 0.093750   q8_0  144/1536 = 0.093750   diff 0.0000 pp
  CORRECTION: the 22:20 compile printed a different depth table (q4_0 d1
  0.607422 ...) that does NOT reproduce from this field - it was derived
  through the same bad aggregator as the 6144 denominator. Use the values
  above.
  READING: the per-depth table is MIXED (q4_0 ahead at depth 2, q8_0 ahead at
  depth 3, tied at 1 and 4). It does NOT show q8_0 winning at every depth, and
  it is NOT the evidence for the verdict. The pooled rates are.

ACCEPTED-COUNT HISTOGRAM (per tf row, field `resp_draft_acc`, out of 1536 each)
  q4_0: {0: 54, 1: 326, 2: 559, 3: 480, 4: 117}
  q8_0: {0: 57, 1: 315, 2: 549, 3: 483, 4: 132}
  Weighted sums check: q4_0 0x54+1x326+2x559+3x480+4x117 = 3352 (matches);
  q8_0 0x57+1x315+2x549+3x483+4x132 = 3390 (matches). The histogram is
  consistent with the pooled numerators above. Both arms are dominated by 1-3
  accepted tokens per step, so neither is degenerate or wildly divergent.

LIMITS
  - ONE drafter (DFlash2), ONE target (Qwen3.8-27B Q4_0), depth M=4 only.
  - No f16-KV control arm (terminated). Path comparison is q4_0 vs q8_0 only.
  - The q4_0 and q8_0 arms drafted slightly different token counts
    (10122 vs 10083). The rates are pooled, not paired token-for-token; a
    per-cell paired test is not derivable from this schema.
  - Repetition evidence for arm-specific text is `freegen` (32 samples), a
    small sample; the `ref` distinct-n set is shared by construction and is not
    independent evidence about either arm.


================================================================================
ITEM 3 - PINNED-DRAFTER NONDETERMINISM:  DETERMINISTIC-BUT-SEED-SENSITIVE
================================================================================
SOURCE: /root/taskR8-LEDGER.md ; /root/r8b/logs/*.acc ; cited source lines in
        /root/int4-noquant-wt (re-opened and read directly for this ledger).

VERDICT: DETERMINISTIC-BUT-SEED-SENSITIVE. The claim of nondeterminism is REAL
as a symptom but MISATTRIBUTED: it is caused by an UNPINNED seed, not a
cross-device race. The suspected cross-device ordering hazard is covered by
three explicit barriers and is, by the code's own comment, already fixed.

SETUP
  Binary   /root/build-g4p2b/bin/llama-server (17888 B, 2026-11-17 13:16)
  Target   Qwen3.8-27B-Q4_0.gguf ; Drafter DFlash2-Q4_0-ownhead-q4.gguf (STOCK)
  Flags    --spec-type draft-dflash --spec-draft-n-max 4 -ctk q4_0 -ctv q4_0
           -ngl 99 -fa on --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio
  Sampling temperature 1.0, top_k 20, top_p 0.95, seeded dist sampler
           (deliberately NOT greedy, so the real sampling path is exercised).

FIXED-SEED RESULT (seed 42): BIT-IDENTICAL ON BOTH ARMS
  SINGLE-CARD (--device ROCm2 -devd ROCm2 -sm none)
    13/13 runs IDENTICAL, 105 steps, accepted-file sha256
      7ec57750c4d61bd4... (full sha verified by re-hashing r0 and r12: equal)
    acceptance 0.35800 = 150 accepted / 419 drafted  (re-verified from .acc)
  CROSS-CARD (--device ROCm1,ROCm2 -devd ROCm1 -sm tensor)
    13/13 runs IDENTICAL, 98 steps, accepted-file sha256
      d38af7fa0bc26cba... (full sha verified by re-hashing r0 and r12: equal)
    acceptance 0.40464 = 157 accepted / 388 drafted  (re-verified from .acc)
  Placement is genuinely cross-device: single r0 log shows q8_repack on
  "device 2" only; cross r0 log shows devices 1 AND 2.
  The 0.35800 vs 0.40464 difference between arms is a CONFIG difference (tensor
  split changes prefill numerics), not nondeterminism.

SEED IS THE ONLY MOVING PART
  Fixed alternate seeds each give a stable but different trajectory:
    single s123  100 steps 66e3f961fe0da609  acc 0.39043
    single s777   95 steps d419d0e519134fda  acc 0.42513
    single s2026  95 steps 6623e1c11dec515c  acc 0.42667
    cross  s123   91 steps 9ba3e97097bd8de8  acc 0.45429
    cross  s777   93 steps 33e9bec9d2114e5a  acc 0.43631
    cross  s2026  98 steps 93dac168ce847462  acc 0.40153

  SENTINEL 0xFFFFFFFF reproduces the original symptom, ON ONE CARD TOO:
    single r20 97 steps 28e417cda48c3694  acc 0.40568 (157/387, re-verified)
    single r21 97 steps 10ebc11f881d20e3  acc 0.41146 (158/384, re-verified)
    single r22 99 steps e25fc03d027ece84  acc 0.39594 (156/394, re-verified)
      -> 3/3 DIFFERENT in step count, per-step sequence AND acceptance
    cross  r20 107 steps / 0.34579 ; r21 93 steps / 0.43631 ; r22 92 / 0.44262
      -> 3/3 DIFFERENT
  Because the divergence appears on the SINGLE CARD, where no cross-device path
  exists, cross-device placement is EXONERATED as the cause.
  Mechanism (code read directly): src/llama-sampler.cpp:340-351, function
  get_rng_seed(uint32_t seed) -- `if (seed == LLAMA_DEFAULT_SEED)` builds a
  `std::random_device rd; return rd();` (falling back to system clock when
  random_device reports zero entropy). LLAMA_DEFAULT_SEED is 0xFFFFFFFF. So the
  sentinel draws a fresh seed EVERY run.

THREE EXPLICIT BARRIERS ON target-write -> drafter-read (all re-read at the
cited lines for this ledger):
  (a) common/speculative.cpp:1783  llama_synchronize(ctx_tgt)
      Unconditional full target-side barrier immediately before the feature
      gather. Comment at 1771-1782: "Must be a full synchronize. The gather
      below reads embd_layer_inp ... while the accum readiness event only
      covers accum rows below p_end ...". A narrow-wait variant measured
      23.4 -> 23.3 tg, i.e. nothing, "so there is no case for the risk".
  (b) src/llama-context.cpp:5161-5162  llama_get_embeddings_layer_inp() calls
      ctx->synchronize() -- a SECOND full barrier inside the exact API the
      drafter calls (call site common/speculative.cpp:1798), so a caller cannot
      skip it.
  (c) src/llama-context.cpp:3384 (block 3380-3388) ggml_backend_event_wait on
      every tap-readback event before the next graph overwrites the
      single-ubatch-deep tap buffer.

  The suspected hazard is documented at src/llama-context.cpp:3037-3045 as a
  COMMENT describing the old defect, ending: "... Handle it where it belongs,
  as a dependency rather than a stall - the event recorded above is handed to
  the next graph_compute, whose copy waits on it on the device." That is a
  historical note about a FIXED issue (the fix being barrier (c)); its presence
  plausibly seeded the race hypothesis. No unprotected write->read window was
  found. The one true peer-copy path (ggml_backend_cuda_cpy_tensor_async)
  records an event on the src stream and stream-waits the dst stream.

LIMITS (untested, NOT exonerated)
  - ONE prompt, ONE n_predict (256), ONE n_max (4).
  - Concurrent multi-slot serving: UNTESTED.
  - Adaptive depth (LLAMA_DFLASH_ADAPTIVE): UNTESTED.
  - Deferred prefill with prompts >~16k (lag-one flush active): UNTESTED.
  - "Deterministic" here means fixed-seed reproducibility on this config; it is
    NOT a claim that any two seeds agree, nor that acceptance is seed-invariant.

DISCREPANCY (item 3)
  - /root/taskR8-LEDGER.md section 2 states "40 .acc files present". The
    directory /root/r8b/logs contains 38 .acc files (counted directly:
    single s42 13, cross s42 13, 6 alternate-seed r0 files, 6 sentinel files).
    40 vs 38 - the extra 2 are not on disk. Does not affect any verdict number
    (the 42-seed and sentinel sets used above are all present and re-verified).
  - Minor: R8 ledger section 3 cites barrier (c) at "3383-3386"; the actual
    ggml_backend_event_wait call is at line 3384 within loop lines 3383-3386.
    Consistent.


================================================================================
ITEM 4 / 5A - FLOOR-ON + RP_Q4_NC_LDS=8:  NOT PROVEN / NOT ADOPTION-SAFE
================================================================================
SOURCES:
  [contended, superseded] /root/taskR6/champ-p-d4.bench-r{1,2,3}.jsonl.agg.json
        /root/taskR6/q4kvlds-p-d4.bench-r{1,2,3}.jsonl.agg.json
        /root/taskR6/probe-champ-p-d4.sha16, probe-q4kvlds-p-d4.sha16
  [quiet, DECISIVE] /root/taskR9-LEDGER.md  (landed 22:20-22:30 PST 2026-11-23,
        AFTER the 22:20 compile; this is the quiet-box lane the earlier pass
        recorded as missing)
        /root/r9/parsed.json                   (14 legs, per-leg timings)
        /root/r9/logs/*.resp.json              (14 raw /completion responses)
        /root/r9/logs/r9_{champ,lds}_r{1..7}.gate.witness  (per-leg gate proof)
        /root/r9/logs/r9_{champ,lds}_r{1..7}.samples       (5s VRAM/util)
        /root/r9/quietness.txt                 (pre-leg util, all zero)

VERDICT: NOT PROVEN / NOT ADOPTION-SAFE. The claimed +11.00% decode win from
floor-ON + RP_Q4_NC_LDS=8 does NOT reproduce under a 14-leg ABBA-counterbalanced
quiet-box sweep. The honest median gain is +5.70% and it is not statistically
significant. Worse, the LDS arm is BIMODAL: it runs in a fast mode near 60 t/s
on some legs and a slow mode at 47-51.5 t/s on others, on IDENTICAL gate state,
IDENTICAL VRAM and IDENTICAL acceptance. A fleet config that can randomly drop to
47 t/s is WORSE than a stable 53.5 t/s champion, even if its median is a bit
higher. DO NOT ADOPT ON THIS EVIDENCE.

*** METRIC CAVEAT - READ BEFORE COMPARING ANY NUMBER ***
  The 14-leg numbers in the brief and below are `tg_incl_draft` (the parsed
  per-response figure that excludes draft overhead), NOT `tg_ts`
  (predicted_n / predicted_ms). The two differ by about 5-10% and are NOT
  interchangeable. /root/taskR9-LEDGER.md section 3 tabulates tg_ts; the
  bimodality analysis below uses tg_incl_draft. Mixing them is how the
  earlier +11% figure was built. State which metric you are using.

MEASURED DECODE - 14 legs, 7 per arm, ABBA counterbalanced
  (source: /root/r9/parsed.json, field tg_incl_draft, from the 14 resp.json)
  arm     legs (tg_incl_draft t/s)                                    median
  champ   50.66  53.46  53.56  53.59  53.39  53.64  53.57             53.56
  lds     60.03  51.52  60.24  47.22  60.11  56.61  51.32             56.61
  Champion core (reps 2-7, dropping the cold pp r1) is TIGHT:
     53.39-53.64, spread 0.25 t/s (0.47%).
  LDS is NOT tight: spread 12.81 t/s (60.24 - 47.22) = 25.0% of its median.

  MEDIAN OF MEDIANS: 56.61 - 53.56 = +3.05 t/s = +5.695% (recorded as +5.70%).
  ALSO COMPUTED, for completeness (tg_ts basis, same 14 legs):
     champ median 48.721 vs lds median 49.390 = +0.669 t/s = +1.373%.
     These are NOT contradictory results - they are the same data on two
     metrics. The bimodality and the non-significance hold on BOTH bases.

  WELCH t ON tg_incl_draft (7 vs 7): t = 1.065 on the means
     (champ mean 53.124, sd 1.090, n=7; lds mean 55.293, sd 5.276, n=7;
      se of difference 2.036). NOT SIGNIFICANT (p ~ 0.31).
     (The R9 ledger reports t=1.85 on the tg_ts basis; same conclusion.)
  PAIRED PER-REP DELTAS (lds - champ, tg_incl_draft):
     +9.37  -1.94  +6.68  -6.37  +6.72  +2.97  -2.25
     3 of 7 are NEGATIVE. A real speedup does not flip sign on 43% of paired
     reps.

  HOW +11% WAS MANUFACTURED (selection artifact, shown to disqualify it)
     Earlier pass: champion mean 52.1895 t/s (contended taskR6) vs lds mean
     57.9284 t/s -> +11.00%. On the quiet box the comparable champion median is
     53.56, not 52.19, and the LDS "mean" 57.93 is only reachable by keeping the
     60.0-60.2 cluster. Median-vs-median gives +5.70%; lds_fast_mode(60.24) vs
     champ_median(53.56) gives +12.5%. That swing IS the artifact.
     Champion on the quiet box measured 48.72 (tg_ts) / 53.56 (tg_incl_draft),
     not 52.19 and not 58.20. The 52.19 champion was a contended-box reading.

BIMODALITY - THE CENTRAL FINDING (tg_incl_draft)
  FAST MODE (3 legs): 60.03, 60.24, 60.11  -> mean 60.13, tight (0.21 spread)
  SLOW MODE (3 legs): 47.22, 51.32, 51.52  -> mean 50.02, spread 4.30
  MID:                56.61 (1 leg)
  The fast cluster is as tight as the champion (0.21 vs 0.25 t/s spread), so
  these are TWO STABLE STATES, not a noisy continuum.

  THE MODES ARE NOT DISTINGUISHED BY ANYTHING IN THE HARNESS - verified:
  1. GATE WITNESSES IDENTICAL across all 7 LDS legs (byte-equal key lines):
       DOT8=3                (all 7)
       MIN_TOKENS=128        (all 7)
       RP_Q4_NC_LDS=8 ACTIVE (all 7), with the same per-width admission text:
         "W8: u4act T>=6, affine T=4..5; W4: u4act T=4,7,8, affine T=4,6..8"
       2 repack lines per leg (q4_0 tensors repacked on device 0 and device 1).
  2. VRAM IDENTICAL: every LDS leg samples g0=13051-13054 MiB,
     g1=9763-9766 MiB. Fast and slow legs are indistinguishable.
  3. ACCEPTANCE IDENTICAL: 0.42553 = 160 accepted / 376 drafted, mean accepted
     length 2.723 (parsed field mean_accepted_len; the log's own mean_len
     variant prints 2.68 as a decimal-summary artifact). All 14 legs.
  4. LOAD: pre-leg GPU util 0 on all three cards for all 14 legs
     (/root/r9/quietness.txt). Late-session load rose to ~7-8 on the last two
     legs, but those were lds r7 (49.39) and champ r7 (48.44/48.63) - mid-pack,
     NOT the slow-mode legs. Load does NOT predict the mode.
  VERIFIED CORRELATION, RECORDED AS A CAVEAT NOT A CAUSE: the parsed n_gen
  (generated tokens in the timed window) is 183/183/183 on the three fast legs
  and 145/156/157 on the three slow legs, so n_gen and speed correlate at
  r=0.999. This is a MEASUREMENT-WINDOW difference, not a proven cause: it says
  the fast legs produced ~26 more tokens in the same request, which is itself a
  symptom of a faster server, not an independent explanation. It does mean the
  two modes are NOT like-for-like token counts, and any future lane must pin
  n_predict/n_gen before claiming a decode rate.

  VERDICT IMPLICATION: the mode is NOT predicted by load, VRAM, gate state or
  acceptance. There is no harness signal that lets an operator know which mode a
  given start will land in. That is precisely what makes it NOT ADOPTION-SAFE.

ACCEPTANCE IS IDENTICAL ACROSS ARMS AND MODES (correctness is fine)
  All 14 legs: accept 160/376 = 0.42553, mean accepted length 2.723.
  The LDS gate is numerically neutral - that claim SURVIVES. What does not
  survive is the claim that it is reliably faster.

GREEDY PROBE
  probe-champ-p-d4.sha16    = 77841cb6a251235d
  probe-q4kvlds-p-d4.sha16  = 77841cb6a251235d
  Identical (see taskD1 correction 3: a reproducibility witness, not a
  discriminator between quant types).

--------------------------------------------------------------------------------
MECHANISM HYPOTHESIS - PER-WIDTH ADMISSION (HYPOTHESIS, UNTESTED)
--------------------------------------------------------------------------------
  The gate admits the LDS kernel PER WIDTH, not globally:
      W8: u4act T>=6, affine T=4..5 ; W4: u4act T=4,7,8, affine T=4,6..8
      (verbatim from every r9 gate witness; matches taskG4-PHASE2-LEDGER.md
      section 1)
  With --spec-draft-n-max 4 and the dflash drafter, the VERIFY WIDTH changes
  step-to-step, so different decode steps select DIFFERENT kernels - some steps
  take the LDS-staged twin, others the generic nc shape. Which widths dominate
  depends on the receipt/acceptance pattern of that particular request.
  PLAUSIBLE EXPLANATION OF BIMODALITY: a run that happens to spend more steps at
  an LDS-favourable width lands in fast mode; one that drifts to an
  LDS-unfavourable width lands in slow mode. This would also explain why
  acceptance is identical (same tokens accepted, different kernel mix) yet
  timing splits into two discrete states.
  STATUS: HYPOTHESIS ONLY, NOT TESTED. It is NOT consistent-by-construction with
  the identical 160/376 acceptance on every leg, so width mix alone does not
  fully determine the mode - the acceptance walk is identical while the mode is
  not. Do not cite this as the cause.
  CHEAP TEST (not run): --spec-draft-n-max 1 pins verify width 2, making LDS
  admission deterministic; if the bimodality vanishes at n-max 1 and returns at
  n-max 4, the width path is implicated.

--------------------------------------------------------------------------------
THE 58.20 DISCREPANCY - EXPLAINED (this was unexplained in the 22:20 compile)
--------------------------------------------------------------------------------
  The quiet sweep measured the champion at 48.72 (tg_ts) on its box and 53.5
  (tg_incl_draft) on mine, vs the historically recorded 58.20 t/s. This is NOT
  contention. It is the documented DRAFT/TARGET WALK REGRESSION.

  PRIMARY SOURCE: /root/wt-g4-phase2/taskG4-PHASE2-LEDGER.md section 3
  ("BASELINE ANOMALY -- uwave->tip walk regression"), verified by direct read.
  That section records, on this exact build lineage:
    - d4-ctrl measured 40.14 t/s at acceptance 0.348, against the adopt
      a-ctrl historical 57.41 t/s at acceptance 0.4377 with a byte-identical
      env/args/files/pair (verified by witness diff in that ledger).
    - Per-prompt acceptance shifts are NON-UNIFORM (dialogue 0.642 -> 0.265,
      econ 0.661 -> 0.299, math 0.578 -> 0.709, code 0.888 -> 0.849), which is
      the signature of a draft/target walk divergence, not a speed effect.
    - The divergence was introduced somewhere in the uwave (e9b3e3197, Nov 14)
      -> tip merge range: k1-headsplit, k2-headkernel, k3-q6lds, fix4,
      d0-determ, dg-graphreplay, d1-vecdot.
    - The 64-token greedy probe is canonical on both eras, so the divergence is
      beyond the probe horizon - which is why the probe hash (77841cb6...) does
      not flag it.

  CORROBORATION BY ACCEPTANCE (acceptance is deterministic; timing is not):
    pre-regression reference : 0.4377  (3888/8883, taskR6/taskW3 family)
    fully-regressed reference : 0.3480  (taskG4-PHASE2-LEDGER.md section 3,
                                         d4-ctrl / d4-cand)
    OBSERVED on this build   : 0.4255  (160/376, all 14 r9 legs)
  0.4255 sits BETWEEN 0.4377 and 0.3480, close to the pre-regression end -
  exactly what a PARTIALLY regressed walk predicts. The build in the quiet
  sweep is /root/build-g4p2b (src_sha 94d1dbaa38..., Nov 17), which lies INSIDE
  the uwave->tip merge range named above.
  CONCLUSION: the 58.20 reference is NOT reproducible on this binary BY
  CONSTRUCTION of the build lineage - it predates the regression. Comparing
  quiet-box absolutes to 58.20 compares across a known regression and will
  always show a ~10-16% "loss" that is not contention and not the LDS gate.
  CONSEQUENCE FOR THIS ITEM: the champion baseline is ~53.5 (tg_incl_draft) /
  48.7 (tg_ts) on this build, NOT 58.20 and NOT the contended 52.19. All LDS
  comparisons above are already made within-build, which is why the +5.70%
  figure is the one to trust.
  A dedicated bisect lane is still owed (taskG4-PHASE2-LEDGER.md section 3
  RECOMMENDs it); until it lands, treat absolute t/s as build-specific.

LIMITS
  - 7 legs per arm; one prompt (112 tokens), n_predict 256, one n-max (4).
  - Two metrics in conflict within the same artifacts (tg_ts vs tg_incl_draft);
    both bases give non-significance, but the effect size differs (+1.4% vs
    +5.7%). Report both.
  - n_gen differs between LDS modes (183 vs 145-157), so the two modes are not
    like-for-like token counts.
  - The external load spike (load ~7-8) hit the last leg of each arm.
  - Acceptance identity is a correctness signal; token-level logit hashes were
    not compared.
  - One build (build-g4p2b), d4 width, TP2 on ROCm0/ROCm1 only.

WHAT WOULD MAKE THIS ADOPTION-SAFE (not done)
  1. >= 10 reps/arm, discard the cold first leg, add a fixed 3 x 256-token
     warm-up before the measured request.
  2. Pin verify width (--spec-draft-n-max 1) to test the per-width hypothesis.
  3. Pin and report n_gen; never compare legs with different generated counts.
  4. Report median-of-medians only; never best-of-N vs median.
  5. Explain WHY the mode splits before shipping the gate to a fleet.

CORRECTION NOTICE (item 4)
  The 22:20 compile recorded this as "MEASURED WIN (+11.00% decode) ... PARKED as
  PENDING CONFIRMATION" and listed /root/taskR9-LEDGER.md as NOT EXISTING. The
  file existed by 22:30 and CONTRADICTS the win. The verdict is DOWNGRADED to
  NOT PROVEN / NOT ADOPTION-SAFE. The +11.00% figure is itself a selection
  artifact and is STRUCK as an effect estimate. See CORRECTIONS C-2 at the top.



================================================================================
ITEM 5 - Q4STACK -16.6%:  MEASURED MARGINAL EFFECT, NOT A BUILD, NOT KERNELS
================================================================================
SOURCE: /root/wt-g4-phase2/taskU-LEDGER.md, section 3.3 (and 3.1 grid)

VERDICT: The -16.6% is NOT "a q4stack build underperforming". It is the MARGINAL
EFFECT of adding the LDS pair (REPACK_Q8_0=1 STOCK_ACC=0 + RP_Q8_NC_LDS=8) ON
TOP OF q4_0-KV at floor-ON at d4. Combining the two levers is worse than either
alone, because at floor-ON they attack the SAME bottleneck (substitutes).

THE NUMBER
  floor-ON, LDS OFF, q4_0 KV : 56.43 t/s  (acceptance 0.4377, mean len 2.7508)
  floor-ON, LDS ON,  q4_0 KV : 47.08 t/s  (acceptance 0.3913, mean len 2.5652)
  Marginal effect of the LDS pair = (47.08 - 56.43) / 56.43 = -16.57%
  = -9.35 t/s.
  At d7 the same marginal is -1.39% (38.20 -> 37.67 t/s = -0.53 t/s), so the
  penalty is width-specific (sharp at width 5, negligible at width 8).

WHY (recorded cause, from section 3.3)
  - At floor-ON, the LDS pair and q4_0-KV are SUBSTITUTES: both attack the same
    verify-width bandwidth bottleneck, so combining them buys nothing.
    Evidence: d4 result 47.08 t/s is LOWER than max(56.43, 48.32) = 56.43 t/s;
    i.e. the combination underperforms the better single lever, not merely
    fails to add. 48.32 t/s is the floor-ON LDS pair with f16 KV.
  - At width 5 the LDS repack path costs more than it gives once q4KV has
    already cheapened the verify pass.

CONTRAST (so -16.6% is not mistaken for "LDS is always bad"): the same LDS pair
  is worth +48.8% on top of floor-0 + q4KV at d7 (34.73 -> 51.69 t/s). The sign
  of the LDS marginal depends on floor and width; at floor-ON it is negative.

WHAT THIS IS NOT (state plainly)
  - NOT a kernel-attribution result. NO rocprof kernel-time diff was ever
    produced for this comparison. It is a measured end-to-end marginal effect
    at the serving-config level only.
  - NOT a claim about which kernel is slower or by how much. Do not claim
    kernel-level causation.
  - Section 3.3 also notes the single-rep nature of these cells.

LIMITS
  - Single rep per cell (bench.py internal 3 reps; r2 spot-check legs starved
    by sibling contention, not run).
  - d7 f16-KV LDS-off cells NOT measured (out of scope).
  - Box shared with sibling lanes (U2 + taskV2-r7): tg cells carry a few pct of
    contention noise. Acceptance is deterministic, so the acceptance columns
    are the trustworthy part.
  - The q4stack arm in taskU was rebased onto the taskQ Q4_0-drafter stack
    (not taskK's literal Q8_0-drafter build-jmerge arm); the LDS tensor-map
    effect is a main-model property (ssm_out), hence drafter-independent, but
    the comparison is not byte-for-byte taskK's original arm.

DISCREPANCY (item 5)
  - None material. The brief's 56.43 -> 47.08 and -16.6% both reconcile to
    section 3.3 exactly. The d7 companion (-1.39%) is added here because the
    brief omitted it and it qualifies the width-dependence.


================================================================================
taskD1 CORRECTIONS  (three; all verified against artifacts for this ledger)
================================================================================

C1. 'v_dot8 does not exist on GCN'  ->  FALSE. STRIKE IT.
    The claim appears in /root/wt-g4-phase2/taskD1-LEDGER.md at line 18
    ("...no scalar nibble loop exists in this codebase; `v_dot8` does not exist
    on GCN...") and again at lines 111-112 ("**v_dot8 does not exist on GCN** -
    the task's v_dot8 option is not on this ISA").
    ARTIFACT EVIDENCE: /root/int4-noquant-wt/ggml/src/ggml-cuda/common.cuh
      line 762 : return __builtin_amdgcn_sdot8(a, b, c, false);   (ggml_cuda_dp8a,
                 inside a `#if defined(GGML_USE_HIP) && defined(__gfx906__)`)
      line 718 : c = __builtin_amdgcn_sdot4(a, b, c, false);      (ggml_cuda_dp4a)
    The codebase's own comment above dp8a: "gfx906-only: later AMD ISAs dropped
    the 8-way int4 dot". So v_dot8 EXISTS on gfx906 (this arch) and is missing
    on LATER ISAs - the opposite of the struck claim.
    Consequence: the struck claim had been used to close the whole int4-dot ALU
    direction. The skinny lane (item 1) was built on the correction and measured
    85.3-93.4% of the weights-only roofline at T=1. Standing rule: grep
    common.cuh before asserting any instruction is absent on this arch; do not
    cite taskD1 as evidence the ALU direction is dead.

C2. i8narrow / skinny prior rejection  ->  NOT TESTED, not "refuted".
    Every launch in the earlier i8narrow lane ran INTERLEAVED=false with no
    split-K, which is NOT the ruquant design under test. Recording it as a
    refutation was invalid.
    ARTIFACT EVIDENCE:
      - /root/int4-r1-wt/taskR1-SKINNY-LEDGER.md section 0 states this
        explicitly and reclassifies the earlier result as "NOT TESTED".
      - /root/wt-ksplit/docs/DESIGN-i8-narrow-kernel.md: status "DESIGN + CPU
        seal DONE; HIP skeleton AUTHORED, NOT COMPILED" - there was no
        interleaved/split-K perf campaign to refute anything.
      - /root/wt-k/taskB2-LEDGER.md did later wire and head-to-head i8narrow,
        and its numbers are poor (i8narrow at 32-77% of stock nc on
        K=5120 N=17408 at widths 1-8; 21-30% on K=17408 N=5120), but that is a
        DIFFERENT, stock-dispatch ratio test at widths 1-16 - it still does not
        test the interleaved ruquant verify design. So the correct label for
        the ORIGINAL rejection remains NOT TESTED; the B2 result is a separate
        negative data point, not a refutation of the design under test.
    Do not cite the earlier i8narrow loss as a refutation of skinny/int4-dot.

C3. Canonical probe hash 77841cb6a251235d  ->  NOT a numerics gate.
    It is IDENTICAL across Q8_0 and Q4_0 mains, so it cannot discriminate quant
    types or certify numerics of a quant change.
    ARTIFACT EVIDENCE:
      - /root/taskR6/probe-champ-p-d4.sha16    = 77841cb6a251235d
      - /root/taskR6/probe-q4kvlds-p-d4.sha16  = 77841cb6a251235d
        (two different configs, same hash)
      - /root/wt-g4-phase2/taskU-LEDGER.md section 4 reports the same sha16 for
        10/10 legs including all four q4_0-KV arms and both floor-0 arms; and
        separately reports KLD 0.011223 (35B) / 0.006855 (27B) for those arms -
        i.e. logits DID change while the greedy probe hash did not.
    Correct use: a greedy-64-token reproducibility witness that a config change
    did not flip the probe's argmax path. NOT evidence that two configs agree
    numerically. Use KLD / logits for the numerics question.


================================================================================
CROSS-CUTTING NOTES FOR THE REVIEWER
================================================================================
 - Supplied-vs-artifact status AFTER the corrections (this is the accurate state):
   item 1  the brief's "32.3% T=8" does NOT survive re-derivation (31.8%
           attn_qkv); also the brief's ledger path was wrong. Verdict unchanged.
   item 2  the brief's 33.12/33.62% SURVIVES EXACTLY (33.1160/33.6209). The
           ERROR WAS IN THIS LEDGER: the 22:20 pass used a 6144 denominator and
           wrote 29.54/30.62. Corrected; see CORRECTIONS C-1.
   item 3  reconciles, except a 40-vs-38 .acc file count in the source ledger.
   item 4  the brief's +10.9% does NOT survive; nor does this ledger's +11.00%.
           The quiet-box lane (taskR9) landed after the compile and REFUTES the
           win: +5.70% median, Welch t n.s., LDS bimodal. Verdict downgraded to
           NOT PROVEN / NOT ADOPTION-SAFE. See CORRECTIONS C-2.
   item 5  reconciles cleanly to taskU-LEDGER.md section 3.3.
 - item 2's CONCLUSION (no +25% from q4_0 KV; q8_0 is HIGHER) is robust and is
   now supported by the CORRECT rates: 33.6209 vs 33.1160 = +0.5049 pp for q8_0.
   The earlier paired t-test (t = -2.756) is WITHDRAWN - it was computed from
   the bad base and is not reproducible from the record schema.
 - STILL OPEN: item 4. The quiet-box confirmation landed and DISCONFIRMED the
   win. Before RP_Q4_NC_LDS=8 can be considered for a fleet, someone must (a)
   explain why the LDS arm is bimodal at identical gate/VRAM/acceptance, and
   (b) run a >=10-rep pinned-width lane as listed in that entry. It should NOT
   be adopted on current evidence.
 - The 58.20 t/s champion reference is NOT a valid baseline on this build: it
   predates the uwave->tip walk regression documented in
   taskG4-PHASE2-LEDGER.md section 3. Compare within-build only.
 - No GPU time was used to produce or correct this ledger. Verification was file
   reads and read-only python3/aggregation over the cited jsonl/parsed dumps;
   no llama-server was started, no lock taken, nothing built or committed.
================================================================================
END taskREVIEW-LEDGER
================================================================================
