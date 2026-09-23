# Retired standalone QKV experiments

These files preserve the September 21 diagnostic detour. They are not imported
by the CSA integration or exposed by `run_csa_validation.sh`.

The integration's computation reference is pypto-lib commit
`205255b4770ee84dfa176bcbc7bbef651953c7e1`,
`models/deepseek_v4_flash_dspark/decode_csa.py::_decode_csa_tp1`.
New computation work must continue from that complete attention chain.

Both NPU experiment tasks failed during compilation. Neither established a
Native/PTO numerical pass. See validation log section 31 for the task IDs and
original failure evidence. The related inherited-subview codegen CPU suite
subsequently passed 223 tests; it has no NPU revalidation from these experiments.
