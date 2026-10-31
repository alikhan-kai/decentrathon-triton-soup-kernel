# db19aca submission comparison

Exact commit inspected: `db19acab9cfcb9944a7ffcd7737383b588247212`.

The user reports 52.7 seconds / 3041 MiB and 10 automatic checks passed. This pass has not independently tied those numbers to a result.json/checks.json pair for this exact commit. The README at the commit still describes 632851f, so it is not evidence for the reported db19aca run.

In `solution/lazy_bnb_compile.py`, `_admit()` calls `ops.read_bytes()` and `init.read_bytes()` on the installed bitsandbytes package, then compiles the read source for identity checks. The official task statement, section 8, prohibits reading files outside the repository and task data. This is a source-review concern; automatic check success does not establish jury approval. It is not a claim that the judges have already rejected this version.

Refined runtime `c580bf8ae53c24b0d4fa08e4d64b08936e3ac1f6` removes those explicit installed-source reads, using actual import-frame/function identities and normal package metadata while preserving the dependency compiler and fallback protocol. Final measurement and check results must be attached before calling it the verified selection.
