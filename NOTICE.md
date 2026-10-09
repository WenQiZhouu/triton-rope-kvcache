# Provenance and licenses

Original project code is Apache-2.0.

Third-party sources are included only to reproduce specific isolated benchmarks. These files were NOT authored by this project's maintainer:

- official/sglang_rope_cache.py: saved SGLang official rope_cache.py source, snapshot retrieved 2026-10-08 (SHA256 a4f191b832b6f81617c5e35e20c4fb02f77a65b5c008a111df84e01c0900e832). Original source: https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/rope_cache.py. Apache-2.0 license: official/SGLANG_LICENSE_APACHE2.txt.
- official/standalone_kernels.py: adapted from FlagGems-vLLM original rotary and reshape/cache sources pinned to commit 4ea1b95b80efa2012fc5d82365f38a03ed691f09. Only program_id API rewritten for standalone tests. Apache-2.0 license: official/LICENSE.

This repository does not claim to be endorsed, used by, or merged into upstream SGLang, vLLM, or FlagGems.
