#!/usr/bin/env python3
"""Run the bounded gfx1151 llama.cpp RMSNorm+Mul quickstart."""

from amd_triton_quickstart import main


if __name__ == "__main__":
    raise SystemExit(main(default_operator="llama_rmsnorm"))
