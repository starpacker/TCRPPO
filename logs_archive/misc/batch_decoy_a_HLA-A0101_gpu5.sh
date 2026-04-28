#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
cd /share/liuyutian/pMHC_decoy_library
echo "=== ATDALMTGY (HLA-A*01:01) ==="
python run_decoy.py ATDALMTGY a --hla "HLA-A*01:01" 2>&1
echo "=== CTELKLNDY (HLA-A*01:01) ==="
python run_decoy.py CTELKLNDY a --hla "HLA-A*01:01" 2>&1
echo "=== CTELKLSDY (HLA-A*01:01) ==="
python run_decoy.py CTELKLSDY a --hla "HLA-A*01:01" 2>&1
echo "=== FRYMNSQGL (HLA-A*01:01) ==="
python run_decoy.py FRYMNSQGL a --hla "HLA-A*01:01" 2>&1
echo "=== FTSDYYQLY (HLA-A*01:01) ==="
python run_decoy.py FTSDYYQLY a --hla "HLA-A*01:01" 2>&1
echo "=== ISDYDYYRY (HLA-A*01:01) ==="
python run_decoy.py ISDYDYYRY a --hla "HLA-A*01:01" 2>&1
echo "=== LTDEMIAQY (HLA-A*01:01) ==="
python run_decoy.py LTDEMIAQY a --hla "HLA-A*01:01" 2>&1
echo "=== PTDNYITTY (HLA-A*01:01) ==="
python run_decoy.py PTDNYITTY a --hla "HLA-A*01:01" 2>&1
echo "=== TTDPSFLGRY (HLA-A*01:01) ==="
python run_decoy.py TTDPSFLGRY a --hla "HLA-A*01:01" 2>&1
echo "=== VTEHDTLLY (HLA-A*01:01) ==="
python run_decoy.py VTEHDTLLY a --hla "HLA-A*01:01" 2>&1
echo "=== YSEHPTFTSQY (HLA-A*01:01) ==="
python run_decoy.py YSEHPTFTSQY a --hla "HLA-A*01:01" 2>&1
echo "DONE"
