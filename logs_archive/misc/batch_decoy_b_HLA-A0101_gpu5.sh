#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
cd /share/liuyutian/pMHC_decoy_library
echo "=== ATDALMTGY (HLA-A*01:01) ==="
python run_decoy.py ATDALMTGY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== CTELKLNDY (HLA-A*01:01) ==="
python run_decoy.py CTELKLNDY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== CTELKLSDY (HLA-A*01:01) ==="
python run_decoy.py CTELKLSDY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== FRYMNSQGL (HLA-A*01:01) ==="
python run_decoy.py FRYMNSQGL b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== FTSDYYQLY (HLA-A*01:01) ==="
python run_decoy.py FTSDYYQLY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== ISDYDYYRY (HLA-A*01:01) ==="
python run_decoy.py ISDYDYYRY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== LTDEMIAQY (HLA-A*01:01) ==="
python run_decoy.py LTDEMIAQY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== PTDNYITTY (HLA-A*01:01) ==="
python run_decoy.py PTDNYITTY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== TTDPSFLGRY (HLA-A*01:01) ==="
python run_decoy.py TTDPSFLGRY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== VTEHDTLLY (HLA-A*01:01) ==="
python run_decoy.py VTEHDTLLY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "=== YSEHPTFTSQY (HLA-A*01:01) ==="
python run_decoy.py YSEHPTFTSQY b --hla "HLA-A*01:01" --skip-structural 2>&1
echo "DONE"
