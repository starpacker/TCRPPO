#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
cd /share/liuyutian/pMHC_decoy_library
echo "=== ALRANSAVK (HLA-A*03:01) ==="
python run_decoy.py ALRANSAVK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== ATVVIGTSK (HLA-A*03:01) ==="
python run_decoy.py ATVVIGTSK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== AVLQSGFRK (HLA-A*03:01) ==="
python run_decoy.py AVLQSGFRK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== GVAMPNLYK (HLA-A*03:01) ==="
python run_decoy.py GVAMPNLYK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== QQQQGQTVTK (HLA-A*03:01) ==="
python run_decoy.py QQQQGQTVTK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== QVPLRPMTYK (HLA-A*03:01) ==="
python run_decoy.py QVPLRPMTYK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== RLFRKSNLK (HLA-A*03:01) ==="
python run_decoy.py RLFRKSNLK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== RLRPGGKKK (HLA-A*03:01) ==="
python run_decoy.py RLRPGGKKK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== RLRPGGKKR (HLA-A*03:01) ==="
python run_decoy.py RLRPGGKKR b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== RLYYDSMSY (HLA-A*03:01) ==="
python run_decoy.py RLYYDSMSY b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== SSNVANYQK (HLA-A*03:01) ==="
python run_decoy.py SSNVANYQK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== TEILPVSMTK (HLA-A*03:01) ==="
python run_decoy.py TEILPVSMTK b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "=== YIFFASFYY (HLA-A*03:01) ==="
python run_decoy.py YIFFASFYY b --hla "HLA-A*03:01" --skip-structural 2>&1
echo "DONE"
