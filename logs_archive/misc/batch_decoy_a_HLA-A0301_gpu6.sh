#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
cd /share/liuyutian/pMHC_decoy_library
echo "=== ALRANSAVK (HLA-A*03:01) ==="
python run_decoy.py ALRANSAVK a --hla "HLA-A*03:01" 2>&1
echo "=== ATVVIGTSK (HLA-A*03:01) ==="
python run_decoy.py ATVVIGTSK a --hla "HLA-A*03:01" 2>&1
echo "=== AVLQSGFRK (HLA-A*03:01) ==="
python run_decoy.py AVLQSGFRK a --hla "HLA-A*03:01" 2>&1
echo "=== GVAMPNLYK (HLA-A*03:01) ==="
python run_decoy.py GVAMPNLYK a --hla "HLA-A*03:01" 2>&1
echo "=== QQQQGQTVTK (HLA-A*03:01) ==="
python run_decoy.py QQQQGQTVTK a --hla "HLA-A*03:01" 2>&1
echo "=== QVPLRPMTYK (HLA-A*03:01) ==="
python run_decoy.py QVPLRPMTYK a --hla "HLA-A*03:01" 2>&1
echo "=== RLFRKSNLK (HLA-A*03:01) ==="
python run_decoy.py RLFRKSNLK a --hla "HLA-A*03:01" 2>&1
echo "=== RLRPGGKKK (HLA-A*03:01) ==="
python run_decoy.py RLRPGGKKK a --hla "HLA-A*03:01" 2>&1
echo "=== RLRPGGKKR (HLA-A*03:01) ==="
python run_decoy.py RLRPGGKKR a --hla "HLA-A*03:01" 2>&1
echo "=== RLYYDSMSY (HLA-A*03:01) ==="
python run_decoy.py RLYYDSMSY a --hla "HLA-A*03:01" 2>&1
echo "=== SSNVANYQK (HLA-A*03:01) ==="
python run_decoy.py SSNVANYQK a --hla "HLA-A*03:01" 2>&1
echo "=== TEILPVSMTK (HLA-A*03:01) ==="
python run_decoy.py TEILPVSMTK a --hla "HLA-A*03:01" 2>&1
echo "=== YIFFASFYY (HLA-A*03:01) ==="
python run_decoy.py YIFFASFYY a --hla "HLA-A*03:01" 2>&1
echo "DONE"
