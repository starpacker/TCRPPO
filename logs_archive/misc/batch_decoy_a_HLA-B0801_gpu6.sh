#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
cd /share/liuyutian/pMHC_decoy_library
echo "=== EIYKRWII (HLA-B*08:01) ==="
python run_decoy.py EIYKRWII a --hla "HLA-B*08:01" 2>&1
echo "=== ELRRKMMYM (HLA-B*08:01) ==="
python run_decoy.py ELRRKMMYM a --hla "HLA-B*08:01" 2>&1
echo "=== FLKEKGGL (HLA-B*08:01) ==="
python run_decoy.py FLKEKGGL a --hla "HLA-B*08:01" 2>&1
echo "=== FLRGRAYGL (HLA-B*08:01) ==="
python run_decoy.py FLRGRAYGL a --hla "HLA-B*08:01" 2>&1
echo "=== HSKKKCDEL (HLA-B*08:01) ==="
python run_decoy.py HSKKKCDEL a --hla "HLA-B*08:01" 2>&1
echo "=== QIKVRVDMV (HLA-B*08:01) ==="
python run_decoy.py QIKVRVDMV a --hla "HLA-B*08:01" 2>&1
echo "=== QIKVRVKMV (HLA-B*08:01) ==="
python run_decoy.py QIKVRVKMV a --hla "HLA-B*08:01" 2>&1
echo "=== RAKFKQLL (HLA-B*08:01) ==="
python run_decoy.py RAKFKQLL a --hla "HLA-B*08:01" 2>&1
echo "DONE"
