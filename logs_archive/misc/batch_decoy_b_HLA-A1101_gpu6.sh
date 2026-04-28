#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
cd /share/liuyutian/pMHC_decoy_library
echo "=== GTSGSPIIDK (HLA-A*11:01) ==="
python run_decoy.py GTSGSPIIDK b --hla "HLA-A*11:01" --skip-structural 2>&1
echo "=== GTSGSPIINR (HLA-A*11:01) ==="
python run_decoy.py GTSGSPIINR b --hla "HLA-A*11:01" --skip-structural 2>&1
echo "=== GTSGSPIVNR (HLA-A*11:01) ==="
python run_decoy.py GTSGSPIVNR b --hla "HLA-A*11:01" --skip-structural 2>&1
echo "=== KTAYSHLSTSK (HLA-A*11:01) ==="
python run_decoy.py KTAYSHLSTSK b --hla "HLA-A*11:01" --skip-structural 2>&1
echo "=== LVVDFSQFSR (HLA-A*11:01) ==="
python run_decoy.py LVVDFSQFSR b --hla "HLA-A*11:01" --skip-structural 2>&1
echo "=== STLPETAVVRR (HLA-A*11:01) ==="
python run_decoy.py STLPETAVVRR b --hla "HLA-A*11:01" --skip-structural 2>&1
echo "DONE"
