#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
cd /share/liuyutian/pMHC_decoy_library
echo "=== FPPTSFGPL (HLA-B*07:02) ==="
python run_decoy.py FPPTSFGPL a --hla "HLA-B*07:02" 2>&1
echo "=== FPQSAPHGVVF (HLA-B*07:02) ==="
python run_decoy.py FPQSAPHGVVF a --hla "HLA-B*07:02" 2>&1
echo "=== GPGHKARVL (HLA-B*07:02) ==="
python run_decoy.py GPGHKARVL a --hla "HLA-B*07:02" 2>&1
echo "=== IPRRNVATL (HLA-B*07:02) ==="
python run_decoy.py IPRRNVATL a --hla "HLA-B*07:02" 2>&1
echo "=== LPRRSGAAGA (HLA-B*07:02) ==="
python run_decoy.py LPRRSGAAGA a --hla "HLA-B*07:02" 2>&1
echo "=== LPRWYFYYL (HLA-B*07:02) ==="
python run_decoy.py LPRWYFYYL a --hla "HLA-B*07:02" 2>&1
echo "=== MPASWVMRI (HLA-B*07:02) ==="
python run_decoy.py MPASWVMRI a --hla "HLA-B*07:02" 2>&1
echo "=== RPHERNGFTVL (HLA-B*07:02) ==="
python run_decoy.py RPHERNGFTVL a --hla "HLA-B*07:02" 2>&1
echo "=== RPPIFIRRL (HLA-B*07:02) ==="
python run_decoy.py RPPIFIRRL a --hla "HLA-B*07:02" 2>&1
echo "=== RPRGEVRFL (HLA-B*07:02) ==="
python run_decoy.py RPRGEVRFL a --hla "HLA-B*07:02" 2>&1
echo "=== TPGPGVRYPL (HLA-B*07:02) ==="
python run_decoy.py TPGPGVRYPL a --hla "HLA-B*07:02" 2>&1
echo "=== TPRVTGGGAM (HLA-B*07:02) ==="
python run_decoy.py TPRVTGGGAM a --hla "HLA-B*07:02" 2>&1
echo "DONE"
