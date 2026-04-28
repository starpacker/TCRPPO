#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
cd /share/liuyutian/pMHC_decoy_library
echo "=== FPPTSFGPL (HLA-B*07:02) ==="
python run_decoy.py FPPTSFGPL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== FPQSAPHGVVF (HLA-B*07:02) ==="
python run_decoy.py FPQSAPHGVVF b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== GPGHKARVL (HLA-B*07:02) ==="
python run_decoy.py GPGHKARVL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== IPRRNVATL (HLA-B*07:02) ==="
python run_decoy.py IPRRNVATL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== LPRRSGAAGA (HLA-B*07:02) ==="
python run_decoy.py LPRRSGAAGA b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== LPRWYFYYL (HLA-B*07:02) ==="
python run_decoy.py LPRWYFYYL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== MPASWVMRI (HLA-B*07:02) ==="
python run_decoy.py MPASWVMRI b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== RPHERNGFTVL (HLA-B*07:02) ==="
python run_decoy.py RPHERNGFTVL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== RPPIFIRRL (HLA-B*07:02) ==="
python run_decoy.py RPPIFIRRL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== RPRGEVRFL (HLA-B*07:02) ==="
python run_decoy.py RPRGEVRFL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== TPGPGVRYPL (HLA-B*07:02) ==="
python run_decoy.py TPGPGVRYPL b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "=== TPRVTGGGAM (HLA-B*07:02) ==="
python run_decoy.py TPRVTGGGAM b --hla "HLA-B*07:02" --skip-structural 2>&1
echo "DONE"
