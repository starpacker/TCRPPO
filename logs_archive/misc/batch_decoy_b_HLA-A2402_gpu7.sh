#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
cd /share/liuyutian/pMHC_decoy_library
echo "=== AYAQKIFKI (HLA-A*24:02) ==="
python run_decoy.py AYAQKIFKI b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== CYTWNQMNL (HLA-A*24:02) ==="
python run_decoy.py CYTWNQMNL b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== IPYNSVTSSI (HLA-A*24:02) ==="
python run_decoy.py IPYNSVTSSI b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== LSPRWYFYYL (HLA-A*24:02) ==="
python run_decoy.py LSPRWYFYYL b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== NYNYLYRLF (HLA-A*24:02) ==="
python run_decoy.py NYNYLYRLF b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== NYSGVVTTVMF (HLA-A*24:02) ==="
python run_decoy.py NYSGVVTTVMF b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== QYDPVAALF (HLA-A*24:02) ==="
python run_decoy.py QYDPVAALF b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== QYIKWPWYI (HLA-A*24:02) ==="
python run_decoy.py QYIKWPWYI b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== RYPLTFGWCF (HLA-A*24:02) ==="
python run_decoy.py RYPLTFGWCF b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== SFHSLHLLF (HLA-A*24:02) ==="
python run_decoy.py SFHSLHLLF b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== VYFLQSINF (HLA-A*24:02) ==="
python run_decoy.py VYFLQSINF b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== VYGIRLEHF (HLA-A*24:02) ==="
python run_decoy.py VYGIRLEHF b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "=== YYQLYSTQL (HLA-A*24:02) ==="
python run_decoy.py YYQLYSTQL b --hla "HLA-A*24:02" --skip-structural 2>&1
echo "DONE"
