#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
cd /share/liuyutian/pMHC_decoy_library
echo "=== AYAQKIFKI (HLA-A*24:02) ==="
python run_decoy.py AYAQKIFKI a --hla "HLA-A*24:02" 2>&1
echo "=== CYTWNQMNL (HLA-A*24:02) ==="
python run_decoy.py CYTWNQMNL a --hla "HLA-A*24:02" 2>&1
echo "=== IPYNSVTSSI (HLA-A*24:02) ==="
python run_decoy.py IPYNSVTSSI a --hla "HLA-A*24:02" 2>&1
echo "=== LSPRWYFYYL (HLA-A*24:02) ==="
python run_decoy.py LSPRWYFYYL a --hla "HLA-A*24:02" 2>&1
echo "=== NYNYLYRLF (HLA-A*24:02) ==="
python run_decoy.py NYNYLYRLF a --hla "HLA-A*24:02" 2>&1
echo "=== NYSGVVTTVMF (HLA-A*24:02) ==="
python run_decoy.py NYSGVVTTVMF a --hla "HLA-A*24:02" 2>&1
echo "=== QYDPVAALF (HLA-A*24:02) ==="
python run_decoy.py QYDPVAALF a --hla "HLA-A*24:02" 2>&1
echo "=== QYIKWPWYI (HLA-A*24:02) ==="
python run_decoy.py QYIKWPWYI a --hla "HLA-A*24:02" 2>&1
echo "=== RYPLTFGWCF (HLA-A*24:02) ==="
python run_decoy.py RYPLTFGWCF a --hla "HLA-A*24:02" 2>&1
echo "=== SFHSLHLLF (HLA-A*24:02) ==="
python run_decoy.py SFHSLHLLF a --hla "HLA-A*24:02" 2>&1
echo "=== VYFLQSINF (HLA-A*24:02) ==="
python run_decoy.py VYFLQSINF a --hla "HLA-A*24:02" 2>&1
echo "=== VYGIRLEHF (HLA-A*24:02) ==="
python run_decoy.py VYGIRLEHF a --hla "HLA-A*24:02" 2>&1
echo "=== YYQLYSTQL (HLA-A*24:02) ==="
python run_decoy.py YYQLYSTQL a --hla "HLA-A*24:02" 2>&1
echo "DONE"
