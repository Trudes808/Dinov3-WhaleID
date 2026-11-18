#!/bin/bash
# CPU: bench numbers once the train days are matched (logs/final_*.log)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=6
P=../../../.venv/bin/python
$P lm_score.py aliked1024 dorsal full hom12 plain > logs/final_dorsal_hom12.log 2>&1 &
$P lm_score.py aliked1024 fluke full hom12 nofb > logs/final_fluke_hom12.log 2>&1 &
$P lm_fuse2.py aliked1024 dorsal hom12 plain > logs/final_dorsal_hom12_isofuse.log 2>&1 &
$P lm_fuse2.py aliked1024 fluke hom12 plain > logs/final_fluke_hom12_isofuse.log 2>&1 &
$P lm_fuse2.py aliked1024 dorsal simconf6 plain > logs/final_dorsal_simconf6_isofuse.log 2>&1 &
wait
echo done
