#!/usr/bin/env bash
# CPU scoring of W3c / W3d / W3e (two seeds each) with W3c's scripts and days, unchanged; waits for w3e.done and
# w3e_scores.done. Marker /var/tmp/sat3737-parts-cache/parts_weak/w3e_compare.done
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
D=experiments/parts_tri_expand
until { [ -f $C/w3e.done ] || [ -f $C/w3e.failed ]; } && [ -f $C/w3e_scores.done ]; do sleep 120; done
M="parts_seg_r05_w3c parts_seg_r05_w3cs1 parts_seg_r05_w3d parts_seg_r05_w3ds1 parts_seg_r05_w3e parts_seg_r05_w3es1"
$PY experiments/parts_weak/compare.py --arms $M --out $D/compare_w3e.json > $C/cmp_w3e.out 2>&1 || echo "compare failed"
$PY experiments/parts_weak/fin_definition.py --models $M --out $D/fin_definition_w3e.json --sheet-name triA_fin_definition.jpg > $C/fd_w3e.out 2>&1 || echo "fin_definition failed"
$PY experiments/parts_weak/finless.py --models $M --out $D/finless_w3e.json > $C/fl_w3e.out 2>&1 || echo "finless failed"
$PY experiments/parts_weak/shares.py --models $M --out $D/shares_w3e.json > $C/sh_w3e.out 2>&1 || echo "shares failed"
$PY $D/pred_fill.py score --models $M --out new_measures_w3e.json > $C/nm_w3e.out 2>&1 || echo "new measures failed"
$PY $D/two_seeds_w3e.py > $C/ts_w3e.out 2>&1 || echo "two_seeds_w3e failed"
$PY $D/w3e_sheet.py > $C/sheet_w3e.out 2>&1 || echo "sheet failed"
date +%T > $C/w3e_compare.done
