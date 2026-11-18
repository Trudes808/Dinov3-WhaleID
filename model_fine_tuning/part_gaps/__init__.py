"""Part-model gap instrument (2026-11-17): why the part model cannot tell which part of a humpback a real photo shows,
ranked by cost and tied to the renders.

  sample.py       stratified real-instance sample (view classifier x size, train + validation days) + blind label sheets
  common.py       loading (real instances, model arms and their rules), taxonomy (CATS, ATTRS)
  report.py       the ranked gap table (real error / cost per category and attribute, arms, render coverage + render error)
  coverage.py     what a render set holds, in the same taxonomy (labels only)
  render_eval.py  a part model on renders, summarised per rendered whale like the real instances
  nearest.py      nearest renders to real instances (frozen DINOv3 whale tokens, silhouette)
  res_test.py     inference-resolution test on the labelled real instances
  sheets.py       failure and nearest-render sheets (scratchpad only)
  run.py          ONE command: coverage -> (render scoring) -> report

Labels: experiments/part_gaps/labels_ai.txt (+ labels_ai_relook.txt) are AI labels made blind to model output, not the
researcher's. Never reads the evaluation days. No photos in the repository: sheets go to the scratchpad.
"""
