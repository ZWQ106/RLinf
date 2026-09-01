# Layouts & initial-state frames (per task)

```
saved_demo/<task>/
  layouts/        the layouts currently attached to this task in the portal
                  (copies of ~/rlinf_data/layouts/<id>.{json,exterior.jpg,wrist.jpg})
  init_layouts/   EVERY episode's first frame, agent (exterior) view, full 1280x720:
                    <dataset>_ep<NN>.exterior.jpg   ZED 2i  (scene)
                    _sheet.jpg                      labelled contact sheet — pick from here
                    index.json                      which SVO file each frame came from
```

Regenerate init frames: `docker exec -i rlinf-eval /opt/venv/openpi/bin/python
/workspace/rlinf/tasl/tools/export_init_frames.py` (skips frames already present).

Promote a pick to a real portal layout:
`python3 tasl/tools/layout_from_dataset.py <dataset> --episode <N> --layout-id <task>-L<k> --task <task>`

## Picks (2026-08-25, hand-reviewed)

Six layouts per task, `<task>-L1..L6`, in the order they were picked.
Older hand-made layouts (`Layout_task_1`, `T1-b-1`, `T2-L1`, `T3-L1`, `T4-L1`, `T5-L1`)
were kept and stay the task default where they existed. `_sheet.jpg` in each
`layouts/` dir shows the six.

| task | L1 | L2 | L3 | L4 | L5 | L6 |
|------|----|----|----|----|----|----|
| T1-a | ep0 | ep16 | ep18 | ep20 | ep21 | ep24 |
| T1-b | ep0 | ep1 | ep3 | ep13 | ep14 | ep7 |
| T2-a | 10ep ep0 | 10ep ep1 | 10ep ep2 | 10ep ep8 | 10ep ep9 | 15ep ep1 |
| T2-b | ep0 | ep1 | ep6 | ep12 | ep10 | ep24 |
| T3-a | ep0 | ep6 | ep7 | ep18 | ep20 | ep24 |
| T3-b | ep0 | ep14 | ep15 | ep17 | ep22 | ep9 |
| T4-a | ep0 | ep5 | ep20 | ep24 | ep8 | ep17 |
| T4-b | ep0 | ep6 | ep7 | ep14 | ep18 | ep24 |
| T5-a | ep0 | ep4 **frame 20** (hand in frames 0–15) | ep7 | ep14 | ep23 | ep21 |
| T5-b | ep0 | ep2 | ep8 | ep13 | ep21 | ep9 |
