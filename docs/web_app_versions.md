# Whale Review web app: the two versions, launching them, rolling back

The web app is the sibling folder `/home/sat3737/NOAA_whales/Whale-Review-Web` (not part of this repository). Two
versions exist:

| | current ("the beta", live since 2026-11-24 23:48) | previous ("the old app", code of 2026-11-23 08:42) |
|---|---|---|
| sign-in | required (invite-only accounts) | none |
| modules | Library, Model Interpretability, Labeling Assistant, Upload, Video Analysis, Admin | Library, Model Interpretability, Labeling Assistant |
| listens on | 127.0.0.1 | 0.0.0.0 by default (this page always starts it on 127.0.0.1) |
| code | `Whale-Review-Web/` | only in `moonshot_cache/switch_root/backup_beta_golive_2026-11-24/code_before/` |

Both read and write the same researcher state files (`Whale-Review-Web/data/<day>/labeling.json`, `pair_marks.json`,
...). Nothing below changes those files.

## 1. Launch the current app (beta)

```bash
cd /home/sat3737/NOAA_whales/Whale-Review-Web
setsid nohup ./serve.sh --port 8765 >> serve.log 2>&1 &     # 127.0.0.1:8765, uses ../Dinov3-WhaleID/.venv
```

Check that it runs:

```bash
ss -ltnp 'sport = :8765'                                         # one line, 127.0.0.1:8765, with the pid
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/healthz     # 200
```

Restart it: stop the process that listens on 8765 by its pid (never `pkill -f`: that can hit other servers), start it
again as above.

```bash
pid=$(ss -ltnpH 'sport = :8765' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1); kill -TERM $pid
```

The log is `Whale-Review-Web/serve.log` (appended to; a start writes a line with the address and the module list).

Reach it from a laptop through an SSH port forward, then open `http://localhost:8765/`:

```bash
ssh -L 8765:localhost:8765 USER@ecen-a58854
```

First admin (no account exists without an invitation; this works on the database directly; text from
`Whale-Review-Web/docs/ACCOUNTS.md`). Use your own address and name:

```bash
cd /home/sat3737/NOAA_whales/Whale-Review-Web
PY=../Dinov3-WhaleID/.venv/bin/python
$PY -m backend.auth create-user --email you@example.org --name "Your Name" --admin --confirmed   # asks for the password
$PY -m backend.auth create-invite --email you@example.org --admin --no-mail                     # or: prints an invite link
```

Later invitations are made on the page `/admin`. Treat invite links like passwords; do not paste them anywhere public.

Background worker (uploads, photo-day processing, video analysis run as jobs in a separate process, one at a time;
the server starts it by itself and restarting the server does not stop a running job):

```bash
cd /home/sat3737/NOAA_whales/Whale-Review-Web; PY=../Dinov3-WhaleID/.venv/bin/python
$PY -m backend.worker --status      # is it running, what is it doing (pid also in data/app/worker.pid)
$PY -m backend.worker --stop        # stop it; its job goes back to the queue and continues when a worker starts again
```

Worker messages: `data/app/jobs/worker.out`; each job's log: `data/app/jobs/<id>.log`.

## 2. Launch the old app next to the beta, without rolling back

```bash
/home/sat3737/NOAA_whales/Dinov3-WhaleID/scripts/web_app_launch_old.sh            # builds the copy on first use, starts it
/home/sat3737/NOAA_whales/Dinov3-WhaleID/scripts/web_app_launch_old.sh --status
/home/sat3737/NOAA_whales/Dinov3-WhaleID/scripts/web_app_launch_old.sh --stop     # stops the process that listens on the port
```

What the script does:

* First use: copies `code_before/` of the backup into `/home/sat3737/NOAA_whales/app_old_2026-11-23/Whale-Review-Web`
  (outside both repositories; `OLD_APP_ROOT=...` changes the folder) and checks the copy against
  `sha256_code_before.txt`. It makes four links: `Whale-Review-Web/data` -> the live `Whale-Review-Web/data`, and
  `Dinov3-WhaleID`, `Whale-Image-Data-Factory`, `moonshot_cache` next to the app folder (the old code looks for the
  model outputs, the label file, the fin-sketch module and the whale mesh through `../Dinov3-WhaleID` and
  `../Whale-Image-Data-Factory`).
* Starts the copy's own (old) `serve.sh` with `--host 127.0.0.1 --port 8766` (`OLD_PORT=...` changes the port; 8765 is
  refused). That `serve.sh` sets the same environment as on 2026-11-23 (trip log on, fin model, test-day RAW roots,
  markers root). It prints the address and the pid; the log is `app_old_2026-11-23/serve_old.log`.
* Reach it through `ssh -L 8766:localhost:8766 USER@ecen-a58854`, then `http://localhost:8766/`.

Read this before using it:

* **The old app has no sign-in.** It must stay on 127.0.0.1 (the script enforces that; never start it with
  `--host 0.0.0.0`). Do not forward its port to anyone but yourself.
* **It reads and writes the same sorting state files as the beta.** Do not sort in both apps at once; a save in one is
  not seen by the other until it reloads a day, and the later save wins.
* It does not know the beta's accounts or access rules: it lists every day folder in the shared days root, including
  days uploaded through the beta (and days an admin marked "restricted"). It has no upload or Video Analysis pages, and
  the beta's log entries that carry names (the old app uses the initials box) are not changed.
* Stop it with `--stop`, or by the pid that listens on 8766 (`ss -ltnp 'sport = :8766'`, `kill -TERM PID`).
* The built folder `app_old_2026-11-23` is small (code and links); delete it whenever you like, the script rebuilds it.

## 3. Roll back to the old app

When: the beta misbehaves in a way that blocks the researchers' sorting (the Library, Labeling Assistant or Interpret
pages fail or lose state) and a fix is not quick. Before that, try the "next to it" launch above: it needs no change to
the live folder. A roll-back puts the old, sign-in-free app on port 8765 (still on 127.0.0.1).

```bash
bash /home/sat3737/NOAA_whales/moonshot_cache/switch_root/ROLLBACK_beta.sh
```

What the script does, in order (it stops at the first failing command):

1. Takes these 18 files of the live `Whale-Review-Web/`: `backend/{app,interpret,labeling,labeling_sketch,library,pairs,signatures}.py`,
   `frontend/{index,labeling,interpret,pairs,bursts,signatures}.html`, `frontend/js/{bursts,lab_lightbox,interp_step_photos}.js`,
   `serve.sh`, `README.md`. Copies them (with times) to
   `backup_beta_golive_2026-11-24/code_at_rollback_<date>_<time>/` in the backup folder.
2. Copies the same 18 files from `code_before/` over the live ones.
3. Compares the live files with `code_before/` by sha256 and prints `code restored` when they match (a mismatch prints
   the differences and the script stops there, before any restart).
4. Sends TERM to the process that listens on 8765 (found by `ss`), waits 2 s.
5. Runs `python -m backend.worker --stop` (the beta's worker; a running job goes back to the queue).
6. Starts `./serve.sh --port 8765 --host 127.0.0.1` (now the old `serve.sh`) in the background, appending to
   `Whale-Review-Web/serve.log`, waits 4 s and prints `ss` for 8765.

What it leaves in place: everything else in `Whale-Review-Web/`: the beta-only backend modules (`auth.py`, `video.py`,
`uploads.py`, `jobs.py`, `worker.py`, ...), the other frontend pages, `pipelines/`, `docs/`, `data/app/` (accounts and
database), and the upload, day, video and export folders (`app_uploads`, `app_days`, `app_video`, `app_exports`). The old
code does not use them. The researchers' state files are not touched; days uploaded through the beta stay in the days
root and show up in the old app. The `tests/` folder is not restored (the beta's tests stay).

Verify afterwards:

```bash
B=/home/sat3737/NOAA_whales/moonshot_cache/switch_root/backup_beta_golive_2026-11-24; W=/home/sat3737/NOAA_whales/Whale-Review-Web
ss -ltnp 'sport = :8765'                                                  # 127.0.0.1:8765
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/           # 200 with no sign-in page
curl -s http://localhost:8765/api/library/days | head -c 200              # a list of days (the beta answers 401 / a sign-in)
# the 18 restored files against the backup's checksums (83 lines in that file also cover tests/, which are not restored):
cd $W && grep -E ' \./(backend/(app|interpret|labeling|labeling_sketch|library|pairs|signatures)\.py|frontend/(index|labeling|interpret|pairs|bursts|signatures)\.html|frontend/js/(bursts|lab_lightbox|interp_step_photos)\.js|serve\.sh|README\.md)$' $B/sha256_code_before.txt | sha256sum -c -
```

Restore one state file by hand (only if a file was damaged; `state/` is the copy of the researchers' state at go-live,
so anything saved since is lost for that file; `sha256_state.txt` lists 418 files, `sha256_state_live_at_backup.txt`
the 12 that were live-edited when the backup ran). Stop the app first, keep the damaged file, then:

```bash
B=/home/sat3737/NOAA_whales/moonshot_cache/switch_root/backup_beta_golive_2026-11-24; W=/home/sat3737/NOAA_whales/Whale-Review-Web
f=data/2025-01-15/labeling.json                                  # example: path as in sha256_state.txt, without the ./
cp -p $W/$f $W/$f.damaged_$(date +%F_%H%M%S) && cp -p $B/state/$f $W/$f
(cd $B/state && grep " \./$f\$" ../sha256_state.txt | sha256sum -c - ) ; sha256sum $W/$f      # the two sums must agree
```

Go forward to the beta again: the roll-back script keeps the beta's version of the 18 files in
`$B/code_at_rollback_<date>_<time>/` (same relative paths). With the app stopped, copy them back and start the beta:

```bash
B=/home/sat3737/NOAA_whales/moonshot_cache/switch_root/backup_beta_golive_2026-11-24; W=/home/sat3737/NOAA_whales/Whale-Review-Web
A=$(ls -d $B/code_at_rollback_* | tail -1)          # the right one if you rolled back more than once: check the date
cp -a $A/. $W/                                       # backend/, frontend/, serve.sh, README.md
(cd $A && sha256sum -c <(grep -F -f <(find . -type f | sed 's|^\./|  |') $B/sha256_code_after.txt))   # only for the files of go-live
# then start as in part 1 (the old process on 8765 must be stopped first)
```

`sha256_code_after.txt` holds the checksums of the beta files that went live on 2026-11-24; the beta has changed since
(a later backup folder, `backup_beta_round2_2026-11-25`, exists), so files from `code_at_rollback_*` made after that
date will differ from it legitimately; compare with the newer backup's own checksum file instead. Accounts, sessions
and the database in `data/app/` were never removed, so people can sign in again as before.
