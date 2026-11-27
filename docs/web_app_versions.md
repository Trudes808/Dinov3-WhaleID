# Whale Review web app: the three versions, launching them, rolling back

The web app is the sibling folder `/home/sat3737/NOAA_whales/Whale-Review-Web` (not part of this repository).

**Its code is not in git.** No repository holds it. The only safety net is the backup folders in
`/home/sat3737/NOAA_whales/moonshot_cache/switch_root/`, and both roll-back scripts and the old-app launcher read from
them. Do not delete, move or edit these folders: `backup_beta_golive_2026-11-24/` (the old app, `code_before/`, and the
researchers' state at go-live, `state/`) and `backup_beta_round3_2026-11-25/` (the beta before round 3). Also keep the
`code_at_rollback_*` folders that a roll-back creates inside them: they hold the code you can go forward to.

Three versions exist:

| | current ("round 3", go-live 2026-11-25) | the beta before round 3 (2026-11-24 23:48 to the round-3 go-live) | the old app (code of 2026-11-23 08:42) |
|---|---|---|---|
| sign-in | required (invite-only accounts) | required | none |
| modules | Library, Model Interpretability, Labeling Assistant, Upload, Video Analysis, Admin | the same | Library, Model Interpretability, Labeling Assistant |
| differences | leaner photo processing, a day opens before its job ends, jobs use every CPU thread but two, dead code removed (`attic/` and `frontend_next/` are gone) | has `attic/` and `frontend_next/` | no uploads, no job worker |
| listens on | 127.0.0.1 | 127.0.0.1 | 0.0.0.0 by default (this page always starts it on 127.0.0.1) |
| code | `Whale-Review-Web/` | only in `backup_beta_round3_2026-11-25/code_before/` (228 files) | only in `backup_beta_golive_2026-11-24/code_before/` (83 files) |
| go there | (current) | `ROLLBACK_round3.sh` (part 4) | `ROLLBACK_beta.sh` (part 3), or launch next to the current app (part 2) |

All three read and write the same researcher state files (`Whale-Review-Web/data/<day>/labeling.json`,
`pair_marks.json`, ...). Nothing below changes those files.

`moonshot_cache/switch_root/golive_round3.sh` is the record of how round 3 was deployed from the build folder
`/home/sat3737/NOAA_whales/Whale-Review-Web-next` (it checks the live code against the pre-round-3 snapshot and refuses
while a job runs). It is not something to run again.

## 1. Launch the current app

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

Processing jobs use every CPU thread of the machine but two (`WHALE_JOB_THREADS=<n>` overrides that number).

Worker messages: `data/app/jobs/worker.out`; each job's log: `data/app/jobs/<id>.log`.

## 2. Launch the old app next to the current app, without rolling back

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

## 3. Roll back to the old app (no sign-in)

When: the web app misbehaves in a way that blocks the researchers' sorting (the Library, Labeling Assistant or Interpret
pages fail or lose state) and a fix is not quick. If the fault came only with round 3, use part 4 first (it keeps the
sign-in). Before either, try the "next to it" launch (part 2): it needs no change to the live folder. A roll-back puts
the old app on port 8765 (still on 127.0.0.1).

**The old app has no sign-in.** The script starts it with `--host 127.0.0.1`; never start it on another address.

```bash
bash /home/sat3737/NOAA_whales/moonshot_cache/switch_root/ROLLBACK_beta.sh
```

What the script does, in order (it stops at the first failing command; the earlier steps change nothing):

1. Refuses if `Whale-Review-Web/backend/auth.py` is missing (the live app is then not the beta: already rolled back).
2. Checks `code_before/` of the backup against `sha256_code_before.txt`.
3. Stops the beta's job worker (`python -m backend.worker --stop`; a running job goes back to the queue), then stops the
   server by the pid that listens on 8765 (TERM, up to 20 s).
4. Moves the live `backend/ frontend/ tests/ serve.sh README.md` to
   `backup_beta_golive_2026-11-24/code_at_rollback_<date>_<time>/` (moved, not deleted), and copies the same five from
   `code_before/` (the 83 files of the old app) into `Whale-Review-Web/`.
5. Checks the live files against `sha256_code_before.txt` and prints `old code in place (83 files match code_before)`.
6. Starts the old `./serve.sh --port 8765 --host 127.0.0.1` in the background (appending to `Whale-Review-Web/serve.log`),
   waits up to 60 s for the day list to answer, prints `ss` for 8765 and the name of the folder that holds the beta's code.

What it leaves in place: `pipelines/`, `docs/`, `requirements.txt`, `data/` (accounts, database, researcher state) and
the folders `app_uploads`, `app_days`, `app_video`, `app_exports`; whatever else is there stays too. The old code does not
use them. The researchers' state files are not touched; days uploaded through the beta stay in the days root and show up
in the old app.

Verify afterwards:

```bash
B=/home/sat3737/NOAA_whales/moonshot_cache/switch_root/backup_beta_golive_2026-11-24; W=/home/sat3737/NOAA_whales/Whale-Review-Web
ss -ltnp 'sport = :8765'                                                  # 127.0.0.1:8765, not 0.0.0.0
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/           # 200 with no sign-in page
curl -s http://localhost:8765/api/library/days | head -c 200              # a list of days (the beta answers 401 / a sign-in)
ls $W/backend/auth.py                                                     # must not exist
cd $W && sha256sum -c --quiet $B/sha256_code_before.txt && echo match     # the 83 old files are in place
```

Go forward again (one command; the same script):

```bash
bash /home/sat3737/NOAA_whales/moonshot_cache/switch_root/ROLLBACK_beta.sh --forward
```

It refuses if `backend/auth.py` already exists. It takes the newest `code_at_rollback_*` folder of the backup (check the
date if you rolled back more than once), stops the old server by pid, moves the old code that ran meanwhile to
`old_code_put_aside_<date>_<time>/`, copies the beta's five items back, starts `./serve.sh --port 8765 --host 127.0.0.1`
and waits for `/healthz` to answer 200 (the job worker starts with the server). Accounts, sessions and the database in
`data/app/` were never removed, so people can sign in again as before. Check with the commands of part 1.

Do not use `backup_beta_golive_2026-11-24/ROLLBACK_beta_v1_in_place.sh`. It is the earlier 18-file version, kept for the
record, and it does not fit the code of round 3.

## 4. Roll back to the beta before round 3 (keeps the sign-in)

When: a fault appeared with round 3 (photo processing, a day opening early, job speed, a page that was changed) and the
beta as it ran from 2026-11-24 23:48 to the round-3 go-live is the known good state.

```bash
bash /home/sat3737/NOAA_whales/moonshot_cache/switch_root/ROLLBACK_round3.sh
```

What the script does, in order (it stops at the first failing command):

1. Refuses if `backend/auth.py` is missing (it is then not for that state; if the old app is live, go forward first with
   `ROLLBACK_beta.sh --forward`), and refuses if `attic/` already exists (the code from before round 3 is already live).
2. Checks `backup_beta_round3_2026-11-25/code_before/` against `sha256_code_before.txt` (228 files).
3. Stops the job worker and the server (pid that listens on 8765). A queued or running job goes back to the queue and
   continues with the code that is then live.
4. Moves `attic backend docs frontend frontend_next pipelines tests README.md requirements.txt serve.sh` of the live folder
   to `backup_beta_round3_2026-11-25/code_at_rollback_<date>_<time>/` (moved, not deleted), and copies the same items from
   `code_before/` into `Whale-Review-Web/`.
5. Checks the live files against `sha256_code_before.txt` and prints `code from before round 3 in place (228 files match
   code_before)`.
6. Starts `./serve.sh --port 8765` (the beta's `serve.sh`, 127.0.0.1) in the background, appending to `serve.log`, and
   waits up to 60 s for `/healthz` to answer 200.

It does not touch `data/` (accounts, database, researcher state) or `app_uploads`, `app_days`, `app_video`,
`app_exports`. The database keeps the two job columns round 3 added (`ready_at`, `ready_note`); the older code ignores
them.

Verify afterwards:

```bash
B=/home/sat3737/NOAA_whales/moonshot_cache/switch_root/backup_beta_round3_2026-11-25; W=/home/sat3737/NOAA_whales/Whale-Review-Web
ss -ltnp 'sport = :8765'                                                  # 127.0.0.1:8765
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/healthz    # 200
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8765/api/library/days   # 401: the sign-in is back
ls -d $W/attic                                                            # exists again
cd $W && sha256sum -c --quiet $B/sha256_code_before.txt && echo match     # the 228 files of the beta before round 3
```

Go forward to round 3 again:

```bash
bash /home/sat3737/NOAA_whales/moonshot_cache/switch_root/ROLLBACK_round3.sh --forward
```

It refuses if `attic/` is missing (round 3 is already live). It takes the newest `code_at_rollback_*` folder (it must hold
`pipelines/cpu.py`), stops the worker and the server, moves the code that is live to
`round2_code_put_aside_<date>_<time>/`, copies round 3 back, starts the server and waits for `/healthz`. Check with the
commands of part 1. The go-live wrote `backup_beta_round3_2026-11-25/sha256_code_after.txt`, which lists the round-3
files: `cd $W && sha256sum -c --quiet $B/sha256_code_after.txt`. If that file is not there, the go-live has not run.

## 5. Which roll-back, and what was tested

* Fault only since round 3, the sign-in is wanted: part 4.
* The beta itself is the problem, or the sorting must go on without it: part 3.
* Only one state file is damaged: the next paragraph, no roll-back.
* Do not sort in two versions at once, and do not run `ROLLBACK_beta_v1_in_place.sh`.

Both roll-back scripts and both `--forward` directions were rehearsed on 2026-11-25 on a copy of the app on another
port: the old app answered without sign-in, the beta answered with sign-in, and the code matched the backups' checksums.
They have not been run on the live app.

Restore one state file by hand (only if a file was damaged; `state/` is the copy of the researchers' state at go-live,
so anything saved since is lost for that file; `sha256_state.txt` lists 418 files, `sha256_state_live_at_backup.txt`
the 12 that were live-edited when the backup ran). Stop the app first, keep the damaged file, then:

```bash
B=/home/sat3737/NOAA_whales/moonshot_cache/switch_root/backup_beta_golive_2026-11-24; W=/home/sat3737/NOAA_whales/Whale-Review-Web
f=data/2025-01-15/labeling.json                                  # example: path as in sha256_state.txt, without the ./
cp -p $W/$f $W/$f.damaged_$(date +%F_%H%M%S) && cp -p $B/state/$f $W/$f
(cd $B/state && grep " \./$f\$" ../sha256_state.txt | sha256sum -c - ) ; sha256sum $W/$f      # the two sums must agree
```
