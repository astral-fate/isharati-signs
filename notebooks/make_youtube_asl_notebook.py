"""Writes notebooks/youtube_asl_keypoints_colab.ipynb: the YouTube-ASL Clip Keypoint Dataset (LINDAT,
hdl:11234/1-5898, CC BY 4.0; 10 zips of ~37 GB = ~374 GB) -> Google Drive -> Isharati poses + manifest -> Hugging Face.

The keypoints are already MediaPipe output (body 33, hands 21+21, face 478, 2D pixels), so no MediaPipe run is needed.
Per zip the notebook writes poses in the Isharati contract ([T,50,3] float16, neck-centred, shoulder-scaled, z = 0)
and a manifest; at the end it joins the English captions, builds a word index for the words our ASL lexicon lacks
(results/coverage_en.json, embedded at generation time) and uploads everything but the raw zips to a private
Hugging Face dataset.

  python notebooks/make_youtube_asl_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "youtube_asl_keypoints_colab.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def missing_words():
    """The uncovered Qur'an/hadith words (most frequent first): the clips to mine signs from."""
    p = ROOT / "results" / "coverage_en.json"
    if not p.exists():
        return []
    return [w for w, _ in json.loads(p.read_text(encoding="utf-8"))["top_missing"]]


INTRO = """
# YouTube-ASL keypoints → Google Drive → Hugging Face (resumable)

Source: **YouTube-ASL Clip Keypoint Dataset** (Železný et al., LINDAT [hdl:11234/1-5898](https://hdl.handle.net/11234/1-5898),
**CC BY 4.0**: redistribution allowed with attribution). 390k sentence clips of ASL, one JSON per clip with
**MediaPipe** 2D keypoints (body 33, hands 21+21, face 478) — MediaPipe has already been run, so this notebook does not
need video or a GPU. 10 zips of ~37 GB (~374 GB) + English captions.

What it does, per zip, one at a time:
1. download to the runtime disk (aria2, 8 connections, retries), check size and zip index;
2. **convert** every clip to the Isharati pose format: `[T, 50, 3]` float16 — nose, neck, R/L shoulder-elbow-wrist,
   left hand 21, right hand 21; neck-centred, shoulder-width-scaled, z = 0 (NaN = not detected); face dropped;
3. write `poses/raw_keypoints_N.npz` (clip id → pose) and `manifest/raw_keypoints_N.csv` (clip, video, frames, frame
   size, hand visibility) to **your Google Drive** (permanent, not the session disk);
4. copy the raw zip to Drive (`KEEP_ZIP`), delete the local copy;
5. upload the poses + manifest part to your **private Hugging Face dataset** (`HF_REPO`).

At the end (cell 6): one `manifest.csv` with each clip's English sentence and split (train/dev), `word_index.json`
(word → clips whose sentence contains it, for the Qur'an/hadith words our lexicon lacks), a dataset card with the
CC BY 4.0 attribution — uploaded too. Later: `snapshot_download(HF_REPO)` gives ~5 GB ready to use.

Survives Colab limits: finished work in Drive (and on HF) is skipped, so after a disconnect run cells 1–4 again.
A Colab VM lives ≤12 h (free) / 24 h (Pro) and closes on long inactivity: keep the tab open with cell 5 running.
Drive takes ≤750 GB of uploads a day. Space: ~374 GB in Drive with `KEEP_ZIP = True`, ~5 GB without.
"""

SETTINGS = """
# 1. Settings + Google Drive (permanent storage)
from google.colab import drive, userdata
drive.mount('/content/drive')

DEST  = '/content/drive/MyDrive/youtube_asl'   # permanent: your Google Drive
LOCAL = '/content/dl'                          # temporary: the runtime disk, one zip at a time
KEEP_ZIP = True   # keep the raw zips (with face points) in Drive; False = only the converted poses (~5 GB)
ONLY = None       # e.g. [1, 2] to process some zips only; None = all 10
HF_REPO = 'youtube-asl-isharati'   # dataset name on Hugging Face (your account); None = no upload
HF_PRIVATE = True                  # private dataset (you can make it public later on huggingface.co)
# Hugging Face token: Colab sidebar -> key icon (Secrets) -> add HF_TOKEN (a *write* token), allow notebook access.

import os, shutil, json
assert os.path.ismount('/content/drive'), 'Google Drive is not mounted: run this cell again'
assert DEST.startswith('/content/drive/MyDrive/'), 'DEST must be inside /content/drive/MyDrive/'
for d in (DEST, DEST + '/poses', DEST + '/manifest', LOCAL):
    os.makedirs(d, exist_ok=True)
open(DEST + '/.write_test', 'w').write('ok'); os.remove(DEST + '/.write_test')
gb = lambda p: shutil.disk_usage(p).free / 1e9
print('saving permanently to Google Drive:', DEST, f'| runtime disk free {gb(LOCAL):.0f} GB')

HF_TOKEN = None
if HF_REPO:
    !pip -q install -U huggingface_hub
    from huggingface_hub import HfApi
    HF_TOKEN = userdata.get('HF_TOKEN')   # never printed or saved in the notebook
    user = HfApi(token=HF_TOKEN).whoami()['name']
    if '/' not in HF_REPO:
        HF_REPO = f'{user}/{HF_REPO}'
    HfApi(token=HF_TOKEN).create_repo(HF_REPO, repo_type='dataset', private=HF_PRIVATE, exist_ok=True)
    print('Hugging Face dataset:', f'https://huggingface.co/datasets/{HF_REPO}', '(private)' if HF_PRIVATE else '')
"""

TOOLS = """
# 2. Tools
!apt-get -qq install -y aria2 > /dev/null && aria2c --version | head -1
"""

FILES = """
# 3. The file list (LINDAT DSpace API; a fixed copy is used if the API is down)
import urllib.request
API = 'https://lindat.mff.cuni.cz/repository/server/api'
ITEM = 'fcb8e746-3a21-4dbe-9d87-8ad40f359ad7'
FALLBACK = {  # name: (bitstream uuid, bytes or None)
 'YT.translations.all.json':   ('ec3632aa-ba9a-400e-b14b-1d15de9b6061', None),
 'YT.translations.train.json': ('f8460818-3605-4f05-9832-90ddc68f22e6', None),
 'YT.translations.dev.json':   ('d5d23d31-c93a-4752-8e5c-e20548f13da0', None),
 'raw_keypoints_1.zip':  ('3dc57bf4-c5fb-491c-8ab2-a9215e0e2fe5', 37504799288),
 'raw_keypoints_2.zip':  ('bfa38e27-bd46-48ae-bf9a-eb1eafaabb95', None),
 'raw_keypoints_3.zip':  ('0c59bda5-908d-4194-93bf-13e13de2ef10', None),
 'raw_keypoints_4.zip':  ('1a4ace36-ed9a-4bfb-a80f-483c0463e02d', None),
 'raw_keypoints_5.zip':  ('6b405702-9f74-4729-8202-55eca36adaea', None),
 'raw_keypoints_6.zip':  ('3b4ef094-bc95-4efb-b8e0-3bc4f63e57b0', None),
 'raw_keypoints_7.zip':  ('74e99da7-c580-4fdf-8363-8024d7a7adf1', None),
 'raw_keypoints_8.zip':  ('43a9146e-0abf-47ec-a3b3-90c01a4d9380', None),
 'raw_keypoints_9.zip':  ('05385788-d459-4f35-92b4-4908b9d86de6', None),
 'raw_keypoints_10.zip': ('d96b874e-72fa-4830-b2f0-a072bb6be31d', None),
}
files = {}
try:
    with urllib.request.urlopen(f'{API}/core/items/{ITEM}/bundles?embed=bitstreams&size=50', timeout=60) as r:
        for b in json.load(r)['_embedded']['bundles']:
            if b['name'] == 'ORIGINAL':
                for x in b['_embedded']['bitstreams']['_embedded']['bitstreams']:
                    files[x['name']] = (x['uuid'], x['sizeBytes'])
except Exception as e:
    print('API unavailable, using the fixed list:', e)
    files = FALLBACK
keep = lambda n: not n.endswith('.zip') or ONLY is None or int(n.split('_')[-1][:-4]) in ONLY
order = sorted((n for n in files if keep(n)), key=lambda n: (n.endswith('.zip'), len(n), n))
for n in order:
    s = files[n][1]
    print(f'{n:28s} {s/1e9:7.2f} GB' if s else n)
json.dump({'api': API, 'order': order, 'files': files, 'dest': DEST, 'local': LOCAL, 'keep_zip': KEEP_ZIP,
           'hf_repo': HF_REPO}, open('/content/job.json', 'w'))
"""

WORKER = r'''
import csv, json, os, shutil, subprocess, sys, time, zipfile, traceback
import numpy as np
job = json.load(open('/content/job.json'))
API, DEST, LOCAL = job['api'], job['dest'], job['local']
HF_REPO, HF_TOKEN = job.get('hf_repo'), os.environ.get('HF_TOKEN')

def drive_ok():  # Drive can drop mid-run; then nothing more is written (it would land on the temporary disk)
    return not DEST.startswith('/content/drive') or os.path.ismount('/content/drive')
if not drive_ok(): raise SystemExit('Google Drive is not mounted: run cell 1 again')
LOG = os.path.join(DEST, 'download_log.txt')
def log(*a):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z: return len(z.namelist())
    except Exception: return 0

def fetch(name, uuid, size):
    url = f'{API}/core/bitstreams/{uuid}/content'
    for attempt in range(1, 9):
        r = subprocess.run(['aria2c', '-c', '-x', '8', '-s', '8', '-k', '64M', '--file-allocation=none',
                            '--max-tries=0', '--retry-wait=30', '--timeout=120', '--console-log-level=warn',
                            '--summary-interval=0', '-d', LOCAL, '-o', name, url])
        p = os.path.join(LOCAL, name)
        if r.returncode == 0 and os.path.exists(p) and (not size or os.path.getsize(p) == size):
            return p
        log(name, f'download attempt {attempt} failed (code {r.returncode}), retrying in 2 min')
        time.sleep(120)
    raise RuntimeError(name + ': download failed')

# MediaPipe pose ids -> Isharati joints 0..7: nose, neck (mid-shoulders), R sh/el/wr, L sh/el/wr;
# 8..28 left hand, 29..49 right hand (MediaPipe's sides, as in src/isharati/pose/keypoints.py)
BODY = [0, None, 12, 14, 16, 11, 13, 15]
def to_pose(frames):
    a = np.full((len(frames), 50, 3), np.nan, np.float32)
    for t, f in enumerate(frames):
        pose = f.get('pose_landmarks') or []
        if len(pose) >= 17:
            for j, i in enumerate(BODY):
                if i is not None: a[t, j, :2] = pose[i][:2]
            a[t, 1] = (a[t, 2] + a[t, 5]) / 2
        for key, start in (('left_hand_landmarks', 8), ('right_hand_landmarks', 29)):
            h = f.get(key) or []
            if len(h) == 21: a[t, start:start + 21, :2] = [p[:2] for p in h]
    a[..., 2] = np.where(np.isnan(a[..., 0]), np.nan, 0.0)
    width = np.linalg.norm(a[:, 2, :2] - a[:, 5, :2], axis=-1)   # shoulder width per frame
    width = width[np.isfinite(width) & (width > 1e-6)]
    if width.size == 0: return None, None
    raw_lh, raw_rh = float(np.isfinite(a[:, 8, 0]).mean()), float(np.isfinite(a[:, 29, 0]).mean())
    a = (a - a[:, 1:2, :]) / float(np.median(width))              # neck-centred, shoulder-scaled
    return a.astype(np.float16), (raw_lh, raw_rh)

def convert(zpath, part, npz_out, csv_out):
    arrays, rows = {}, []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith('.json')]
        for k, n in enumerate(names):
            clip = os.path.splitext(os.path.basename(n))[0]   # <video>.<start>-<end>
            try:
                d = json.loads(z.read(n))
                pose, hands = to_pose(d['keypoints'])
            except Exception:
                pose = None
            if pose is not None:
                arrays[clip] = pose
                video, span = clip.rsplit('.', 1)
                start, end = span.split('-')
                size = d.get('size') or [None, None]
                rows.append([clip, video, int(start), int(end), len(pose), size[0], size[1],
                             round(hands[0], 3), round(hands[1], 3), part])
            if k % 5000 == 0: log(part, f'converted {k}/{len(names)} clips')
    tmp = npz_out + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_out)
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['clip_id', 'video_id', 'start', 'end', 'frames', 'width', 'height', 'left_hand', 'right_hand', 'part'])
        w.writerows(rows)
    return len(arrays), len(names)

def hf_upload(local, remote):
    if not HF_REPO: return True
    from huggingface_hub import HfApi
    for attempt in range(1, 6):
        try:
            HfApi(token=HF_TOKEN).upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=HF_REPO,
                                              repo_type='dataset', commit_message=f'add {remote}')
            return True
        except Exception as e:
            log(remote, f'Hugging Face upload attempt {attempt} failed: {e}'); time.sleep(60 * attempt)
    return False

def hf_mark(part):  # a marker in Drive: this part is on Hugging Face
    return os.path.join(DEST, 'manifest', f'.hf_{part}')

for name in job['order']:
    if not drive_ok():
        print('Google Drive was disconnected: stopping; run cells 1-4 again', flush=True); break
    uuid, size = job['files'][name]
    is_zip = name.endswith('.zip')
    part = name[:-4] if is_zip else name
    final = os.path.join(DEST, name)
    npz_out = os.path.join(DEST, 'poses', part + '.npz')
    csv_out = os.path.join(DEST, 'manifest', part + '.csv')
    need_final = not is_zip or job['keep_zip']
    have_final = os.path.exists(final) and (not size or os.path.getsize(final) == size) and (not is_zip or zip_ok(final))
    have_conv = not is_zip or (os.path.exists(npz_out) and os.path.exists(csv_out))
    have_hf = not is_zip or not HF_REPO or os.path.exists(hf_mark(part))
    if (have_final or not need_final) and have_conv and have_hf:
        log(name, 'already done, skipped'); continue
    try:
        if not have_final or not have_conv:
            if have_final:
                p = final  # the zip is in Drive already: convert from there
            else:
                if shutil.disk_usage(LOCAL).free < (size or 0) * 1.1 + 5e9:
                    log(name, 'not enough runtime disk, stopping'); break
                log(name, 'downloading', f'{(size or 0)/1e9:.1f} GB')
                t0 = time.time()
                p = fetch(name, uuid, size)
                log(name, f'downloaded in {(time.time()-t0)/60:.0f} min')
            if is_zip:
                n = zip_ok(p)
                if not n:
                    os.remove(p)
                    raise RuntimeError(name + ': zip unreadable, deleted; it is fetched again on the next run')
                if not have_conv:
                    ln, lc = os.path.join(LOCAL, part + '.npz'), os.path.join(LOCAL, part + '.csv')
                    kept, total = convert(p, part, ln, lc)
                    log(name, f'converted: {kept}/{total} clips have a usable pose')
                    for src, dst in ((ln, npz_out), (lc, csv_out)):
                        shutil.copyfile(src, dst + '.part'); os.replace(dst + '.part', dst); os.remove(src)
            if need_final and not have_final:
                log(name, 'copying to Drive')
                shutil.copyfile(p, final + '.part')
                if size and os.path.getsize(final + '.part') != size: raise RuntimeError(name + ': Drive copy incomplete')
                if not drive_ok(): raise RuntimeError(name + ': Drive disconnected during the copy; local file kept')
                os.replace(final + '.part', final)
            if p != final: os.remove(p)
        if is_zip and HF_REPO and not os.path.exists(hf_mark(part)):
            log(name, 'uploading poses + manifest to Hugging Face')
            if hf_upload(npz_out, f'poses/{part}.npz') and hf_upload(csv_out, f'manifest/{part}.csv'):
                open(hf_mark(part), 'w').write('ok')
        log(name, 'done')
    except Exception:
        log(name, 'ERROR', traceback.format_exc().replace(chr(10), ' | '))
os.sync()  # hand everything to the Drive client before the session can end
log('worker finished')
'''

START = """
# 4. Start the background worker (safe to run again: finished work is skipped)
import subprocess
open('/content/worker.py', 'w').write(WORKER)
running = subprocess.run('pgrep -f /content/worker.py', shell=True, capture_output=True, text=True).stdout.strip()
if running:
    print('worker already running, pid', running)
else:
    env = dict(os.environ, HF_TOKEN=HF_TOKEN or '')   # the token goes to the worker only, never to a file
    subprocess.Popen(['python', '-u', '/content/worker.py'], stdout=open('/content/worker.out', 'a'),
                     stderr=subprocess.STDOUT, start_new_session=True, env=env)  # outlives this cell
    print('worker started; progress in', DEST + '/download_log.txt')
"""

PROGRESS = """
# 5. Progress (keep this running; stop it any time, the worker keeps going)
import time, glob
def status():
    out = subprocess.run('tail -n 6 ' + DEST + '/download_log.txt', shell=True, capture_output=True, text=True).stdout
    now = ', '.join(f'{os.path.basename(p)} {os.path.getsize(p)/1e9:.1f} GB'
                    for p in sorted(glob.glob(LOCAL + '/*')) if not p.endswith('.aria2'))
    done = len(glob.glob(DEST + '/poses/*.npz'))
    alive = bool(subprocess.run('pgrep -f /content/worker.py', shell=True, capture_output=True).stdout)
    return out, now, done, alive
while True:
    out, now, done, alive = status()
    print(time.strftime('%H:%M'), '| worker', 'running' if alive else 'STOPPED', f'| converted parts in Drive: {done}/10',
          '| local:', now or '-', f'| disk free {gb(LOCAL):.0f} GB')
    print(out)
    if not alive:
        print('worker stopped: see the log above (all done -> run cell 6; otherwise re-run cells 1-4 to resume)'); break
    time.sleep(300)
"""

FINISH = """
# 6. When all parts are done: one manifest with the English sentences, the word index, the dataset card -> Drive + HF
import csv, re, glob
MISSING = __MISSING__   # Qur'an/hadith words our ASL lexicon lacks (most frequent first)
split = {}
for s in ('train', 'dev'):
    p = f'{DEST}/YT.translations.{s}.json'
    if os.path.exists(p):
        for v in json.load(open(p)).values():
            for c in v['clip_order']: split[c] = s
caps = json.load(open(f'{DEST}/YT.translations.all.json'))
text = {c: v[c]['translation'] for v in caps.values() for c in v['clip_order']}
norm = lambda t: re.sub(r'\\s+', ' ', re.sub(r"[^a-z0-9\\s]", ' ', t.lower().replace("'s ", ' ').replace("'", ''))).split()

rows, index = [], {w: [] for w in MISSING}
for part in sorted(glob.glob(DEST + '/manifest/raw_keypoints_*.csv')):
    for r in csv.DictReader(open(part)):
        r['text'] = text.get(r['clip_id'], ''); r['split'] = split.get(r['clip_id'], '')
        rows.append(r)
        for w in set(norm(r['text'])) & index.keys():
            index[w].append(r['clip_id'])
cols = ['clip_id', 'video_id', 'start', 'end', 'frames', 'width', 'height', 'left_hand', 'right_hand', 'part', 'split', 'text']
with open(DEST + '/manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
json.dump(index, open(DEST + '/word_index.json', 'w'))
found = sum(1 for v in index.values() if v)
print(f'{len(rows)} clips in manifest.csv; {found}/{len(MISSING)} missing lexicon words occur in some sentence')

CARD = '''---
license: cc-by-4.0
language: [en]
tags: [sign-language, asl, pose, keypoints, mediapipe]
pretty_name: YouTube-ASL keypoints (Isharati pose format)
---
# YouTube-ASL keypoints in the Isharati pose format

Derived from the **YouTube-ASL Clip Keypoint Dataset** by Tomáš Železný et al., LINDAT/CLARIAH-CZ,
http://hdl.handle.net/11234/1-5898, licensed **CC BY 4.0**; captions from YouTube-ASL (Uthus, Tanzer, Georg, 2023).
Changes: MediaPipe 2D keypoints converted to 50 joints (nose, neck, shoulders, elbows, wrists, both hands), face
dropped, neck-centred and shoulder-scaled, float16.

* `poses/raw_keypoints_N.npz` — clip id -> `[T, 50, 3]` (x, y, z = 0; NaN = not detected). Joints: 0 nose, 1 neck,
  2-4 right shoulder/elbow/wrist, 5-7 left, 8-28 left hand, 29-49 right hand (MediaPipe sides). T = source frames.
* `manifest.csv` — clip_id, video_id, start, end, frames, width, height, left_hand/right_hand (share of frames with
  the hand detected), part (which npz), split (train/dev), text (English sentence).
* `word_index.json` — word -> clip ids whose sentence contains it.
'''
open(DEST + '/README.md', 'w').write(CARD)
if HF_REPO:
    api = HfApi(token=HF_TOKEN)
    for f in ('manifest.csv', 'word_index.json', 'README.md', 'YT.translations.all.json',
              'YT.translations.train.json', 'YT.translations.dev.json'):
        if os.path.exists(f'{DEST}/{f}'):
            api.upload_file(path_or_fileobj=f'{DEST}/{f}', path_in_repo=f, repo_id=HF_REPO, repo_type='dataset')
    print('uploaded:', f'https://huggingface.co/datasets/{HF_REPO}')
drive.flush_and_unmount()   # waits until every file is in Google Drive
print('all files are in Google Drive:', DEST)
"""

USE = """
## Later, on any machine

```python
from huggingface_hub import snapshot_download
import numpy as np, pandas as pd, json
root = snapshot_download('YOUR_NAME/youtube-asl-isharati', repo_type='dataset')   # ~5 GB, once
man = pd.read_csv(f'{root}/manifest.csv')
words = json.load(open(f'{root}/word_index.json'))
clip = words['mercy'][0]
part = man.set_index('clip_id').loc[clip, 'part']
pose = np.load(f'{root}/poses/{part}.npz')[clip]   # [T, 50, 3], the Isharati format
```
"""

PEEK = """
# 7. Peek: a few clips with their sentences, and the word index
import numpy as np
npz = sorted(glob.glob(DEST + '/poses/*.npz'))
if npz:
    z = np.load(npz[0])
    for k in list(z.keys())[:5]:
        print(k, z[k].shape, '|', text.get(k, '?'))
for w in MISSING[:15]:
    print(f'{w:14s} {len(index[w]):6d} clips')
"""


def main():
    cells = [md(INTRO), code(SETTINGS), code(TOOLS), code(FILES),
             code("# 4a. The worker script (runs in the background)\nWORKER = r'''" + WORKER + "'''"),
             code(START), code(PROGRESS), code(FINISH.replace("__MISSING__", json.dumps(missing_words()))),
             md(USE), code(PEEK)]
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(OUT, len(missing_words()), "missing words embedded")


if __name__ == "__main__":
    main()
