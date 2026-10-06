#!/usr/bin/env sh
# Publish the current commit's files to the public repository as one new commit on its main, without the private
# history (which held secrets): the public history is a chain of snapshots, each with the previous snapshot as parent.
#   origin  -> astral-fate/isharati-signs-archive  (private, full history)
#   public  -> astral-fate/isharati-signs          (public, snapshots only)
# Never `git push public <branch>`: that would publish the private history.
#   sh scripts/hub/publish_public.sh "message"
set -e
git remote get-url public >/dev/null 2>&1 || git remote add public https://github.com/astral-fate/isharati-signs.git
git fetch -q public main
msg=${1:-"Snapshot of $(git rev-parse --short HEAD)"}
c=$(git commit-tree "HEAD^{tree}" -p public/main -m "$msg")
git push public "$c:refs/heads/main"
echo "public main -> $c"
