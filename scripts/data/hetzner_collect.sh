#!/usr/bin/env bash
# Run scripts/data/collect_finepdfs.py on N Hetzner Cloud servers (one worker each).
#
#   scripts/data/hetzner_collect.sh up <config> <n> [server-type] [location]   # create servers, start workers
#   scripts/data/hetzner_collect.sh status <config>                             # last progress line per worker
#   scripts/data/hetzner_collect.sh attach <config> <worker>                    # watch a worker live (tmux)
#   scripts/data/hetzner_collect.sh down <config>                               # delete the servers
#
# Each worker runs inside a tmux session named "collect" and prints a progress line every 30 s
# (also logged to /root/collect.log). Detach from an attached session with Ctrl-b d.
# Needs: hcloud with an active context, an SSH key at ~/.ssh/id_ed25519.pub, and S3_ACCESS_KEY /
# S3_SECRET_KEY in the environment. The S3 keys are passed to each worker over SSH only; they are
# never written to Hetzner metadata (cloud-init) or to disk on the servers.
# Without a server type, the cheapest non-deprecated type at the location is used. Workers resume
# from their last uploaded shard, so `up` after a crash or `down` continues where they stopped.
set -euo pipefail

cmd=${1:?up|status|down}; config=${2:?FinePDFs config, e.g. srp_Cyrl}
label="role=collector,config=${config//_/-}"
prefix=${PREFIX:-unshredder/collect/finepdfs}  # PREFIX / EXTRA_ARGS (e.g. "--max-records 2000") for test runs
repo="https://github.com/nikola-rahman/unshredder.git"
ssh_opts=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes)

servers() { hcloud server list -l "$label" -o noheader -o columns=name,ipv4; }

case "$cmd" in
up)
  n=${3:?number of workers}; loc=${5:-fsn1}
  : "${S3_ACCESS_KEY:?}" "${S3_SECRET_KEY:?}"
  type=${4:-$(hcloud server-type list -o json | python3 -c "
import json, sys
loc = sys.argv[1]
ok = [(float(p['price_hourly']['gross']), t['name']) for t in json.load(sys.stdin) if not t.get('deprecation')
      for p in t['prices'] if p['location'] == loc]
print(min(ok)[1])" "$loc")}
  hcloud ssh-key describe unshredder >/dev/null 2>&1 ||
    hcloud ssh-key create --name unshredder --public-key-from-file ~/.ssh/id_ed25519.pub >/dev/null
  echo "creating $n x $type in $loc for $config"
  for w in $(seq 0 $((n - 1))); do
    hcloud server create --name "collect-${config//_/-}-$w" --type "$type" --location "$loc" \
      --image ubuntu-24.04 --ssh-key unshredder --label "$label" --label "worker=$w" >/dev/null &
  done
  wait
  servers | while read -r name ip; do
    w=${name##*-}
    until ssh "${ssh_opts[@]}" "root@$ip" true 2>/dev/null; do sleep 5; done
    ssh "${ssh_opts[@]}" "root@$ip" "S3_ACCESS_KEY='$S3_ACCESS_KEY' S3_SECRET_KEY='$S3_SECRET_KEY' bash -s" <<EOF &
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git tmux >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="\$HOME/.local/bin:\$PATH"
[ -d unshredder ] || git clone -q $repo
cd unshredder && git pull -q
tmux kill-session -t collect 2>/dev/null || true
tmux new-session -d -s collect -e S3_ACCESS_KEY="\$S3_ACCESS_KEY" -e S3_SECRET_KEY="\$S3_SECRET_KEY" \
  "cd /root/unshredder && uv run --no-project --with pyarrow --with huggingface_hub --with boto3 --with pymupdf \
   python scripts/data/collect_finepdfs.py --config $config --worker $w --workers $n --prefix $prefix \
   --exclude-sha256 configs/finepdfs_eval_exclude.sha256 ${EXTRA_ARGS:-} 2>&1 | tee /root/collect.log; echo FINISHED; sleep infinity"
echo "started worker $w on $name (tmux session: collect)"
EOF
  done
  wait
  ;;
status)
  servers | while read -r name ip; do
    echo "$name: $(ssh "${ssh_opts[@]}" "root@$ip" "grep -E '^(progress|DONE|listed|worker)|Error|Traceback' /root/collect.log | tail -1" 2>&1)"
  done
  ;;
attach)
  w=${3:?worker number}
  ip=$(servers | awk -v n="collect-${config//_/-}-$w" '$1 == n {print $2}')
  exec ssh -t "${ssh_opts[@]}" "root@$ip" tmux attach -t collect
  ;;
down)
  servers | while read -r name _; do hcloud server delete "$name" >/dev/null && echo "deleted $name"; done
  ;;
*) echo "unknown command $cmd" >&2; exit 2 ;;
esac
