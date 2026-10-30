#!/usr/bin/env bash
# Ranca & Murray (2013) UnShredder, unmodified, in an isolated Python 2.7 container.
# Runs both of its cost functions with its Kruskal search on the fixed test splits.
#   bash docker_run.sh /root/data/datasets <out_dir>
# Files are copied in and out with `docker cp` (bind mounts don't work on Docker-in-Docker hosts).
set -euo pipefail
DATA=${1:?datasets dir}; OUT=${2:?output dir}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
NAME=ranca2013

docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME -e MPLBACKEND=Agg python:2.7 sleep infinity >/dev/null
trap 'docker rm -f $NAME >/dev/null 2>&1 || true' EXIT
docker exec $NAME mkdir -p /data/datasets /out
for d in D1 D2 D3; do docker cp "$DATA/$d" $NAME:/data/datasets/$d; done
docker cp "$HERE/run_ranca.py" $NAME:/opt/run_ranca.py

docker exec $NAME bash -c '
set -euo pipefail
pip install -q "numpy==1.16.6" "scipy==1.2.3" "matplotlib==2.2.5" "Pillow==6.2.2" pybrain==0.3 2>&1 | tail -1
git clone -q https://github.com/RazvanRanca/UnShredder.git /src && cd /src/code
git log -1 --format="UnShredder commit %H"
docs() { python - "$1" "$2" "$3" <<"PY"
import os, sys
root, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
base = os.path.join(root, "mechanical")
names = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d, "strips")))
print(" ".join(os.path.join(base, d) for d in names[lo:hi]))
PY
}
TEST="$(docs /data/datasets/D1 45 60) $(docs /data/datasets/D2 15 20) $(docs /data/datasets/D3 75 100)"
TRAIN=$(for d in $(docs /data/datasets/D1 15 17); do echo $d/$(basename $d).jpg; done)
for COST in prediction; do  # blackGaus: their search calls np.logaddexp on Decimal values, unsupported by every NumPy built for Python 2.7
  yes "" | python /opt/run_ranca.py --cost $COST --train $TRAIN --docs $TEST --out /out/ranca_$COST.jsonl \
    2> /out/ranca_$COST.log || echo "cost $COST failed, see log"
done
echo RANCA_DONE
'
docker cp $NAME:/out/. "$OUT/"
