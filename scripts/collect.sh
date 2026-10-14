#!/usr/bin/env bash
# Collect one shard per town, with a fresh CARLA server per town (switching maps in a
# running server can segfault it).
#
# Usage: scripts/collect.sh <split> <scenarios_per_town> <cands> <base_seed> [port=2000] [gpu=0]
#   scripts/collect.sh train  150 2  0
#   scripts/collect.sh rerank  30 16 100 3000 1   # a server also binds port+1/+2: space parallel ones out
# Shards land in data/carla/<split>_<town>_s<seed>.npz. Override towns with TOWNS="...".
# Extra collect.py flags via EXTRA="...", e.g. EXTRA=--clean for unperturbed reference rollouts.
set -u
SPLIT=$1; N=$2; CANDS=$3; SEED=$4; PORT=${5:-2000}; GPU=${6:-0}
TOWNS=${TOWNS:-"Town10HD_Opt Town03_Opt Town05_Opt"}  # non-_Opt maps segfault the 0.9.16 server on load
cd "$(dirname "$0")/.."
mkdir -p data/carla logs

stop_server() { pkill -f -- "[C]arlaUE4-Linux-Shipping.*-carla-rpc-port=$PORT" ; sleep 3; }
trap stop_server EXIT

i=0
for town in $TOWNS; do
    seed=$((SEED + i))
    stop_server
    scripts/carla_server.sh "$PORT" "$GPU" > "logs/carla_server_$PORT.log" 2>&1 &
    uv run python -m envs.carla.collect --town "$town" --scenarios "$N" --cands "$CANDS" \
        --seed "$seed" --port "$PORT" --tm-port $((PORT + 6000)) \
        ${EXTRA:-} --out "data/carla/${SPLIT}_${town}_s${seed}.npz" 2>&1 | tee "logs/collect_${SPLIT}_${town}_s${seed}.log"
    i=$((i + 1))
done
