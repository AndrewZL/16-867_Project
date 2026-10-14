#!/usr/bin/env bash
# Launch a headless CARLA server.  Usage: scripts/carla_server.sh [port=2000] [gpu=0]
CARLA_ROOT=${CARLA_ROOT:-$HOME/carla/CARLA_0.9.16}
PORT=${1:-2000}
GPU=${2:-0}
exec "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -quality-level=Low \
    -carla-rpc-port="$PORT" -graphicsadapter="$GPU"
