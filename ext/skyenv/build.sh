#!/bin/bash
# Build skyenv.dll (training: no drawing) and skyenv_draw.dll (with the port's renderer, for watching):
# the skyroads-sdl game logic (unchanged) plus our headless driver.
# Run from an MSYS2 UCRT64 shell:  bash ext/skyenv/build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/../skyroads-sdl/src"
gcc -std=c11 -O2 -w -shared -I"$S" -o "$HERE/skyenv.dll" \
    "$S/compat.c" "$S/game_play.c" "$HERE/sky_env.c" -lm -static-libgcc
gcc -std=c11 -O2 -w -shared -I"$S" -DSKYENV_DRAW -o "$HERE/skyenv_draw.dll" \
    "$S/compat.c" "$S/game_play.c" "$S/render.c" "$HERE/sky_env.c" -lm -static-libgcc
echo "built $HERE/skyenv.dll and skyenv_draw.dll"
