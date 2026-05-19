#!/usr/bin/env bash
#
# Sincroniza firmware/ del Mac al Pi vía rsync usando el alias SSH "raspi".
#
# Uso:
#   ./sync_to_pi.sh              # sync real
#   ./sync_to_pi.sh --dry-run    # ver qué se haría, sin tocar nada
#
# Mac source:  ~/Documents/tesis_maestria/firmware/
# Pi target:   ~/projects/stm32g4/firmware/
#
# Excluye:
#   build/         — artefactos generados en el Pi, no se sincronizan
#   .DS_Store      — basura de macOS
#   *.swp          — swap de vim

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/"
DEST="raspi:projects/stm32g4/firmware/"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[dry-run] no se modifica nada en el Pi"
fi

rsync -av --delete \
    --exclude='build/' \
    --exclude='.DS_Store' \
    --exclude='*.swp' \
    $DRY_RUN \
    "${SRC}" "${DEST}"

echo
echo "Sync completo. Próximos pasos en el Pi:"
echo "  ssh raspi"
echo "  cd ~/projects/stm32g4/firmware/blink"
echo "  cmake -B build && cmake --build build"
