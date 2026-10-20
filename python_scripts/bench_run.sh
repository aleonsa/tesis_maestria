#!/usr/bin/env bash
# Flashea un .elf de 07_m2pc, captura N segundos del UART y resume las métricas.
# Uso: bench_run.sh <elf> <segundos> <log_destino>
set -u
elf=$1; secs=$2; log=$3
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
    -c "reset_config srst_only connect_assert_srst" \
    -c "program $elf verify reset exit" 2>&1 | grep -q "Verified OK" || { echo "FLASH FALLÓ"; exit 1; }
stty -F /dev/ttyACM0 921600 raw -echo
timeout "$secs" cat /dev/ttyACM0 > "$log" 2>/dev/null
python3 "$(dirname "$0")/bench_summary.py" "$log"
