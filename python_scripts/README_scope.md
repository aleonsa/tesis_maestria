# Scopes del banco FCS-M²PC

Visor inicial con el stream existente de `07_m2pc`. No cambia el firmware,
no envía comandos y no reanuda una placa detenida. Requiere una sesión gráfica,
Python, NumPy, Matplotlib y pyserial (dependencias en `requirements-scope.txt`).
En este equipo ya están disponibles; no se instaló software adicional.

Desde la raíz del repositorio:

```bash
python3 python_scripts/scope_m2pc.py
```

Lee `/dev/ttyACM0` a 921600 baud y guarda automáticamente un log nuevo en
`datos_banco/m2pc_live_<fecha>.log` (ignorado por Git). Cerrar la ventana cierra
el puerto, pero **no detiene el motor**. No ejecutar otro lector serie a la vez.

Dos cosas aprendidas en banco (sesión 22): abrirlo **con la placa ya corriendo `07_m2pc`**
(si corre otra app a 115200 baud el log se llena de basura y no se recupera solo), y
`--save` **no sobreescribe** un log existente (borra el viejo o cambia el nombre).

## Controles y señales

- Ocho paneles sincronizados: velocidad, error de velocidad, id, iq, errores
  de corriente, corrientes αβ, tensiones comandadas y salida del PI.
- **Pausar vista** permite hacer zoom/pan con la barra de Matplotlib sin
  detener adquisición ni grabación. **Seguir** vuelve a la ventana deslizante.
- Selectores de ventana para observar dinámica lenta o ciclos eléctricos.
- Diagnóstico textual: estado de conexión, antigüedad de datos, descartes del
  visor, overruns, I²C, tiempos y offsets, cuando llegan las líneas de estado.
- El contador de descartes solo cubre la cola del visor: el stream actual no
  lleva secuencia y no permite cuantificar de forma fiable pérdidas previas.

La referencia de velocidad se extrae del banner de arranque. Si se conecta
con el motor ya funcionando, se **supone 300 rpm**, marcado explícitamente;
usar `--speed-ref 600` si corresponde. No es una referencia transmitida por
muestra, y durante el primer segundo el PI todavía no está activo.

**Las corrientes dq y sus errores son aproximaciones**, no métricas finales
de tesis: el logger promedia αβ durante 1 ms, transmite el ángulo al final y
lee iq* al imprimir. No se corrige ese desfase usando una latencia supuesta.
El stream no permite observar rizado PWM a 50 kHz. Las tensiones son comandos,
no mediciones. No se representa par medido ni ADALINE inexistente.

## Reproducir datos, sin mover el motor

```bash
python3 python_scripts/scope_m2pc.py --replay datos_banco/m2pc_s21_valle_on.log
python3 python_scripts/scope_m2pc.py --replay datos_banco/m2pc_s21_valle_on.log --rate 2
```

Los reinicios del dispositivo limpian la historia para no unir corridas.
No se generan logs nuevos al reproducir. La ventana permanece abierta al
terminar, para inspeccionar la última ventana.

Pruebas sin hardware:

```bash
python3 -m unittest discover -s python_scripts -p 'test_scope_m2pc.py'
```

Prueba gráfica con cierre y captura automáticos:

```bash
python3 python_scripts/scope_m2pc.py --replay datos_banco/m2pc_s21_valle_on.log \
  --duration 5 --snapshot /tmp/opencode/scope_m2pc.png
```
