# Guión — reunión con el asesor, 23 de octubre de 2026

Presentación: `resultados_asesor.pdf` (11 láminas). Tiempo estimado: 15–20 minutos más discusión.
Las cifras salen de `fig/asesor_numeros.json`; si cambia algún log se regeneran con
`python3 python_scripts/make_advisor_figs.py`.

---

## Lámina 1 — Portada

Hoy quiero contarle lo que salió al cerrar el lazo con el ADALINE en el motor real. El resultado
no es el que esperábamos, pero creo que es un resultado sólido y con una explicación clara, y al
final me gustaría que decidamos entre un par de caminos para cerrar la tesis.

## Lámina 2 — Qué queríamos comprobar

La idea de la tesis es que si el controlador predictivo aprende la forma real de la BEMF mientras
el motor gira, predice mejor y el rizo baja. En simulación eso funcionó: el ADALINE redujo el rizo
13 % respecto del modelo trapezoidal y 3 % respecto del sinusoidal.

Para probarlo en el motor hicimos cuatro cosas. Primero medimos la BEMF con el inversor apagado,
para tener una referencia que no dependa de nada del controlador. Segundo, corrimos el control con
tres modelos: el nominal, uno equivocado a propósito y el que aprende el ADALINE. Tercero, no nos
quedamos sólo con el error RMS: también miramos el valor medio de la corriente y sus armónicos,
que resultaron ser mucho más sensibles. Y cuarto, probamos la estabilidad del aprendizaje con
distintas ganancias.

La respuesta corta es que en este motor el modelo de BEMF casi no influye en el resultado del
control. Lo importante es que eso tiene una explicación numérica, no es un "no salió".

## Lámina 3 — La BEMF del motor es prácticamente una sinusoide

Esta es la medición con el inversor apagado, girando la llanta a mano. A la izquierda, la curva
medida y la sinusoide ideal están una encima de la otra; abajo, el residuo: no pasa del 0.6 % del
pico. A la derecha, los armónicos por fase en escala logarítmica, con su intervalo de confianza.
La distorsión total es de 0.4 a 0.7 % según la fase, y sólo el tercero y el quinto se distinguen
del ruido.

Para ponerlo en contexto: el motor que usamos en simulación tiene 2.6 % de distorsión, y el
trapecio con el que comparábamos, 4.6 %. Aquí hay muy poco que aprender. Esto ya lo
sospechábamos; lo que no sabíamos era si "poco" era suficiente para notarse en el control.

## Lámina 4 — Cuánto pesa un error de BEMF dentro del predictor

Para responder eso hay que ver cómo entra la BEMF en el predictor. El predictor avanza la
corriente un paso con la tensión aplicada menos la BEMF, multiplicado por G = Ts/L, que en nuestro
banco vale 0.023 amperes por volt. Como compensamos un retardo, la predicción abarca dos pasos:
un error de δe volts en la BEMF mueve la corriente predicha 2·G·δe.

Con eso podemos poner todo en la misma escala de corriente, que es lo que muestra la gráfica.
Arriba, en rojo, lo que un error de BEMF mueve en la predicción: la distorsión real del motor
equivale a 0.2 miliamperes; un modelo equivocado un 25 %, a 9 miliamperes; quitar la BEMF del
modelo por completo, a 36. Abajo, en azul, con lo que trabaja el controlador: pasos de 19
miliamperes, porque sólo puede elegir entre diez subdivisiones de un vector; un error RMS medido
de 48; y un rizo pico a pico de 284.

La escala es logarítmica. Lo que el modelo de BEMF puede mover está entre cinco y doscientas
veces por debajo del piso del controlador.

## Lámina 5 — Lo que predice esa cuenta y lo que medimos

Si el error de modelo y el piso son independientes, el RMS total es la suma cuadrática de los
dos. Esa es la curva azul, y no tiene ningún parámetro ajustado: sólo G, la BEMF de pico y el
piso medido. Es plana hasta errores de modelo de varias decenas de por ciento. Para que el RMS
subiera un 10 % habría que equivocarse un 60 % en la BEMF; incluso quitándola por completo
subiría sólo a 60 miliamperes.

Las líneas punteadas marcan dónde caen los casos reales: el banco, en verde, está en 0.5 %; el
motor de simulación, en naranja, en 2.6 %; y el trapecio frente al motor real, en rojo, en 4 %.
Todos en la parte plana.

Y los dos puntos medidos confirman la curva: con el modelo nominal medimos 48 ± 2.5
miliamperes, y con el modelo equivocado un 25 %, 45 ± 2. El modelo equivocado no empeoró nada.

## Lámina 6 — En el experimento, el error RMS no distingue un modelo de otro

Aquí están las cuatro corridas, todas a 300 rpm, en la misma sesión y con la misma carga de
telemetría. A la izquierda, el RMS por bloques de 10 segundos a lo largo del tiempo; a la derecha,
la media con su dispersión. Nominal 48, equivocado 45, ADALINE desde el nominal 45. Las
diferencias son de 3 o 4 miliamperes, que es lo que varía una corrida de la otra.

Un dato más: el error en el eje d es casi igual al del eje q, 0.9 a 1. Si el error viniera del
modelo de BEMF tendría una estructura; que sea igual en los dos ejes apunta a ruido de medida y
resolución del controlador.

La corrida naranja, el ADALINE partiendo de un modelo equivocado, es la única que sube con el
tiempo, hasta 52 ± 4. Eso lo retomo en la lámina de estabilidad.

## Lámina 7 — Donde sí se nota el modelo: en el valor medio de la corriente

Entonces, ¿el modelo no importa para nada? Sí importa, pero no en el RMS. Importa en el valor
medio de la corriente, que se mide con 0.1 miliamperes de precisión, 25 veces mejor que el RMS.

El modelo equivocado desplaza la corriente media 12.6 miliamperes en el eje d y 10.3 en el eje q.
La misma cuenta de dos pasos, menos 2·G·δe, predice 5.0 y 7.6. No coincide exactamente —estamos
a un factor de dos en d— pero da el signo y el orden de magnitud sin ajustar nada. Lo que quiero
rescatar es la conclusión física: un error en el fundamental de la BEMF se convierte en un sesgo
constante, no en rizo. Y un sesgo constante lo corrige el lazo de velocidad sin problema.

## Lámina 8 — El rizo que queda no viene de la forma de la BEMF

Esta es la gráfica que más me costó y creo que es la más contundente. Descompusimos la corriente
por armónicos del ángulo eléctrico, usando 26 mil ventanas sincronizadas con el encoder; la
resolución es de 0.12 miliamperes. El rizo de par, que es lo que nos importa, aparece en el sexto
armónico: 2 miliamperes en iq y 15 en id.

La barra roja es la cota de lo que la forma de la BEMF puede explicar en ese armónico, con la
distorsión que medimos: 0.2 miliamperes. Una décima parte del rizo en iq y una centésima en id.
El resto lo ponen el lazo y el inversor: la estructura de sectores, los tiempos muertos, la
resolución de diez subdivisiones. Aunque el ADALINE aprendiera la forma perfectamente, el 90 %
del rizo seguiría ahí.

## Lámina 9 — El ADALINE aprende la amplitud, pero arrastra un sesgo de fase

Ahora, sobre el ADALINE en sí. Lo arrancamos con una BEMF un 20 % más chica de lo real y con 10
grados de desfase. Panel a: la amplitud la corrige en unos 110 segundos y termina al 3 % del
valor que medimos con el inversor apagado. Eso es lo que queríamos: el identificador funciona en
lazo cerrado.

Panel b: la fase no la corrige. Arranque en 10 grados o arranque en cero, termina entre 10 y 12.
Y panel c: aparece un desbalance entre los ejes alfa y beta del 6 al 8 %, que crece con el
tiempo. Nuestra sospecha es que el observador reconstruye la BEMF a partir de la tensión que se
le pide al inversor, no de la que realmente aplica, y los tiempos muertos y las asimetrías se
cuelan ahí. Pero esto no lo hemos comprobado; lo pongo como pendiente.

## Lámina 10 — La adaptación en línea sólo es segura con ganancia pequeña y buena señal

Último resultado: estabilidad. Todo normalizado al RMS de su referencia. En rojo, con la ganancia
que usábamos en simulación, el lazo se vuelve inestable: el error se triplica y salta la protección
de sobrecorriente. Con una ganancia 20 veces menor, en verde, a velocidad constante, el
aprendizaje no estorba. Pero en naranja, con velocidad variable y pasando por cero, donde la
BEMF es pequeña y el ruido del observador domina, el error sube más del doble.

Y la curva azul oscura, partiendo del modelo equivocado, sube despacio hasta un 11 % al final. Esto
es lo que no me cierra: la cuenta de sensibilidad diría que el aprendizaje mueve apenas 1
miliampere. Hay algo en la interacción entre lo que aprende y el controlador que todavía no
entendemos.

## Lámina 11 — Qué concluyo y qué quisiera decidir hoy

Lo que queda establecido: la BEMF de este motor es sinusoidal con menos de 0.7 % de distorsión; el
error de corriente tiene un piso de 48 miliamperes que ponen el controlador y la medición, y un
error de modelo de hasta 25 % no lo mueve; y el ADALINE identifica la amplitud en lazo cerrado.

Lo que queda acotado: aprender la forma de la BEMF sólo paga cuando el error de modelo es
comparable al piso del controlador. En este motor estamos más de un orden de magnitud por debajo
de ese punto. Por eso yo no diría que la hipótesis se refuta; se delimita, y la delimitación es
cuantitativa.

Lo que no sabemos: de dónde sale el sesgo de fase del observador y por qué el aprendizaje degrada
más de lo que la sensibilidad predice. Y hay que decir que todo se midió a 300 rpm y casi sin
carga.

Y lo que me gustaría decidir con usted. Tengo cuatro opciones y no son excluyentes:

1. Cerrar con este resultado. Las contribuciones serían la BEMF medida con incertidumbre, el
   identificador validado en lazo cerrado y la frontera cuantitativa de cuándo importa el modelo.
2. Cambiar la métrica a lo que sí responde al modelo —valor medio y armónicos— y repetir con
   carga, donde el rizo de par sí escala con la corriente.
3. Forzar un desajuste grande, emulando una BEMF distorsionada en el modelo, para comprobar la
   curva de sensibilidad en la zona donde sí sube.
4. Corregir el observador antes de seguir adaptando en línea.

Mi inclinación es la primera más la tercera: cerrar con lo que hay y agregar un experimento que
confirme la curva por el lado donde el modelo sí importa. Pero quisiera su opinión.

---

## Preguntas que pueden salir, y qué contestar

**"¿No será que el ADALINE está mal implementado?"**
Hicimos un control: el ADALINE activo pero con ganancia cero, para ver si el solo hecho de tenerlo
en el lazo cambiaba algo. Dio lo mismo que sin ADALINE. Además, partiendo de un modelo 20 % más
chico, recupera la amplitud. Lo que hace, lo hace bien; el problema es que no hay nada que ganar.

**"¿Y la fase de 10 grados?"**
Ésa es la parte que no entendemos. Apunta al observador. No afecta la conclusión principal porque
el modelo con 10 grados de error dio el mismo RMS que el nominal.

**"¿Por qué en simulación sí funcionó?"**
Dos razones. En simulación el motor tenía 2.6 % de distorsión y se comparaba contra un trapecio
que se equivocaba 4 %, y aun así la ganancia frente al seno fue sólo 3 %. Y ahí el piso del
controlador era otro: 240 volts de bus con una inductancia distinta. Habría que correr la
ablación amplitud contra forma que ya está escrita para separar cuánto de esa ganancia es de
amplitud y cuánto de forma.

**"¿Con carga cambiaría?"**
El efecto sobre la corriente no depende de la carga, pero el rizo de par por forma de BEMF sí
escala con iq. Con 10 miliamperes de iq no lo vemos. Es un experimento razonable, pero la cota
seguiría siendo del orden de la distorsión, 0.5 %.

**"¿Qué tan confiable es la referencia de BEMF?"**
Usa la red de sensado del propio inversor, con un divisor cuya razón tiene ±5 % de
incertidumbre. Eso mueve la amplitud, no la forma; los armónicos relativos están bien medidos.

**"¿Cuánto pesan las corridas?"**
Pocas: 5, 3 y 14 bloques de 10 segundos para las referencias. La variación de ±2.5 miliamperes
entre corridas es una estimación con pocos datos. Repetir es barato si hace falta.
