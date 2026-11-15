# Patrones de prosa que se evitan (español académico)

El revisor `scripts/check_prosa.py` marca los patrones de las secciones 1 a 4. La sección 5 es juicio humano.

## 1. Muletillas prohibidas por el autor
"es importante destacar / señalar / notar", "cabe mencionar / señalar / destacar / resaltar", "vale la pena", "en este sentido", "por otro lado", "además", "asimismo", "por consiguiente".

## 2. Relleno y énfasis vacío
- Conectores que no conectan: "de esta manera", "de este modo", "en definitiva", "en resumen" a mitad de sección, "en conclusión" fuera de las conclusiones, "como se mencionó anteriormente".
- Adverbios de énfasis: "claramente", "evidentemente", "notablemente", "sin lugar a dudas", "indudablemente", "fundamentalmente".
- "Significativamente" y "sustancialmente" sin una cifra al lado. Si hay cifra, la cifra basta.
- Adjetivos de folleto: "robusto", "sólido", "novedoso", "innovador", "crucial", "clave", "poderoso", "elegante", "versátil", "panorama", "paradigma".
- Perífrasis: "juega / desempeña un papel (clave, fundamental)", "en el marco de", "en el contexto de", "a lo largo de", "con el fin de" (usar "para"), "llevar a cabo" (usar el verbo), "realizar un análisis de" (usar "analizar").

## 3. Estructuras de máquina
- "No solo X, sino también Y".
- Contraste de pizarrón: "no es X, sino Y" repetido en un mismo apartado.
- Cierre de moraleja: la última frase del párrafo repite la idea con "lo que permite / lo cual garantiza / lo que demuestra que".
- Anuncio: "A continuación se presenta…", "En esta sección se describe…" cuando la sección empieza justo debajo. Una frase de ruta al inicio de un capítulo sí vale; una por sección, no.
- Cadena de gerundios: tres o más en una oración ("analizando, comparando y evaluando…").
- Tríadas por reflejo: tres sustantivos o tres adjetivos en cada frase. Una tríada real (tres cosas que de verdad hay que nombrar) está bien; diez en una página no.
- Oraciones de más de 45 palabras.
- Guiones largos (—) en cadena: más de dos por párrafo.

## 4. Lo que sí se hace
- Primero el hecho, luego su lectura. "El ADALINE recupera $\lambda$ en 90 s. Converge a la tensión efectiva, no a la BEMF del imán."
- Sujeto concreto: "medimos", "el LMS converge", "la ISR tarda 13 µs". Sin sujeto abstracto ("la implementación del esquema permite…").
- El número lleva unidad, condición y fuente. Si no tiene fuente, `\pend{}`.
- Un párrafo = una idea. Si hay un "y además", son dos párrafos.
- Un resultado negativo se escribe como cualquier otro: qué se midió, con qué controles, qué acota.

## 5. Antes y después

Relleno:
> Cabe señalar que, de esta manera, el algoritmo LMS desempeña un papel fundamental en la adaptación de los coeficientes, lo que permite una convergencia robusta.

> El LMS ajusta los coeficientes en cada periodo. Con $R = \tfrac12 \mathbf{I}$ la convergencia es monótona.

Moraleja:
> El M2PC evalúa 66 combinaciones por periodo, lo cual demuestra la gran capacidad de cómputo del microcontrolador.

> El M2PC evalúa 66 combinaciones por periodo. La ISR completa tarda 13 µs de 20.

Resultado negativo:
> Desafortunadamente, el ADALINE no logró mejorar el seguimiento, probablemente debido a limitaciones del banco.

> Con un modelo de BEMF equivocado en 20 %, el error de seguimiento no cambió (44.1 contra 46.7 mA). El rizo de conmutación, de unos 300 mA pico a pico, domina al efecto de la forma de la BEMF.

Anuncio:
> A continuación se presenta la caracterización de la BEMF del motor, la cual se llevó a cabo con la red de la placa.

> Medimos la BEMF en circuito abierto con la red de la placa.
