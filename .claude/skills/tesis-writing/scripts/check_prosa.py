#!/usr/bin/env python3
"""Revisor de prosa para la tesis. Marca muletillas, relleno y estructuras de máquina.

Uso: python3 check_prosa.py archivo.tex [más.tex ...]
Solo biblioteca estándar. Salida: archivo:línea [categoría] texto. Código de salida 1 si hay hits.
Los patrones están descritos en patrones.md.
"""
import re
import sys

BANNED = [
    r"es importante (destacar|señalar|notar|mencionar|resaltar)",
    r"cabe (mencionar|señalar|destacar|resaltar|notar)",
    r"vale la pena",
    r"en este sentido",
    r"por otro lado",
    r"\bademás\b",
    r"\basimismo\b",
    r"por consiguiente",
]
FILLER = [
    r"de esta manera", r"de este modo", r"en definitiva", r"en conclusión",
    r"como se mencionó anteriormente", r"sin lugar a dudas", r"indudablemente",
    r"\bclaramente\b", r"\bevidentemente\b", r"\bnotablemente\b", r"fundamentalmente",
    r"\bsignificativamente\b", r"\bsustancialmente\b",
    r"\brobust[oa]s?\b", r"\bsólid[oa]s?\b", r"\bnovedos[oa]s?\b", r"\binnovador(a|es|as)?\b",
    r"\bcrucial(es)?\b", r"\bpoderos[oa]s?\b", r"\bversátil(es)?\b", r"\bpanorama\b", r"\bparadigma\b",
    r"(juega|juegan|desempeña|desempeñan) un papel", r"en el marco de", r"en el contexto de",
    r"a lo largo de", r"con el fin de", r"llevar a cabo", r"llevó a cabo", r"llevamos a cabo",
    r"realizar un análisis",
]
STRUCT = [
    (r"no (solo|sólo) .{1,80}sino (también|que)", "no solo... sino también"),
    (r"(,|;) (lo (que|cual)) (permite|garantiza|demuestra|asegura|posibilita|evidencia)", "cierre de moraleja"),
    (r"a continuación se (presenta|describe|muestra|expone)", "anuncio"),
    (r"en esta sección se (presenta|describe|muestra|expone)", "anuncio"),
]


def limpiar(texto):
    """Quita comentarios, matemáticas, entornos no prosa y comandos. Conserva el número de línea."""
    # comentarios: % no precedido por backslash
    texto = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in texto.split("\n"))

    def blanquear(m):
        return re.sub(r"[^\n]", " ", m.group(0))

    for env in ("equation", "align", "eqnarray", "tikzpicture", "tabular", "table", "figure", "algorithmic", "lstlisting"):
        texto = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env), blanquear, texto, flags=re.S)
    texto = re.sub(r"\$\$.*?\$\$", blanquear, texto, flags=re.S)
    texto = re.sub(r"\$[^$]*\$", blanquear, texto)
    texto = re.sub(r"\\(cite[pt]?|ref|eqref|label|gls|acrshort|acrlong|pend|input|subfile|graphicspath|documentclass)\*?(\[[^\]]*\])?\{[^}]*\}", blanquear, texto)
    texto = re.sub(r"\\(begin|end)\{[^}]*\}", blanquear, texto)
    texto = re.sub(r"\\(emph|textbf|textit|item|section|subsection|subsubsection|chapter|caption)\*?(\[[^\]]*\])?", " ", texto)
    texto = re.sub(r"\\[a-zA-Z]+", " ", texto)
    return texto


def oraciones(texto):
    """Genera (línea, oración). Corta en . ? ! y en líneas en blanco (títulos y párrafos)."""
    linea = 1
    for parte in re.split(r"(\n\s*\n)", texto):
        if re.fullmatch(r"\n\s*\n", parte):
            linea += parte.count("\n")
            continue
        for m in re.finditer(r"[^.?!]+[.?!]?", parte):
            frag = m.group(0)
            lead = len(frag) - len(frag.lstrip())
            if frag.strip():
                yield linea + parte[: m.start()].count("\n") + frag[:lead].count("\n"), frag.strip()
        linea += parte.count("\n")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    hits = 0
    for ruta in argv[1:]:
        crudo = open(ruta, encoding="utf-8").read()
        texto = limpiar(crudo)
        lineas = texto.split("\n")

        def reportar(n, cat, frag):
            nonlocal hits
            hits += 1
            print(f"{ruta}:{n} [{cat}] {frag.strip()[:110]}")

        for n, l in enumerate(lineas, 1):
            bajo = l.lower()
            for p in BANNED:
                if re.search(p, bajo):
                    reportar(n, "muletilla", l)
            for p in FILLER:
                if re.search(p, bajo):
                    reportar(n, "relleno", l)
            if l.count("—") > 2:
                reportar(n, "guiones largos", l)

        plano = "\n".join(lineas)
        for p, nombre in STRUCT:
            for m in re.finditer(p, plano.lower(), flags=re.S):
                n = plano[: m.start()].count("\n") + 1
                reportar(n, nombre, plano[m.start(): m.end()].replace("\n", " "))

        for n, o in oraciones(plano):
            palabras = o.split()
            if len(palabras) > 45:
                reportar(n, f"oración de {len(palabras)} palabras", o)
            gerundios = re.findall(r"\b\w+(?:ando|iendo)\b", o.lower())
            if len(gerundios) >= 3:
                reportar(n, "cadena de gerundios", o)
    print(f"\n{hits} hallazgos" if hits else "\nsin hallazgos")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
