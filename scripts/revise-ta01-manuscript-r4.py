#!/usr/bin/env python3
"""Update the manuscript with repeated, paired input-perturbation results."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Articulo_G25_revision_integrada_DeepSeek_Claude_R3.docx"
OUTPUT = ROOT / "Articulo_G25_revision_integrada_DeepSeek_Claude_R4.docx"
FIGURES = ROOT / "artifacts/manuscript-review-2026-10-29"


def find_para(doc: Document, prefix: str):
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise RuntimeError(f"Paragraph not found: {prefix}")


def set_text(paragraph, value: str) -> None:
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(value)


def main() -> None:
    doc = Document(SOURCE)

    # Report the prespecified PI3 stress levels using 30 paired stochastic draws.
    p = find_para(doc, "Al introducir un 5 % de ruido gaussiano")
    set_text(
        p,
        "La PI3 se reanalizó con 30 perturbaciones independientes por condición y horizonte, aplicadas de forma pareada a los mismos modelos y ventanas de prueba. Con ruido gaussiano del 5 %, el aumento medio del MAE respecto de la entrada limpia fue 6.1 % y 5.8 % para la LSTM (1 h y 6 h), y 60.0 % y 45.8 % para el híbrido sin OOF. Con 30 % de faltantes MCAR imputados mediante la media de entrenamiento, los aumentos fueron 97.0 % y 71.0 % para la LSTM, y 132.0 % y 82.6 % para el híbrido sin OOF. La figura 5 añade el híbrido OOF y un MLP libre OOF: bajo ruido del 5 %, sus aumentos medios fueron 343 %/31.1 % y 828.5 %/334.9 %, respectivamente; bajo 30 % MCAR, 390.9 %/81.2 % y 701.9 %/348.4 % (1 h/6 h). La tabla 6 presenta MAE promediados entre sorteos para las condiciones principales de la PI3. Las barras de error de la figura son IC bootstrap del 95 % sobre sorteos, condicionados a las fechas de prueba y ajustes fijos; no cuantifican incertidumbre de muestreo de fechas ni de inicialización de modelos. Los correctores OOF y MLP libre promedian tres ajustes por sorteo. Las perturbaciones independientes de canales y pasos constituyen una prueba de estrés, no un proceso realista de fallas de sensor; el desplazamiento actual de la restricción de salida permaneció limpio. No se reentrenó bajo perturbación. Por ello, la magnitud de sensibilidad es específica del protocolo y del generador semisintético TA-01, y no se extrapola al campo. Fuente: data/validation/ta01-robustness-repeated.json y data/validation/ta01-robustness-repeated-draws.csv."
    )
    set_text(
        p,
        p.text.replace(
            "No se reentrenó bajo perturbación. Por ello, la magnitud de sensibilidad",
            "No se reentrenó bajo perturbación. El ruido independiente en cada paso también altera diferencias derivadas de presión de poros e índice de seguridad: con el nivel nominal de 1 %, la desviación estándar inducida en la diferencia último−primero equivale a 164.6 y 146.7 desviaciones estándar de entrenamiento de la característica derivada de presión (1 h y 6 h), y a 133.3 y 140.9 para la característica derivada de seguridad. Este diagnóstico de escala puede ayudar a interpretar la sensibilidad de los correctores, pero no identifica su causa; convendría contrastarlo con perturbaciones temporalmente coherentes y variables estáticas preservadas. Por ello, la magnitud de sensibilidad",
        ),
    )
    p = find_para(doc, "Tabla 6. MAE del híbrido principal sin OOF")
    set_text(
        p,
        "Tabla 6. MAE medio de la LSTM y del híbrido principal sin OOF en 30 sorteos de perturbación pareados; la condición limpia es el valor de referencia. Se muestran las ventanas de prueba y el MAE en mm. Fuente: data/validation/ta01-robustness-repeated.json."
    )
    p = find_para(doc, "Figura 5. Cambio porcentual del MAE")
    set_text(
        p,
        "Figura 5. Aumento medio del MAE respecto de entradas limpias en 30 sorteos de ruido gaussiano o faltantes MCAR, para LSTM, híbrido sin OOF, híbrido OOF y MLP libre OOF. Los puntos representan medias y las barras, IC bootstrap del 95 % entre sorteos, condicionados a fechas de prueba y ajustes fijos. El ruido se escala con la desviación estándar de entrenamiento; los faltantes se imputan con la media de entrenamiento. Las perturbaciones se aplican independientemente a todos los canales y pasos, y el desplazamiento usado en la restricción de salida permanece limpio. No hubo reentrenamiento. Fuente: data/validation/ta01-robustness-repeated.json."
    )

    # Keep methods, interpretation, and figure cross-references aligned with R4.
    p = find_para(doc, "Los escaladores de entrada y la normalización del cambio del objetivo")
    old_protocol = "La evaluación de robustez de la PI3 se realizó con el híbrido ajustado en la partición principal sin OOF; no se evaluó aquí la robustez del corrector entrenado con predicciones OOF ni se reentrenaron los modelos bajo perturbación. En la evaluación de robustez, el ruido y el enmascaramiento MCAR se aplicaron de forma independiente a los 15 canales de entrada y a cada paso del historial; el límite de salida se calculó con el desplazamiento actual original, sin perturbar."
    new_protocol = "La PI3 se amplió con 30 sorteos independientes por condición y horizonte; cada sorteo generó las mismas entradas perturbadas para LSTM, híbrido sin OOF, híbrido OOF y MLP libre OOF. Se mantuvieron fijos las fechas de prueba y los ajustes, y se promediaron por sorteo tres correctores OOF o tres MLP libres. Los IC del 95 % se estimaron con 20 000 remuestreos bootstrap de los 30 sorteos y son condicionales a esos ajustes y fechas; no incorporan variabilidad entre fechas ni entre inicializaciones. El ruido gaussiano y el enmascaramiento MCAR se aplicaron de forma independiente a los 15 canales y a cada uno de los seis pasos; el límite de salida utilizó el desplazamiento actual original, sin perturbar. No hubo reentrenamiento bajo perturbación."
    if old_protocol not in p.text:
        raise RuntimeError("Could not locate prior robustness protocol for replacement")
    set_text(p, p.text.replace(old_protocol, new_protocol))

    p = find_para(doc, "La PI1 se examina en las tablas 3 y 4")
    set_text(
        p,
        "La PI1 se examina en las tablas 3 y 4; la PI2, mediante las comparaciones pareadas agrupadas por fecha descritas en resultados y el contraste exploratorio con el comparador proporcional de la figura 3; la PI3, en la tabla 6 y la figura 5; y la PI4, mediante las verificaciones del MEF y las pruebas espaciales de la tabla 5 y la figura 2. Las conclusiones se circunscriben a estas evaluaciones."
    )

    p = find_para(doc, "Validez de las conclusiones.")
    set_text(
        p,
        p.text.replace(
            "La unidad de remuestreo fue la fecha de lluvia y no la ventana horaria.",
            "Para los contrastes predictivos por fecha, la unidad de remuestreo fue la fecha de lluvia y no la ventana horaria. En la PI3, los intervalos resumen variación entre 30 sorteos de perturbación, condicionada a fechas de prueba y ajustes fijos, y no deben interpretarse como incertidumbre entre fechas o inicializaciones.",
        )
    )

    p = find_para(doc, "Respecto a la PI2, no se observó una mejora uniforme")
    set_text(
        p,
        p.text.replace(
            "En la partición principal, tanto el ajuste original como las sensibilidades OOF favorecieron con mayor frecuencia al híbrido, pero sus intervalos pareados incluyeron cero.",
            "En la partición principal sin OOF, el híbrido obtuvo menor MAE; el IC pareado por fecha excluyó cero a 1 h e incluyó cero a 6 h. Las sensibilidades multisemilla y OOF favorecieron con mayor frecuencia al híbrido, pero sus intervalos incluyeron cero.",
        ).replace(
            "Para la PI3, los resultados de robustez corresponden al híbrido principal sin OOF: ambos modelos se degradaron con las perturbaciones y el aumento fue mayor para el híbrido en las condiciones reportadas. No se evaluó la robustez del corrector OOF ni se entrenó con entradas perturbadas.",
            "Para la PI3, se evaluaron LSTM, híbrido sin OOF, híbrido OOF y MLP libre OOF en 30 sorteos de perturbación. La sensibilidad varió entre modelos y horizontes; en varios escenarios los correctores OOF y el MLP libre mostraron aumentos de MAE mayores que la LSTM. Estos intervalos son condicionales a ajustes y fechas fijos y el protocolo no incluyó reentrenamiento con entradas perturbadas.",
        )
    )

    # Remove an accidental repeated sentence from the prior editing pass.
    duplicate = (
        " En particular, el generador calcula por nodo la norma euclidiana de la diferencia entre el desplazamiento húmedo y el estado seco, redondea esa magnitud a 10⁻⁵ mm y después selecciona el máximo espacial; no redondea primero el máximo global ni calcula la norma a partir de componentes previamente redondeadas."
    )
    p = find_para(doc, "El MEF supone deformación plana y elasticidad lineal")
    if p.text.endswith(duplicate + duplicate):
        set_text(p, p.text[:-len(duplicate)])

    # Update the data-availability paragraph so that the new evidence is auditable.
    p = find_para(doc, "El repositorio público sslope")
    if "ta01-robustness-repeated.json" not in p.text:
        set_text(
            p,
            p.text.replace(
                "data/validation/ta01-primary-influence-and-target-regimes.json; los métodos",
                "data/validation/ta01-primary-influence-and-target-regimes.json, data/validation/ta01-robustness-repeated.json y data/validation/ta01-robustness-repeated-draws.csv; los métodos",
            ).replace(
                "scripts/evaluate-ta01-proportional-rainfall-baseline.py",
                "scripts/evaluate-ta01-proportional-rainfall-baseline.py y scripts/evaluate-ta01-robustness-repeated.py",
            ),
        )

    # Refresh the table values from repeated-draw means; preserve its geometry.
    repeated = json.loads((ROOT / "data/validation/ta01-robustness-repeated.json").read_text(encoding="utf-8"))
    expected = {}
    for result in repeated["results"]:
        horizon_label = f"{result['horizonHours']} h"
        expected[(horizon_label, "Limpias")] = (
            result["cleanMaeMmByModel"]["lstm"],
            result["cleanMaeMmByModel"]["in_sample_hybrid"],
        )
        for condition in result["conditions"]:
            if condition["condition"] == "NOISE_5PCT_STD":
                label = "5% de ruido"
            elif condition["condition"] == "MISSING_30PCT_MCAR":
                label = "30% de datos faltantes"
            else:
                continue
            expected[(horizon_label, label)] = (
                condition["summary"]["lstm"]["meanMaeMm"],
                condition["summary"]["in_sample_hybrid"]["meanMaeMm"],
            )
    table = doc.tables[5]
    if len(table.rows) != 7 or len(table.columns) != 5:
        raise RuntimeError("Unexpected Table 6 geometry")
    for row in table.rows[1:]:
        key = (row.cells[0].text.strip(), row.cells[1].text.strip())
        if key not in expected:
            raise RuntimeError(f"Unexpected Table 6 row: {key}")
        lstm, hybrid = expected[key]
        for cell, value in ((row.cells[3], lstm), (row.cells[4], hybrid)):
            if cell.paragraphs and cell.paragraphs[0].runs:
                cell.paragraphs[0].runs[0].text = f"{value:.8f}"
                for run in cell.paragraphs[0].runs[1:]:
                    run.text = ""
            else:
                cell.text = f"{value:.8f}"

    # Do not alter quotations, references, section order, table/figure count, or layout geometry.
    doc.save(OUTPUT)

    replacement = {"word/media/image5.jpg": FIGURES / "figura5.jpg"}
    temp = OUTPUT.with_suffix(".tmp.docx")
    with ZipFile(OUTPUT, "r") as src, ZipFile(temp, "w", ZIP_DEFLATED) as dst:
        for item in src.infolist():
            content = replacement[item.filename].read_bytes() if item.filename in replacement else src.read(item.filename)
            dst.writestr(item, content)
    temp.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
