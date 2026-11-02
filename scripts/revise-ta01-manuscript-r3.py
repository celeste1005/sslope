#!/usr/bin/env python3
"""Create an evidence-calibrated R3 manuscript without changing references."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "Articulo_G25_revision_integrada_DeepSeek_Claude_R2.docx"
OUTPUT = ROOT / "Articulo_G25_revision_integrada_DeepSeek_Claude_R3.docx"
FIGURES = ROOT / "artifacts/manuscript-review-2026-10-29"


def find_para(doc: Document, prefix: str):
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise RuntimeError(f"Paragraph not found: {prefix}")


def set_text(paragraph, value: str) -> None:
    # Keep paragraph-level style and first-run character style.
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(value)


def main() -> None:
    doc = Document(SOURCE)
    paragraphs = doc.paragraphs

    # Method: an exploratory, history-only comparator that uses the same idealized
    # future rainfall information available to the forecasting models.
    p = find_para(doc, "Los escaladores de entrada y la normalización del cambio")
    set_text(p, p.text + " Como control exploratorio adicional, se evaluó un comparador proporcional basado exclusivamente en seis pares históricos de lluvia acumulada y desplazamiento almacenado. En cada ventana se estimó por mínimos cuadrados, sin intercepto, la pendiente β = Σ(rᵢyᵢ)/Σ(rᵢ²), y se pronosticó max(yₜ, βrₜ₊ₕ); rₜ₊ₕ es la lluvia acumulada conocida en este experimento y yₜ el desplazamiento actual. Si la lluvia histórica acumulada fue cero, se usó persistencia. El comparador no utiliza la etiqueta futura para ajustar β y refleja la forma simplificada del generador TA-01; no se propone como ley constitutiva ni como modelo validado para taludes de campo. La violación de sensibilidad se contabiliza cuando, al aumentar artificialmente la entrada de lluvia en 10 mm, la corrección disminuye más de 10⁻¹², o cuando, al aumentar el índice de seguridad en 0.1, la corrección aumenta más de 10⁻¹²; el conteo evalúa la salida del corrector y no constituye un criterio de desempeño predictivo.")

    # Abstract: include the new result without implying field performance.
    p = find_para(doc, "Este estudio evalúa un prototipo semisintético")
    set_text(p, p.text.replace("TA-01 combina geometría y materiales supuestos con precipitación diaria de NASA POWER (2020–2025) y genera 500 escenarios horarios.", "TA-01 combina geometría y materiales supuestos con precipitación diaria de NASA POWER (2020–2025) y genera 500 escenarios horarios; cada escenario representa una fecha con un perfil rectangular supuesto, y la eficiencia de drenaje reduce la presión de lluvia del generador.").replace("No se valida el método en campo; los resultados caracterizan el generador semisintético TA-01.", "Un comparador proporcional basado en la lluvia acumulada tuvo menor MAE puntual que LSTM e híbridos en las seis combinaciones de partición y horizonte; en cinco combinaciones, los intervalos por fecha separaron a los tres modelos de ese comparador. No se valida el método en campo; los resultados caracterizan el generador semisintético TA-01."))

    p = find_para(doc, "La precipitación diaria de reanálisis se obtuvo de NASA POWER/MERRA-2")
    set_text(p, p.text + " Cada escenario de TA-01 está asociado con una fecha de lluvia y 24 estados horarios; el total diario se distribuye en un perfil rectangular hipotético de 6, 12 o 24 h. En el generador, la eficiencia de drenaje reduce la componente de presión de lluvia mediante el factor (1 − eficiencia); no simula el drenaje transitorio ni convierte el perfil supuesto en observaciones horarias.")

    # Main-split result and limits of the new comparison.
    p = find_para(doc, "En la comparación principal sin OOF, con fechas de lluvia disjuntas")
    set_text(p, p.text + " Como contraste exploratorio, el comparador proporcional obtuvo un MAE de 1.53 × 10⁻⁶ mm a 1 h y 1.49 × 10⁻⁵ mm a 6 h, frente a 8.58 × 10⁻⁶ y 6.33 × 10⁻⁵ mm de la LSTM, 7.98 × 10⁻⁶ y 6.18 × 10⁻⁵ mm del híbrido sin OOF, y 4.86 × 10⁻⁶ y 6.27 × 10⁻⁵ mm del híbrido OOF, respectivamente. Los IC bootstrap por fecha de las tres diferencias modelo menos comparador fueron positivos a 1 h; a 6 h incluyeron cero. La comparación OOF corresponde a una ejecución por partición y horizonte. Estos intervalos no se ajustaron por multiplicidad y el resultado solo caracteriza TA-01. En la muestra principal completa, MAE[LSTM] − MAE[híbrido sin OOF] fue 0.597 × 10⁻⁶ mm a 1 h (IC del 95 %: 0.013 a 1.575 × 10⁻⁶) y 1.562 × 10⁻⁶ mm a 6 h (−0.673 a 5.359 × 10⁻⁶). El escenario TA01-S0001 (14-01-2021; 102.88 mm de lluvia diaria) aportó el 71.6 % del error absoluto de la LSTM y el 71.5 % del híbrido a 1 h, y el 75.4 % y 74.5 %, respectivamente, a 6 h. Al excluirlo únicamente como sensibilidad, los MAE fueron 2.472 × 10⁻⁶ frente a 2.308 × 10⁻⁶ mm a 1 h (LSTM e híbrido) y 15.79 × 10⁻⁶ frente a 15.94 × 10⁻⁶ mm a 6 h; la dirección relativa se invirtió a 6 h. La muestra completa permanece como análisis principal; no se excluyó ninguna fecha para elegir el modelo.")

    p = find_para(doc, "En la comparación principal, los 500 escenarios se distribuyen entre entrenamiento")
    set_text(p, p.text + " La fecha extrema asociada con TA01-S0001 se reservó en la prueba y no participó en el ajuste; su exclusión posterior se informa solo como análisis de influencia, no como selección de resultados.")

    # Replace Figure 3 caption and its interpretation; source-date bootstrap.
    p = find_para(doc, "Figura 3. Diferencia pareada del MAE entre LSTM")
    set_text(p, "Figura 3. Diferencia del MAE de cada modelo respecto del comparador proporcional de lluvia acumulada, expresada como porcentaje del MAE de la LSTM en la misma partición. Los puntos muestran las diferencias estimadas y las barras, los IC del 95 % agrupados por fecha; los valores positivos favorecen al comparador. La línea vertical marca diferencia cero. Los resultados OOF proceden de un ajuste por partición y horizonte; no se ajustó por comparaciones múltiples. Fuente: data/validation/ta01-proportional-rainfall-baseline.json.")

    # Chronological benchmark results and interval construction details.
    p = find_para(doc, "Los modelos de referencia también permiten contextualizar las pruebas cronológicas")
    set_text(p, p.text + " El comparador proporcional presentó un MAE puntual menor que LSTM, híbrido sin OOF e híbrido OOF en los cuatro pares año-horizonte. En conjunto con la partición principal, su error puntual fue menor en las seis combinaciones; en cinco, los 15 IC por fecha correspondientes a los tres modelos quedaron por encima de cero (Δ = MAE del modelo − MAE del comparador). Los tres IC de la partición principal a 6 h incluyeron cero. La lectura es exploratoria y no incorpora ajuste por multiplicidad; además, el comparador comparte con los modelos la lluvia futura idealizada del experimento.")

    # Make the selection and coverage descriptions unambiguous about the variant.
    p = find_para(doc, "La selección del modelo con el menor MAE de validación")
    set_text(p, p.text.replace("La selección del modelo con el menor MAE de validación", "La selección entre LSTM e híbrido sin OOF con el menor MAE de validación"))
    p = find_para(doc, "Figura 4. Cobertura empírica de los intervalos de predicción")
    set_text(p, "Figura 4. Cobertura empírica retrospectiva de los intervalos del modelo híbrido con corrección ajustada sin OOF en las pruebas cronológicas, frente a la cobertura global y la de fechas con lluvia ≥5 mm/día. Las etiquetas bajo las barras indican fechas y ventanas evaluadas; la línea discontinua marca el 95 % nominal. Los subconjuntos de lluvia intensa incluyen solo dos fechas en 2024 y tres en 2025; la cobertura no es una garantía de calibración. Fuente: data/validation/ta01-rolling-interval-coverage.csv.")
    p = find_para(doc, "Figura 5. Sensibilidad predictiva al ruido")
    set_text(p, "Figura 5. Cambio porcentual del MAE frente a entradas limpias al perturbar los pronósticos del híbrido principal sin OOF y de la LSTM. Los paneles separan horizontes y las categorías distinguen ruido gaussiano de datos faltantes MCAR, imputados con la media de entrenamiento. Los niveles del 1 % de ruido y 10 % de faltantes son exploratorios; la PI3 se centra en 5 % y 30 %. El ruido se escala por la desviación estándar de entrenamiento; las perturbaciones se aplican independientemente a los canales y pasos, mientras el desplazamiento usado en el límite de salida permanece limpio. No hubo reentrenamiento. Fuente: data/validation/ta01-robustness-summary.csv.")

    # Reproduction trail for the newly added analysis.
    p = find_para(doc, "La reproducción de todas las tablas requiere los prefijos de salida")
    set_text(p, p.text + " El comparador proporcional y sus intervalos agrupados por fecha se reproducen con scripts/evaluate-ta01-proportional-rainfall-baseline.py; los resultados y las predicciones por ventana están en data/validation/ta01-proportional-rainfall-baseline.json y data/validation/ta01-proportional-rainfall-baseline-predictions.csv. Las figuras revisadas se generan con scripts/generate-ta01-review-figures.py.")

    p = find_para(doc, "El repositorio público sslope")
    set_text(p, "El repositorio público sslope (https://github.com/RobertoA1/sslope) contiene el código base del prototipo. Los análisis suplementarios que respaldan este manuscrito se documentan en data/validation/ta01-seed-sensitivity-summary.json, data/validation/ta01-oof-comparison-bootstrap.json, data/validation/ta01-oof-all-splits-comparison.json, data/validation/ta01-oof-seed-sensitivity.json, data/validation/ta01-oof-seed-sensitivity-all-splits.json, data/validation/ta01-oof-target-regime-summary.json, data/validation/ta01-oof-corrector-ablation.json, data/validation/ta01-rounding-sensitivity.json, data/validation/ta01-solver-tolerance-sensitivity.json, data/validation/ta01-solver-tolerance-retrained-comparison.json, data/validation/ta01-proportional-rainfall-baseline.json y data/validation/ta01-primary-influence-and-target-regimes.json; los métodos y las predicciones por ventana incluyen scripts/evaluate-ta01-proportional-rainfall-baseline.py y data/validation/ta01-proportional-rainfall-baseline-predictions.csv. Estos archivos, sus manifiestos y predicciones se conservan en la copia de trabajo y aún no se han incorporado al remoto público; deberán depositarse allí antes de afirmar su disponibilidad pública. El manifiesto histórico data/validation/ta01-replication-manifest.json se conserva para sus archivos originales y no cubre por sí solo todos los análisis añadidos. El inventario complementario data/validation/ta01-review-supplement-manifest.json registra hashes SHA-256 y se verifica con scripts/verify-ta01-supplement-manifest.py. El código y la documentación utilizan la licencia MIT; las contribuciones originales en data/generated/, data/models/ y data/validation/ utilizan CC BY 4.0 (data/LICENSE.md). La precipitación de NASA POWER y las demás fuentes externas conservan la procedencia y las condiciones de uso documentadas por separado. El archivo README describe la ejecución y Docker Compose reproduce el servicio web local. Sin embargo, todavía no se fijan las versiones de Python para todos los experimentos ni se ha depositado una versión de publicación inmutable, lo que limita la reproducción exacta. TA-01 no procesó datos personales ni confidenciales.")

    p = find_para(doc, "En una sensibilidad exploratoria de la partición principal, la corrección se entrenó")
    set_text(p, p.text + " La prueba principal contiene 39 de 75 fechas con lluvia diaria igual a cero: las 702/702 ventanas a 1 h y las 507/507 a 6 h de esos días tienen objetivo almacenado cero. En días con lluvia positiva, también fueron cero 130/648 objetivos a 1 h y 45/468 a 6 h. El código define la presión de lluvia como función de la lluvia acumulada y resta el desplazamiento del equilibrio seco; por ello, para días secos el objetivo generado es cero, mientras que ceros en días húmedos también pueden resultar del redondeo. Cada corrector usa 19 entradas y 16 unidades ocultas: el MLP libre tiene 337 parámetros y el corrector monótono, 339. Con el mismo límite de desplazamiento no decreciente aplicado al final, el MAE medio del MLP libre fue 8.85 × 10⁻⁶ mm a 1 h y 6.80 × 10⁻⁵ mm a 6 h; la LSTM registró 8.58 × 10⁻⁶ y 6.33 × 10⁻⁵ mm, respectivamente. El MLP libre tuvo mayor MAE que la LSTM en las tres semillas por horizonte. Las violaciones reportadas se obtuvieron con las perturbaciones de entrada descritas en Métodos y no son conteos de ventanas con error alto.")

    p = find_para(doc, "El MEF supone deformación plana y elasticidad lineal")
    set_text(p, p.text + " En particular, el generador calcula por nodo la norma euclidiana de la diferencia entre el desplazamiento húmedo y el estado seco, redondea esa magnitud a 10⁻⁵ mm y después selecciona el máximo espacial; no redondea primero el máximo global ni calcula la norma a partir de componentes previamente redondeadas.")

    p = find_para(doc, "El MEF supone deformación plana y elasticidad lineal")
    set_text(p, p.text + " En particular, el generador calcula por nodo la norma euclidiana de la diferencia entre el desplazamiento húmedo y el estado seco, redondea esa magnitud a 10⁻⁵ mm y después selecciona el máximo espacial; no redondea primero el máximo global ni calcula la norma a partir de componentes previamente redondeadas.")

    # Discussion and conclusions: no claim that the neural methods add value
    # beyond the generator-matched baseline.
    p = find_para(doc, "El uso de conocimiento físico en predictores de estabilidad de taludes")
    set_text(p, p.text + " El control proporcional, construido a partir del historial almacenado y la lluvia acumulada, presentó menor MAE puntual que LSTM e híbridos en todas las particiones y horizontes. Los intervalos por fecha fueron inconclusos únicamente para la partición principal a 6 h; no se corrigieron por multiplicidad y las comparaciones OOF usan una sola ejecución. Este resultado reduce la evidencia de valor predictivo incremental de las redes en TA-01 y motiva comparar cualquier nueva arquitectura contra este control antes de atribuir mejoras a la corrección residual.")

    p = find_para(doc, "En relación con la PI1, la comparación principal no demuestra")
    set_text(p, p.text + " El comparador proporcional alcanzó menor MAE puntual que los tres modelos temporales en las seis combinaciones examinadas; sus contrastes por fecha fueron inconclusos a 6 h en la partición principal. La evidencia disponible, por tanto, no establece un aporte predictivo incremental de LSTM o del corrector sobre la relación proporcional inducida por TA-01.")

    # The repeated Spanish abstract at the end is a duplicate of the front matter.
    duplicate_start = next((i for i, q in enumerate(doc.paragraphs) if q.text.strip() == "RESUMEN"), None)
    if duplicate_start is not None:
        duplicate_end = next((i for i in range(duplicate_start + 1, len(doc.paragraphs)) if doc.paragraphs[i].text.strip().startswith("Contribuciones de los autores")), None)
        if duplicate_end is None:
            raise RuntimeError("Could not find end of repeated abstract block")
        for q in list(doc.paragraphs[duplicate_start:duplicate_end]):
            q._element.getparent().remove(q._element)

    doc.save(OUTPUT)

    # Keep existing relationship IDs and geometry; only replace the raster bytes.
    replacement = {"word/media/image3.jpg": FIGURES / "figura3.jpg", "word/media/image4.jpg": FIGURES / "figura4.jpg", "word/media/image5.jpg": FIGURES / "figura5.jpg"}
    temp = OUTPUT.with_suffix(".tmp.docx")
    with ZipFile(OUTPUT, "r") as src, ZipFile(temp, "w", ZIP_DEFLATED) as dst:
        for item in src.infolist():
            content = replacement[item.filename].read_bytes() if item.filename in replacement else src.read(item.filename)
            dst.writestr(item, content)
    temp.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
