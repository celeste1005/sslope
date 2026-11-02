#!/usr/bin/env python3
"""Render refreshed manuscript figures from archived TA-01 results."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "manuscript-review-2026-10-29"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"lstm": "#1f77b4", "in_sample_hybrid": "#d55e00", "oof_hybrid": "#009e73"}
LABELS = {"lstm": "LSTM", "in_sample_hybrid": "Híbrido sin OOF", "oof_hybrid": "Híbrido OOF"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.0f}" y="44" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="34" font-weight="bold" fill="#111">{esc(title)}</text>',
    ]


def save_svg(name: str, pieces: list[str]) -> None:
    (OUT / name).write_text("\n".join(pieces + ["</svg>"]), encoding="utf-8")


def figure1() -> None:
    width, height = 2048, 922
    s = svg_header(width, height, "")
    s = s[:2]
    boxes = [
        (55, 175, "Lluvia externa", "NASA POWER diaria"),
        (755, 175, "Escenarios supuestos", "Geometría y propiedades\nmediante LHS"),
        (1455, 175, "MEF 2D", "24 estados cuasiestáticos"),
        (1455, 525, "Ventanas temporales", "Historial de 6 h\nHorizontes de 1 h y 6 h"),
        (755, 525, "LSTM y corrección", "Corrección residual\nmonótona"),
        (55, 525, "Evaluación", "Fechas disjuntas\n2024 / 2025"),
    ]
    for x,y,title,body in boxes:
        s.append(f'<rect x="{x}" y="{y}" width="535" height="190" fill="white" stroke="#555" stroke-width="4"/>')
        s.append(f'<text x="{x+267.5}" y="{y+72}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="36" font-weight="bold" fill="#111">{esc(title)}</text>')
        for j,line in enumerate(body.split("\n")):
            s.append(f'<text x="{x+267.5}" y="{y+125+j*42}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="30" fill="#111">{esc(line)}</text>')
    arrows = [(590,270,755,270),(1290,270,1455,270),(1722,365,1722,525),(1455,620,1290,620),(755,620,590,620)]
    for x1,y1,x2,y2 in arrows:
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" stroke-width="7" marker-end="url(#arrow)"/>')
    s.insert(2, '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#555"/></marker></defs>')
    s.append(f'<text x="1024" y="850" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="30" fill="#333">Flujo semisintético TA-01; no representa monitoreo de campo.</text>')
    save_svg("figura1.svg", s)


def figure2() -> None:
    rows=list(csv.DictReader((ROOT / "data/validation/ta01-fem-mesh-summary.csv").open(encoding="utf-8-sig",newline="")))
    width,height=2048,922
    s=svg_header(width,height,"")[:2]
    panels=[(130,970,"maximum_rainfall_displacement_mm","a) Magnitud calculada","Desplazamiento máximo inducido por lluvia (mm)",0.0,0.06,0.01),(1090,1930,"displacement_difference_against_48x32_percent","b) Referencia interna","Diferencia frente a malla 48 × 32 (%)",0.0,90.0,15.0)]
    for left,right,key,title,ylabel,ymin,ymax,step in panels:
        top,bottom=170,700
        vals=[float(r[key]) for r in rows]; xs=[int(r["node_count"]) for r in rows]
        xlo,xhi=0,950
        def xp(v): return left+(v-xlo)/(xhi-xlo)*(right-left)
        def yp(v): return bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
        for v in [ymin+i*step for i in range(int((ymax-ymin)/step)+1)]:
            y=yp(v);s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e6e6e6"/>');s.append(f'<text x="{left-18}" y="{y+7:.1f}" text-anchor="end" font-family="Times New Roman,Times,serif" font-size="20" fill="#333">{v:g}</text>')
        for v in range(0,901,200):
            x=xp(v);s.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#f0f0f0"/>');s.append(f'<text x="{x:.1f}" y="{bottom+30}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="20" fill="#333">{v}</text>')
        s.append(f'<text x="{(left+right)/2}" y="110" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="30" font-weight="bold" fill="#111">{esc(title)}</text>')
        s.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2"/>')
        s.append(f'<polyline fill="none" stroke="#2878b5" stroke-width="5" points="{" ".join(f"{xp(x):.1f},{yp(v):.1f}" for x,v in zip(xs,vals))}"/>')
        for x,v in zip(xs,vals):s.append(f'<circle cx="{xp(x):.1f}" cy="{yp(v):.1f}" r="11" fill="#2878b5"/>')
        s.append(f'<text x="{(left+right)/2}" y="{bottom+67}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="22" fill="#222">Nodos de malla</text>')
        s.append(f'<text x="{left-85}" y="{(top+bottom)/2}" transform="rotate(-90 {left-85} {(top+bottom)/2})" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="22" fill="#222">{esc(ylabel)}</text>')
    save_svg("figura2.svg",s)


def figure6() -> None:
    rows=list(csv.DictReader((ROOT / "data/validation/ta01-ablation-summary.csv").open(encoding="utf-8-sig",newline="")))
    variants=["LSTM","HYBRID_NO_HYDROLOGY","HYBRID_NO_FEM","HYBRID_FULL"]
    labels=["LSTM","Máscara de\nhidrología","Máscara\nmecánica","Híbrido\ncompleto"]
    width,height=2048,922;s=svg_header(width,height,"")[:2]
    colors=["#2878b5","#888888","#bbbbbb","#d55e00"]
    for panel,(h,left,right) in enumerate(((1,150,980),(6,1120,1950))):
        top,bottom=165,710;scale=1e-5 if h==1 else 1e-4;maxv=2.8 if h==1 else 1.7
        s.append(f'<text x="{(left+right)/2}" y="112" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="30" font-weight="bold" fill="#111">Horizonte {h} h · sin reentrenamiento</text>')
        for k in range(6):
            v=maxv*k/5;y=bottom-v/maxv*(bottom-top);s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e6e6e6"/>');s.append(f'<text x="{left-16}" y="{y+7:.1f}" text-anchor="end" font-family="Times New Roman,Times,serif" font-size="20" fill="#333">{v:.1f}</text>')
        data=[]
        for variant in variants:data.append(float(next(r["mae_mm"] for r in rows if int(r["horizon_hours"])==h and r["variant_id"]==variant))/scale)
        gap=(right-left)/4;bw=90
        for i,(label,v,color) in enumerate(zip(labels,data,colors)):
            x=left+(i+0.5)*gap-bw/2;y=bottom-v/maxv*(bottom-top);s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{bottom-y:.1f}" fill="{color}" stroke="#555" stroke-width="2"/>')
            for j,line in enumerate(label.split("\n")):s.append(f'<text x="{x+bw/2}" y="{bottom+35+j*23}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="17" fill="#333">{esc(line)}</text>')
        s.append(f'<text x="{left-80}" y="{(top+bottom)/2}" transform="rotate(-90 {left-80} {(top+bottom)/2})" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="22" fill="#222">MAE (× {"10⁻⁵" if h==1 else "10⁻⁴"} mm)</text>')
    save_svg("figura6.svg",s)


def figure3() -> None:
    payload = json.loads((ROOT / "data/validation/ta01-proportional-rainfall-baseline.json").read_text(encoding="utf-8"))
    result_map = {(r["split"], r["horizonHours"]): r for r in payload["results"]}
    split_info = [("primary", "Partición principal"), ("backtest-2024", "Prueba cronológica 2024"), ("chronological-2025", "Prueba cronológica 2025")]
    width, height = 2048, 922
    s = svg_header(width, height, "Error relativo al comparador proporcional de lluvia acumulada")
    legend_y = 94
    for j, key in enumerate(("lstm", "in_sample_hybrid", "oof_hybrid")):
        x = 540 + j * 470
        s.append(f'<circle cx="{x}" cy="{legend_y}" r="10" fill="{COLORS[key]}"/>')
        s.append(f'<text x="{x+20}" y="{legend_y+8}" font-family="Times New Roman,Times,serif" font-size="24" fill="#222">{esc(LABELS[key])}</text>')

    methods = [("lstm", "lstm"), ("in_sample_hybrid", "in_sample_hybrid"), ("oof_hybrid", "oof_hybrid")]
    plot_l, plot_r, plot_t, plot_b = 600, 1995, 150, 812
    low, high = -20.0, 250.0
    def xp(v: float) -> float:
        return plot_l + (v - low) / (high - low) * (plot_r - plot_l)
    split_label_x, method_label_x = 30, 515
    row_height = (plot_b - plot_t) / 18
    ordered = [(split, label, horizon) for split, label in split_info for horizon in (1, 6)]
    for tick in range(-20, 251, 30):
        tx = xp(tick)
        s.append(f'<line x1="{tx:.1f}" y1="{plot_t}" x2="{tx:.1f}" y2="{plot_b}" stroke="#e5e5e5" stroke-width="1"/>')
        s.append(f'<text x="{tx:.1f}" y="{plot_b+31}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="20" fill="#333">{tick}</text>')
    zero_x = xp(0.0)
    s.append(f'<line x1="{zero_x:.1f}" y1="{plot_t}" x2="{zero_x:.1f}" y2="{plot_b}" stroke="#555" stroke-width="3" stroke-dasharray="9 8"/>')
    s.append(f'<line x1="{plot_l}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="#222" stroke-width="2"/>')
    for group_index, (split, split_label, horizon) in enumerate(ordered):
        d = result_map[(split, horizon)]
        lstm_mae = d["models"]["lstm"]["maeMm"]
        group_top = plot_t + group_index * 3 * row_height
        if group_index > 0:
            s.append(f'<line x1="20" y1="{group_top:.1f}" x2="2010" y2="{group_top:.1f}" stroke="#c8c8c8" stroke-width="1.5"/>')
        s.append(f'<text x="{split_label_x}" y="{group_top + 1.5*row_height + 8:.1f}" font-family="Times New Roman,Times,serif" font-size="20" font-weight="bold" fill="#222">{esc(split_label)} · {horizon} h</text>')
        for method_index, (key, model_key) in enumerate(methods):
            yy = group_top + (method_index + 0.5) * row_height
            obj = d["models"][model_key]["pairedBootstrapVsProportionalMm"]
            delta = 100 * obj["observedDifferenceMm"] / lstm_mae
            ci0, ci1 = [100 * v / lstm_mae for v in obj["confidenceInterval95Mm"]]
            s.append(f'<text x="{method_label_x}" y="{yy+7:.1f}" text-anchor="end" font-family="Times New Roman,Times,serif" font-size="19" fill="#222">{esc(LABELS[key])}</text>')
            s.append(f'<line x1="{xp(ci0):.1f}" y1="{yy:.1f}" x2="{xp(ci1):.1f}" y2="{yy:.1f}" stroke="{COLORS[key]}" stroke-width="6" stroke-linecap="round"/>')
            s.append(f'<line x1="{xp(ci0):.1f}" y1="{yy-9:.1f}" x2="{xp(ci0):.1f}" y2="{yy+9:.1f}" stroke="{COLORS[key]}" stroke-width="2.5"/>')
            s.append(f'<line x1="{xp(ci1):.1f}" y1="{yy-9:.1f}" x2="{xp(ci1):.1f}" y2="{yy+9:.1f}" stroke="{COLORS[key]}" stroke-width="2.5"/>')
            s.append(f'<circle cx="{xp(delta):.1f}" cy="{yy:.1f}" r="9" fill="{COLORS[key]}"/>')
    s.append(f'<text x="{(plot_l+plot_r)/2:.1f}" y="{height-23}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="22" fill="#222">Diferencia del MAE frente al comparador (% del MAE de la LSTM); valores positivos favorecen al comparador</text>')
    save_svg("figura3.svg", s)


def figure4() -> None:
    rows = list(csv.DictReader((ROOT / "data/validation/ta01-rolling-interval-coverage.csv").open(encoding="utf-8-sig", newline="")))
    data = [r for r in rows if r["segment"] in ("ALL", "RAIN_HIGH_GE_5")]
    width, height = 2048, 922
    s = svg_header(width, height, "Cobertura empírica de los intervalos en pruebas cronológicas")
    s.append('<rect x="680" y="77" width="20" height="20" fill="#2878b5"/><text x="710" y="96" font-family="Times New Roman,Times,serif" font-size="24" fill="#222">Todas las fechas</text>')
    s.append('<rect x="990" y="77" width="20" height="20" fill="#e07a22"/><text x="1020" y="96" font-family="Times New Roman,Times,serif" font-size="24" fill="#222">Lluvia ≥5 mm/día</text>')
    left, right, top, bottom = 190, 1990, 155, 755
    def yp(v: float) -> float: return bottom - v/110*(bottom-top)
    for v in range(0, 101, 20):
        y=yp(v)
        s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e4e4e4"/>')
        s.append(f'<text x="{left-15}" y="{y+7:.1f}" text-anchor="end" font-family="Times New Roman,Times,serif" font-size="21" fill="#333">{v}</text>')
    s.append(f'<line x1="{left}" y1="{yp(95):.1f}" x2="{right}" y2="{yp(95):.1f}" stroke="#555" stroke-width="3" stroke-dasharray="9 7"/>')
    s.append(f'<text x="{right-4}" y="{yp(95)-12:.1f}" text-anchor="end" font-family="Times New Roman,Times,serif" font-size="19" fill="#555">Nivel nominal 95 %</text>')
    s.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" stroke-width="2"/>')
    s.append(f'<text x="47" y="{(top+bottom)/2:.1f}" transform="rotate(-90 47 {(top+bottom)/2:.1f})" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="27" fill="#222">Cobertura empírica (%)</text>')
    groups=[("2024",1),("2024",6),("2025",1),("2025",6)]
    centers=[420,880,1330,1770]
    by={(r["test_year"],int(r["horizon_hours"]),r["segment"]):r for r in data}
    for (year,h),cx in zip(groups,centers):
        for offset,seg,color in [(-84,"ALL","#2878b5"),(84,"RAIN_HIGH_GE_5","#e07a22")]:
            r=by[(year,h,seg)]
            v=float(r["empirical_coverage"])*100
            n_dates=int(r["source_date_count"]); n=int(r["sample_count"])
            x=cx+offset; y=yp(v); bar_w=74
            s.append(f'<rect x="{x-bar_w/2}" y="{y:.1f}" width="{bar_w}" height="{bottom-y:.1f}" rx="3" fill="{color}"/>')
            label=f'{v:.1f}%'
            s.append(f'<text x="{x}" y="{max(y-13,top+25):.1f}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="21" font-weight="bold" fill="#222">{label}</text>')
            s.append(f'<text x="{x}" y="{bottom+29}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="17" fill="#444">{n_dates} f. / {n} v.</text>')
        s.append(f'<text x="{cx}" y="{bottom+72}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="24" font-weight="bold" fill="#222">{year} · {h} h</text>')
    s.append(f'<text x="{(left+right)/2}" y="{height-22}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="20" fill="#444">f. = fechas; v. = ventanas de pronóstico. La lluvia intensa comprende pocas fechas.</text>')
    save_svg("figura4.svg", s)


def figure5() -> None:
    payload=json.loads((ROOT / "data/validation/ta01-robustness-repeated.json").read_text(encoding="utf-8"))
    conditions=[("NOISE_1PCT_STD","Ruido 1 %"),("NOISE_5PCT_STD","Ruido 5 %"),("MISSING_10PCT_MCAR","MCAR 10 %"),("MISSING_30PCT_MCAR","MCAR 30 %")]
    models=[("lstm","LSTM","#2878b5"),("in_sample_hybrid","Híbrido sin OOF","#d55e00"),("oof_hybrid","Híbrido OOF","#009e73"),("free_mlp_oof","MLP libre OOF","#7b61a8")]
    width,height=2048,922
    s=svg_header(width,height,"Aumento del MAE en 30 perturbaciones pareadas")
    legend_x=[430,690,1050,1320]
    for (key,label,color),x in zip(models,legend_x):
        s.append(f'<circle cx="{x}" cy="84" r="10" fill="{color}"/><text x="{x+19}" y="92" font-family="Times New Roman,Times,serif" font-size="21" fill="#222">{esc(label)}</text>')
    panels=[(1,35),(6,1050)]
    for horizon,x0 in panels:
        plot_l=x0+100; plot_r=x0+960; plot_t=170; plot_b=755
        s.append(f'<text x="{(plot_l+plot_r)/2}" y="139" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="30" font-weight="bold" fill="#111">Horizonte {horizon} h</text>')
        maxv=850 if horizon==1 else 400
        step=100 if horizon==1 else 50
        for v in range(0,maxv+1,step):
            yy=plot_b-v/maxv*(plot_b-plot_t)
            s.append(f'<line x1="{plot_l}" y1="{yy:.1f}" x2="{plot_r}" y2="{yy:.1f}" stroke="#e6e6e6"/>')
            s.append(f'<text x="{plot_l-16}" y="{yy+7:.1f}" text-anchor="end" font-family="Times New Roman,Times,serif" font-size="18" fill="#333">{v}</text>')
        s.append(f'<line x1="{plot_l}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" stroke="#222" stroke-width="2"/>')
        s.append(f'<text x="{x0+23}" y="{(plot_t+plot_b)/2}" transform="rotate(-90 {x0+23} {(plot_t+plot_b)/2})" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="22" fill="#222">Aumento medio del MAE (%)</text>')
        result=next(r for r in payload["results"] if int(r["horizonHours"])==horizon)
        by_condition={row["condition"]:row for row in result["conditions"]}
        offsets=[-30,-10,10,30]
        for i,(condition,label) in enumerate(conditions):
            cx=plot_l+(i+0.5)*(plot_r-plot_l)/len(conditions)
            row=by_condition[condition]
            for off,(key,_model_label,color) in zip(offsets,models):
                obj=row["summary"][key]
                val=float(obj["meanIncreasePercentVsSameModelClean"])
                lo,hi=[float(v) for v in obj["bootstrapMeanIncrease95Percent"]]
                def yp(v): return plot_b-v/maxv*(plot_b-plot_t)
                x=cx+off; y=yp(val); y0=yp(lo); y1=yp(hi)
                s.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="4"/>')
                s.append(f'<line x1="{x-5:.1f}" y1="{y0:.1f}" x2="{x+5:.1f}" y2="{y0:.1f}" stroke="{color}" stroke-width="2"/>')
                s.append(f'<line x1="{x-5:.1f}" y1="{y1:.1f}" x2="{x+5:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="2"/>')
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>')
            s.append(f'<text x="{cx:.1f}" y="{plot_b+31}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="18" fill="#333">{esc(label)}</text>')
        s.append(f'<text x="{(plot_l+plot_r)/2}" y="{plot_b+75}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="18" fill="#444">Perturbación de entrada</text>')
    s.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Times New Roman,Times,serif" font-size="17" fill="#444">Puntos: media de 30 sorteos; barras: IC bootstrap del 95 % sobre sorteos, condicionados a fechas y ajustes fijos.</text>')
    save_svg("figura5.svg",s)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print(f"SVGs written to {OUT}")
