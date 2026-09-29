"""
dashboard.py | A static, single-page research dashboard (no server needed).

Builds docs/index.html + docs/assets/plotly.min.js. Open the HTML file directly,
or serve it free with GitHub Pages (Settings -> Pages -> Branch: main, /docs).

    d = Dashboard("Regime & Risk Lab", "EIA daily benchmarks · as of 2026-09-22")
    t = d.tab("regimes", "Regimes")
    t.kpis([("Current regime", "Turbulent", "WTI, P = 0.83", "warning")])
    t.chart(fig, "Smoothed regime probability", "2-state Gaussian Markov-switching")
    t.chart_set("Price & regimes", {"WTI": fig1, "Brent": fig2})    # segmented control
    t.table(df, "Backtest verdicts")
    d.write("docs")
    d.screenshot("docs", ["regimes", "risk"])      # docs/screenshots/<tab>.png

Tabs are addressed by URL hash (index.html#regimes), which is also how the
screenshots are taken. Charts are Plotly, themed to match the PDF figures.
"""
from __future__ import annotations

import html as _html
import json
import re
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from . import browser
from .report import NUM_RE
from .style import SERIES

LIGHT = dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
             grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)")
DARK = dict(surface="#1a1a19", page="#0d0d0d", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
            grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)")


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True, unified: bool = True) -> go.Figure:
    """Apply the house chart style: hairline grid, recessive axes, system sans, hover layer."""
    fig.update_layout(
        height=height, margin=dict(l=52, r=16, t=12, b=40),
        paper_bgcolor=LIGHT["surface"], plot_bgcolor=LIGHT["surface"],
        font=dict(family='system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif', size=12, color=LIGHT["ink2"]),
        hovermode="x unified" if unified else "closest",
        hoverlabel=dict(font_size=12, bgcolor="#ffffff", bordercolor=LIGHT["axis"], font_color=LIGHT["ink"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)"),
        colorway=SERIES,
    )
    fig.update_xaxes(showgrid=False, linecolor=LIGHT["axis"], ticks="outside", tickcolor=LIGHT["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=LIGHT["grid"], gridwidth=1, linecolor=LIGHT["axis"], zeroline=False)
    return fig


class Tab:
    def __init__(self, key: str, label: str, intro: str = ""):
        self.key, self.label, self.intro = key, label, intro
        self.blocks: list[tuple[str, object]] = []

    def kpis(self, items):
        """items: (label, value, note[, status]) with status in good|warning|serious|critical."""
        self.blocks.append(("kpis", list(items)))

    def chart(self, fig, title, subtitle="", half=False, note=""):
        self.blocks.append(("chart", (fig, title, subtitle, half, note)))

    def chart_set(self, title, figs: dict, subtitle="", half=False, note=""):
        """Several versions of one chart (e.g. per asset) behind a one-row segmented control."""
        self.blocks.append(("chartset", (figs, title, subtitle, half, note)))

    def table(self, df, title, subtitle="", half=False, note=""):
        self.blocks.append(("table", (df, title, subtitle, half, note)))

    def text(self, md_html: str, half=False):
        self.blocks.append(("text", (md_html, half)))

    def image(self, src: str, title: str, subtitle="", half=False, note=""):
        self.blocks.append(("image", (src, title, subtitle, half, note)))


class Dashboard:
    def __init__(self, title, subtitle, links=None, footer=""):
        self.title, self.subtitle = title, subtitle
        self.links = links or []
        self.footer = footer
        self.tabs: list[Tab] = []
        self.specs: list = []

    def tab(self, key, label, intro="") -> Tab:
        t = Tab(key, label, intro)
        self.tabs.append(t)
        return t

    # ---------------------------------------------------------------- render
    def _spec(self, fig) -> int:
        self.specs.append(json.loads(pio.to_json(fig, validate=False, pretty=False, engine="json")))
        return len(self.specs) - 1

    @staticmethod
    def _cell(v) -> str:
        raw = str(v).strip()
        s = _html.escape(str(v))
        cls = {"✔": "ok", "✖": "bad", "▲": "warn"}.get(s[:1], "")
        if not cls and NUM_RE.fullmatch(raw):
            cls = "num"
        return f'<td class="{cls}">{s}</td>' if cls else f"<td>{s}</td>"

    def _card(self, title, subtitle, inner, half, note, controls=""):
        h = f'<div class="card{" half" if half else ""}"><div class="card-head"><div><h3>{_html.escape(title)}</h3>'
        if subtitle:
            h += f'<p class="cs">{_html.escape(subtitle)}</p>'
        h += f"</div>{controls}</div>{inner}"
        if note:
            h += f'<p class="note">{note}</p>'
        return h + "</div>"

    def _render_tab(self, t: Tab) -> str:
        out = [f'<section class="tab" id="tab-{t.key}" data-key="{t.key}">']
        if t.intro:
            out.append(f'<p class="intro">{t.intro}</p>')
        grid = []
        for kind, x in t.blocks:
            if kind == "kpis":
                tiles = []
                for it in x:
                    lab, val, note = it[:3]
                    st = it[3] if len(it) > 3 else ""
                    icon = {"good": "✔", "warning": "▲", "serious": "▲", "critical": "✖"}.get(st, "")
                    badge = f'<span class="st {st}">{icon}</span>' if st else ""
                    tiles.append(f'<div class="kpi"><div class="kl">{_html.escape(lab)}</div>'
                                 f'<div class="kv">{badge}{_html.escape(val)}</div><div class="kn">{_html.escape(note)}</div></div>')
                if grid:
                    out.append('<div class="grid">' + "".join(grid) + "</div>")
                    grid = []
                out.append('<div class="kpis">' + "".join(tiles) + "</div>")
            elif kind == "chart":
                fig, title, sub, half, note = x
                i = self._spec(fig)
                grid.append(self._card(title, sub, f'<div class="plot" data-spec="{i}"></div>', half, note))
            elif kind == "chartset":
                figs, title, sub, half, note = x
                ids = [self._spec(f) for f in figs.values()]
                btns = "".join(f'<button data-spec="{i}" class="{"on" if j == 0 else ""}">{_html.escape(k)}</button>'
                               for j, (k, i) in enumerate(zip(figs.keys(), ids)))
                grid.append(self._card(title, sub, f'<div class="plot" data-spec="{ids[0]}"></div>', half, note,
                                       controls=f'<div class="seg" role="group">{btns}</div>'))
            elif kind == "table":
                df, title, sub, half, note = x
                from .report import _numeric_col
                head = "".join(f'<th class="{"num" if _numeric_col(df[c]) else ""}">{_html.escape(str(c))}</th>'
                               for c in df.columns)
                rows = "".join("<tr>" + "".join(self._cell(v) for v in r) + "</tr>"
                               for r in df.astype(object).itertuples(index=False))
                grid.append(self._card(title, sub, f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>', half, note))
            elif kind == "text":
                txt, half = x
                grid.append(f'<div class="card text{" half" if half else ""}">{txt}</div>')
            elif kind == "image":
                src, title, sub, half, note = x
                grid.append(self._card(title, sub, f'<img class="img" src="{src}" alt="{_html.escape(title)}">', half, note))
        if grid:
            out.append('<div class="grid">' + "".join(grid) + "</div>")
        out.append("</section>")
        return "".join(out)

    def html(self) -> str:
        nav = "".join(f'<a href="#{t.key}" data-key="{t.key}">{_html.escape(t.label)}</a>' for t in self.tabs)
        links = "".join(f'<a class="lnk" href="{h}">{_html.escape(l)}</a>' for l, h in self.links)
        self.specs = []
        tabs = "".join(self._render_tab(t) for t in self.tabs)
        js = JS_DASH.replace("__L__", json.dumps(LIGHT)).replace("__D__", json.dumps(DARK))
        page = TEMPLATE
        for k, v in dict(title=_html.escape(self.title), subtitle=_html.escape(self.subtitle), nav=nav,
                         links=links, footer=self.footer, built=f"{date.today():%Y-%m-%d}", css=CSS_DASH).items():
            page = page.replace("{" + k + "}", v)
        # tabs, data and script last, so their contents are never scanned for placeholders
        page = page.replace("{tabs}", tabs, 1)
        page = page.replace("{specs}", json.dumps(self.specs, separators=(",", ":")), 1)
        return page.replace("{js}", js, 1)

    def write(self, out_dir="docs") -> Path:
        out = Path(out_dir)
        (out / "assets").mkdir(parents=True, exist_ok=True)
        js = out / "assets" / "plotly.min.js"
        if not js.exists():
            from plotly.offline import get_plotlyjs
            js.write_text(get_plotlyjs(), encoding="utf-8")
        p = out / "index.html"
        p.write_text(self.html(), encoding="utf-8")
        return p

    def screenshot(self, out_dir="docs", keys=None, width=1440, height=900) -> list[str]:
        out = Path(out_dir)
        keys = keys or [t.key for t in self.tabs]
        shots = [(k, str(out / "screenshots" / f"{k}.png")) for k in keys]
        return browser.screenshots(out / "index.html", shots, width=width, height=height)


CSS_DASH = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--border:rgba(11,11,11,.10);--accent:#2a78d6;--ok:#006300;--bad:#b52e2e;--warn:#8a5a00;--chip:#eef3fa}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--accent:#3987e5;--ok:#0ca30c;--bad:#e66767;--warn:#fab219;--chip:#1f2a38}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--accent:#3987e5;--ok:#0ca30c;--bad:#e66767;--warn:#fab219;--chip:#1f2a38}
*{box-sizing:border-box}html,body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
header{padding:22px 28px 0;max-width:1440px;margin:0 auto}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-.2px}.subtitle{color:var(--ink2);margin:0;font-size:14px}
.links{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.lnk,.theme{font-size:13px;color:var(--ink);text-decoration:none;border:1px solid var(--border);background:var(--surface);padding:6px 11px;border-radius:7px;cursor:pointer}
.lnk:hover,.theme:hover{border-color:var(--accent)}
nav{display:flex;gap:4px;margin-top:18px;border-bottom:1px solid var(--grid);overflow-x:auto}
nav a{padding:10px 14px;color:var(--ink2);text-decoration:none;font-size:14px;border-bottom:2px solid transparent;white-space:nowrap}
nav a.on{color:var(--ink);border-bottom-color:var(--accent);font-weight:600}
main{max-width:1440px;margin:0 auto;padding:18px 28px 28px}
.tab{display:none}.tab.on{display:block}
.intro{color:var(--ink2);font-size:14px;max-width:980px;margin:0 0 14px;line-height:1.5}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin-bottom:14px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.kl{font-size:12px;color:var(--ink2);text-transform:uppercase;letter-spacing:.3px}
.kv{font-size:26px;font-weight:600;margin:4px 0 2px;display:flex;align-items:center;gap:8px}
.kn{font-size:12px;color:var(--muted)}
.st{font-size:12px;width:22px;height:22px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;color:#fff}
.st.good{background:#0ca30c}.st.warning{background:#fab219;color:#0b0b0b}.st.serious{background:#ec835a}.st.critical{background:#d03b3b}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:14px}
.card{grid-column:span 2;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px 10px;min-width:0}
.card.half{grid-column:span 1}
.card-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:6px;flex-wrap:wrap}
.card h3{font-size:15px;margin:0}.cs{font-size:12.5px;color:var(--ink2);margin:3px 0 0}
.note{font-size:12px;color:var(--muted);margin:6px 0 2px;line-height:1.45}
.seg{display:flex;gap:2px;background:var(--page);border:1px solid var(--border);border-radius:8px;padding:2px;flex-wrap:wrap}
.seg button{font:inherit;font-size:12.5px;border:0;background:transparent;color:var(--ink2);padding:5px 10px;border-radius:6px;cursor:pointer}
.seg button.on{background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--border);font-weight:600}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--axis);padding:7px 8px;white-space:nowrap;background:var(--chip)}
td{border-bottom:1px solid var(--grid);padding:6px 8px}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.ok{color:var(--ok);font-weight:600;white-space:nowrap}td.bad{color:var(--bad);font-weight:600;white-space:nowrap}td.warn{color:var(--warn);font-weight:600;white-space:nowrap}
.card.text{font-size:14px;line-height:1.55;color:var(--ink2)}.card.text b{color:var(--ink)}
.img{width:100%;border-radius:6px;display:block}
footer{max-width:1440px;margin:0 auto;padding:0 28px 30px;color:var(--muted);font-size:12px;line-height:1.5}
@media (max-width:900px){.grid{grid-template-columns:1fr}.card,.card.half{grid-column:span 1}header,main,footer{padding-left:16px;padding-right:16px}}
"""

JS_DASH = """
const L=__L__, D=__D__;
const drawn=new Set();
function theme(){const t=document.documentElement.dataset.theme;
  if(t) return t; return matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}
function chrome(){const c=theme()==='dark'?D:L; const u={paper_bgcolor:c.surface,plot_bgcolor:c.surface,'font.color':c.ink2,
  'hoverlabel.bgcolor':theme()==='dark'?'#262625':'#ffffff','hoverlabel.font.color':c.ink,'hoverlabel.bordercolor':c.axis};
  return [u,c];}
function themed(el){const [u,c]=chrome(); const lay=el.layout||{}; Object.keys(lay).forEach(k=>{
  if(/^xaxis\\d*$/.test(k)){u[k+'.linecolor']=c.axis;u[k+'.tickcolor']=c.axis;u[k+'.gridcolor']=c.grid;}
  if(/^yaxis\\d*$/.test(k)){u[k+'.linecolor']=c.axis;u[k+'.gridcolor']=c.grid;}}); return u;}
function draw(el){const s=SPECS[+el.dataset.spec]; const cfg={displaylogo:false,responsive:true,
  modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d','toggleSpikelines']};
  Plotly.react(el,s.data,s.layout,cfg).then(()=>Plotly.relayout(el,themed(el)));}
function show(key){const tabs=[...document.querySelectorAll('.tab')];
  if(!tabs.some(t=>t.dataset.key===key)) key=tabs[0].dataset.key;
  tabs.forEach(t=>t.classList.toggle('on',t.dataset.key===key));
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('on',a.dataset.key===key));
  document.querySelectorAll('#tab-'+key+' .plot').forEach(el=>{ if(!drawn.has(el)){drawn.add(el);draw(el);} else Plotly.Plots.resize(el);});}
window.addEventListener('hashchange',()=>show(location.hash.slice(1)));
document.addEventListener('click',e=>{const b=e.target.closest('.seg button'); if(!b) return;
  const card=b.closest('.card'); card.querySelectorAll('.seg button').forEach(x=>x.classList.toggle('on',x===b));
  const el=card.querySelector('.plot'); el.dataset.spec=b.dataset.spec; draw(el);});
document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=theme()==='dark'?'light':'dark';
  drawn.forEach(el=>Plotly.relayout(el,themed(el)));});
show(location.hash.slice(1));
"""

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style><script src="assets/plotly.min.js"></script></head><body>
<div id="shot"><header><div class="top"><div><h1>{title}</h1><p class="subtitle">{subtitle}</p></div>
<div class="links">{links}<button class="theme" id="theme" title="Toggle light/dark">Light / dark</button></div></div><nav>{nav}</nav></header>
<main>{tabs}</main></div>
<footer>{footer}<br>Built {built}. Static page: every number comes from the CSVs in this repository's outputs/ folders.</footer>
<script>const SPECS={specs};</script><script>{js}</script></body></html>"""
