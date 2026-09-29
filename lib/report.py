"""
report.py | Build one report twice: GitHub Markdown and a styled HTML/PDF.

    rep = Report("Regime & Risk Summary", "Energy benchmarks, as of 2026-09-22", out_dir, "risk_summary")
    rep.section("Headline numbers")
    rep.kpis([("1-day 99% VaR", "$41.2k", "WTI, $1mm long")])
    rep.table(df, caption="Backtest results")
    rep.figure("figures/02_regimes.png", "Smoothed turbulent-regime probability")
    md, html = rep.write()
    rep.pdf("reports/")

Values are formatted by the caller (see fmt_* helpers) so the report never
re-rounds a number differently from the CSV it came from.
"""
from __future__ import annotations

import html as _html
import math
import os
import re
from datetime import date
from pathlib import Path

import markdown as _md

from . import browser

CSS = Path(__file__).with_name("report.css")


# ---------------------------------------------------------------- formatting
def fmt_money(x, d=1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    a = abs(x)
    s = f"${a/1e6:,.{max(d,2)}f}mm" if a >= 1e6 else (f"${a/1e3:,.{d}f}k" if a >= 1e3 else f"${a:,.0f}")
    return ("−" if x < 0 else "") + s


def fmt_pct(x, d=1, signed=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{abs(x)*100:.{d}f}%"
    return ("−" if x < 0 else ("+" if signed and x > 0 else "")) + s


def fmt_num(x, d=2, signed=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{abs(x):,.{d}f}"
    return ("−" if x < 0 else ("+" if signed and x > 0 else "")) + s


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def verdict(ok: bool | None, yes="Pass", no="Reject", warn=None) -> str:
    """Verdict label with an icon, so status is never carried by colour alone."""
    if warn:
        return f"▲ {warn}"
    if ok is None:
        return "– n/a"
    return f"✔ {yes}" if ok else f"✖ {no}"


NUM_RE = re.compile(r"[−+\-$]?[\d,.]+(%|k|mm|bp|x|×|pp)?|–|<0\.001")


def _numeric_col(col) -> bool:
    vals = [str(v).strip() for v in col]
    return bool(vals) and all(NUM_RE.fullmatch(v) for v in vals)


# ---------------------------------------------------------------- the builder
class Report:
    def __init__(self, title: str, subtitle: str, out_dir, slug: str):
        self.title, self.subtitle, self.slug = title, subtitle, slug
        self.out_dir = Path(out_dir)
        self.blocks: list[tuple[str, object]] = []

    # -- content -----------------------------------------------------------
    def section(self, title: str, new_page: bool = False):
        self.blocks.append(("h2", (title, new_page)))

    def sub(self, title: str):
        self.blocks.append(("h3", title))

    def text(self, md: str):
        self.blocks.append(("p", md.strip()))

    def bullets(self, items):
        self.blocks.append(("ul", list(items)))

    def kpis(self, items):
        """items: (label, value, note) tuples. Rendered as tiles in HTML, a table in Markdown."""
        self.blocks.append(("kpi", list(items)))

    def table(self, df, caption: str | None = None, index: bool = False, small: bool = False):
        self.blocks.append(("table", (df.reset_index() if index else df, caption, small)))

    def figure(self, path, caption: str = "", width: str = "100%"):
        # relative to the report's folder, so the Markdown renders on GitHub and the PDF finds its images
        rel = os.path.relpath(Path(path).resolve(), self.out_dir.resolve())
        self.blocks.append(("fig", (str(rel).replace("\\", "/"), caption, width)))

    def callout(self, md: str, kind: str = "note"):
        self.blocks.append(("callout", (md.strip(), kind)))

    # -- renderers ----------------------------------------------------------
    @staticmethod
    def _md_table(df) -> str:
        cols = [str(c) for c in df.columns]
        esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines += ["| " + " | ".join(esc(v) for v in row) + " |" for row in df.astype(object).itertuples(index=False)]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = [f"# {self.title}", "", f"*{self.subtitle} · generated {date.today():%Y-%m-%d}*", ""]
        for kind, x in self.blocks:
            if kind == "h2":
                out += [f"## {x[0]}", ""]
            elif kind == "h3":
                out += [f"### {x}", ""]
            elif kind == "p":
                out += [x, ""]
            elif kind == "ul":
                out += [f"- {i}" for i in x] + [""]
            elif kind == "kpi":
                out += ["| Metric | Value | Context |", "|---|---|---|"]
                out += [f"| {a} | **{b}** | {c} |" for a, b, c in x] + [""]
            elif kind == "table":
                df, cap, _ = x
                if cap:
                    out += [f"**{cap}**", ""]
                out += [self._md_table(df), ""]
            elif kind == "fig":
                path, cap, _ = x
                out += [f"![{cap}]({path})", ""]
                if cap:
                    out += [f"*{cap}*", ""]
            elif kind == "callout":
                md, k = x
                out += ["> **" + {"note": "Note", "warn": "Caveat", "good": "Takeaway"}.get(k, "Note") + ":** " + md, ""]
        return "\n".join(out)

    @staticmethod
    def _cell(v) -> str:
        raw = str(v).strip()
        s = _html.escape(str(v))
        if s.startswith("✔"):
            return f'<td class="ok">{s}</td>'
        if s.startswith("✖"):
            return f'<td class="bad">{s}</td>'
        if s.startswith("▲"):
            return f'<td class="warn">{s}</td>'
        num = NUM_RE.fullmatch(raw)
        return f'<td class="num">{s}</td>' if num else f"<td>{s}</td>"

    def to_html(self) -> str:
        md = lambda t: _md.markdown(t, extensions=["tables", "sane_lists"])
        body = [f'<header class="cover"><h1>{_html.escape(self.title)}</h1>'
                f'<p class="sub">{_html.escape(self.subtitle)} · generated {date.today():%Y-%m-%d}</p></header>']
        for kind, x in self.blocks:
            if kind == "h2":
                cls = ' class="newpage"' if x[1] else ""
                body.append(f"<h2{cls}>{_html.escape(x[0])}</h2>")
            elif kind == "h3":
                body.append(f"<h3>{_html.escape(x)}</h3>")
            elif kind == "p":
                body.append(md(x))
            elif kind == "ul":
                body.append(md("\n".join(f"- {i}" for i in x)))
            elif kind == "kpi":
                tiles = "".join(f'<div class="kpi"><div class="kl">{_html.escape(a)}</div>'
                                f'<div class="kv">{_html.escape(b)}</div><div class="kn">{_html.escape(c)}</div></div>'
                                for a, b, c in x)
                body.append(f'<div class="kpis">{tiles}</div>')
            elif kind == "table":
                df, cap, small = x
                head = "".join(f'<th class="{"num" if _numeric_col(df[c]) else ""}">{_html.escape(str(c))}</th>'
                               for c in df.columns)
                rows = "".join("<tr>" + "".join(self._cell(v) for v in r) + "</tr>"
                               for r in df.astype(object).itertuples(index=False))
                capt = f"<caption>{_html.escape(cap)}</caption>" if cap else ""
                body.append(f'<table class="{"small" if small else ""}">{capt}<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>')
            elif kind == "fig":
                path, cap, w = x
                body.append(f'<figure><img src="{path}" style="width:{w}"><figcaption>{_html.escape(cap)}</figcaption></figure>')
            elif kind == "callout":
                t, k = x
                label = {"note": "Note", "warn": "Caveat", "good": "Takeaway"}.get(k, "Note")
                body.append(f'<div class="callout {k}"><b>{label}.</b> {md(t)[3:-4]}</div>')
        css = CSS.read_text() if CSS.exists() else ""
        return ("<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>{_html.escape(self.title)}</title><style>{css}</style></head>"
                f"<body>{''.join(body)}</body></html>")

    # -- output ---------------------------------------------------------------
    def write(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.out_dir / f"{self.slug}.md"
        html_path = self.out_dir / f"{self.slug}.html"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        html_path.write_text(self.to_html(), encoding="utf-8")
        return md_path, html_path

    def pdf(self, reports_dir) -> Path | None:
        _, html_path = self.write()
        pdf_path = Path(reports_dir) / f"{self.slug}.pdf"
        ok = browser.html_to_pdf(html_path, pdf_path)
        return pdf_path if ok else None
