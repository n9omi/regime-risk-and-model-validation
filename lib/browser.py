"""
browser.py | Headless-browser helpers: HTML -> PDF and dashboard screenshots.

Tries, in order:
  1. Playwright with the browser in $CHROME_PATH (if set)
  2. Playwright driving your installed Google Chrome or Microsoft Edge
  3. Playwright's own Chromium (`python -m playwright install chromium`)
  4. The Chrome / Edge command line (no Playwright needed; PDFs only)
If none is available the HTML is still written and a clear message is printed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ARGS = ["--no-sandbox", "--disable-gpu", "--hide-scrollbars"]

CHROME_CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chromium-browser") or "",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def _chrome_exe() -> str | None:
    for c in CHROME_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def _launch(p):
    """Return a launched Playwright browser, trying the options above in order."""
    errors = []
    exe = os.environ.get("CHROME_PATH")
    if exe and Path(exe).exists():
        try:
            return p.chromium.launch(executable_path=exe, args=ARGS + ["--single-process", "--no-zygote"])
        except Exception as e:  # pragma: no cover - depends on machine
            errors.append(f"CHROME_PATH: {e}")
    for channel in ("chrome", "msedge"):
        try:
            return p.chromium.launch(channel=channel, args=ARGS)
        except Exception as e:
            errors.append(f"{channel}: {str(e).splitlines()[0]}")
    try:
        return p.chromium.launch(args=ARGS)
    except Exception as e:
        errors.append(f"bundled chromium: {str(e).splitlines()[0]}")
    raise RuntimeError("No headless browser found.\n  " + "\n  ".join(errors))


def html_to_pdf(html_path, pdf_path) -> bool:
    """Print an HTML file to PDF. Returns True on success."""
    html_path, pdf_path = Path(html_path).resolve(), Path(pdf_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            b = _launch(p)
            page = b.new_page()
            page.goto(html_path.as_uri(), wait_until="load")
            page.wait_for_timeout(300)
            page.pdf(path=str(pdf_path), format="Letter", print_background=True,
                     prefer_css_page_size=True, display_header_footer=True,
                     header_template="<span></span>",
                     footer_template=("<div style='font-size:7pt;color:#898781;width:100%;"
                                      "text-align:center;font-family:Helvetica'>"
                                      "<span class='pageNumber'></span> / <span class='totalPages'></span></div>"))
            b.close()
        return True
    except Exception as e:
        exe = _chrome_exe()
        if exe:
            cmd = [exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                   f"--print-to-pdf={pdf_path}", html_path.as_uri()]
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if pdf_path.exists():
                return True
            print(f"  ! Chrome CLI could not print the PDF: {r.stderr.decode()[:300]}")
        print(f"  ! PDF skipped ({str(e).splitlines()[0]}). Open {html_path.name} in a browser and use Print -> Save as PDF.")
        return False


def screenshots(html_path, shots: list[tuple[str, str]], width=1440, height=900, wait_ms=1600) -> list[str]:
    """
    Screenshot one page several times, once per URL hash (dashboard tab).
    shots: list of (hash, output_png). Crops to the <main> element of the page.
    """
    html_path = Path(html_path).resolve()
    done = []
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  ! Playwright not installed: skipping screenshots (pip install playwright).")
        return done
    try:
        with sync_playwright() as p:
            b = _launch(p)
            page = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=1.5)
            for h, out in shots:
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                page.goto(html_path.as_uri() + ("#" + h if h else ""), wait_until="load")
                page.evaluate("window.dispatchEvent(new HashChangeEvent('hashchange'))")
                page.wait_for_timeout(wait_ms)
                target = page.locator("#shot") if page.locator("#shot").count() else page.locator("body")
                target.screenshot(path=str(out))
                done.append(str(out))
            b.close()
    except Exception as e:
        print(f"  ! Screenshots skipped: {str(e).splitlines()[0]}")
    return done
