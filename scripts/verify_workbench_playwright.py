"""Playwright end-to-end browser verification suite for SpinnyBall Scientific Workbench.

Validates:
1. Scenario switching (Orbital motion, Free spin, Momentum exchange)
2. Simulation execution, play/pause, scrub, and reset
3. Exact-sample plot inspection and live readout
4. Orbital speed sweep execution, parameter updates, table generation, and row selection
5. JSON and CSV export/download for simulations and sweeps
6. Deterministic JSON import recomputing (single runs and sweeps)
7. Physical invariant audit modal and model notes dialog
8. Responsive mobile viewport (390 x 844) without horizontal overflow
9. Console error and unhandled rejection tracking (zero allowed)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output" / "playwright"
PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"


def wait_for_badge(page, timeout=10000):
    page.wait_for_function(
        "() => { const el = document.getElementById('balanceBadge'); return el && el.textContent.includes('0.1%'); }",
        timeout=timeout
    )
    return page.locator("#balanceBadge").inner_text()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server_process = None

    # Check if server is already running on port
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_running = (sock.connect_ex(('127.0.0.1', PORT)) == 0)
    sock.close()

    if not is_running:
        print(f"Starting workbench server on port {PORT}...")
        server_process = subprocess.Popen(
            [sys.executable, "-m", "workbench", "--port", str(PORT), "--no-browser"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(1.2)
    else:
        print(f"Workbench server already running on port {PORT}.")

    console_errors = []
    page_errors = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                accept_downloads=True
            )
            page = context.new_page()

            page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            print("\n--- 1. Loading Workbench ---")
            page.goto(BASE_URL, wait_until="networkidle")
            assert "SpinnyBall" in page.title(), f"Unexpected title: {page.title()}"
            print("Page loaded successfully.")

            # Wait for initial simulation run to complete
            badge_text = wait_for_badge(page)
            print(f"Initial Orbit Balance Badge: {badge_text}")
            assert "< 0.1%" in badge_text

            print("\n--- 2. Testing Playback Controls ---")
            play_btn = page.locator("#play")
            reset_btn = page.locator("#reset")
            timeline = page.locator("#timeline")

            # Click play
            play_btn.click()
            time.sleep(0.3)
            # Click pause
            play_btn.click()
            time.sleep(0.1)

            # Scrub timeline
            timeline.fill("0.5")
            page.dispatch_event("#timeline", "input")
            time.sleep(0.1)

            # Reset
            reset_btn.click()
            time.sleep(0.1)
            clock_text = page.locator("#clock").inner_text()
            assert "t = 0" in clock_text, f"Clock did not reset: {clock_text}"
            print("Playback, scrub, and reset verified.")

            print("\n--- 3. Testing Plot Inspection ---")
            signal_plot = page.locator("#signalPlot")
            signal_plot.focus()
            page.keyboard.press("ArrowRight")
            time.sleep(0.1)
            inspection_text = page.locator("#plotInspection").inner_text()
            print(f"Plot Inspection Readout (focused): {inspection_text}")
            assert "sample" in inspection_text.lower() or "t =" in inspection_text, "Inspection readout did not activate"

            print("\n--- 4. Testing Orbit Speed Sweep ---")
            sweep_panel = page.locator("#sweepPanel")
            sweep_panel.scroll_into_view_if_needed()

            # Set sweep parameters
            page.fill("#sweepMinSpeed", "1.3")
            page.fill("#sweepMaxSpeed", "1.5")
            page.fill("#sweepCount", "5")

            # Run sweep
            page.click("#runSweepButton")
            page.wait_for_selector("#sweepTableBody tr:nth-child(5)", timeout=20000)
            rows = page.locator("#sweepTableBody tr").all()
            assert len(rows) == 5, f"Expected 5 sweep rows, got {len(rows)}"
            print(f"Sweep completed with {len(rows)} points.")

            # Verify row 1 is bound and row 5 is unbound
            row1_text = rows[0].inner_text()
            row5_text = rows[4].inner_text()
            print(f"Row 1 (s=1.30): {row1_text}")
            print(f"Row 5 (s=1.50): {row5_text}")
            assert "bound" in row1_text.lower()
            assert "unbound" in row5_text.lower()

            # Click row 4 (s=1.45) to load into simulation
            load_btn = rows[3].locator("button")
            load_btn.click()
            wait_for_badge(page)
            speed_val = page.locator("#param-speed").input_value()
            print(f"Loaded sweep row into orbit parameters (param-speed: {speed_val})")
            assert speed_val == "1.45"

            # Take sweep desktop screenshot
            page.screenshot(path=str(OUTPUT_DIR / "sweep-desktop.png"))

            print("\n--- 5. Testing Export & Download (Sim & Sweep) ---")
            # Sweep exports
            with page.expect_download() as download_info:
                page.click("#exportSweepJSON")
            download = download_info.value
            sweep_json_path = OUTPUT_DIR / "downloaded_sweep.json"
            download.save_as(str(sweep_json_path))
            with open(sweep_json_path, "r", encoding="utf-8") as f:
                sweep_data = json.load(f)
            assert sweep_data.get("sweepSchemaVersion") == 1
            assert len(sweep_data.get("rows", [])) == 5
            print("Sweep JSON export verified.")

            with page.expect_download() as download_info:
                page.click("#exportSweepCSV")
            download = download_info.value
            sweep_csv_path = OUTPUT_DIR / "downloaded_sweep.csv"
            download.save_as(str(sweep_csv_path))
            with open(sweep_csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 6  # Header + 5 data rows
            print("Sweep CSV export verified.")

            # Sim exports
            with page.expect_download() as download_info:
                page.click("#exportJSON")
            download = download_info.value
            sim_json_path = OUTPUT_DIR / "downloaded_orbit.json"
            download.save_as(str(sim_json_path))
            with open(sim_json_path, "r", encoding="utf-8") as f:
                sim_data = json.load(f)
            assert sim_data.get("schemaVersion") == 1
            assert sim_data.get("config", {}).get("model") == "orbit"
            print("Sim Orbit JSON export verified.")

            with page.expect_download() as download_info:
                page.click("#exportCSV")
            download = download_info.value
            sim_csv_path = OUTPUT_DIR / "downloaded_orbit.csv"
            download.save_as(str(sim_csv_path))
            with open(sim_csv_path, "r", encoding="utf-8") as f:
                csv_lines = f.readlines()
            assert len(csv_lines) > 100
            print("Sim Orbit CSV export verified.")

            print("\n--- 6. Testing Deterministic JSON Import ---")
            # Trigger import with the exported simulation file
            page.set_input_files("#importFile", str(sim_json_path))
            recomputed_badge = wait_for_badge(page)
            print(f"Recomputed import balance badge: {recomputed_badge}")
            assert "< 0.1%" in recomputed_badge

            # Trigger import with the exported sweep file
            page.set_input_files("#importFile", str(sweep_json_path))
            page.wait_for_selector("#sweepTableBody tr:nth-child(5)", timeout=20000)
            print("Recomputed sweep import verified.")

            print("\n--- 7. Testing Modals (Audit & Notes) ---")
            # Model notes
            page.click("#notesButton")
            page.wait_for_selector("#notes[open]")
            notes_header = page.locator("#notes h2").inner_text()
            assert "Observe the model" in notes_header
            page.click("#closeNotes")
            page.wait_for_function("() => !document.getElementById('notes').hasAttribute('open')")

            # Audit dialog with external sample
            page.set_input_files("#auditFile", str(sim_csv_path))
            page.wait_for_selector("#auditModal[open]", timeout=5000)
            audit_content = page.locator("#auditContent").inner_text()
            assert "PASSED" in audit_content
            page.click("#closeAudit")
            page.wait_for_function("() => !document.getElementById('auditModal').hasAttribute('open')")
            print("Dialogs verified.")

            print("\n--- 8. Testing Scenario 02: Free Spin ---")
            page.click("button[data-model='spin']")
            spin_badge = wait_for_badge(page)
            print(f"Free spin balance badge: {spin_badge}")
            assert "< 0.1%" in spin_badge

            # Choose Intermediate axis preset (second option)
            page.select_option("#preset", index=1)
            page.click("#runButton")
            intermediate_badge = wait_for_badge(page)
            print(f"Intermediate axis spin balance badge: {intermediate_badge}")
            assert "< 0.1%" in intermediate_badge
            page.screenshot(path=str(OUTPUT_DIR / "spin-desktop.png"))

            print("\n--- 9. Testing Scenario 03: Momentum Exchange ---")
            page.click("button[data-model='exchange']")
            exchange_badge = wait_for_badge(page)
            print(f"Momentum exchange balance badge: {exchange_badge}")
            assert "< 0.1%" in exchange_badge
            page.screenshot(path=str(OUTPUT_DIR / "exchange-desktop.png"))

            print("\n--- 10. Responsive Viewport Check (390 x 844 Mobile) ---")
            page.set_viewport_size({"width": 390, "height": 844})
            page.click("button[data-model='orbit']")
            wait_for_badge(page)
            time.sleep(0.5)

            # Check for horizontal overflow
            scroll_width = page.evaluate("() => document.documentElement.scrollWidth")
            client_width = page.evaluate("() => document.documentElement.clientWidth")
            print(f"Mobile Viewport: clientWidth={client_width}, scrollWidth={scroll_width}")
            assert scroll_width <= client_width, f"Horizontal overflow detected! scrollWidth ({scroll_width}) > clientWidth ({client_width})"
            page.screenshot(path=str(OUTPUT_DIR / "mobile-390x844.png"))
            print("Mobile responsive test passed with 0 horizontal overflow.")

            # Desktop orbit screenshot
            page.set_viewport_size({"width": 1280, "height": 900})
            page.screenshot(path=str(OUTPUT_DIR / "orbit-desktop.png"))

            browser.close()

        print("\n--- Console & Error Inspection ---")
        if console_errors:
            print(f"Console errors caught: {console_errors}")
        if page_errors:
            print(f"Page errors caught: {page_errors}")

        assert len(console_errors) == 0, f"Found {len(console_errors)} console errors: {console_errors}"
        assert len(page_errors) == 0, f"Found {len(page_errors)} page errors: {page_errors}"

        print("\nAll Playwright E2E tests PASSED perfectly with 0 console errors!")

    finally:
        if server_process is not None:
            server_process.terminate()
            server_process.wait()
            print("Server stopped.")


if __name__ == "__main__":
    main()
