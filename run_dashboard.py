"""
THERMO-X Dashboard Launcher.

Run from the project root:

    python run_dashboard.py

This launcher:

1. Starts Streamlit.
2. Waits for the local server to become available.
3. Opens Google Chrome automatically.
4. Keeps the Streamlit process running.

The explicit Chrome launch is used so that the project does not
depend on Windows' default browser configuration.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

DASHBOARD_FILE = (
    PROJECT_ROOT
    / "dashboard"
    / "dashboard.py"
)

HOST = "localhost"

PORT = 8501

URL = (
    f"http://{HOST}:{PORT}"
)


# ============================================================
# CHROME DISCOVERY
# ============================================================


def find_chrome() -> Path | None:
    """
    Try to find Google Chrome on Windows.

    Common installation locations are checked first.
    """

    candidates: list[Path] = []

    # Environment-variable based locations.
    program_files = os.environ.get(
        "PROGRAMFILES"
    )

    program_files_x86 = os.environ.get(
        "PROGRAMFILES(X86)"
    )

    local_app_data = os.environ.get(
        "LOCALAPPDATA"
    )

    if program_files:

        candidates.append(
            Path(program_files)
            / "Google"
            / "Chrome"
            / "Application"
            / "chrome.exe"
        )

    if program_files_x86:

        candidates.append(
            Path(program_files_x86)
            / "Google"
            / "Chrome"
            / "Application"
            / "chrome.exe"
        )

    if local_app_data:

        candidates.append(
            Path(local_app_data)
            / "Google"
            / "Chrome"
            / "Application"
            / "chrome.exe"
        )

    # Check PATH as an additional possibility.
    chrome_from_path = shutil.which(
        "chrome"
    )

    if chrome_from_path:

        candidates.append(
            Path(
                chrome_from_path
            )
        )

    for candidate in candidates:

        if candidate.exists():

            return candidate

    return None


# ============================================================
# SERVER CHECK
# ============================================================


def is_server_ready() -> bool:
    """
    Check whether Streamlit is responding on localhost:8501.
    """

    try:

        with urllib.request.urlopen(
            URL,
            timeout=0.75,
        ):

            return True

    except Exception:

        return False


def wait_for_server(
    timeout_s: float = 30.0,
) -> bool:
    """
    Wait until the Streamlit server responds.

    Returns True when ready, otherwise False.
    """

    start = time.monotonic()

    while (
        time.monotonic()
        - start
        < timeout_s
    ):

        if is_server_ready():

            return True

        time.sleep(
            0.25
        )

    return False


# ============================================================
# CHROME LAUNCH
# ============================================================


def open_chrome() -> None:
    """Open the THERMO-X page in Google Chrome."""

    chrome = find_chrome()

    if chrome:

        print(
            f"Google Chrome found:\n{chrome}"
        )

        subprocess.Popen(
            [
                str(chrome),

                "--new-window",

                URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return

    # Fallback only if Chrome could not be found.
    print(
        "Google Chrome executable was not found."
    )

    print(
        "Opening the default Windows browser instead."
    )

    webbrowser.open_new(
        URL
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Start THERMO-X."""

    if not DASHBOARD_FILE.exists():

        raise FileNotFoundError(
            "THERMO-X dashboard was not found:\n"
            f"{DASHBOARD_FILE}"
        )

    print()
    print(
        "============================================"
    )

    print(
        "        THERMO-X DASHBOARD LAUNCHER"
    )

    print(
        "============================================"
    )

    print()

    print(
        f"Project : {PROJECT_ROOT}"
    )

    print(
        f"Dashboard : {DASHBOARD_FILE}"
    )

    print(
        f"URL : {URL}"
    )

    print()

    # --------------------------------------------------------
    # If Streamlit is already running, simply open Chrome.
    # --------------------------------------------------------

    if is_server_ready():

        print(
            "Existing Streamlit server detected."
        )

        open_chrome()

        print(
            "THERMO-X is already running."
        )

        return

    # --------------------------------------------------------
    # Start Streamlit.
    #
    # We explicitly force headless mode for this subprocess
    # because THIS launcher is responsible for opening Chrome.
    #
    # This prevents duplicate browser windows.
    # --------------------------------------------------------

    command = [
        sys.executable,

        "-m",

        "streamlit",

        "run",

        str(DASHBOARD_FILE),

        "--server.headless=true",

        f"--server.port={PORT}",
    ]

    print(
        "Starting Streamlit..."
    )

    process = subprocess.Popen(
        command,

        cwd=PROJECT_ROOT,
    )

    # --------------------------------------------------------
    # Wait for the server.
    # --------------------------------------------------------

    print(
        "Waiting for THERMO-X to become ready..."
    )

    ready = wait_for_server()

    if not ready:

        print(
            "ERROR: Streamlit did not become ready "
            "within the expected time."
        )

        process.terminate()

        sys.exit(1)

    print(
        "THERMO-X server is ready."
    )

    # --------------------------------------------------------
    # Open Chrome.
    # --------------------------------------------------------

    open_chrome()

    print()

    print(
        "THERMO-X is running."
    )

    print(
        "Press Ctrl+C in this terminal to stop it."
    )

    print()

    try:

        process.wait()

    except KeyboardInterrupt:

        print()

        print(
            "Stopping THERMO-X..."
        )

        process.terminate()

        try:

            process.wait(
                timeout=5
            )

        except subprocess.TimeoutExpired:

            process.kill()


if __name__ == "__main__":
    main()