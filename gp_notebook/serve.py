"""Portable marimo launcher with remote-GPU connection hints."""

from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from gp_notebook.device import detect_device_info
from gp_notebook.paths import PROJECT_ROOT

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 2718
NOTEBOOK_PATH: Path = PROJECT_ROOT / "notebook.py"

ServeScenario = Literal["local", "remote_ssh", "remote_compute"]


class HeadlessMode(Enum):
    """Tri-state headless setting before CLI/env resolution."""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class ServeContext:
    """Detected runtime environment for marimo serving and tunnel hints."""

    hostname: str
    scenario: ServeScenario
    ssh_remote: bool
    has_display: bool
    ssh_gateway: str | None
    ssh_user: str | None


@dataclass(frozen=True, slots=True)
class ServeOptions:
    """Resolved marimo launch configuration."""

    host: str
    port: int
    headless: bool
    edit_mode: bool
    notebook_path: Path


def _parse_ssh_connection() -> tuple[str | None, str | None]:
    """Return SSH client user and server address from standard SSH env vars."""
    connection = os.environ.get("SSH_CONNECTION", "").strip()
    if connection:
        parts = connection.split()
        if len(parts) >= 3:
            return os.environ.get("USER"), parts[2]
    client = os.environ.get("SSH_CLIENT", "").strip()
    if client:
        parts = client.split()
        if parts:
            return os.environ.get("USER"), parts[0]
    return os.environ.get("USER"), None


def detect_serve_context() -> ServeContext:
    """Classify the environment using portable SSH, display, and hostname signals."""
    hostname = socket.gethostname()
    ssh_remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
    has_display = bool(os.environ.get("DISPLAY"))
    ssh_user, ssh_gateway = _parse_ssh_connection()

    if ssh_remote:
        scenario: ServeScenario = "remote_ssh"
    elif not has_display:
        scenario = "remote_compute"
    else:
        scenario = "local"

    return ServeContext(
        hostname=hostname,
        scenario=scenario,
        ssh_remote=ssh_remote,
        has_display=has_display,
        ssh_gateway=ssh_gateway,
        ssh_user=ssh_user,
    )


def _parse_headless_env() -> HeadlessMode:
    """Parse GP_MARIMO_HEADLESS when set; otherwise defer to auto-detection."""
    raw = os.environ.get("GP_MARIMO_HEADLESS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return HeadlessMode.ON
    if raw in {"0", "false", "no", "off"}:
        return HeadlessMode.OFF
    return HeadlessMode.AUTO


def _auto_headless(ctx: ServeContext) -> bool:
    """Infer headless mode when not explicitly configured."""
    if ctx.scenario == "local":
        return not ctx.has_display
    return True


def resolve_serve_options(
    ctx: ServeContext,
    *,
    host: str | None = None,
    port: int | None = None,
    headless_mode: HeadlessMode = HeadlessMode.AUTO,
    edit_mode: bool = False,
    notebook_path: Path | None = None,
) -> ServeOptions:
    """Merge CLI flags, environment variables, and detected context into launch options."""
    resolved_host = host or os.environ.get("GP_MARIMO_HOST", DEFAULT_HOST)
    resolved_port = port or int(os.environ.get("GP_MARIMO_PORT", str(DEFAULT_PORT)))
    if headless_mode is HeadlessMode.AUTO:
        resolved_headless = _auto_headless(ctx)
    else:
        resolved_headless = headless_mode is HeadlessMode.ON

    return ServeOptions(
        host=resolved_host,
        port=resolved_port,
        headless=resolved_headless,
        edit_mode=edit_mode,
        notebook_path=notebook_path or NOTEBOOK_PATH,
    )


def print_connection_hints(ctx: ServeContext, options: ServeOptions) -> None:
    """Print environment-specific instructions for reaching the marimo server."""
    host = options.host
    port = options.port
    url = f"http://{host}:{port}"
    device = detect_device_info()
    device_line = (
        f"Accelerator: {device['kind'].upper()}"
        + (f" ({device['device_name']})" if device["device_name"] else " (CPU only)")
    )

    print("\n--- GP notebook serve ---")
    print(f"Host: {ctx.hostname} · Scenario: {ctx.scenario} · {device_line}")
    print(f"Marimo URL on this machine: {url}\n")

    if ctx.scenario == "local":
        print("Open in your browser:")
        print(f"  {url}")
        if options.headless:
            print("\nHeadless mode is on (no DISPLAY). Use the URL above on this machine.")
        return

    if ctx.scenario == "remote_ssh":
        user = ctx.ssh_user or "USER"
        gateway = ctx.ssh_gateway or ctx.hostname
        print("Marimo is on the same host as your SSH session.")
        print("On your local machine, create a tunnel:")
        print(f"  ssh -N -L {port}:{host}:{port} {user}@{gateway}")
        print(f"Then open http://127.0.0.1:{port} in your browser.")
        return

    user = ctx.ssh_user or "USER"
    gateway = ctx.ssh_gateway or "GATEWAY"
    print(f"Marimo is running on compute host '{ctx.hostname}'.")
    print("If your IDE forwards ports from an SSH gateway, bridge gateway → compute:")
    print(f"  ssh -N -L {port}:localhost:{port} {ctx.hostname}")
    print(f"Then open http://127.0.0.1:{port} on the gateway (or via your IDE port forward).")
    print("\nFrom your laptop (when ProxyJump or direct compute SSH is allowed):")
    print(
        f"  ssh -N -L {port}:{host}:{port} -J {user}@{gateway} {user}@{ctx.hostname}"
    )
    print(f"Then open http://127.0.0.1:{port} in your browser.")


def build_marimo_argv(options: ServeOptions) -> list[str]:
    """Build the argv list passed to ``python -m marimo``."""
    command = "edit" if options.edit_mode else "run"
    argv = [
        sys.executable,
        "-m",
        "marimo",
        command,
        str(options.notebook_path),
        "--host",
        options.host,
        "--port",
        str(options.port),
    ]
    if options.headless:
        argv.append("--headless")
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the portable marimo launcher."""
    parser = argparse.ArgumentParser(
        description="Launch the garden-path marimo notebook with remote-GPU-friendly defaults.",
    )
    parser.add_argument("--host", default=None, help=f"Bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=None, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Use marimo edit instead of marimo run.",
    )
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument(
        "--headless",
        action="store_const",
        const=HeadlessMode.ON,
        dest="headless_mode",
        help="Do not open a browser on the server.",
    )
    headless.add_argument(
        "--no-headless",
        action="store_const",
        const=HeadlessMode.OFF,
        dest="headless_mode",
        help="Allow marimo to open a browser (local development).",
    )
    parser.set_defaults(headless_mode=HeadlessMode.AUTO)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Detect environment, print connection hints, and exec marimo."""
    args = parse_args(argv)
    ctx = detect_serve_context()
    env_headless = _parse_headless_env()
    headless_mode = args.headless_mode
    if headless_mode is HeadlessMode.AUTO and env_headless is not HeadlessMode.AUTO:
        headless_mode = env_headless

    options = resolve_serve_options(
        ctx,
        host=args.host,
        port=args.port,
        headless_mode=headless_mode,
        edit_mode=args.edit,
    )
    if not options.notebook_path.exists():
        print(f"Notebook not found: {options.notebook_path}", file=sys.stderr)
        raise SystemExit(1)

    print_connection_hints(ctx, options)
    marimo_argv = build_marimo_argv(options)
    os.execv(sys.executable, marimo_argv)


if __name__ == "__main__":
    main()
