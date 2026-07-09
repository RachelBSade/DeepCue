"""
Windows launcher: sets SelectorEventLoop before Daphne imports its reactor.

Daphne installs Twisted's asyncioreactor at import time, which locks in
whatever event loop is active at that moment. On Windows the default is
ProactorEventLoop, which breaks redis.asyncio (WSAECONNABORTED / 10053).
This launcher forces SelectorEventLoop before any Daphne code runs.

Usage:
    python run_daphne.py -b 0.0.0.0 -p 8000 deepcue_backend.asgi:application
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from daphne.cli import CommandLineInterface  # noqa: E402

if __name__ == "__main__":
    CommandLineInterface.entrypoint()
