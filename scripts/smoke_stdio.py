"""Smoke-test a doc-index-mcp install over stdio.

Starts the server binary, runs a real MCP `initialize` + `tools/list`
exchange, and fails if the server dies or advertises no tools.

This exists because the pytest suite runs against `uv sync` (fully pinned by
uv.lock) and so cannot detect a dependency constraint in pyproject.toml that
is too loose for a fresh install. Run it against a wheel installed *without*
the lockfile.

Usage:
    python scripts/smoke_stdio.py path/to/doc-index-mcp
"""

import json
import subprocess
import sys

TIMEOUT_SECONDS = 120

EXPECTED_TOOLS = {
    "doc_index",
    "doc_search",
    "doc_list",
    "doc_chunk",
    "doc_toc",
    "doc_get_content",
    "read_document",
    "list_tables",
    "extract_table",
}


def fail(message, proc=None):
    print(f"FAIL: {message}", file=sys.stderr)
    if proc is not None:
        stderr = proc.stderr.read()
        if stderr:
            print("--- server stderr ---", file=sys.stderr)
            print(stderr, file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    server_path = sys.argv[1]

    proc = subprocess.Popen(
        [server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send(payload):
        if proc.poll() is not None:
            fail(f"server exited early with code {proc.returncode}", proc)
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def read():
        line = proc.stdout.readline()
        if not line:
            fail("server closed stdout without responding", proc)
        return json.loads(line)

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke", "version": "0"},
                },
            }
        )
        init = read()
        if "error" in init:
            fail(f"initialize returned an error: {init['error']}", proc)
        print(f"initialize OK: {init['result']['serverInfo']}")

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listing = read()
        if "error" in listing:
            fail(f"tools/list returned an error: {listing['error']}", proc)

        found = {tool["name"] for tool in listing["result"]["tools"]}
        print(f"tools/list OK: {len(found)} tools")
        for name in sorted(found):
            print(f"  - {name}")

        missing = EXPECTED_TOOLS - found
        if missing:
            fail(f"missing expected tools: {sorted(missing)}", proc)
    finally:
        if proc.poll() is None:
            proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print("smoke test passed")


if __name__ == "__main__":
    main()
