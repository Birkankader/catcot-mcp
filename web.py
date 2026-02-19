"""Lightweight HTTP server for the Catcot dashboard."""

import json
import os
import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import chromadb

from savings import get_savings_summary

CHROMA_DIR = os.path.expanduser("~/.code-rag-mcp/chroma_db")
DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"
DEFAULT_PORT = 9847


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the Catcot dashboard and API endpoints."""

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_dashboard()
        elif self.path == "/api/status":
            self._api_status()
        elif self.path == "/api/projects":
            self._api_projects()
        elif self.path == "/api/savings":
            self._api_savings()
        elif self.path == "/api/embeddings":
            self._api_embeddings()
        else:
            self.send_error(404)

    def _serve_dashboard(self):
        try:
            content = DASHBOARD_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(500, "Dashboard file not found")

    def _api_status(self):
        status = {"mcp": True, "mcp_detail": "Catcot running on stdio"}

        # Check ChromaDB
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collections = client.list_collections()
            status["chromadb"] = True
            status["chroma_detail"] = f"{len(collections)} collection(s)"
        except Exception:
            status["chromadb"] = False
            status["chroma_detail"] = "Not available"

        self._send_json(status)

    def _api_projects(self):
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collections = client.list_collections()
            projects = []
            for col in collections:
                meta = col.metadata or {}
                projects.append({
                    "name": col.name,
                    "project_path": meta.get("project_path", "unknown"),
                    "chunks": col.count(),
                })
            self._send_json(projects)
        except Exception as e:
            self._send_json([])

    def _api_savings(self):
        """Return Catcot savings summary."""
        try:
            summary = get_savings_summary()
            self._send_json(summary)
        except Exception:
            self._send_json({
                "total_searches": 0,
                "tokens_saved": 0,
                "tokens_used": 0,
                "dollars_saved": 0.0,
                "trend": [],
            })

    def _api_embeddings(self):
        """Return 2-D projected embeddings for all indexed code chunks."""
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collections = client.list_collections()

            points = []
            for col in collections:
                meta = col.metadata or {}
                project = meta.get("project_path", col.name)

                # Fetch everything — embeddings + metadata
                try:
                    result = col.get(include=["embeddings", "metadatas"])
                except Exception:
                    continue

                embeddings = result.get("embeddings") or []
                metadatas = result.get("metadatas") or []

                if not embeddings:
                    continue

                # --- Deterministic random projection: 768-dim -> 2-dim ---
                # Build a fixed pseudo-random 768x2 projection matrix using
                # a simple LCG seeded at 42.  This preserves relative distances
                # well enough for visual clustering (Johnson-Lindenstrauss).
                DIM = len(embeddings[0])
                seed = 42
                proj = []
                for _ in range(DIM * 2):
                    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
                    # Map uint32 to [-1, 1]
                    proj.append((seed / 0x7FFFFFFF) - 1.0)

                # proj is a flat list of DIM*2 values; treat as DIM rows of 2
                # i.e. proj_matrix[d] = (proj[2*d], proj[2*d+1])
                # Normalise each column by 1/sqrt(DIM) for unit variance
                scale = 1.0 / (DIM ** 0.5)

                for emb, md in zip(embeddings, metadatas):
                    if not emb:
                        continue
                    x = sum(emb[d] * proj[2 * d] for d in range(min(DIM, len(emb)))) * scale
                    y = sum(emb[d] * proj[2 * d + 1] for d in range(min(DIM, len(emb)))) * scale
                    points.append({
                        "x": round(x, 6),
                        "y": round(y, 6),
                        "file_path": md.get("file_path", ""),
                        "symbol_name": md.get("symbol_name", ""),
                        "language": md.get("language", ""),
                        "project": project,
                    })

            self._send_json(points)
        except Exception as e:
            self._send_json([])

    def _send_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def _find_free_port(start: int = DEFAULT_PORT, tries: int = 10) -> int:
    import socket
    for port in range(start, start + tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start


def start_dashboard(open_browser: bool = True) -> int:
    """Start the Catcot dashboard HTTP server in a background thread.

    Returns the port number the server is listening on.
    """
    port = _find_free_port()
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}"

    if open_browser:
        webbrowser.open(url)

    return port
