"""Run with python -m workbench; only Python's standard library is needed."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser


class LabHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript'}

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description='Open the SpinnyBall physics laboratory.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('Port must be between 0 and 65535 (0 selects a free port).')
    directory = Path(__file__).resolve().parent
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(LabHandler, directory=str(directory)))
    except OSError as exc:
        parser.exit(1, f'Could not start the laboratory: {exc}\nTry --port 8766 or --port 0.\n')
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'SpinnyBall laboratory: {url}\nPress Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
