import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from petitcoding import benchmark

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        size = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(size))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for event in ({'choices': [{'delta': {'content': 'hello'}}]}, {'choices': [], 'usage': {'completion_tokens': 1}}):
            self.wfile.write(b'data: '+json.dumps(event).encode()+b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')

class BenchmarkTest(unittest.TestCase):
    def test_stream_and_concurrent_report(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                args = type('Args', (), dict(prompt_file=None, prompt='hello', requests=4,
                    output_tokens=10, timeout=5, concurrency=[1, 2], output=d, warmup=0,
                    url=f'http://127.0.0.1:{server.server_port}/v1', model='test'))()
                self.assertEqual(benchmark.run(args), 0)
                data = json.loads(Path(d, 'summary.json').read_text())
                self.assertEqual([x['successes'] for x in data['results']], [4, 4])
                self.assertEqual(len(Path(d, 'requests.jsonl').read_text().splitlines()), 8)
        finally:
            server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
