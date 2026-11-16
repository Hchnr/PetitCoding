import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from petitcoding import benchmark

class Handler(BaseHTTPRequestHandler):
    prompts = []
    def log_message(self, *args): pass
    def do_POST(self):
        size = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(size))
        Handler.prompts.append(data['messages'][0]['content'])
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for tok in ('a', 'b', 'c'):
            self.wfile.write(b'data: '+json.dumps({'choices': [{'delta': {'content': tok}}]}).encode()+b'\n\n')
        self.wfile.write(b'data: '+json.dumps({'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 3}}).encode()+b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')

class BenchmarkTest(unittest.TestCase):
    def _args(self, port, d, **kw):
        base = dict(prompt_file=None, prompt='hello', requests=4, output_tokens=10, timeout=5,
                    concurrency=[1, 2], output=d, warmup=0, url=f'http://127.0.0.1:{port}/v1', model='test')
        base.update(kw)
        return type('Args', (), base)()

    def test_stream_and_concurrent_report(self):
        Handler.prompts = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                args = self._args(server.server_port, d)
                self.assertEqual(benchmark.run(args), 0)
                data = json.loads(Path(d, 'summary.json').read_text())
                self.assertEqual([x['successes'] for x in data['results']], [4, 4])
                self.assertEqual(len(Path(d, 'requests.jsonl').read_text().splitlines()), 8)
                for x in data['results']:
                    self.assertIsNotNone(x['chunk_interval_p50_s'])
                    self.assertIsNotNone(x['tpot_s'])
                    self.assertEqual(x['input_tokens'], 4*5)
                    self.assertIsNotNone(x['output_tokens_per_s'])
                self.assertEqual(len(Handler.prompts), 8)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_cold_cache_varies_prompts(self):
        Handler.prompts = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                args = self._args(server.server_port, d, cold_cache=True, concurrency=[1])
                self.assertEqual(benchmark.run(args), 0)
                self.assertEqual(len(Handler.prompts), 4)
                self.assertEqual(len(set(Handler.prompts)), 4)
        finally:
            server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
