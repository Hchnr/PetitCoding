"""Closed-loop streaming benchmark for an OpenAI-compatible chat endpoint."""
import argparse
import concurrent.futures
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path


def request(url, model, prompt, output, timeout):
    start = time.perf_counter()
    first = None
    chunks = []
    usage = {}
    done = False
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': output,
               'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(url.rstrip('/') + '/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                if not data:
                    continue
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning') or delta.get('tool_calls'):
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        chunks.append(now - start)
        if first is None or not done:
            raise RuntimeError('stream ended without output or completion marker')
        return {'ok': True, 'latency_s': time.perf_counter()-start, 'ttft_s': first-start,
                'output_chunks_s': chunks, 'usage': usage}
    except Exception as e:
        return {'ok': False, 'latency_s': time.perf_counter()-start, 'error': str(e)}


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[math.ceil(len(xs)*p)-1]


def run(args):
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    if not prompt:
        raise ValueError('prompt must not be empty')
    if args.requests < 1 or args.output_tokens < 1 or args.timeout <= 0 or any(c < 1 for c in args.concurrency):
        raise ValueError('requests, output tokens, timeout and concurrency must be positive')
    if args.requests < max(args.concurrency):
        raise ValueError('requests must be at least the largest concurrency')
    if args.warmup < 0:
        raise ValueError('warmup cannot be negative')
    cold = getattr(args, 'cold_cache', False)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    if args.warmup:
        for _ in range(args.warmup):
            warm = request(args.url, args.model, prompt, args.output_tokens, args.timeout)
            if not warm['ok']:
                raise RuntimeError('warmup failed: '+warm['error'])
    summary = []
    with (output/'requests.jsonl').open('w') as raw:
        for c in args.concurrency:
            start = time.perf_counter()
            rows = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
                def rp():
                    return prompt if not cold else 'cold-cache:'+uuid.uuid4().hex+':\n'+prompt
                futures = [executor.submit(request, args.url, args.model, rp(), args.output_tokens, args.timeout)
                           for _ in range(args.requests)]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    row['concurrency'] = c
                    rows.append(row)
                    raw.write(json.dumps(row)+'\n')
                    raw.flush()
            elapsed = time.perf_counter()-start
            ok = [r for r in rows if r['ok']]
            ttfts = [r['ttft_s'] for r in ok]
            latencies = [r['latency_s'] for r in ok]
            intervals = []
            tpots = []
            for r in ok:
                cs = r['output_chunks_s']
                intervals.extend(cs[i+1]-cs[i] for i in range(len(cs)-1))
                count = r['usage'].get('completion_tokens')
                if isinstance(count, int) and count > 1 and len(cs) > 1:
                    tpots.append((cs[-1]-cs[0])/(count-1))
            tokens = sum(r['usage'].get('completion_tokens', 0) for r in ok)
            has_usage = bool(ok) and all('completion_tokens' in r['usage'] for r in ok)
            has_input_usage = bool(ok) and all('prompt_tokens' in r['usage'] for r in ok)
            item = {'concurrency': c, 'requests': len(rows), 'successes': len(ok), 'elapsed_s': elapsed,
                    'requests_per_s': len(ok)/elapsed, 'output_tokens_per_s': tokens/elapsed if has_usage else None,
                    'input_tokens': sum(r['usage']['prompt_tokens'] for r in ok) if has_input_usage else None,
                    'ttft_p50_s': percentile(ttfts, .5), 'ttft_p95_s': percentile(ttfts, .95),
                    'latency_p50_s': percentile(latencies, .5), 'latency_p95_s': percentile(latencies, .95),
                    'chunk_interval_p50_s': percentile(intervals, .5), 'chunk_interval_p95_s': percentile(intervals, .95),
                    'tpot_s': sum(tpots)/len(tpots) if tpots else None}
            summary.append(item)
            print(json.dumps(item), flush=True)
    (output/'summary.json').write_text(json.dumps({'settings': vars(args), 'results': summary}, indent=2)+'\n')
    return 0 if all(r['successes'] == r['requests'] for r in summary) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:30000/v1')
    p.add_argument('--model', default='petit-qwen')
    p.add_argument('--prompt', default='Explain the binary search algorithm briefly.')
    p.add_argument('--prompt-file')
    p.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--requests', type=int, default=8, help='number of requests per concurrency')
    p.add_argument('--output-tokens', type=int, default=256)
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--cold-cache', action='store_true', help='give each request a unique prompt to defeat prefix caching')
    p.add_argument('--output', default='benchmark-results')
    args = p.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError) as e:
        p.exit(2, f'benchmark: {e}\n')

if __name__ == '__main__':
    raise SystemExit(main())
