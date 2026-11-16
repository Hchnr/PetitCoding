import argparse
import json
import os
import shlex
import sys
from .config import command, load


def main():
    parser = argparse.ArgumentParser(prog='petitcoding')
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('serve', 'show', 'client'):
        p = sub.add_parser(name)
        p.add_argument('--config', default='configs/sglang.toml')
        if name == 'client':
            p.add_argument('frontend', choices=['claude', 'opencode', 'codex'])
    sub.add_parser('benchmark', help='see python -m petitcoding.benchmark --help')
    args = parser.parse_args()
    if args.action == 'benchmark':
        parser.error('use python -m petitcoding.benchmark --help')
    try:
        s = load(args.config)
        cmd, env = command(s)
        if args.action == 'show':
            names = ['CUDA_VISIBLE_DEVICES', 'SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN', 'VLLM_ALLOW_LONG_MAX_MODEL_LEN']
            prefix = ' '.join(f'{name}={shlex.quote(env[name])}' for name in names if name in env and (name == 'CUDA_VISIBLE_DEVICES' or env[name] != os.environ.get(name)))
            print(prefix, shlex.join(cmd))
        elif args.action == 'serve':
            os.execvpe(cmd[0], cmd, env)
        else:
            root = f"http://{s['host']}:{s['port']}"
            model = s['model_name']
            if args.frontend == 'claude':
                model += '[1m]' if s['backend'] == 'sglang' and s['context_length'] > 262144 else ''
                values = {'ANTHROPIC_BASE_URL': root, 'ANTHROPIC_AUTH_TOKEN': 'dummy',
                          'ANTHROPIC_DEFAULT_HAIKU_MODEL': model, 'ANTHROPIC_DEFAULT_SONNET_MODEL': model,
                          'ANTHROPIC_DEFAULT_OPUS_MODEL': model, 'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
                          'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'API_TIMEOUT_MS': '3000000',
                          'CLAUDE_CODE_AUTO_COMPACT_WINDOW': str(s['context_length'])}
                print(json.dumps({'claudeCode.disableLoginPrompt': True, 'claudeCode.environmentVariables': [{'name': k, 'value': v} for k, v in values.items()]}, indent=2))
            elif args.frontend == 'opencode':
                print(json.dumps({'$schema': 'https://opencode.ai/config.json', 'provider': {'petitcoding': {
                    'npm': '@ai-sdk/openai-compatible', 'name': 'PetitCoding', 'options': {'baseURL': root+'/v1'},
                    'models': {model: {'name': model, 'limit': {'context': s['context_length'], 'output': 16384}}}}},
                    'model': 'petitcoding/'+model}, indent=2))
            else:
                print('# Experimental: verify your Codex version supports a custom Responses endpoint')
                print('model = "'+model+'"\nmodel_provider = "petitcoding"\n\n[model_providers.petitcoding]')
                print('name = "PetitCoding"\nbase_url = "'+root+'/v1"\nwire_api = "responses"')
    except (ValueError, OSError) as e:
        print(f'petitcoding: {e}', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
