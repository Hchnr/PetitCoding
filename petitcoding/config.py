import json
import os
import tomllib
from pathlib import Path

NATIVE = 262144
MAX_CONTEXT = 1000000


def load(path):
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    if set(config) != {'server'}:
        raise ValueError('configuration must contain only [server]')
    s = config['server']
    keys = {'backend', 'model_path', 'model_name', 'host', 'port', 'gpu_ids',
            'context_length', 'max_running_requests', 'memory_fraction'}
    if set(s) != keys:
        raise ValueError(f'wrong server keys; missing={keys-set(s)}, unknown={set(s)-keys}')
    if s['backend'] not in ('sglang', 'vllm'):
        raise ValueError('backend must be sglang or vllm')
    if not Path(s['model_path']).is_dir():
        raise ValueError('model_path must exist')
    ids = s['gpu_ids']
    if not isinstance(ids, list) or len(ids) not in (1, 2, 4) or any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('gpu_ids must contain 1, 2 or 4 distinct nonnegative IDs')
    if type(s['context_length']) is not int or not 1 <= s['context_length'] <= MAX_CONTEXT:
        raise ValueError('context_length must be between 1 and 1000000')
    if type(s['port']) is not int or not 1 <= s['port'] <= 65535:
        raise ValueError('invalid port')
    if type(s['max_running_requests']) is not int or s['max_running_requests'] < 1:
        raise ValueError('max_running_requests must be positive')
    if not isinstance(s['memory_fraction'], (float, int)) or not 0 < s['memory_fraction'] < 1:
        raise ValueError('memory_fraction must be between 0 and 1')
    if not isinstance(s['model_name'], str) or not s['model_name'] or not isinstance(s['host'], str) or not s['host']:
        raise ValueError('model_name and host must be nonempty strings')
    return s


def command(s):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, s['gpu_ids']))
    yarn = None
    if s['context_length'] > NATIVE:
        yarn = {'text_config': {'rope_parameters': {'mrope_interleaved': True,
                'mrope_section': [11, 11, 10], 'rope_type': 'yarn', 'rope_theta': 10000000,
                'partial_rotary_factor': 0.25, 'factor': 2.0 if s['context_length'] <= 524288 else 4.0,
                'original_max_position_embeddings': NATIVE}}}
    if s['backend'] == 'sglang':
        cmd = ['python', '-m', 'sglang.launch_server', '--model-path', s['model_path'],
               '--served-model-name', s['model_name'], '--host', s['host'], '--port', str(s['port']),
               '--tp-size', str(len(s['gpu_ids'])), '--dtype', 'bfloat16', '--kv-cache-dtype', 'bf16',
               '--context-length', str(s['context_length']), '--max-running-requests', str(s['max_running_requests']),
               '--mem-fraction-static', str(s['memory_fraction']), '--reasoning-parser', 'qwen3',
               '--tool-call-parser', 'qwen3_coder', '--mamba-ssm-dtype', 'float32']
        if yarn:
            env['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN'] = '1'
            cmd += ['--json-model-override-args', json.dumps(yarn)]
    else:
        cmd = ['vllm', 'serve', s['model_path'], '--served-model-name', s['model_name'],
               '--host', s['host'], '--port', str(s['port']), '--tensor-parallel-size', str(len(s['gpu_ids'])),
               '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16', '--max-model-len', str(s['context_length']),
               '--max-num-seqs', str(s['max_running_requests']), '--gpu-memory-utilization', str(s['memory_fraction']),
               '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3']
        if yarn:
            env['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
            cmd += ['--hf-overrides', json.dumps(yarn)]
    return cmd, env
