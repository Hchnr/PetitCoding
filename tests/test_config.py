import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path
from petitcoding.config import command, load

class ConfigTest(unittest.TestCase):
    def test_backend_commands_and_precision(self):
        for name in ('sglang', 'vllm'):
            s = load('configs/'+name+'.toml')
            cmd, env = command(s)
            self.assertIn('bfloat16', cmd)
            self.assertIn('bf16' if name == 'sglang' else 'bfloat16', cmd)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
            self.assertNotIn('--quantization', cmd)
            if name == 'sglang':
                self.assertEqual(cmd[cmd.index('--mamba-ssm-dtype')+1], 'float32')

    def test_extended_context(self):
        s = load('configs/sglang.toml')
        s['context_length'] = 1000000
        cmd, env = command(s)
        self.assertIn('--json-model-override-args', cmd)
        self.assertEqual(env['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN'], '1')
        self.assertIn('"factor": 4.0', cmd[-1])

    def test_claude_profiles_declare_model_window_and_effort(self):
        for path, model, window in [('configs/sglang.toml', 'petit-qwen', '262144'),
                                    ('configs/sglang-1m.toml', 'petit-qwen[1m]', '1000000')]:
            result = subprocess.check_output([sys.executable, '-m', 'petitcoding', 'client',
                                              '--config', path, 'claude'], text=True)
            values = {item['name']: item['value'] for item in json.loads(result)['claudeCode.environmentVariables']}
            self.assertEqual(values['ANTHROPIC_MODEL'], model)
            self.assertEqual(values['CLAUDE_CODE_MAX_CONTEXT_TOKENS'], window)
            self.assertEqual(values['CLAUDE_CODE_EFFORT_LEVEL'], 'medium')
            self.assertEqual(values['CLAUDE_CODE_AUTO_MODE_SERVER'], '0')
            self.assertEqual(values['CLAUDE_CODE_AUTO_MODE_MODEL'], model)

    def test_qwen_client_provider(self):
        for path, window in [('configs/sglang.toml', 262144), ('configs/sglang-1m.toml', 1000000)]:
            result = subprocess.check_output([sys.executable, '-m', 'petitcoding', 'client', '--config', path, 'qwen'], text=True)
            cfg = json.loads(result)
            provider = cfg['modelProviders']['openai'][0]
            self.assertEqual(provider['id'], 'petit-qwen')
            self.assertTrue(provider['baseUrl'].endswith('/v1'))
            self.assertEqual(provider['generationConfig']['contextWindowSize'], window)
            self.assertEqual(cfg['env'][provider['envKey']], 'dummy')
            self.assertEqual(cfg['security']['auth']['selectedType'], 'openai')
            self.assertEqual(cfg['model']['name'], 'petit-qwen')

    def test_reject_unknown_options(self):
        source = Path('configs/sglang.toml').read_text() + '\nquantization = "fp8"\n'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as f:
            f.write(source); f.flush()
            with self.assertRaises(ValueError): load(f.name)

if __name__ == '__main__': unittest.main()
