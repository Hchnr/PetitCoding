import tempfile
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

    def test_extended_context(self):
        s = load('configs/sglang.toml')
        s['context_length'] = 1000000
        cmd, env = command(s)
        self.assertIn('--json-model-override-args', cmd)
        self.assertEqual(env['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN'], '1')
        self.assertIn('"factor": 4.0', cmd[-1])

    def test_reject_unknown_options(self):
        source = Path('configs/sglang.toml').read_text() + '\nquantization = "fp8"\n'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml') as f:
            f.write(source); f.flush()
            with self.assertRaises(ValueError): load(f.name)

if __name__ == '__main__': unittest.main()
