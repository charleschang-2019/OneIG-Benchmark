"""GPU-free regression tests for dispatch visibility and numbering."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oneig_bench.parallel import dispatch


def visible_worker(shard, common):
    return [(i, (item, os.environ.get('CUDA_VISIBLE_DEVICES'), os.getpid()))
            for i, item in shard]


class DeviceDispatchTests(unittest.TestCase):
    def run_dispatch(self, devices, visible, items=None):
        with patch.dict(os.environ):
            if visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible
            before = os.environ.get('CUDA_VISIBLE_DEVICES')
            result = dispatch('test_parallel_devices:visible_worker',
                              items if items is not None else list(range(6)), {}, devices)
            self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), before)
            return result

    def test_single_existing_visibility(self):
        result = self.run_dispatch([1], '3,5')
        self.assertEqual({r[1] for r in result}, {'5'})
        self.assertTrue(all(r[2] != os.getpid() for r in result))

    def test_single_without_visibility(self):
        self.assertEqual({r[1] for r in self.run_dispatch([1], None)}, {'1'})

    def test_multi_mapping_and_order(self):
        result = self.run_dispatch([1, 0], '3,5')
        self.assertEqual([r[0] for r in result], list(range(6)))
        self.assertEqual([r[1] for r in result], ['5', '3'] * 3)
        self.assertEqual(len({r[2] for r in result}), 2)

    def test_multi_without_visibility(self):
        self.assertEqual({r[1] for r in self.run_dispatch([2, 4], None)}, {'2', '4'})

    def test_uuid_tokens(self):
        self.assertEqual({r[1] for r in self.run_dispatch([1], 'GPU-abc,MIG-def')}, {'MIG-def'})

    def test_one_item_mapping(self):
        self.assertEqual(self.run_dispatch([1, 0], '3,5', [42])[0][:2], (42, '5'))

    def test_default_and_repeated_dispatch(self):
        self.assertEqual({r[1] for r in self.run_dispatch([], '3,5')}, {'3'})
        self.assertEqual({r[1] for r in self.run_dispatch([1], '3,5')}, {'5'})

    def test_invalid_selection(self):
        for devices, visible in [([2], '3,5'), ([-1], None), ([0, 0], None),
                                 ([0], ''), ([0], '-1'), ([0, 2], '3,5')]:
            with self.subTest(devices=devices, visible=visible):
                with self.assertRaises(ValueError):
                    self.run_dispatch(devices, visible, [42])

    def test_empty_items(self):
        self.assertEqual(self.run_dispatch([0], '', []), [])
