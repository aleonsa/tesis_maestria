"""Pruebas del parser y reinicios, sin GUI ni hardware."""

import argparse
import math
import unittest

from scope_m2pc import K_RAW_PER_A, Reader, parse_sample


class ScopeTests(unittest.TestCase):
    def test_units_and_park_zero(self):
        sample = parse_sample("L 1000 0 256 512 1200 -300 3000 -42")
        self.assertEqual(sample.t, 1)
        self.assertEqual(sample.rpm, 300)
        self.assertAlmostEqual(sample.id_ma, 1000 / K_RAW_PER_A)
        self.assertAlmostEqual(sample.iq_ma, 2000 / K_RAW_PER_A)
        self.assertEqual((sample.u_alpha, sample.u_beta, sample.ref_ma), (1.2, -.3, -42))

    def test_park_quarter_turn(self):
        sample = parse_sample("L 1000 16384 256 512 0 0 0 0")
        self.assertTrue(math.isclose(sample.id_ma, 2000 / K_RAW_PER_A))
        self.assertTrue(math.isclose(sample.iq_ma, -1000 / K_RAW_PER_A))

    def test_invalid_lines(self):
        for line in ("[off] cal a=1", "L 1 2 3", "L 1 2 3 4 5 6 7 8 extra", ""):
            self.assertIsNone(parse_sample(line))

    def test_reference_banner_and_resets(self):
        reader = Reader(argparse.Namespace(speed_ref=300))
        reader.consume("[m2pc] lazo de velocidad: w* = 600 rpm, Kp=432")
        self.assertEqual(reader.speed_ref, 600)
        reader.consume("L 1000 0 0 0 0 0 0 0")
        reader.consume("L 1 0 0 0 0 0 0 0")
        self.assertEqual(reader.epoch, 1)
        self.assertEqual(len(reader.pending), 1)
        reader.consume("[TRIP] sobrecorriente")
        self.assertTrue(reader.trip)
        reader.consume("[boot] STM32G431")
        self.assertFalse(reader.trip)
        self.assertEqual(reader.speed_ref, 300)
        self.assertFalse(reader.pending)

    def test_queue_is_bounded_and_counts_overflow(self):
        reader = Reader(argparse.Namespace(speed_ref=300))
        for _ in range(20001):
            reader.consume("L 1 0 0 0 0 0 0 0")
        self.assertEqual(len(reader.pending), 20000)
        self.assertEqual(reader.dropped, 1)


if __name__ == "__main__":
    unittest.main()
