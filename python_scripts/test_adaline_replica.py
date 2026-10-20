"""Host C vs Python: python3 -m unittest discover -s python_scripts -p 'test_adaline_replica.py'."""
import ctypes as ct
import math
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import numpy as np

import adaline_replica as ar


ROOT = Path(__file__).resolve().parents[1]


class AdalineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="adaline-", dir="/tmp/opencode")
        lib = Path(cls.tmp.name) / "adaline.so"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-shared", "-fPIC", "-fsanitize=undefined", "-fno-sanitize-recover=all",
            "-I", str(ROOT / "firmware/lib"), str(ROOT / "firmware/lib/adaline.c"),
            "-lm", "-o", str(lib),
        ], check=True)
        cls.c = ct.CDLL(str(lib))
        cls.c.adaline_init.argtypes = [ct.c_uint8]
        cls.c.adaline_update.argtypes = [ct.c_int32]*6 + [ct.c_uint32, ct.c_int32]
        cls.c.adaline_update.restype = ct.c_uint8
        cls.c.adaline_w_q27.restype = ct.POINTER(ct.c_int32)
        cls.c.adaline_eval.argtypes = [ct.c_int32]*3 + [ct.POINTER(ct.c_int32)]*2

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def weights(self):
        return np.ctypeslib.as_array(self.c.adaline_w_q27(), shape=(28,)).copy().reshape(14, 2)

    def test_electrical_center_both_directions(self):
        for omega in (858993, -858993):
            close = 12345
            center = ar.theta_centro(close, omega)
            delta = ((close-center) & 0xFFFFFFFF)
            self.assertEqual(delta, (omega*7*25) & 0xFFFFFFFF)

    def test_initialization_and_axes(self):
        self.c.adaline_init(0)
        np.testing.assert_array_equal(self.weights(), ar.AdalineQ(7, "modelo").W)
        for th in (0, 0x40000000, 0x80000000, 0xC0000000, 0xFFFFFFFF):
            for epk in (-1540, 0, 1540):
                q = ar.AdalineQ(7, "modelo")
                s, c = ar.sincos_modelo(th)
                ea, eb = ct.c_int32(), ct.c_int32()
                self.c.adaline_eval(s, c, epk, ct.byref(ea), ct.byref(eb))
                ref_a, ref_b = ar.eval_isr(q.W, s, c, epk, 7)
                self.assertEqual((ea.value, eb.value), (ref_a, ref_b))
                # la vía Q30 de la ISR y la Q31 del update difieren solo por redondeo
                x = ar.basis_recurrencia(th, 7)
                self.assertLessEqual(abs(ea.value - ((epk*q.eval_g(x, 0)) >> ar.W_FRAC)), 1)

    def test_random_updates_bit_exact_and_speed_gate(self):
        rng = random.Random(25)
        for init in ("modelo", "cero"):
            self.c.adaline_init(init == "cero")
            q = ar.AdalineQ(7, init)
            for _ in range(3000):
                ia, ib, dia, dib = [rng.randint(-10000, 10000) for _ in range(4)]
                ua, ub = [rng.randint(-12000, 12000) for _ in range(2)]
                th, omega = rng.randrange(2**32), rng.randint(-1500000, 1500000)
                epk = (omega*ar.KE_MV_Q18) >> 18
                accepted = 0
                if abs(epk) >= ar.EPK_MIN_MV:
                    ea = ua - (((ia >> 4)*ar.RK_Q8) >> 12) - ((dia*ar.LK_Q15) >> 15)
                    eb = ub - (((ib >> 4)*ar.RK_Q8) >> 12) - ((dib*ar.LK_Q15) >> 15)
                    q.paso(ar.basis_recurrencia(ar.theta_centro(th, omega), 7), epk, ea, eb)
                    accepted = q.accepted
                self.assertEqual(self.c.adaline_update(ia, ib, dia, dib, ua, ub, th, omega), accepted)
                np.testing.assert_array_equal(self.weights(), q.W)

    def test_synthetic_above_unity(self):
        self.c.adaline_init(1)
        omega = 858993  # approximately 600 rpm
        epk = (omega*ar.KE_MV_Q18) >> 18
        # τ de convergencia del fundamental con μ=7 ≈ 2/μ ≈ 19k updates;
        # 150k ≈ 8τ. Solo se ejercita el C (el espejo bit-exact lo cubren
        # los tests aleatorios y de replay).
        for n in range(150000):
            close = (omega*7*50*n) & 0xFFFFFFFF
            th = (ar.theta_centro(close, omega))*2*math.pi/2**32
            ua, ub = round(-1.2*epk*math.sin(th)), round(1.2*epk*math.cos(th))
            self.c.adaline_update(0, 0, 0, 0, ua, ub, close, omega)
        w = self.weights()/ar.W_ONE
        self.assertAlmostEqual(w[1, 0], -1.2, delta=0.003)
        self.assertAlmostEqual(w[0, 1], 1.2, delta=0.003)
        self.assertLess(np.abs(w).sum(axis=0).max(), 4)

    def test_bench_log_replay_bit_exact(self):
        paths = sorted((ROOT / "datos_banco").glob("m2pc_s23_exp2_*.log"))
        if not paths:
            self.skipTest("logs de banco no disponibles")
        self.c.adaline_init(0)
        q = ar.AdalineQ(7, "modelo")
        count = 0
        for path in paths:
            prev = None
            for t, close, omega, ia, ib, ua, ub in ar.ventanas(path, 5000):
                if prev is not None and t-prev[0] == 1:
                    dia, dib = ia-prev[1], ib-prev[2]
                    epk = (omega*ar.KE_MV_Q18) >> 18
                    accepted = 0
                    if abs(epk) >= ar.EPK_MIN_MV:
                        ea = ua - (((ia >> 4)*ar.RK_Q8) >> 12) - ((dia*ar.LK_Q15) >> 15)
                        eb = ub - (((ib >> 4)*ar.RK_Q8) >> 12) - ((dib*ar.LK_Q15) >> 15)
                        q.paso(ar.basis_recurrencia(ar.theta_centro(close, omega), 7), epk, ea, eb)
                        accepted = q.accepted
                        count += 1
                    self.assertEqual(self.c.adaline_update(ia, ib, dia, dib, ua, ub, close, omega), accepted)
                    np.testing.assert_array_equal(self.weights(), q.W)
                prev = t, ia, ib
        self.assertGreater(count, 100)

    def test_extreme_observer_inputs_bounded(self):
        self.c.adaline_init(0)
        for omega in (-2**31, -100000, 100000, 2**31-1):
            for v in (-2**31, 2**31-1):
                self.c.adaline_update(v, v, v, v, v, v, 0xFFFFFFFF, omega)
                w = self.weights()
                self.assertLessEqual(np.abs(w).sum(axis=0).max(), 4*ar.W_ONE)

    def test_rejected_update_does_not_publish_partial_weights(self):
        self.c.adaline_init(1)
        omega = 100000
        close = omega*7*25  # center = 0, all cosines positive
        rejected = False
        # Con μ=7 la norma L1 crece ~4e5/update con eps saturado; el límite
        # (4·W_ONE) se alcanza en ~1.3k updates. Antes con μ=131 bastaban 150.
        for _ in range(3000):
            before = self.weights()
            accepted = self.c.adaline_update(0, 0, 0, 0, 12000, -12000, close, omega)
            if not accepted:
                np.testing.assert_array_equal(before, self.weights())
                rejected = True
        self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
