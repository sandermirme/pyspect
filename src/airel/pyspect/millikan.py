import math

import numpy as np


class Millikan:
    def __init__(self):
        self.A = 1.05
        self.B = 0.558
        self.C = 0.999
        self.free_path = 66.5
        self.k1 = 0.00472
        self.rad_mult = 1e9
        self.mob_mult = 10000.0

    def ktor(self, k):
        k1 = self.k1
        free_path = self.free_path
        A = self.A
        B = self.B
        C = self.C
        rad_mult = self.rad_mult

        kcm = k * self.mob_mult

        if isinstance(k, np.ndarray):
            dar = np.empty_like(k)
            for i in range(len(k)):
                a = 100
                b = a  # a on nmeetrit
                kcmi = kcm[i]
                while True:
                    b = 0.8 * a + 0.2 * b
                    a = k1 / kcmi * (1 + free_path / b * (A + B * math.exp(-b * C / free_path)))
                    if abs(b - a) < 0.001 * b:
                        break
                dar[i] = a / rad_mult
            return dar
        else:
            a = 100
            b = a  # a on nmeetrit
            while True:
                b = 0.8 * a + 0.2 * b
                a = k1 / kcm * (1 + free_path / b * (A + B * math.exp(-b * C / free_path)))
                if abs(b - a) < 0.001 * b:
                    break

            return a / rad_mult

    def ktod(self, k):
        return self.ktor(k) * 2

    def rtok(self, r):
        rnm = r * self.rad_mult
        k = self.k1 / rnm * (1 + self.free_path / rnm * (self.A + self.B * np.exp(
            -rnm * self.C / self.free_path)))  # //cm^2 /(V*s)   //=q*elaeng/6/pii/visk*Canparand;visk=1.81e-5 kg/m/s
        return k / self.mob_mult  # liikuvus on m/s V/m kohta

    def dtok(self, d):
        return self.rtok(d * 0.5)

    def dlogkdlogd(self, r):
        rnm = r * self.rad_mult
        val = -(rnm + 2 * self.free_path * self.A + 2 * self.free_path * self.B * np.exp(
            -rnm * self.C / self.free_path) + self.B * self.C * math.exp(-rnm * self.C / self.free_path) * rnm) / (
                          rnm * (rnm + self.free_path * self.A + self.free_path * self.B * math.exp(
                      -rnm * self.C / self.free_path))) * rnm
        return val
