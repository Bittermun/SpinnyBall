import unittest
import numpy as np
from src.sgms_anchor_metabolism import simulate_fusion_metabolism

class TestSGMSAnchorMetabolism(unittest.TestCase):
    def test_compression_ratio_and_geometry_scaling(self):
        # 10x magnetic field ratio -> 100x magnetic compression kappa
        B_max = 20.0
        B_in = 2.0
        T0 = 2.0
        
        # Cylindrical (2D) geometry: kappa_V = sqrt(kappa) = 10
        # T_ion = T0 * kappa_V^(2/3) = 2.0 * 10^(2/3) = 9.28317766
        res_cyl = simulate_fusion_metabolism(
            B_max=B_max, B_in=B_in, T0_keV=T0,
            compression_geometry="cylindrical"
        )
        self.assertAlmostEqual(res_cyl['magnetic_compression_ratio'], 100.0, places=5)
        self.assertAlmostEqual(res_cyl['volumetric_compression_ratio'], 10.0, places=5)
        self.assertAlmostEqual(res_cyl['ion_temperature_keV'], 2.0 * (10.0 ** (2.0 / 3.0)), places=5)

        # Spherical (3D) geometry: kappa_V = kappa^0.75 = 100^0.75 = 31.6227766
        # T_ion = T0 * kappa_V^(2/3) = T0 * kappa^0.5 = 2.0 * 10 = 20.0
        res_sph = simulate_fusion_metabolism(
            B_max=B_max, B_in=B_in, T0_keV=T0,
            compression_geometry="spherical"
        )
        self.assertAlmostEqual(res_sph['magnetic_compression_ratio'], 100.0, places=5)
        self.assertAlmostEqual(res_sph['volumetric_compression_ratio'], 100.0 ** 0.75, places=5)
        self.assertAlmostEqual(res_sph['ion_temperature_keV'], 20.0, places=5)

    def test_bosch_hale_reactivity_limits(self):
        # Test that Bosch-Hale reactivity for D-T returns valid, positive, non-zero values
        # for standard fusion temperatures.
        for T in [1.0, 5.0, 10.0, 20.0, 50.0]:
            res = simulate_fusion_metabolism(
                B_max=20.0, B_in=2.0, T0_keV=T,
                compression_geometry="spherical"
            )
            sigma_v = res['reactivity_m3_s']
            self.assertGreater(sigma_v, 0.0)
            self.assertLess(sigma_v, 1e-14) # Reactivity should not exceed non-physical limits

    def test_stirling_cryocooler_cop_at_carnot_bound(self):
        # Stirling COP = efficiency * T_cryo / (300 - T_cryo)
        # At 77 K and efficiency 12%: 0.12 * 77 / 223 = 0.041435
        res = simulate_fusion_metabolism(
            B_max=20.0, B_in=2.0, cryocooler_efficiency=0.12, cryo_temp_K=77.0
        )
        expected_cop = 0.12 * (77.0 / (300.0 - 77.0))
        simulated_cop = res['cryocooler_efficiency'] * (77.0 / (300.0 - 77.0))
        self.assertAlmostEqual(simulated_cop, expected_cop, places=6)

        # Cryocooler power P_cryo = heat_leak_W / COP
        expected_power = 100.0 / expected_cop
        self.assertAlmostEqual(res['power_cryocooler_W'], expected_power, places=4)

    def test_power_density_deficit(self):
        # Verify that for T0 = 1.991553 keV and density = 2e20 m^-3, 
        # the ratio of spherical to cylindrical fusion power density is 51.7x
        B_max = 20.0
        B_in = 2.0
        T0 = 1.991553
        density = 2e20
        Ef = 2.818e-12
        
        res_sph = simulate_fusion_metabolism(
            B_max=B_max, B_in=B_in, T0_keV=T0, density_in_m3=density,
            compression_geometry="spherical"
        )
        res_cyl = simulate_fusion_metabolism(
            B_max=B_max, B_in=B_in, T0_keV=T0, density_in_m3=density,
            compression_geometry="cylindrical"
        )
        
        pd_sph = 0.25 * (res_sph['volumetric_compression_ratio'] * density)**2 * res_sph['reactivity_m3_s'] * Ef
        pd_cyl = 0.25 * (res_cyl['volumetric_compression_ratio'] * density)**2 * res_cyl['reactivity_m3_s'] * Ef
        
        ratio = pd_sph / pd_cyl
        self.assertAlmostEqual(ratio, 51.7, delta=0.01)

if __name__ == "__main__":
    unittest.main()
