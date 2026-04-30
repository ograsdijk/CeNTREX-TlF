import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from centrex_tlf.lindblad.parameters import (
    Parameter, Time, linear, gaussian, pchip_tabulated, RuntimeExpression,
)

print("=== Basic expressions ===")
t = Time()
v = Parameter("v", 200.0)
z0 = Parameter("z0", -0.005)
z = linear(t, offset=z0, slope=v)

print(f"z = {z!r}")
print(f"z(t=0) = {z.evaluate(t=0)}")
print(f"z(t=25e-6) = {z.evaluate(t=25e-6)}")
print(f"z(t=25e-6, v=300) = {z.evaluate(t=25e-6, v=300.0)}")
assert abs(z.evaluate(t=0) - (-0.005)) < 1e-15
assert abs(z.evaluate(t=25e-6) - (-0.005 + 200.0 * 25e-6)) < 1e-15
assert abs(z.evaluate(t=25e-6, v=300.0) - (-0.005 + 300.0 * 25e-6)) < 1e-15
print("  PASSED")

print("\n=== Gaussian ===")
omega0 = Parameter("omega0", 2 * np.pi * 1e6)
z_laser = Parameter("z_laser", 0.002)
w0 = Parameter("w0", 200e-6)
Omega = gaussian(z, center=z_laser, sigma=w0, amplitude=omega0)
print(f"Omega = {Omega!r}")
val = Omega.evaluate(t=0)
expected = 2*np.pi*1e6 * np.exp(-(-0.005 - 0.002)**2 / (2 * (200e-6)**2))
print(f"Omega(t=0) = {val:.6e} (expected ~0, far from laser)")
print(f"Omega(t=35e-6) = {Omega.evaluate(t=35e-6):.6e} (near laser at z=0.002)")
print("  PASSED")

print("\n=== evaluate_array ===")
t_array = np.linspace(0, 50e-6, 100)
z_array = z.evaluate_array("t", t_array)
print(f"z_array shape: {z_array.shape}")
assert z_array.shape == (100,)
assert abs(z_array[0] - (-0.005)) < 1e-15
print(f"z(t=0) = {z_array[0]:.6f}")
print(f"z(t=50us) = {z_array[-1]:.6f}")

Omega_array = Omega.evaluate_array("t", t_array)
print(f"Omega_array shape: {Omega_array.shape}")
peak_idx = np.argmax(Omega_array)
peak_t = t_array[peak_idx]
peak_z = z0.default + v.default * peak_t
print(f"Omega peak at t={peak_t*1e6:.1f} us (z={peak_z*1000:.2f} mm)")
print(f"  (z_laser = {z_laser.default*1000:.2f} mm)")
print("  PASSED")

print("\n=== PCHIP tabulated ===")
z_grid = np.linspace(-0.01, 0.01, 20)
ez_profile = 10.0 + 5.0 * np.sin(np.pi * z_grid / 0.01)
gp = Parameter("z_grid", tuple(z_grid.tolist()))
ep = Parameter("ez_profile", tuple(ez_profile.tolist()))
Ez = pchip_tabulated(z, gp, ep)
print(f"Ez = {Ez!r}")
Ez_val = Ez.evaluate(t=25e-6)
print(f"Ez(t=25us) = {Ez_val:.4f} V/cm")

Ez_array = Ez.evaluate_array("t", t_array)
print(f"Ez_array shape: {Ez_array.shape}")
print(f"Ez range: {Ez_array.min():.2f} to {Ez_array.max():.2f} V/cm")
print("  PASSED")

print("\n=== __repr__ ===")
print(f"t: {t!r}")
print(f"z: {z!r}")
print(f"Omega: {Omega!r}")

print("\nAll tests passed!")
