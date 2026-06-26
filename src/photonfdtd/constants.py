"""Physical constants (SI units throughout)."""
import math

C_0 = 299_792_458.0          # vacuum speed of light, m/s
EPS_0 = 8.854_187_8128e-12   # vacuum permittivity, F/m
MU_0 = 1.256_637_062_12e-6   # vacuum permeability, H/m
ETA_0 = math.sqrt(MU_0 / EPS_0)   # vacuum impedance, ~376.73 ohm
Q_E = 1.602_176_634e-19      # elementary charge, C (exact, SI redefinition 2019)
