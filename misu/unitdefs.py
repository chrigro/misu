

# Power

# Pressure
createUnit('atm atmosphere_standard', 101325 * Pa)
createUnit('atmosphere_technical  at ', 9.80665e4 * Pa)
createUnit('bar', 1e5 * Pa)
createUnit('barye_cgs_unit ', 0.1 * Pa)
createUnit('centimetre_of_mercury  cmHg ', 1.33322e3 * Pa)
createUnit('centimetre_of_water_4degC  cmH2O ', 98.0638 * Pa)
createUnit('foot_of_mercury_conventional  ftHg ', 40.63666e3 * Pa)
createUnit('foot_of_water_39_2_F  ftH2O ', 2.98898e3 * Pa)
createUnit('inch_of_mercury_conventional  inHg ', 3.386389e3 * Pa)
createUnit('inch_of_water_39_2_F  inH2O ', 249.082 * Pa)
createUnit('kilogram_force_per_square_millimetre ', 9.80665e6 * Pa)
createUnit('kip_per_square_inch  ksi ', 6.894757e6 * Pa)
createUnit('micron_micrometre_of_mercury  mHg ', 0.1333224 * Pa)
createUnit('mmHg millimetre_of_mercury', 133.3224 * Pa)
createUnit('millimetre_of_water_3_98_C  mmH2O ', 9.80638 * Pa)
createUnit('pz pieze_mts_unit  ', 1e3 * Pa)
createUnit('psf pound_per_square_foot', 47.88026 * Pa)
createUnit('psi pound_per_square_inch', 6.894757e3 * Pa)
createUnit('poundal_per_square_foot ', 1.488164 * Pa)
createUnit('short_ton_per_square_foot ', 95.760518e3 * Pa)
createUnit('torr', 133.3224 * Pa)

# Velocity
createUnit('metre_per_second_SI_unit ', 1 * m/s, unitCategory='Velocity')
metre_per_second_SI_unit.setRepresent(as_unit=metre_per_second_SI_unit, symbol='m/s')

createUnit('fph foot_per_hour', 8.466667e-5 * m/s)
createUnit('fpm foot_per_minute', 5.08e-3 * m/s)
createUnit('fps foot_per_second', 3.048e-1 * m/s)
createUnit('furlong_per_fortnight ', 1.663095e-4 * m/s)
createUnit('inch_per_hour  iph ', 7.05556e-6 * m/s)
createUnit('inch_per_minute  ipm ', 4.23333e-4 * m/s)
createUnit('inch_per_second  ips ', 2.54e-2 * m/s)
createUnit('kph kilometre_per_hour', 2.777778e-1 * m/s)
createUnit('kt kn knot', 0.514444 * m/s)
createUnit('knot_Admiralty', 0.514773 * m/s)
createUnit('M mach_number', 340 * m/s)
createUnit('mph mile_per_hour', 0.44704 * m/s)
createUnit('mile_per_minute  mpm ', 26.8224 * m/s)
createUnit('mile_per_second  mps ', 1609.344 * m/s)
createUnit('c speed_of_light_in_vacuum', 299792458 * m/s)
createUnit('speed_of_sound_in_air', 340 * m/s)

# Additional derived quantities
createUnit('kg_m3', kg/m3, unitCategory="Mass density")
kg_m3.setRepresent(as_unit=kg_m3, symbol='kg/m3')
#import pdb; pdb.set_trace()
createUnit('kg_hr', kg/hr, unitCategory="Mass flowrate")
kg_hr.setRepresent(as_unit=kg_hr, symbol='kg/hr')
createUnit('kmol_hr', kmol/hr, unitCategory="Molar flowrate")
kmol_hr.setRepresent(as_unit=kmol_hr, symbol='kmol/hr')
ncm = Ncm = (m3) * (101325*Pa) / (8.314*J/mol/K) / (273.15*K)
ncmh = Ncmh = ncm / hr
# NOTE 15 C:
scf = (ft) * (101325*Pa) / (8.314*J/mol/K) / ((273.15+15)*K)
scfm = scf / minute
scfd = scf / day
MMSCFD = scfd / 1e6
SP_OPEC = 101.560 * kPa
# NOTE! http://goldbook.iupac.org/S05910.html
SP_STP = 1e5 * Pa # http://goldbook.iupac.org/S06036.html
MMbbl = bbl / 1e6
MMscf = scf / 1e6
bcf = Bcf = scf / 1e9

# Engineering quantities
createUnit('kJ_kg_K', kJ/kg/K, unitCategory="Heat capacity mass")
kJ_kg_K.setRepresent(as_unit=kJ_kg_K, symbol='kJ/kg/K')
createUnit('kJ_kmol_K', kJ/kmol/K, unitCategory="Heat capacity mole")
kJ_kmol_K.setRepresent(as_unit=kJ_kmol_K, symbol='kJ/kmol/K')

createUnit('kJ_kg', kJ/kg, unitCategory="Specific enthalpy mass")
kJ_kg.setRepresent(as_unit=kJ_kg, symbol='kJ/kg')
createUnit('kJ_kmol', kJ/kmol, unitCategory="Specific enthalpy mole")
kJ_kmol.setRepresent(as_unit=kJ_kmol, symbol='kJ/kmol')


createUnit('W_m_K', W/m/K, unitCategory="Thermal conductivity")
W_m_K.setRepresent(as_unit=W_m_K, symbol='W/m/K')

createUnit('N_m', N/m, unitCategory="Surface tension")
N_m.setRepresent(as_unit=N_m, symbol='N/m')

createUnit('g_mol', g/mol, unitCategory="Molecular weight")
g_mol.setRepresent(as_unit=g_mol, symbol='g/mol')
