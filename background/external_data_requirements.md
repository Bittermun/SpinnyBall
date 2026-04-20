# External Data Requirements for Production Deployment

## Critical Missing Data

To transition from research simulation to production system, the following external data must be obtained:

### 1. Superconductor Material Properties (CRITICAL)

**GdBCO (Gadolinium Barium Copper Oxide) Characterization:**
- [ ] Critical current density J_c(B, T) curves at 77K and varying magnetic fields
- [ ] Flux-pinning force density F_p vs magnetic field strength
- [ ] Thermal conductivity κ(T) in superconducting state
- [ ] Specific heat capacity C_p(T) near transition temperature
- [ ] Quench propagation velocity at operating conditions
- [ ] AC loss coefficients at high-frequency field variations

**Search Terms:**
```
"GdBCO critical current density 77K magnetic field"
"type-II superconductor flux pinning force GdBCO"
"REBCO coated conductor AC loss high frequency"
"superconductor quench propagation velocity measurement"
```

**Likely Sources:**
- SuperPower Inc. technical datasheets
- AMSC (American Superconductor) product specifications
- Papers from IEEE Transactions on Applied Superconductivity
- NIST superconductor database

---

### 2. Eddy Current Heating Parameters (HIGH PRIORITY)

**Required Coefficients:**
- [ ] Eddy current loss coefficient k_eddy for rotor/stator laminations
- [ ] Magnetic permeability μ_r of composite structural materials
- [ ] Electrical conductivity σ(T) of housing materials at cryogenic temperatures
- [ ] Lamination thickness optimization data
- [ ] Hysteresis loss coefficients for ferromagnetic components

**Formula Reference:**
```
P_eddy = k_eddy · B² · f² · t² · V
where:
  k_eddy = material-dependent coefficient
  B = magnetic flux density
  f = frequency of field variation
  t = lamination thickness
  V = volume of conducting material
```

**Search Terms:**
```
"eddy current loss coefficient cryogenic temperature"
"magnetic bearing eddy current heating calculation"
"lamination steel loss separation model"
"Steinmetz equation parameters cryogenic"
```

---

### 3. Cryocooler Performance Data (HIGH PRIORITY)

**Required Specifications:**
- [ ] Cooling power vs temperature curves for Stirling/GM cryocoolers at 70-90K range
- [ ] Input power consumption at various cooling loads
- [ ] Warm-up time constants during quench events
- [ ] Vibration characteristics (microphonics) affecting flux-pinning stability
- [ ] Mean time between failure (MTBF) for space-qualified units
- [ ] Mass and volume constraints for orbital deployment

**Candidate Systems to Research:**
- Thales Cryogenics LPT9310 series
- Sunpower CryoTel GT series
- Chart Industries (formerly Cryomech) PT415
- Sumitomo Heavy Industries RDK series

**Search Terms:**
```
"space qualified cryocooler 77K cooling power curve"
"Stirling cryocooler vibration microphonics superconductor"
"cryocooler specific power W/W at 80K"
```

---

### 4. SiC Power Electronics Characteristics (MEDIUM PRIORITY)

**Required Data:**
- [ ] Switching losses vs frequency for SiC MOSFETs at cryogenic temperatures
- [ ] Conduction losses at operating currents (100A-1000A range)
- [ ] Gate drive requirements at low temperature
- [ ] Thermal resistance junction-to-case for cryogenic mounting
- [ ] Radiation hardness data for lunar orbital environment

**Search Terms:**
```
"SiC MOSFET cryogenic operation switching loss"
"wide bandgap semiconductor low temperature characterization"
"space qualified SiC power module radiation hard"
```

---

### 5. Magnetic Bearing/Flux-Pinning Stiffness (MEDIUM PRIORITY)

**Required Measurements:**
- [ ] Transverse stiffness k_trans vs gap distance for GdBCO-magnet pairs
- [ ] Damping coefficients c_trans from flux-pinning hysteresis
- [ ] Temperature dependence of stiffness near T_c
- [ ] Long-term degradation from thermal cycling
- [ ] Load capacity limits before flux creep dominates

**Search Terms:**
```
"flux-pinning stiffness measurement GdBCO permanent magnet"
"superconducting magnetic bearing damping coefficient"
"flux creep rate high temperature superconductor bearing"
```

**Likely Sources:**
- ESA (European Space Agency) technology reports
- NASA Glenn Research Center superconducting bearing tests
- Journal of Physics: Conference Series (EUCAS proceedings)

---

### 6. Lunar Orbital Environment Parameters (MEDIUM PRIORITY)

**Required Environmental Data:**
- [ ] Geomagnetic field strength and gradient at 100-500 km lunar altitude
- [ ] Solar wind plasma density and velocity distributions
- [ ] Micrometeoroid flux vs particle size (for reliability modeling)
- [ ] Thermal environment: solar flux, albedo, planetary IR
- [ ] Atomic oxygen concentration (if any at lunar orbit)
- [ ] Radiation belt particle fluxes (protons, electrons)

**Search Terms:**
```
"lunar orbital magnetic field strength altitude profile"
"lunar exosphere plasma density solar wind interaction"
"micrometeoroid flux model lunar orbit MASTER-2009"
"thermal environment cis-lunar space solar constant"
```

**Likely Sources:**
- NASA LADEE mission data
- ARTEMIS mission measurements
- NASA SP-8027 (space environment standards)
- ECSS-E-ST-10-04C (space environment)

---

### 7. Structural Materials at Cryogenic Temperatures (LOW-MEDIUM PRIORITY)

**Required Properties:**
- [ ] Young's modulus E(T) for carbon fiber composites at 77K
- [ ] Coefficient of thermal expansion (CTE) mismatch stresses
- [ ] Fracture toughness K_IC at cryogenic temperatures
- [ ] Fatigue life under thermal cycling (77K ↔ 300K)
- [ ] Outgassing rates in vacuum for adhesive bonds

**Search Terms:**
```
"carbon fiber composite mechanical properties 77K"
"cryogenic thermal expansion coefficient mismatch stress"
"vacuum outgassing rate adhesive space application"
```

---

## Data Acquisition Strategy

### Phase 1: Literature Review (Week 1-2)
1. Search IEEE Xplore for superconductor characterization papers
2. Query NASA Technical Reports Server (NTRS)
3. Review ESA ESTEC technical notes
4. Contact manufacturers directly for datasheets

### Phase 2: Expert Consultation (Week 3-4)
1. Reach out to university superconductivity labs (MIT, Cambridge, Tokyo U)
2. Contact NASA Glenn Research Center superconducting systems group
3. Engage with industry (SuperPower, AMSC, Bruker HTS)
4. Post queries on ResearchGate for unpublished data

### Phase 3: Empirical Estimation (Week 5-6)
If exact data unavailable:
1. Use analogous material properties (YBCO → GdBCO scaling)
2. Apply physics-based models with conservative assumptions
3. Perform sensitivity analysis on uncertain parameters
4. Design margins for worst-case scenarios

---

## Minimum Viable Dataset for Initial Deployment

**Absolute Minimum to Proceed:**
1. ✅ GdBCO J_c(B) at 77K (single curve sufficient for initial model)
2. ✅ Cryocooler cooling power at 80K (one data point per candidate unit)
3. ✅ Eddy current loss estimate (order-of-magnitude from similar systems)
4. ✅ Lunar magnetic field strength (average value acceptable)

**Can Be Refined Later:**
- Full temperature-dependent curves
- Frequency-dependent loss separation
- Detailed vibration spectra
- Long-term degradation rates

---

## Recommended Contacts

### Academic Experts
- **Prof. David Larbalestier** (NHMFL, FSU) - Superconductor applications
- **Prof. Arno Godeke** (Lawrence Berkeley Lab) - HTS magnet systems
- **Dr. James Bray** (NASA Glenn) - Superconducting space systems

### Industry Contacts
- **SuperPower Inc.** - GdBCO wire manufacturing
- **Thales Cryogenics** - Space cryocoolers
- **Cryomagnetics** - Superconducting magnets and bearings

### Government Labs
- **NASA Glenn Research Center** - Superconducting systems for space
- **ESA ESTEC** - European space technology development
- **NIST** - Superconductor property measurements

---

*Document Status: Draft*  
*Last Updated: $(date)*  
*Action Required: User to initiate literature search and expert outreach*
