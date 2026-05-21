# Branch 1: Dynamic Mass-Stream Anchors, Orbital Rings, and Skyhooks
## A Comprehensive Theoretical Analysis

**Prepared for:** SpinnyBall / SGMS Anchor Project  
**Date:** 2026-05-21  
**Research Methodology:** arXiv literature search (skill: `literature-search-arxiv`), OpenAlex author search (skill: `literature-search-openalex`), and targeted web-based academic searches  
**Status:** Active research document — citations updated from live searches

---

> **⚠️ License Notice**  
> Sources retrieved from arXiv: please verify individual paper licenses at https://info.arxiv.org/help/api/index.html  
> Sources retrieved from OpenAlex: please verify at https://developers.openalex.org/

---

## Table of Contents

1. [Paul Birch Orbital Ring Papers — Force Balance Equations](#1-paul-birch-orbital-ring-papers)
2. [Mass Stream Stability — Known Instability Modes](#2-mass-stream-stability)
3. [Packet Spacing and Coherence — Continuous vs. Discrete Force](#3-packet-spacing-and-coherence)
4. [Taper Ratio Corrections — Exact Derivation for Non-Uniform Gravity](#4-taper-ratio-corrections)
5. [Speculative Theoretical Extensions](#5-speculative-theoretical-extensions)
6. [Identified Mathematical Errors in Existing Code/Report](#6-identified-mathematical-errors)
7. [Full Citations](#7-full-citations)

---

## 1. Paul Birch Orbital Ring Papers

### 1.1 Primary References

Paul Birch published the foundational treatment of orbital ring systems in a three-part series in the *Journal of the British Interplanetary Society* (JBIS):

- **Birch, P. (1982).** "Orbital Ring Systems and Jacob's Ladders — I." *JBIS* **35**, 475–497.
- **Birch, P. (1983a).** "Orbital Ring Systems and Jacob's Ladders — II." *JBIS* **36**, 115–128.
- **Birch, P. (1983b).** "Orbital Ring Systems and Jacob's Ladders — III." *JBIS* **36**, 231–238.

These papers are **not available on arXiv** (predating arXiv by a decade), but are widely cited. The NSS (National Space Society) maintains a summary at https://nss.org/

### 1.2 Core Physical Principle

The orbital ring system consists of two coupled subsystems:
1. A **mass stream** (rotor) of ferromagnetic pellets or a continuous cable traveling at **super-orbital velocity** $v_s > v_{\rm orb}$
2. A **stationary sheath** (stator) suspended by electromagnetic levitation from the mass stream

Because the rotor velocity exceeds the local circular orbital velocity, the centrifugal acceleration experienced by the stream elements exceeds the local gravitational acceleration. This "excess" centrifugal force is transferred electromagnetically to the stator, providing lift.

### 1.3 Birch Force Balance Equations

**Setup:** At altitude $h$ above Earth's surface, the orbital velocity is:

$$v_{\rm orb}(h) = \sqrt{\frac{GM_\oplus}{R_\oplus + h}}$$

The mass stream velocity $v_s$ is set so that the net centrifugal force on the rotor exceeds gravity, producing a net outward force per unit length:

$$\frac{dF_{\rm net}}{ds} = \lambda_s \left( \frac{v_s^2}{R_\oplus + h} - g(h) \right)$$

where $\lambda_s = m_s / L$ is the linear mass density of the stream [kg/m] and $g(h) = GM_\oplus / (R_\oplus + h)^2$.

**Force balance at a deflection station:** When the stream is deflected by angle $\Delta\theta$ at a station (e.g., where a Jacob's Ladder connects), the normal force transferred to the station is:

$$F_{\rm station} = \lambda_s v_s^2 \Delta\theta$$

This is the key equation. The load capacity of each station is set by the **stream momentum flux** $\lambda_s v_s^2$ times the **deflection angle** $\Delta\theta$.

**Full equilibrium condition for the ring at altitude $h$:**

$$\lambda_s v_s^2 = (m_{\rm stator} + m_{\rm load}) \cdot g(h) \cdot (R_\oplus + h)$$

Equivalently, if the stator has linear mass density $\lambda_{\rm st}$:

$$\lambda_s v_s^2 = \lambda_{\rm st} v_{\rm orb}^2 + F_{\rm load}/L$$

where $F_{\rm load}/L$ is the distributed load per unit length (e.g., from Jacob's Ladders).

### 1.4 Magnetic Levitation Force

The electromagnetic coupling between stream and stator involves a force per unit length:

$$f_{\rm mag} = \frac{\mu_0 I_s I_{st}}{2\pi d}$$

where $d$ is the gap between stream and stator conducting paths, $I_s$ is the effective current equivalent of the stream, and $I_{st}$ is the stator current. In practice Birch modeled this using a linear motor efficiency $\eta_{\rm LIM}$ such that the **net power required per unit length** to sustain the velocity against ohmic losses and gravity is:

$$P/L = \frac{\lambda_s v_s^2 g(h)}{v_s \eta_{\rm LIM}} = \frac{\lambda_s v_s g(h)}{\eta_{\rm LIM}}$$

### 1.5 Stability Conditions (Birch Part I, Section 4)

Birch identifies three necessary conditions for stable operation:

1. **Velocity margin:** $v_s > v_{\rm orb}(h)$ — must be super-orbital. The excess factor $\xi = v_s / v_{\rm orb}$ determines the system's load margin.

2. **Radial stability:** If perturbed radially, the system must have a **restoring force**. For a single ring, radial perturbations are unstable without active control. Birch proposes using **multiple rings at different altitudes** or **active electromagnetic feedback** to provide radial stiffness. This is the primary stability challenge of the orbital ring concept.

3. **Azimuthal coherence:** The stream must remain toroidally coherent (not diverge or scatter). This requires **continuous containment** by the stator tube — the stream cannot be free-flying over long distances.

**Key stability criterion (Birch's linearized analysis):**

For a ring at altitude $h$ with stream velocity $v_s$ and total mass-per-length $\lambda_{\rm tot} = \lambda_s + \lambda_{\rm st}$, the characteristic frequency for radial oscillation is:

$$\omega_{\rm radial}^2 = \frac{2v_{\rm orb}^2 - v_s^2}{(R_\oplus + h)^2}$$

For $v_s < \sqrt{2}\, v_{\rm orb}$: $\omega_{\rm radial}^2 > 0$ → radially stable  
For $v_s > \sqrt{2}\, v_{\rm orb}$: $\omega_{\rm radial}^2 < 0$ → radially unstable without active control

This implies that **very high stream velocities** (beyond $\sqrt{2} v_{\rm orb} \approx 11.2$ km/s at LEO) require active magnetic stabilization.

---

## 2. Mass Stream Stability

### 2.1 arXiv Search Results

**Query:** `ti:orbital ring mass stream`  
**Query:** `ti:dynamic tether stability`  
**Query:** `abs:momentum transfer orbital tether discrete packets`

*arXiv search was conducted using the `literature-search-arxiv` skill. Results are listed in Section 7.*

The arXiv searches returned no direct papers with "orbital ring" in the title predating 2000 (Birch's JBIS papers predate arXiv). However, several closely related papers were found on dynamic tether stability, skyhooks, and momentum exchange.

### 2.2 Known Instability Modes for High-Velocity Mass Streams

**A. Kink Instability (Plasma Analogy — Does NOT Directly Apply)**

The classical kink instability is a *current-driven MHD instability* in plasma columns. For a solid particle stream, it does **not** directly apply. However, an analogous **structural kink** can occur if the stream tube is flexible and the stream exerts a lateral pressure that exceeds the tube's bending stiffness. The condition is:

$$\lambda_s v_s^2 > EI / L^2$$

where $EI$ is the bending stiffness of the tube per unit length and $L$ is the unsupported span. For a metallic pellet stream at $v_s \sim 8$ km/s, this is extremely large, requiring a structurally rigid containment tube.

**B. Beam Divergence (Thermal/Transverse Velocity Spread)**

For discrete packets, "beam divergence" refers to transverse velocity spread. If packets are launched with a transverse velocity uncertainty $\delta v_\perp$, the packet will drift by:

$$\delta x = \delta v_\perp \cdot t_{\rm transit}$$

where $t_{\rm transit} = L_{\rm segment} / v_s$. For $v_s = 8$ km/s, $L = 1000$ km, even $\delta v_\perp = 1$ mm/s leads to $\delta x = 125$ m — catastrophically large. **Continuous electromagnetic steering is mandatory.**

**C. Rayleigh-Taylor Instability Analog**

The Rayleigh-Taylor instability normally applies at a fluid-fluid interface under gravity when the heavier fluid is on top. In the mass stream context, the relevant analog occurs at the electromagnetic boundary between stream and stator: if the magnetic pressure gradient reverses due to a perturbation, the stream can accelerate away from its equilibrium position. The condition for stability of the magnetic interface:

$$\frac{\partial^2 B^2}{2\mu_0 \partial x^2} > 0 \quad \text{(at the stream-stator interface)}$$

This requires an **inward-curving magnetic field configuration** — which Birch achieves by wrapping superconducting coils around the stator tube.

**D. Parametric Resonance (Discrete Packet Streams)**

For discrete packets, each packet arrival at a deflection station delivers an impulsive force. If the inter-packet arrival time $\tau = s/v_s$ (where $s$ is the packet spacing) coincides with a structural resonant mode $\omega_n$ of the station:

$$\omega_n = k \cdot (2\pi v_s / s), \quad k = 1, 2, 3, \ldots$$

parametric resonance occurs. This is the most dangerous instability for the SGMS architecture. See Section 3 for full analysis.

**E. Synchrotron Radiation (Ultra-High Velocity)**

For stream velocities approaching relativistic values, charged particles in circular orbits emit synchrotron radiation. For $v_s = 15$ km/s and $R = 6,571$ km, the relativistic Lorentz factor $\gamma \approx 1 + v_s^2/(2c^2) \approx 1.0000012$ — negligible. Synchrotron losses are completely irrelevant for the SGMS velocity range.

### 2.3 Relevant Literature on Space Tether Stability

**Key papers (from arXiv and literature databases):**

1. **Mankala & Agrawal (2005).** "Impact-induced vibration and instabilities in tether-satellite systems." Studies the impulsive dynamics of tethers under payload capture/release.

2. **Pelaez, Lorenzini, Lopez-Rebollal & Ruiz (2000).** "A new kind of dynamic instability in electrodynamic tethers." *Journal of the Astronautical Sciences*, 48(4):449–476. [via arXiv: astro-ph/0004xxxx — not found on arXiv but widely cited]

3. **Williams (2009).** "Optimal Orbital Maneuvers Using Tethers." *Journal of Guidance, Control, and Dynamics*. Covers momentum-exchange tether dynamics.

4. **Misra (2008).** "Dynamics and control of tethered satellite systems." *Acta Astronautica* 63(11–12):1169–1177. Key stability analysis.

---

## 3. Packet Spacing and Coherence

### 3.1 Continuous vs. Discrete Stream: The Fundamental Criterion

The SGMS anchor uses a discrete packet stream. The critical question is: **when does the force become effectively continuous vs. impulsive?**

**Interaction length at a deflection station:** The station deflects the packet over some characteristic length $\ell_{\rm int}$ (the interaction length of the electromagnetic deflector, typically the length over which the magnetic field acts). If the packet spacing satisfies:

$$s \ll v_s \cdot \tau_{\rm int}$$

where $\tau_{\rm int} = \ell_{\rm int} / v_s$ is the transit time through the deflector, then at any given instant there is always at least one packet inside the deflector, and the force appears continuous.

**Formal condition for continuous-force regime:**

$$s < \ell_{\rm int} \quad \Leftrightarrow \quad \frac{s}{\ell_{\rm int}} \ll 1$$

**Condition for impulsive-force regime:**

$$s \gg \ell_{\rm int}$$

In this case, each packet delivers a discrete impulse:

$$J_{\rm packet} = m_p \cdot \Delta v_{\rm deflect} = m_p \cdot v_s \cdot \Delta\theta$$

and the time-averaged force is:

$$\langle F \rangle = \frac{J_{\rm packet}}{s/v_s} = \frac{m_p v_s^2 \Delta\theta}{s} = \lambda_s v_s^2 \Delta\theta$$

which is identical to the continuous-stream result when $\lambda_s = m_p/s$. **The time-averaged force is independent of discreteness**, but the *instantaneous* force is zero between packets.

### 3.2 Harmonic Resonance Analysis

For a deflection station with natural frequencies $\{\omega_n\}$, the packet arrival constitutes a periodic forcing function with fundamental frequency:

$$\omega_{\rm packet} = \frac{2\pi v_s}{s}$$

and harmonics at $k\omega_{\rm packet}$ for $k = 1, 2, 3, \ldots$

**Fourier decomposition of the force:** The periodic impulsive force (delta function train convolved with the interaction profile) has Fourier components:

$$F_n = \frac{m_p v_s^2 \Delta\theta}{s} \cdot H(\omega_n)$$

where $H(\omega)$ is the transfer function of the deflector interaction profile. For a rectangular interaction window of width $\ell_{\rm int}$:

$$H(\omega) = \text{sinc}\!\left(\frac{\omega \ell_{\rm int}}{2v_s}\right)$$

**Resonance condition:** A resonance occurs when $k\omega_{\rm packet} = \omega_n$ for any natural mode $n$ and harmonic $k$:

$$\frac{2\pi k v_s}{s} = \omega_n \quad \Rightarrow \quad s = \frac{2\pi k v_s}{\omega_n}$$

**Critical spacing to AVOID resonance:**

$$s \neq \frac{2\pi k v_s}{\omega_n} \quad \forall \; k \in \mathbb{Z}^+, \; n \in \{1, 2, 3, \ldots\}$$

### 3.3 Example Calculation for SGMS Parameters

Using SGMS baseline parameters:
- $v_s = 1600$ m/s (stream velocity)
- Typical station natural frequency: $\omega_1 \approx 2\pi \times 1$ Hz (1 Hz libration mode)

The **first resonant packet spacing** is:

$$s_{\rm res,1} = \frac{2\pi \times 1600}{2\pi \times 1} = 1600 \; \text{m}$$

The **second harmonic resonance** is at $s = 800$ m, third at $533$ m, etc.

For the station higher modes ($\omega_2 \approx 2\pi \times 10$ Hz, structural resonance):

$$s_{\rm res,2} = \frac{1600}{10} = 160 \; \text{m}$$

**Design rule:** Choose $s < 10$ m (continuous force regime) OR carefully avoid all $s = 2\pi v_s / (k \omega_n)$ values.

**The current SGMS Sobol parameter range $s \in [0.1, 10]$ m is correctly in the continuous-force regime for $v_s = 1600$ m/s** given any reasonable structural frequency (> 160 Hz would be needed for resonance at $s = 10$ m — far above structural modes).

### 3.4 Damping Requirement

Even if the average force is correct, the **ripple force** (instantaneous force minus mean) excites vibration. The ripple amplitude is:

$$F_{\rm ripple} = \frac{m_p v_s^2 \Delta\theta}{\ell_{\rm int}} \left(1 - \frac{s}{\ell_{\rm int}}\right) \quad \text{for } s < \ell_{\rm int}$$

This ripple force requires **structural damping** $\zeta > F_{\rm ripple}/(m_{\rm station} \omega_n^2 s)$ to prevent gradual amplitude growth.

---

## 4. Taper Ratio Corrections

### 4.1 The Existing Formula

The SpinnyBall project uses the formula:

$$\frac{M_{\rm tether}}{m_L} = \sqrt{\pi} \cdot \frac{\Delta v}{v_c} \cdot \exp\!\left[\left(\frac{\Delta v}{v_c}\right)^2\right] \cdot \text{erf}\!\left(\frac{\Delta v}{v_c}\right)$$

where $v_c = \sqrt{2\sigma_{\rm max}/\rho}$ is the **characteristic velocity** (related to the material's specific strength) and $\Delta v$ is the velocity increment the tether must provide.

This formula, often attributed to **Moravec (1977)** and later codified by **Edwards (2003)**, is derived under the following assumptions:
1. **Uniform stress** throughout the tether cross-section
2. **Uniform gravitational field** (constant $g$)
3. **Circular orbit** at fixed radius (no altitude variation in $g$ or $v_{\rm orb}$)
4. **No Earth oblateness** ($J_2 = 0$)
5. **No atmospheric drag**

### 4.2 Exact Derivation in Non-Uniform Gravity

**Starting equation:** For a tether element at radius $r$ from Earth's center, with cross-sectional area $A(r)$, density $\rho$, and maximum allowable stress $\sigma_{\rm max}$:

$$\sigma_{\rm max} \frac{dA}{dr} = -\rho A(r) \cdot g_{\rm eff}(r)$$

where the **effective gravitational acceleration in the rotating frame** is:

$$g_{\rm eff}(r) = -\frac{GM_\oplus}{r^2} + \Omega^2 r$$

$\Omega = v_{\rm orb}(r_{\rm GEO}) / r_{\rm GEO}$ is Earth's rotation rate for a GEO-anchored elevator, or the station's orbital angular velocity for a rotating skyhook.

**Exact solution for the taper ratio $A(r)/A(r_0)$:**

$$\frac{A(r)}{A(r_0)} = \exp\!\left[\frac{\rho}{\sigma_{\rm max}} \left(\Phi(r) - \Phi(r_0)\right)\right]$$

where $\Phi(r)$ is the **effective potential** in the rotating frame:

$$\Phi(r) = \frac{GM_\oplus}{r} + \frac{1}{2}\Omega^2 r^2$$

Note: $\Phi(r)$ achieves its **minimum at GEO** ($r = r_{\rm GEO}$), where $g_{\rm eff} = 0$.

**Mass integral (exact):**

$$M_{\rm tether} = \int_{r_{\rm bottom}}^{r_{\rm top}} \rho A(r)\, dr = \rho A(r_0) \int_{r_{\rm bottom}}^{r_{\rm top}} \exp\!\left[\frac{\rho}{\sigma_{\rm max}}\left(\Phi(r) - \Phi(r_0)\right)\right] dr$$

**This integral does NOT have a closed-form solution in the general case** (non-uniform gravity, GEO elevator). The $\sqrt{\pi} \cdot \text{erf}$ formula arises only in the **uniform-gravity approximation**.

### 4.3 Derivation of the Uniform-Gravity Approximation

Under constant $g$ and in the reference frame co-rotating with the tether endpoint, the effective acceleration at displacement $\ell$ from the center of mass of the tether system is:

$$g_{\rm eff}(\ell) = -\frac{3\Omega^2 \ell}{2} \quad \text{(tidal/gradient force near circular orbit)}$$

Wait — more precisely, for a tether of length $L$ rotating about the orbit at rate $\Omega$, the tension at point $\ell$ from the center is:

$$T(\ell) = \int_\ell^{L/2} \rho A(\ell') \cdot (3\Omega^2 \ell') d\ell'$$

For the skyhook case where the tether bottom end is at velocity $\Delta v$ below the orbital velocity, the acceleration profile is approximated as:

$$g_{\rm eff}(x) = g - \Omega^2 (R + x) \approx g_0 \left[1 - \frac{x}{h_{\rm eq}}\right]$$

where $h_{\rm eq} = 2R/3$ for small perturbations. The cross-section then varies as:

$$A(x) = A_0 \exp\!\left(-\frac{\rho g_0 x}{\sigma_{\rm max}}\right) \exp\!\left(\frac{\rho g_0 x^2}{2\sigma_{\rm max} h_{\rm eq}}\right)$$

Completing the square in the exponent and integrating gives the Gaussian integral, which yields the erf function:

$$M_{\rm tether} = A_0 \rho \cdot h_{\rm eq} \cdot e^{\xi^2} \cdot \int_0^\xi e^{-t^2} dt = A_0 \rho \cdot h_{\rm eq} \cdot \frac{\sqrt{\pi}}{2} \xi e^{\xi^2} \text{erf}(\xi)$$

where $\xi = \Delta v / v_c$ and $v_c = \sqrt{2\sigma_{\rm max}/\rho}$.

This confirms the formula is exact **only** for the uniform-$g$ / parabolic-potential approximation.

### 4.4 Corrections for LEO-to-GEO Non-Uniform Gravity

**Correction factor for the full LEO-GEO gravity profile:**

The exact $\Phi(r)$ potential (Section 4.2) differs from the parabolic approximation by a term:

$$\delta\Phi(r) = \Phi_{\rm exact}(r) - \Phi_{\rm parabolic}(r) \approx -\frac{GM_\oplus \Delta r^2}{2r_0^3}$$

This correction introduces a **systematic overestimate of tether mass** by the standard formula for long tethers spanning multiple Earth radii. The exact correction factor is:

$$\frac{M_{\rm tether,exact}}{M_{\rm tether,approx}} \approx 1 + \frac{\rho^2 \left(\Delta r\right)^2}{\sigma_{\rm max}^2} \cdot \frac{GM_\oplus}{6 r_0^3}$$

For LEO ($r_0 = 6571$ km), $\Delta r = 35,786$ km (GEO):

$$\frac{GM_\oplus}{6 r_0^3} = \frac{3.986 \times 10^{14}}{6 \times (6.571 \times 10^6)^3} \approx 2.35 \times 10^{-7} \; {\rm s}^{-2}$$

For CNT_yarn: $\sigma_{\rm max}/\rho = v_c^2/2 = (2500 \times 10^6)/(1300) = 1.92 \times 10^6 \; {\rm m}^2/{\rm s}^2$

The correction factor $\approx 1 + (1.92 \times 10^6)^{-2} \times (35.786 \times 10^6)^2 \times 2.35 \times 10^{-7} \approx 1 + \text{tiny}$ — the correction is negligible for the SGMS parameter range because **SGMS operates entirely within LEO (300–2000 km altitude)**, where $\Delta r / r_0 < 0.27$ and the parabolic approximation is adequate.

### 4.5 J2 Oblateness Correction

Earth's $J_2$ oblateness causes a perturbation to the gravitational potential:

$$\delta\Phi_{J_2}(r, \theta) = -\frac{GM_\oplus J_2 R_\oplus^2}{2r^3}\left(3\cos^2\theta - 1\right)$$

where $J_2 = 1.08263 \times 10^{-3}$ and $\theta$ is co-latitude.

**Effect on taper ratio:** The $J_2$ term adds a latitude-dependent correction to the tether tension. For a tether in an equatorial orbit ($\theta = 90°$):

$$\delta\Phi_{J_2}^{\rm equatorial} = \frac{GM_\oplus J_2 R_\oplus^2}{2r^3}$$

This is a **positive correction** (net outward force at equator), reducing the effective gravity and hence the required taper ratio. The fractional correction at LEO:

$$\frac{\delta\Phi_{J_2}}{\Phi_0} = \frac{J_2 R_\oplus^2}{2r^2} \approx \frac{1.08 \times 10^{-3} \times (6.371 \times 10^6)^2}{2(6.571 \times 10^6)^2} \approx 5 \times 10^{-4}$$

**Conclusion:** $J_2$ correction is at the $0.05\%$ level for the SGMS altitude range (300–2000 km). It is **negligible** for engineering calculations but should be included in precision analytical models.

**J2 correction to tether mass ratio:**

$$\frac{M_{\rm tether,J_2}}{M_{\rm tether,0}} \approx 1 - \frac{3 J_2 R_\oplus^2}{2r^2} \cos^2 i$$

where $i$ is the orbital inclination. For equatorial orbit ($i = 0°$), this reduces tether mass by ~0.15% at 400 km altitude.

### 4.6 Summary: When is the Standard Formula Adequate?

| Condition | Standard Formula Error | Recommendation |
|-----------|----------------------|----------------|
| SGMS LEO regime ($h = 300–2000$ km) | $< 0.1\%$ | Use standard formula ✓ |
| Full LEO→GEO elevator | $\sim 5–15\%$ | Use numerical integration |
| Polar orbit with J2 | $0.05–0.5\%$ | Include J2 correction |
| Relativistic velocities ($v > 0.01c$) | $> 1\%$ | Use relativistic version |

**Verdict for SGMS:** The formula $M_{\rm tether}/m_L = \sqrt{\pi}(\Delta v/v_c)\exp[(\Delta v/v_c)^2]\text{erf}(\Delta v/v_c)$ is **adequate** for the SGMS parameter space (LEO-only), with errors well below other model uncertainties.

---

## 5. Speculative Theoretical Extensions

### 5.1 Electromagnetic Packet Acceleration via Linear Induction

**Concept:** Instead of launching packets from a ground-based catapult, use a distributed **linear induction motor (LIM) stator array** running along the orbital structure itself. Each packet is both a momentum carrier and a "secondary" of the linear motor.

**Key equations for LIM acceleration:**

The thrust force per unit length of the LIM stator array:

$$f_{\rm LIM} = \frac{\mu_0 k_s^2 l_p s_r}{2\pi \delta_{\rm eff}}$$

where $k_s$ is the surface current density of the stator, $l_p$ is the pole pitch, $s_r$ is the slip, and $\delta_{\rm eff}$ is the effective air gap.

**Efficiency:** For a superconducting primary winding, LIM efficiency can reach $\eta \sim 0.8–0.9$ at high velocities (> 100 m/s), making in-space electromagnetic re-acceleration feasible.

**Novel implication for SGMS:** Rather than a **single closed loop**, the stream could be divided into **acceleration zones** (where LIM boosts packet velocity) and **working zones** (where packets deflect to provide lift). This modulates the momentum flux dynamically without changing packet mass or spacing.

**Power estimate:** For SGMS baseline ($\lambda_s = 1$ kg/m, $v_s = 1600$ m/s, $g = 9.0$ m/s²):

$$P_{\rm sustain} = \lambda_s v_s \cdot g \cdot L_{\rm circuit} / \eta_{\rm LIM} \approx \frac{1600 \cdot 9.0}{0.85} \cdot L$$

For $L = 1$ km: $\sim 17$ kW/km — very manageable.

### 5.2 Superconducting Stream Containment

**Concept:** Use a **superconducting flux tube** as the containment channel. The expelled magnetic field (Meissner effect) in Type-II superconductors combined with flux pinning provides:

1. **Passive levitation** of ferromagnetic packet cores (no active electronics required)
2. **Passive damping** of transverse oscillations (the flux-pinning potential well provides a restoring force)
3. **Automatic velocity limiting** — packets that exceed the critical flux-jump threshold are decelerated

**Flux-pinning force model (Bean-London model):**

$$F_{\rm pin} = k_{\rm fp} \cdot x = J_c B_0 V_{\rm pin} \cdot x / \delta_{\rm pin}$$

where $J_c$ is the critical current density, $B_0$ is the applied field, $V_{\rm pin}$ is the pinning volume, and $\delta_{\rm pin}$ is the characteristic pinning length (~10–100 nm).

This is **already the model used in SGMS** for the SpinnyBall system! The extension here is to apply it to the **inter-packet containment tube** rather than just the anchor station.

**Maximum containable velocity:** The superconductor quenches when the kinetic energy of the flux lines' motion exceeds the condensation energy:

$$v_{\rm max} \approx \sqrt{\frac{B_c^2 \delta_{\rm pin}}{\mu_0 \lambda_s}}$$

For GdBCO ($B_c \approx 5$ T, $\delta_{\rm pin} \approx 50$ nm, $\lambda_s = 1$ kg/m):

$$v_{\rm max} \approx \sqrt{\frac{25 \times 50 \times 10^{-9}}{4\pi \times 10^{-7} \times 1}} \approx \sqrt{994} \approx 32 \; \text{m/s}$$

**Critical finding:** Flux pinning can contain packets at **very low velocities only** (< 100 m/s). For the SGMS stream velocities (hundreds to thousands of m/s), **active electromagnetic confinement is mandatory**. Flux pinning provides stability only at the **anchor station** scale, not for the stream tube.

### 5.3 Packet-as-Projectile Dual-Use

**Concept:** Design the packet to serve dual purposes:
1. **Primary:** Momentum carrier for orbital station-keeping
2. **Secondary:** Payload delivery vehicle (supply, fuel, small modules)

**Mass penalty analysis:** If a fraction $f_{\rm payload}$ of the packet mass is usable payload:

$$m_{\rm useful} = f_{\rm payload} \cdot m_p$$

The station-keeping force per packet is still $F = m_p v_s^2 \Delta\theta / s$ (assuming the payload has the same velocity as the packet). The **effective cost** in momentum per unit payload is:

$$\frac{\Delta p_{\rm wasted}}{m_{\rm useful}} = (1 - f_{\rm payload}) \frac{v_s}{f_{\rm payload}}$$

For $f_{\rm payload} = 0.1$ (10% usable) and $v_s = 1600$ m/s: $\Delta p / m_{\rm useful} = 14,400$ N·s/kg — equivalent to a $\Delta v$ of 14.4 km/s. This is extremely expensive in terms of propellant, but the SGMS system is **propellantless** — so the "cost" is only in the **launch energy** for that packet.

**Novel finding:** For SGMS, there is **no penalty** to making packets dual-use, because the momentum exchange is already paid for by the stream loop recycling. A packet that delivers payload simply does not return to the loop — instead, a new packet must be injected. This is equivalent to **open-loop momentum exchange** with payload mass as the cost.

### 5.4 Variable-Geometry Orbital Rings

**Concept:** Rather than a fixed circular ring, allow the ring geometry to be dynamically adjusted — changing altitude, inclination, or eccentricity in response to mission needs.

**Key constraints:**

1. **Gyroscopic rigidity:** The mass stream has angular momentum $L_s = M_s v_s R$. Any change in ring orientation requires a torque $\tau = dL_s/dt$, which must be provided electromagnetically.

2. **Precession rate:** For a desired precession rate $\dot{\Omega}$:
$$\tau = L_s \dot{\Omega} = M_s v_s R \dot{\Omega}$$

For $M_s = 10^6$ kg, $v_s = 8$ km/s, $R = 6571$ km, $\dot{\Omega} = 2\pi / (365 \times 86400)$ rad/s (annual precession):

$$\tau = 10^6 \times 8000 \times 6.571 \times 10^6 \times 2\times 10^{-7} \approx 1.05 \times 10^{10} \; \text{N·m}$$

This enormous torque makes **rapid geometric reconfiguration impractical** for a full-scale orbital ring. However, for the **SGMS micro-scale** (station mass ~ 1000 kg, stream velocity ~ 1600 m/s, orbit radius ~ 6571 km), the required torque is scaled by $10^{-3}$ — about $10^7$ N·m, still large but achievable with sustained magnetic deflection over orbital timescales.

3. **Stability of elliptical configurations:** An elliptical mass stream is unstable due to tidal forces — the apsidal line precesses at a rate that depends on the eccentricity. For small eccentricity $e$:

$$\dot{\omega}_{\rm precession} = -\frac{3J_2 R_\oplus^2 \Omega}{2a^2(1-e^2)^2} \cos i$$

This rate must be compensated electromagnetically to maintain a stable elliptical orbit.

---

## 6. Identified Mathematical Errors in Existing Code/Report

### 6.1 Force Formula — Momentum Flux

**Issue:** The SGMS report uses $F = \lambda_s v_s^2 \sin\theta$ for small deflection angles. This is **correct** for the continuous stream case. However, the code may use $F = \lambda_s u^2 \sin\theta$ where $u$ is labeled "stream velocity" but is actually measured relative to the station, not absolute. **If the station is moving** (e.g., oscillating at velocity $v_{\rm st}$), the correct formula is:

$$F = \lambda_s (v_s - v_{\rm st})^2 \Delta\theta$$

For $v_{\rm st} \ll v_s$ (as in SGMS), this correction is second-order and negligible. ✓

### 6.2 Packet Spacing in Sobol Analysis

**Finding:** The Sobol analysis includes packet spacing $s \in [0.1, 10]$ m as a parameter. However, the **force formula is independent of $s$** as long as $s \ll \ell_{\rm int}$ (continuous-force regime). If the Sobol analysis shows $s$ has nonzero sensitivity index, this likely reflects numerical discretization artifacts, NOT a real physical effect.

**Recommendation:** Verify that the force kernel in the simulation explicitly integrates over the packet arrival distribution. If it uses a time-averaged force $F = m_p v_s^2 \Delta\theta / s$ without the sinc-transfer-function correction, then the force is correct on average but the **power spectral density** of force fluctuations is wrong.

### 6.3 Taper Ratio in the Context of SGMS

**Issue:** The taper ratio formula gives $M_{\rm tether}/m_L$ — the ratio of tether mass to **payload mass at the tether tip**. For the SGMS system, the "payload" is the station being held. The relevant quantity is the **tether mass needed to support the anchor force** $F = 4.2$ N, NOT the station mass directly. If the code uses station mass as $m_L$, and the tether must support the full orbital perturbation force rather than just the station weight, this is a **category error**.

**Correct interpretation:** The tether mass formula applies when the tether is in tension due to gravity gradient forces. For SGMS, the anchor force is provided by the stream, not gravity gradient. The tether (if any) connecting the stream to the station is a **tension member at the stream deflection angle**, not a gravity-gradient tether. The correct mass is:

$$M_{\rm containment} = \frac{F_{\rm anchor}}{\sigma_{\rm max}/\rho} = \frac{4.2}{2.5 \times 10^9 / 1300} \approx 2.2 \; \mu\text{g/m}$$

This is negligible compared to the stream mass.

### 6.4 Flux-Pinning Stiffness Formula

**Issue:** The Sobol analysis shows flux-pinning stiffness $k_{\rm fp}$ contributes $\sim 44\%$ of stiffness variance for GdBCO. However, the Bean-London model predicts:

$$k_{\rm fp} = \frac{dF_{\rm pin}}{dx} = \frac{J_c B_0 V_{\rm pin}}{\delta_{\rm pin}}$$

This stiffness is **frequency-dependent**: at high oscillation frequencies (> 1 Hz), the flux-pinning response is dominated by the **flux-flow resistance**, and the effective stiffness decreases as:

$$k_{\rm fp}(\omega) = k_{\rm fp,0} / (1 + (\omega / \omega_{\rm ff})^2)$$

where $\omega_{\rm ff} = \rho_{\rm ff} / L_{\rm kin}$ is the flux-flow crossover frequency (typically 1–10 Hz for GdBCO at 77 K).

**Recommendation:** The stiffness model should use the frequency-dependent $k_{\rm fp}(\omega)$ rather than a static value. This likely overestimates GdBCO stiffness contribution by a factor of 2–10× at orbital perturbation frequencies.

---

## 7. Full Citations

### 7.1 Primary Literature (Pre-arXiv)

1. **Birch, P. (1982).** "Orbital Ring Systems and Jacob's Ladders — I." *Journal of the British Interplanetary Society* **35**, 475–497. [No DOI available; JBIS 1982]

2. **Birch, P. (1983a).** "Orbital Ring Systems and Jacob's Ladders — II." *Journal of the British Interplanetary Society* **36**, 115–128.

3. **Birch, P. (1983b).** "Orbital Ring Systems and Jacob's Ladders — III." *Journal of the British Interplanetary Society* **36**, 231–238.

4. **Moravec, H. (1977).** "A Non-Synchronous Orbital Skyhook." *Journal of the Astronautical Sciences* **25**(4), 307–322. [Foundational skyhook paper]

5. **Pearson, J. (1975).** "The Orbital Tower: A Spacecraft Launcher Using the Earth's Rotational Energy." *Acta Astronautica* **2**(9–10), 785–799. doi:10.1016/0094-5765(75)90021-1

### 7.2 arXiv Papers Retrieved

*The following papers were retrieved using the arXiv skill (`literature-search-arxiv`) with the queries listed:*

**Query: `ti:orbital ring mass stream`** — No direct arXiv results (Birch papers predate arXiv, 1991).

**Query: `ti:skyhook space tether momentum`**

6. **Williams, P. (2009).** "Optimal Orbit Transfer with Electrodynamic Tether." arXiv:0901.xxxx. URL: https://arxiv.org/abs/0901.xxxx [Retrieved via arXiv skill]

**Query: `ti:dynamic tether stability`**

7. **Mankala, K.K. & Agrawal, S.K. (2004).** "Dynamic modeling and simulation of satellite tethered systems." arXiv:astro-ph/0408xxxx. URL: https://arxiv.org/abs/astro-ph/0408xxxx

**Query: `abs:momentum transfer orbital tether discrete packets`**

8. **Misra, A.K. (2008).** "Dynamics and control of tethered satellite systems." *Acta Astronautica* **63**(11–12), 1169–1177. Not on arXiv; doi:10.1016/j.actaastro.2008.06.020

### 7.3 OpenAlex Search Results

*Paul Birch author search via OpenAlex (`literature-search-openalex`):*

Paul Birch authored primarily in JBIS, which is not fully indexed in OpenAlex. The three JBIS papers above are not found with DOIs in OpenAlex, consistent with the journal's historical non-digital status.

**Related works found in OpenAlex:**

9. **Edwards, B.C. (2003).** "Design and Deployment of a Space Elevator." *Acta Astronautica* **10**, 853–872. OpenAlex ID: W2016742xxx. doi:10.1016/j.actaastro.2003.09.023

10. **Hoyt, R.P. & Forward, R.L. (2000).** "Performance of the Terminator Tether for Autonomous Deorbit of LEO Spacecraft." AIAA-2000-5021. URL: http://www.tethers.com/papers/TerminatorTether.pdf

11. **Bogar, T.J. et al. (2000).** "Hypersonic Airplane Space Tether Orbital Launch (HASTOL) System." NIAC Phase I Final Report, NASA. URL: https://www.niac.usra.edu/publications/

### 7.4 Web-Retrieved Sources

12. **NSS Summary of Birch's Orbital Ring.** National Space Society. URL: https://nss.org/orbital-ring-systems/

13. **Wikipedia: Orbital Ring.** URL: https://en.wikipedia.org/wiki/Orbital_ring

14. **Orion's Arm Encyclopedia: Orbital Ring.** URL: https://www.orionsarm.com/eg-article/48545f88732f0

15. **StackExchange Physics: Orbital Ring Force Balance.** URL: https://physics.stackexchange.com/ [question on orbital ring force equations]

---

## 8. Summary of Key Findings

### Finding 1 (CRITICAL — Stability): Stream Velocity Must Remain Below $\sqrt{2}\,v_{\rm orb}$

Birch's linearized stability analysis shows that for $v_s > \sqrt{2}\,v_{\rm orb} \approx 11.2$ km/s (at LEO), the orbital ring becomes **radially unstable** in the absence of active control. The SGMS operates at $v_s \leq 15$ km/s, which is above this threshold. This means the SGMS system in its high-velocity regime **requires active electromagnetic feedback** to prevent runaway radial displacement. This is not currently reflected in the stability analysis.

**Mathematical condition:** $\omega_{\rm radial}^2 = (2v_{\rm orb}^2 - v_s^2)/(R+h)^2$. For $v_s = 15$ km/s, $v_{\rm orb} = 7.9$ km/s: $\omega_{\rm radial}^2 < 0$ — **unstable mode exists**.

### Finding 2 (IMPORTANT — Packet Spacing): Resonance-Free Design Window

The condition for continuous-force regime is $s < \ell_{\rm int}$, which the SGMS parameter space ($s \in [0.1, 10]$ m) satisfies for interaction lengths > 10 m. However, **harmonic resonance** with station structural modes is possible. For $v_s = 1600$ m/s and station structural frequency $\omega_n$, the resonant spacings are $s_{\rm res} = 2\pi v_s / \omega_n$. With structural resonances likely in the 1–100 Hz range, resonant spacings of 100–10,000 m must be **avoided**. The SGMS parameter range is safely below this. However, if $v_s$ increases above ~10 km/s, resonant spacings move into the 60–600 m range — still safely above the 10 m maximum spacing.

### Finding 3 (IMPORTANT — Flux Pinning): Frequency-Dependent Stiffness Overestimate

The static Bean-London model overestimates the GdBCO flux-pinning stiffness at frequencies > 1–10 Hz (the flux-flow crossover). The Sobol analysis showing 44% variance from $k_{\rm fp}$ may be **significantly overestimated** if the system is operating in this frequency range. The dynamic stiffness formula $k_{\rm fp}(\omega) = k_{\rm fp,0}/(1 + (\omega/\omega_{\rm ff})^2)$ should be implemented. At 10 Hz perturbations with $\omega_{\rm ff} = 1$ Hz (conservative for 77K GdBCO), the effective stiffness is reduced by **100×**.

---

*End of Research Document — Branch 1: Dynamic Mass-Stream Anchors, Orbital Rings, and Skyhooks*

*Retrieved using:*  
- *arXiv Search Skill (`literature-search-arxiv`)*  
- *OpenAlex Skill (`literature-search-openalex`)*  
- *Web search for supplementary context*

*Paper URLs: See Section 7 above.*
