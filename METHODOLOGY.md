# Data Extraction and Processing Methodology

This document explains how simulation data is extracted from Lumerical and processed to calculate the output metrics.

---

## Simulation Flow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT PARAMETERS                              │
│  w_r, h_si, S, doping, lambda, length                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CHARGE SIMULATION (Electrical)                   │
│  Lumerical DEVICE solver                                         │
│  Output: n(V), p(V) - carrier concentrations vs voltage          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    charge_data.mat
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FDE SIMULATION (Optical)                        │
│  Lumerical MODE solver                                           │
│  Input: carrier data from CHARGE                                 │
│  Output: neff(V) - complex effective index vs voltage            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                               │
│  Calculate: C, α, Δφ, V_π, V_π*L, cost                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. CHARGE Simulation - Electrical Data

### 1.1 Lumerical API Call

```python
charge_data = charge_session.getresult("CHARGE::monitor_charge", "total_charge")
```

### 1.2 Extracted Data

| Key | Description | Shape |
|-----|-------------|-------|
| `V_drain` | Applied voltage array | (N,) |
| `n` | Electron count at each voltage | (N,) |
| `p` | Hole count at each voltage | (N,) |

### 1.3 Capacitance Calculation

**Step 1: Convert carrier counts to charge**

```
Qn = q × n
Qp = q × p
```

Where:
- `q` = 1.602 × 10⁻¹⁹ C (elementary charge)
- `n`, `p` = carrier counts from Lumerical

**Step 2: Calculate differential capacitance**

```
Cn = dQn/dV
Cp = dQp/dV
```

Using `np.gradient(Q, V)` for numerical differentiation.

**Step 3: Total capacitance**

```
C_total = Cn + Cp
```

**Step 4: Unit conversion to pF/cm**

```
C_total_pF_cm = C_total × 1e10
```

The factor 1e10 converts from F/m to pF/cm:
- F → pF: × 1e12
- /m → /cm: × 1e-2
- Combined: × 1e10

---

## 2. FDE Simulation - Optical Data

### 2.1 Lumerical API Call

```python
sweep_result = fde_session.getsweepresult("voltage", "neff")
neff = sweep_result['neff']  # Complex array
```

### 2.2 Voltage Reconstruction

The FDE sweep returns `neff` values but not the voltage array. We reconstruct it:

```python
V = np.linspace(0, V_MAX, len(neff))  # V_MAX = 2.5 V
```

### 2.3 Effective Index Change (Δneff)

```
Δneff(V) = Re[neff(V)] - Re[neff(0)]
```

The change in effective index relative to zero bias.

---

## 3. Optical Loss Calculation

### 3.1 Formula

```
α = 2 × k₀ × Im(neff) × (10/ln(10)) × 10⁻²  [dB/cm]
```

Where:
- `k₀ = 2π/λ` - wave number in vacuum
- `Im(neff)` - imaginary part of effective index (related to absorption)
- `10/ln(10) ≈ 4.343` - converts Nepers to dB
- `10⁻²` - converts /m to /cm

### 3.2 Derivation

Starting from the field propagation:

```
E(z) = E₀ × exp(i × k₀ × neff × z)
     = E₀ × exp(i × k₀ × Re(neff) × z) × exp(-k₀ × Im(neff) × z)
```

Power decay:
```
P(z) = P₀ × exp(-2 × k₀ × Im(neff) × z)
```

Loss coefficient in Np/m:
```
α_Np/m = 2 × k₀ × Im(neff)
```

Convert to dB/cm:
```
α_dB/cm = α_Np/m × (10/ln(10)) × (1/100)
```

### 3.3 Code Implementation

```python
def calc_alpha(neff, wavelength):
    k0 = 2 * np.pi / wavelength
    alpha = 2 * k0 * np.imag(neff) * (10 / np.log(10)) * 1e-2
    return alpha  # dB/cm
```

---

## 4. Phase Shift Calculation

### 4.1 Formula

```
Δφ = (2π × Δneff × L) / λ  [radians]
```

Where:
- `Δneff` - change in effective index
- `L` - device length (actual, from params)
- `λ` - wavelength (actual, from params)

### 4.2 Code Implementation

```python
def calc_dphi(d_neff, length, wavelength):
    delta_phi = (2 * np.pi * d_neff * length) / wavelength
    return delta_phi  # radians
```

---

## 5. V_π Calculation

### 5.1 Definition

V_π is the voltage required to achieve a π-radian phase shift.

### 5.2 Method

Use linear interpolation to find V where |Δφ| = π:

```python
def calculate_v_pi(voltages, abs_dphi):
    if np.max(abs_dphi) < np.pi:
        return np.nan  # Phase shift never reaches π

    v_pi = np.interp(np.pi, abs_dphi, voltages)
    return v_pi
```

### 5.3 V_π × L Product

```
V_π × L = V_π × L × 1000  [V·mm]
```

Where L is converted from meters to millimeters.

---

## 6. Values at V_π

Once V_π is determined, we interpolate other metrics at this operating point:

```python
loss_at_v_pi = np.interp(v_pi, V_fde, alpha_dB_per_cm)
C_at_v_pi = np.interp(v_pi, V_cap, C_total_pF_cm)
```

---

## 7. Cost Function

### 7.1 Success Case (|Δφ_max| ≥ π)

Weighted quadratic cost:

```
cost = w_loss × (α/T_loss)² + w_vpil × (V_π×L/T_vpil)²
```

Where:
- `w_loss = 0.3` - weight for loss
- `w_vpil = 0.7` - weight for V_π×L
- `T_loss = 2.0 dB/cm` - loss target
- `T_vpil = 1.0 V·mm` - V_π×L target

### 7.2 Penalty Case (|Δφ_max| < π)

If phase shift doesn't reach π:

```
cost = C_base + β × (π - Δφ_max)²
```

Where:
- `C_base` = max cost from valid simulations
- `β = 9 × C_base / π²`

---

## 8. Data Flow Summary

```
CHARGE Simulation
    │
    ├── V_drain[] ──────────────────────────────┐
    │                                            │
    ├── n[] ──► Qn = q×n ──► Cn = dQn/dV ──┐    │
    │                                       │    │
    └── p[] ──► Qp = q×p ──► Cp = dQp/dV ──┼────┼──► C_total = Cn + Cp
                                           │    │         │
                                           │    │         ▼
FDE Simulation                             │    │    C_at_v_pi (interpolated)
    │                                      │    │
    └── neff[] ───┬─► Δneff = Re(neff - neff[0])
                  │         │
                  │         ├──► Δφ = 2π×Δneff×L/λ ──► V_π (interpolated)
                  │         │                              │
                  │         │                              ▼
                  │         │                         V_π×L = V_π × L
                  │         │
                  └─► α = 2×k₀×Im(neff)×4.343×0.01 ──► loss_at_v_pi (interpolated)
                                                              │
                                                              ▼
                                                    cost = f(α, V_π×L)
```

---

## 9. Units Summary

| Metric | Internal Unit | Output Unit | Conversion |
|--------|---------------|-------------|------------|
| Voltage | V | V | - |
| Length | m | mm | × 1000 |
| Wavelength | m | nm | × 1e9 |
| Capacitance | F/m | pF/cm | × 1e10 |
| Loss | Np/m | dB/cm | × 4.343 × 0.01 |
| Phase shift | rad | rad | - |
| V_π × L | V·m | V·mm | × 1000 |
| Doping | cm⁻³ | m⁻³ | × 1e6 (for Lumerical) |

---

## 10. Key Code References

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `extract_capacitance()` | data_processor.py | 53 | Get C from CHARGE |
| `extract_optical_parameters()` | data_processor.py | 103 | Get neff from FDE |
| `calc_alpha()` | data_processor.py | 166 | Calculate optical loss |
| `calc_dneff()` | data_processor.py | 188 | Calculate Δneff |
| `calc_dphi()` | data_processor.py | 202 | Calculate phase shift |
| `calculate_v_pi()` | data_processor.py | 220 | Find V_π by interpolation |
| `set_charge_parameters()` | sim_handler.py | 10 | Configure CHARGE sim |
| `set_fde_parameters()` | sim_handler.py | 212 | Configure FDE sim |
| `import_charge_data()` | sim_handler.py | 364 | Transfer CHARGE→FDE |
