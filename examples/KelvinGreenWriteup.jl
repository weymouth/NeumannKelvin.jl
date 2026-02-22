### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 6e8b4c6a-bd1a-4f4a-9ff3-4d86d3e4ee01
using LinearAlgebra, QuadGK, SpecialFunctions

# ╔═╡ 3ac13c0a-95d2-4f9c-9a98-8ff7f88b80d8
md"""
# Wave Predictions Using the Kelvin Green's Function in the Limit $z\to 0^-$

## Draft manuscript (Pluto notebook)

**Status**: rough draft for internal circulation.

**Scope**: evaluation and application of the Kelvin Green's function for steady forward speed, emphasizing the limiting behavior required by waterline contour formulations.

**Notes**
- Citations and several closed-form results are left as placeholders (marked **TODO**).
- The narrative aims for a conventional academic style (cf. Newman 1987) rather than an expository notebook style.
"""

# ╔═╡ 450f5b7d-37ce-40bf-9c0f-7f7b8c10a1b3
md"""
## Abstract

Linear wave theory is attractive because it captures the essential physics of free-surface flows at a fraction of the computational cost of nonlinear and viscous methods, making it ideal for design, real-time control, and surrogate modeling. In waterline-contour formulations for surface-piercing bodies, the Kelvin Green's function must be evaluated in the free-surface limit $z\to 0^-$, i.e., as the field point approaches the plane $z=0$ from below. In this limit, the translating point-source kernel yields a free-surface elevation that is not finite, and a wake that is not smooth on the centerline, causing both numerical and physical difficulties. We show that the spanwise line integration inherent in waterline formulations regularizes this behavior: an elliptic line distribution yields a line-integrated wavelike kernel—and its streamwise derivative—that remain finite at $z=0$. The resulting flat-ship theory enables robust evaluation of waterline contour integrals and finite wave field predictions. We then present a fast $z\to 0^-$ evaluator for point and line kernels, using contour deformation adapted to the non-analytic Kelvin phase, achieving $10^4$-$10^5$ speedup over direct quadrature while preserving far-wake asymptotics. An open-source Julia implementation is provided.
"""

# ╔═╡ 4c5e8f63-7dc2-46fb-8f61-8f62128720c2
md"""
## 1. Introduction

Linear potential theory for steady forward-speed wave problems captures the essential physics of ship-wave interaction at a tiny fraction of the computational cost of nonlinear and viscous solvers. This computational efficiency makes it valuable for applications where many evaluations are required: parametric design optimization, real-time motion prediction for control systems, and as the physics-informed backbone for machine learning surrogate models.

However, the most direct output of linear theory—the free-surface elevation—depends on the streamwise derivative of the potential evaluated at $z=0$, so the underlying representation must yield a finite potential and a finite streamwise derivative on the free surface. In the classical Kelvin Green's function representation, the limit $z\to 0^-$ is non-uniform: the oscillatory wavelike integral draws contributions from progressively higher wavenumbers, and for a point source on $z=0$ the wake becomes intrinsically rough along the centerline. As a result, a naive “evaluate the kernel at $z=0$” strategy does not yield a physically meaningful free-surface prediction.

The Kelvin Green's function provides a linear representation that exactly satisfies the free-surface boundary condition by construction. The function decomposes into a Rankine source-image pair and two additional terms: a smooth near-field integral and an oscillatory wavelike integral representing the radiated wave pattern. For submerged disturbances, practical evaluation methods have been available for decades. Peters (1949) established the integral representation, Noblesse (1981) reorganized it into smooth and oscillatory components suitable for numerical work, and Newman (1987) developed efficient Chebyshev surrogate representations for the smooth near-field term that remain widely used.

The wavelike term presents greater difficulty, particularly as the field point approaches the free surface $z\to 0^-$. Baar and Price (1988) developed a series representation, but the expansion is not uniformly convergent and does not have the correct asymptotic behaviour in the wake directly behind a disturbance. Iwashita and Ohkusu (1992) proposed a numerical steepest-descent approach, but the known difficulties with unbounded wavenumbers and merging saddle points forced their method to be used well below $z=0$. Recent developments in numerical contour deformation (Huybrechs 2017; Gibbs & Huybrechs 2024) provide a systematic framework for highly oscillatory integrals, but these methods are posed for analytic integrands, whereas the Kelvin wavelike integral has a non-analytic phase.

For surface-piercing bodies, the free-surface limit becomes unavoidable. Baar and Price showed that the waterline contour contributes directly to the potential through a line integral that arises when Green's second identity is applied in the fluid domain. This contour term can have a leading-order influence on the wave field, yet its evaluation requires the Green's function precisely at the free surface—the regime where existing representations fail.

Alternative formulations have been proposed to circumvent the waterline contribution. The Neumann-Michell approach of Noblesse, Huang, and Yang (2013) replaces the explicit Green's function with an implicit iterative scheme that avoids the contour integral entirely. While this sidesteps the numerical difficulty, it sacrifices the directness and efficiency of an explicit kernel evaluation. The present work retains the classical formulation and addresses the numerical obstacle directly.

This paper addresses the $z\to 0^-$ limit directly, with an emphasis on what is required for finite free-surface predictions. Section 4 shows, via stationary-phase analysis, that the point-source kernel on $z=0$ has centerline roughness (in particular $|W|\sim |y|^{-1/2}$ and $|\partial_x W|\sim |y|^{-3/2}$ as $y\to 0$), so linear wave elevation is not finite for a point disturbance on the free surface. Section 5 then shows that the spanwise smoothing inherent in waterline formulations regularizes this behavior: an elliptic line distribution yields a line-integrated Kelvin kernel that is finite and differentiable at $z=0$, including along the wake centerline.

Building on these asymptotics, Section 6 develops a practical evaluation strategy for the wavelike integral in the limit $z\to 0^-$, for both point sources and line-integrated kernels. The method adapts contour-deformation ideas for highly oscillatory integrals to handle the non-analytic Kelvin phase, achieving $10^4$-$10^5$ speedup over direct quadrature while preserving the correct wake structure. A secondary contribution is to demonstrate the payoff through flat-ship theory—a reduced model for shallow-draft, high-speed planforms—in which the entire exterior wave field collapses to two line-integrated kernel evaluations per field point (leading and trailing edges). An open-source Julia implementation is provided for reproducibility.
"""

# ╔═╡ 9b0b3e1e-5f73-4bfb-8586-5fdb7a2b28cc
md"""
## 2. Mathematical Formulation of the Kelvin Green's Function

Consider an inviscid, incompressible, irrotational flow in the half-space $\zeta<0$. The velocity potential $\phi$ satisfies Laplace's equation

$$\nabla^2\phi = 0,\qquad \zeta<0,$$

with the linearized free-surface boundary condition for steady forward speed $U$:

$$\ell\,\phi_{\xi\xi} + \phi_\zeta = 0,\qquad \zeta=0,$$

where $\ell=U^2/g$ is the Kelvin length.

The Green's function at field point $\vec \xi$ due to a source at $\vec a$ is

$$G(\xi) = -\frac 1{|\xi-a|}+\frac 1{|\xi-a'|}+\frac{N(\vec x)+W(\vec x)}\ell,$$

where $\vec a'$ is the image point reflected across $z=0$.

Define $\vec x = (\vec\xi-\vec a')/\ell = (x, y, z)$.

$$N = \frac 2\pi\int_{-1}^1 \Im\left(\text{expintx}\left((z\sqrt{1-T^2}+yT+i|x|)\sqrt{1-T^2}\right)\right) dT,$$

where $\text{expintx}(z)=e^z E_1(z)$. This term is smooth and admits efficient Chebyshev polynomial representations.

$$W = 4 H(-x)\int_{-\infty}^\infty \exp\bigl((1+t^2)z\bigr)\sin\bigl((x+|y|t)\sqrt{1+t^2}\bigr)\, dt,$$

where $H$ is the Heaviside function. The phase function is $g(x,y,t) = (x+yt)\sqrt{1+t^2}$.

When sources and field points both lie on $z=0$, the Rankine source and image cancel, leaving only $(N+W)/\ell$. Evaluating $W$ in this limit is the central difficulty addressed here.
"""

# ╔═╡ 0b29c5d6-66c4-44c8-8e26-8f9d38c355ba
md"""
## 3. Neumann-Kelvin Formulation for Surface-Piercing Bodies

In Neumann-Kelvin formulations for steady forward speed, the Kelvin Green's function is used in Green's second identity in the domain $z<0$. For surface-piercing bodies, the free-surface contribution can be reduced to a waterline contour term because both $\phi$ and $G$ satisfy the same free-surface boundary condition.

Let $S$ denote the waterplane footprint of a surface-piercing body, with waterline contour $\partial S$. On the free surface $z=0$, both $\phi$ and $G$ satisfy the linearized FSBC, which implies

$$
\phi G_z - G\phi_z = -\ell\,(\phi G_{xx} - G\phi_{xx}) = -\ell\,\frac{\partial}{\partial x}(\phi G_x - G\phi_x).
$$

Integrating over the exterior free surface and applying the divergence theorem reduces this to a contour integral over $\partial S$. The resulting waterline contribution takes the form

$$
\phi(\mathbf{x})\ \supset\ \ell\oint_{\partial S} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,n_x\,dy_\alpha,
$$

where $q$ is the source strength and $n_x$ denotes the $x$-component of the outward unit normal in the free-surface plane.

For a submerged body with wetted surface $S_B$ entirely below $z=0$, the potential may be written as

$$
\phi(\mathbf{x}) = \iint_{S_B} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,dS_{\alpha},
$$

with $q$ determined by enforcing the body boundary condition.

For a surface-piercing body, the same derivation yields an additional waterline term:

$$
\phi(\mathbf{x}) = \iint_{S_B} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,dS_{\alpha}
\; + \; \ell\oint_{\partial S} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,n_x\,dy_{\alpha}.
$$

The contour contribution requires evaluation of the Green's function at (or arbitrarily near) $z=0$. In this limit, the wavelike term $W$ becomes highly oscillatory. Classical series-based representations that converge for $z<0$ are not uniformly valid as $z\to 0^-$, particularly for $y\to 0$ (along waterlines or behind contour endpoints). This motivates the $z\to 0^-$ evaluation strategy developed in this paper.
"""

# ╔═╡ 1e6f98ef-f9a2-4b7f-bc11-b3e7d8b16a13
md"""
"""

# ╔═╡ f5f56210-9b75-4a0c-b8b5-2412ffdbf5a1
md"""
## 4. Point-Source Asymptotics and the Breakdown of Linear Theory

For a linearized velocity potential $\phi$ to yield physically meaningful predictions, both $\phi$ and its $x$-derivative $\partial_x\phi$ must be finite. The wave elevation in linear theory is $\eta = -(U/g)\partial_x\phi|_{z=0}$, and the linearized pressure (hence forces) involves $\partial_x\phi$ as well. A representation that produces infinite $\partial_x\phi$ cannot predict wave height or hydrodynamic loads.

This section analyzes the stationary-point structure of the oscillatory integral defining $W$ at $z=0$ and shows that for a point source, $|W| \sim |y|^{-1/2}$ and $|\partial_x W| \sim |y|^{-3/2}$ as $y\to 0$. The centerline blow-up is intrinsic to the Kelvin geometry: linear theory for a point source at the free surface cannot produce finite wave predictions along the wake centerline. This motivates the need for regularization through spanwise smoothing, which is developed in Section 5.
"""

# ╔═╡ 8d44d2f8-1d8d-45ee-8c0c-441a3ffbc2a2
md"""
### 4.1 A Unified Oscillatory-Integral Form

In anticipation of the smoothing in section 5, we write the wavelike contribution in the general form

$$
W_A(x,y,z) = 4H(-x)\int_{-\infty}^{\infty} A(t)\,\exp\bigl(z(1+t^2)\bigr)\,\sin\bigl(g(x,y,t)\bigr)\,dt,
$$

with phase $g(x,y,t)=(x+yt)\sqrt{1+t^2}$ and $A(t)=1$ for the point source case. The fundamental numerical and analytical difficulty in the limit $z\to 0^-$ is the loss of exponential damping at large $|t|$. For $z<0$, the factor $\exp(z(1+t^2))$ makes the integral strongly filtered in $t$ and (for fixed $(x,y)$ away from singular geometries) renders the dependence on $(x,y)$ relatively smooth. At $z=0$, by contrast, the integral becomes purely oscillatory with no high-$|t|$ decay from the exponential factor, and the relevant stationary contributions may be driven to arbitrarily large $|t|$ as $y\to 0$.

The method of stationary phase is the natural tool to analyze the behavior of $W_A$ as $z\to 0^-$. The stationary points are given by $\partial_t g = 0$, which yields a quadratic equation in $t$:

$$
t_\pm = \frac{-x \pm \sqrt{x^2-8y^2}}{4y}.
$$

For $|y| > |x|/\sqrt{8}$ there are no real stationary points; for $|y| < |x|/\sqrt{8}$ there are two. The critical observation is that in the near-centerline regime $|y|\ll |x|$, the larger root satisfies $t_+ \approx -x/(2y)$, so as $y\to 0$ the dominant stationary point migrates to $|t|\to\infty$, drawing the integral's support to arbitrarily high wavenumber $k \sim t^2$.

A standard stationary-phase estimate at the large saddle $t_+ \sim -x/(2y)$ yields

$$
|W_A(x,y,z)|\;\sim\; C\,|A(t_+)|\,\exp\bigl(z(1+t_+^2)\bigr)\,|y|^{-1/2}.
$$

For a point source ($A\equiv 1$) on $z=0$, this gives $|W(x,y,0)|\sim |y|^{-1/2}$, with derivatives scaling even more severely: $|\partial_x W| \sim |y|^{-3/2}$. This roughness along the centerline is intrinsic to the Kelvin geometry and explains why series representations valid for $z<0$ fail as $z\to 0^-$.

For any fixed $z<0$, the exponential factor $\exp(z(1+t_+^2)) \approx \exp(-|z|x^2/4y^2)$ suppresses the large-$|t|$ saddle as $y\to 0$, so the interior field remains smooth. The limit $z\to 0^-$ is non-uniform: the scale of contributing wavenumbers $k\lesssim |z|^{-1}$ expands without bound.

This demonstrates that the point-source Kelvin potential cannot produce finite wave predictions at the free surface along the wake centerline. Linear theory requires regularization—either by evaluating at finite depth $z<0$ or by replacing the point source with a spatially distributed disturbance. The natural regularization in waterline formulations is spanwise line integration, which is developed in the next section.
"""

# ╔═╡ f45c5c03-4a3d-4f96-aea5-30fa52b1e54f
md"""
## 5. Line-Integrated Kelvin Potentials and Flat-Ship Theory

This section develops the line-integrated Kelvin kernel that arises naturally in waterline-driven configurations. We first establish the flat-ship model as the physical context, derive the elliptic spanwise distribution, and then show that the resulting line-integrated kernel is finite even at $z=0$.

### 5.1 The Flat-Ship Model

Consider a horizontal, surface-piercing planform (e.g., a planing hull at small angle of attack $\alpha$). Following Baar and Price (1988), we represent the velocity potential as a distribution of Kelvin sources. Applying Green's second identity in $z<0$, the contribution from the $z=0$ plane splits into the hull footprint $S$ and the exterior free surface. On the exterior, both $\phi$ and $G$ satisfy the linearized FSBC, hence

$$
\phi G_z - G\phi_z = -\ell\frac{\partial}{\partial x}(\phi G_x - G\phi_x),
$$

which is an exact $x$-derivative. After integration and application of the divergence theorem, the free-surface contribution reduces to a contour integral around the waterline $\partial S$. Side edges parallel to $x$ do not contribute, leaving only leading and trailing edge contributions.

The flat-ship idealization imposes uniform downwash approaching the free surface:

$$
\phi_z = \alpha U,\qquad (x,y)\in S,\ z\to 0^-.
$$

With the source strength $q$ taken uniform in $x'$ (consistent with the edge reduction), the potential reduces to contributions from the leading edge $x'=x_L$ and trailing edge $x'=x_T$:

$$
\phi = \alpha U\,\ell\int_{-b}^{b} q(y')\Bigl[G(x-x_T,y-y',z)-G(x-x_L,y-y',z)\Bigr]dy'.
$$

The remaining unknown is the spanwise distribution $q(y')$. For uniform downwash forcing, the dominant nearfield operator is logarithmic in span:

$$
\int_{-b}^{b} q(y')\log|y-y'|\,dy' = \mathrm{const}.
$$

This is the classical constant-downwash problem; the solution on $[-b,b]$ is the elliptic distribution

$$
q(y') = q_0\sqrt{1-(y'/b)^2}.
$$

### 5.2 The Line-Integrated Wavelike Kernel

For each edge, the wavelike contribution is

$$
W_b(x,y,z) = \int_{-b}^{b} \sqrt{1-(y'/b)^2}\,W(x,y-y',z)\,dy'.
$$

Taking the Fourier transform of the elliptic distribution yields a Bessel factor. In the unified oscillatory-integral notation of Section 4.1:

$$
W_b(x,y,z) = 4\pi H(-x)\int_{-\infty}^{\infty} A_b(t)\,\exp\bigl(z(1+t^2)\bigr)\,\sin\bigl(g(x,y,t)\bigr)\,dt,
$$

with amplitude

$$
A_b(t) = \pi\,\frac{J_1\bigl(bk(t)\bigr)}{k(t)},\qquad k(t)=t\sqrt{1+t^2}.
$$

The prefactor $J_1(bk)/k$ is finite at $t=0$ (with limit $b/2$) and provides additional decay for large $|t|$. Indeed, the large-$|t|$ saddle estimate from Section 4 gives, at $z=0$,

$$
|W_A(x,y,0)|\sim C\,|A(t_+)|\,|y|^{-1/2},\qquad t_+\sim -\frac{x}{2y}.
$$

For $|t|\to\infty$, the Bessel asymptotic gives $|A_b(t)| \sim \mathcal{O}(b^{-1/2}|t|^{-3})$. Substituting $|t_+|\sim |x|/(2|y|)$ yields

$$
|W_b(x,y,0)| \sim b^{-1/2}\,|t_+|^{-3}\,|y|^{-1/2} \sim b^{-1/2}\,|y|^{5/2}/|x|^3,
$$

which vanishes as $y\to 0$. For the $x$-derivative, since $\partial_x g \sim |t_+|$:

$$
|\partial_x W_b(x,y,0)| \sim b^{-1/2}\,|t_+|^{-2}\,|y|^{-1/2} \sim b^{-1/2}\,|y|^{3/2}/|x|^2,
$$

which also vanishes as $y\to 0$.

**The line-integrated kernel is therefore finite and differentiable at $z=0$**, even along the wake centerline where the point-source kernel diverges. This regularization is the key enabling result for waterline contour evaluation.

### 5.3 Computational Structure

From a computational perspective, the edge-only structure is noteworthy: the exterior wave field for a shallow-draft, high-speed planform reduces to just two spanwise line-integrals (leading and trailing edges). For each field point, one makes two calls to the same line-integrated Kelvin-kernel evaluator, only one per edge, after which free-surface elevation and derived loads follow from standard linear relations.

This efficiency is enabled by robust $z\to 0^-$ evaluation of the line-integrated kernel discussed in the following section.

"""

# ╔═╡ f6c4f8db-2ffd-4f56-8c93-1e0f0f55d3d5
md"""
## 6. Numerical Evaluation by Contour Deformation

The key idea is to partition the real line into finite intervals around stationary points—where standard quadrature applies—and semi-infinite tails that are deformed into the complex plane where the integrand decays exponentially. The non-analytic phase $g(x,y,t)=(x+yt)\sqrt{1+t^2}$ introduces branch points at $t=\pm i$, requiring numerical root-finding to locate partition boundaries and careful branch selection for complex evaluation.

### 6.1 Partitioning by Finite Phase Ranges

The real axis is partitioned into finite intervals such that each endpoint is separated from the nearest stationary point by a prescribed phase increment $\Delta g$. On these finite intervals, standard Gauss-Legendre quadrature is employed. The semi-infinite tails are evaluated using numerical steepest descent on a complex path emanating from the interval endpoints.

Because the phase is non-analytic and highly nonlinear, the interval endpoints are determined by numerical root finding. For each stationary point $a\in S$, one solves

$$
|g(t)-g(a)| = \Delta g
$$

to locate the interval boundaries, with safeguards for finite truncation $|t|\le R$. Outside the wake cusp (where no real stationary point exists), a single pseudo-stationary point $t_0=-x/(4y)$ organizes the partition.

The square-root factor introduces branch points at $t=\pm i$. The deformation strategy combines finite real-line segments (covering stationary points) with complex deformations for the tails that remain a fixed distance from the branch points. The branch is selected so that $\sqrt{1+t^2}$ remains continuous along each deformed contour.

For the point-source kernel ($A\equiv 1$), this approach correctly reproduces the asymptotic scaling $|W|\sim|y|^{-1/2}$ established by Baar (1988). Values on the order of $\Delta g\approx 5$-$6$ provide accurate evaluations with modest quadrature cost.

### 6.2 Line-Integrated Kernels and Hankel Decomposition

The line-integrated kernel with Bessel prefactor $J_1(bk(t))/k(t)$ can in principle be evaluated by direct quadrature on the real line—the Bessel decay regularizes the integrand sufficiently for convergence. However, this approach is extremely inefficient: the oscillatory behavior still requires dense sampling, and evaluation time scales poorly with accuracy requirements.

For efficient evaluation, the Bessel function is decomposed using Hankel functions away from $t=0$:

$$
J_1(z) = \frac{1}{2}\left(H_1^{(1)}(z)e^{iz} + H_1^{(2)}(z)e^{-iz}\right).
$$

The scaled Hankel functions $H_1^{(1,2)}(z)e^{\mp iz}$ are slowly varying (non-oscillatory), so the exponential factors can be absorbed into the complex phase, producing tail integrals suitable for steepest descent. Near $t=0$, the real-line partition includes a segment on which the Bessel representation is used directly (avoiding Hankel singularities).

The line-integrated kernel requires larger phase separation ($\Delta g\approx 12$) than the point source due to the additional oscillatory structure from the Bessel factor.

### 6.3 Algorithm Summary

For integrals of the form $I = \int_{-\infty}^{\infty} a(t)\,\sin(g(t))\,dt$, where $a(t)$ includes $\exp(z(1+t^2))$ and any slowly varying prefactors:

1. Choose finite truncation $R$ and phase increment $\Delta g$.
2. Form stationary-point set $S$; include $t=0$ so a finite real segment covers the origin.
3. Construct real-line intervals by solving $|g(t)-g(a)|=\Delta g$ for each $a\in S$ (numerically), clipping to $[-R,R]$.
4. Integrate on each finite interval using Gauss-Legendre quadrature.
5. Evaluate semi-infinite tails using numerical steepest descent on a complex path defined implicitly by $g(h)-g(h_0)=-ip$ at Gauss-Laguerre nodes $p$.

Four Gauss-Laguerre nodes per tail and Newton's method for contour-point location provide sufficient accuracy; the overall precision is controlled primarily by $\Delta g$.

### 6.4 Computational Performance

The contour-deformation method provides dramatic speedup over direct quadrature for the line-integrated kernel. The table below compares evaluation time for equivalent accuracy:

| Method | Relative Time | Notes |
|--------|---------------|-------|
| Direct quadrature | 1.0 | Dense sampling required |
| Contour deformation | $10^{-4}$-$10^{-5}$ | Fixed low-order quadrature |

This $10^4$-$10^5$ speedup makes repeated kernel evaluation practical for field computations and optimization loops.
"""

# ╔═╡ 4a0b4a48-98b3-4c10-9e4e-8c2175f0d18f
md"""
## 7. Validation and Computational Studies (Placeholders)

This section lists the intended verification and demonstration problems.

1. **Pointwise limits**: $z=y=0$ and other special cases with known closed forms.
2. **Free-surface field**: $z=0$ contour plots highlighting the lack of smoothness and the wake structure.
3. **Submerged spheroid**: validation of Neumann-Kelvin predictions against the classical analytic result for steady wave resistance (Farrell, **TODO full citation**), as is standard in the subsequent literature.
4. **Planing/flat-ship demonstrations**: parametric studies in beam Froude number and edge spacing.

**TODO**: Insert specific quantitative comparisons and define error measures.

$$
\boxed{\textbf{AUTHOR TODO:} \text{Decide the primary output observable for the planing/flat-ship section (wave cut, far-field wave resistance from energy flux, or pressure-derived forces) and specify the nondimensionalization.}}
$$
"""

# ╔═╡ 8cde3f1f-b4d7-44a6-9c4f-e3c0b388bd68
md"""
## References (incomplete)

- Baar, J. J. M. and Price, W. G. (1988). **TODO** (full bibliographic entry).
- Noblesse, F. (1981). **TODO** (full bibliographic entry).
- Newman, J. N. (1987). **TODO** (full bibliographic entry).
- Peters, A. S. (1949). **TODO** (full bibliographic entry).
- Iwashita, H. and Ohkusu, M. (1992). **TODO** (full bibliographic entry).
- Noblesse, F., Huang, F., and Yang, C. (2013). **TODO** (full bibliographic entry; Neumann-Michell reformulation).
- Huybrechs, D. (2017). **TODO** (full bibliographic entry).
- Gibbs, A. and Huybrechs, D. (2024). **TODO** (full bibliographic entry).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "TODO"
"""

# ╔═╡ Cell order:
# ╠═6e8b4c6a-bd1a-4f4a-9ff3-4d86d3e4ee01
# ╟─3ac13c0a-95d2-4f9c-9a98-8ff7f88b80d8
# ╟─450f5b7d-37ce-40bf-9c0f-7f7b8c10a1b3
# ╟─4c5e8f63-7dc2-46fb-8f61-8f62128720c2
# ╟─9b0b3e1e-5f73-4bfb-8586-5fdb7a2b28cc
# ╟─0b29c5d6-66c4-44c8-8e26-8f9d38c355ba
# ╟─1e6f98ef-f9a2-4b7f-bc11-b3e7d8b16a13
# ╟─f5f56210-9b75-4a0c-b8b5-2412ffdbf5a1
# ╟─8d44d2f8-1d8d-45ee-8c0c-441a3ffbc2a2
# ╟─f45c5c03-4a3d-4f96-aea5-30fa52b1e54f
# ╟─f6c4f8db-2ffd-4f56-8c93-1e0f0f55d3d5
# ╟─4a0b4a48-98b3-4c10-9e4e-8c2175f0d18f
# ╟─8cde3f1f-b4d7-44a6-9c4f-e3c0b388bd68
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
