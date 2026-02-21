### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 6e8b4c6a-bd1a-4f4a-9ff3-4d86d3e4ee01
using LinearAlgebra, QuadGK, SpecialFunctions

# ╔═╡ 3ac13c0a-95d2-4f9c-9a98-8ff7f88b80d8
md"""
# Wave Predictions Using the Kelvin Green’s Function in the Limit $z\to 0^-$

## Draft manuscript (Pluto notebook)

**Status**: rough draft for internal circulation.

**Scope**: evaluation and application of the Kelvin Green’s function for steady forward speed, emphasizing the limiting behavior required by waterline contour formulations.

**Notes**
- Citations and several closed-form results are left as placeholders (marked **TODO**).
- The narrative aims for a conventional academic style (cf. Newman 1987) rather than an expository notebook style.
"""

# ╔═╡ 450f5b7d-37ce-40bf-9c0f-7f7b8c10a1b3
md"""
## Abstract

Linear wave predictions at steady forward speed remain valuable because they capture the essential dispersive physics at low cost. The Kelvin Green’s function provides an exact linear forward-speed *Green’s function* satisfying the free-surface boundary condition on the exterior free surface by construction, but its routine use has been limited by the numerical difficulty of evaluating the wavelike contribution as the field point approaches the free surface.

This work develops a practical and robust evaluation of the wavelike term in the limit $z\to 0^-$, targeted to the regime required by waterline-contour (Neumann–Kelvin) formulations for surface-piercing bodies. The method combines a decomposition that isolates the oscillatory structure with a partition of the real line and complex tail evaluation by numerical steepest descent / numerical stationary phase, and it explicitly accommodates the non-analytic square-root dispersion factor.

The approach is intended to be verified against classical special cases and is demonstrated on a horizontal thin-ship (“flat-ship”) model whose reduced linearization is posed in the $z\to 0^-$ sense. This application is used to illustrate the methodology gap and the resulting computational efficiency; an exhaustive assessment of planing-model fidelity is deferred.
"""

# ╔═╡ 4c5e8f63-7dc2-46fb-8f61-8f62128720c2
md"""
## 1. Introduction and Motivation

The Kelvin Green’s function provides an exact linear forward-speed Green’s function satisfying the free-surface boundary condition on the exterior free surface by construction. Its practical use has been limited primarily by numerical difficulties in evaluating the wavelike contribution as $z\to 0^-$, precisely the regime required by waterline-contour formulations for surface-piercing configurations.

### 1.1 Contributions and Outline

The purpose of this paper is to provide a computationally practical evaluation of the Kelvin wavelike term in the regime required by waterline-contour formulations, and to demonstrate how such an evaluation enables rapid linear wave predictions at forward speed.

A second contribution is to make explicit a computational “payoff” that becomes available once a robust $z\to 0^-$ evaluator exists: in waterline-reduced models the planform influence can collapse to a small number of spanwise-smoothed Kelvin-kernel evaluations. In particular, the reduced flat-ship demonstration in Section 8 leads to an exterior wave-field predictor that, *given a spanwise closure for the source strength*, reduces to the difference of two line-integrated kernel evaluations (leading and trailing edges). This is highlighted as an efficiency/enabling result rather than as a claim of planing-model correctness.

The methodological emphasis is not on contour deformation in the abstract, but on an evaluation strategy tailored to the Kelvin oscillatory integral structure: (i) a non-analytic square-root dispersion factor with branch points; (ii) stationary-point behavior tied to the wake geometry and responsible for severe free-surface roughness; and (iii) spanwise line integration expressed through a decaying amplitude $A(t)$, which regularizes (but does not eliminate) $z=0$ roughness.

The flat-ship application is included to “lean into” the difficulty: it produces a line-integrated kernel posed directly on $z=0$ and therefore serves as a sharp demonstration that robust $z\to 0^-$ evaluation unlocks a range of efficient forward-speed predictions. The objective in that section is methodological feasibility and computational efficiency, not a comprehensive quantification of planing-hull modeling error.
"""

# ╔═╡ 9b0b3e1e-5f73-4bfb-8586-5fdb7a2b28cc
md"""
## 2. Governing Problem and Notation

Consider an inviscid, incompressible, irrotational flow in the half-space $z<0$ with a steady translating disturbance. The velocity potential $\phi$ satisfies Laplace’s equation

$$
\nabla^2\phi = 0,\qquad z<0,
$$

with the linearized free-surface boundary condition for steady forward speed

$$
\ell\,\phi_{xx} + \phi_z = 0,\qquad z=0,
$$

where $\ell=U^2/g$ is the Kelvin length. The coordinate system is chosen so that the undisturbed free surface is at $z=0$, and the mean flow is in the negative $x$ direction.

Throughout, we follow the standard conventions used in Noblesse (1981): all distances are nondimensionalized by $\ell$; the background flow is in the $-x$ direction so that $-x$ is downstream and $+x$ is upstream; and distances are formed relative to the reflected image-source geometry used in the classical decomposition. In particular, the limiting evaluation $z\to 0^-$ becomes essential in waterline-contour formulations, where sources lie on $z=0$ and the induced free-surface quantities are required on $z=0$.

The Green’s function $G(\mathbf{x};\boldsymbol{\alpha})$ is defined as the potential induced at field point $\mathbf{x}=(x,y,z)$ due to a unit source at $\boldsymbol{\alpha}=(\alpha,\beta,\zeta)$, satisfying the same FSBC.

$$
\boxed{\textbf{AUTHOR TODO:} \text{Confirm the exact normalization/sign convention for }G\text{ (e.g. }1/4\pi\text{ factors) and the image-source definition as stated in Noblesse (1981).}}
$$
"""

# ╔═╡ 0b29c5d6-66c4-44c8-8e26-8f9d38c355ba
md"""
## 3. Kelvin Green’s Function: Decomposition and Representations

The forward-speed Green’s function can be written in the conventional decomposition

$$
G = G_0 + G_0^* + N + W,
$$

where $G_0$ is the Rankine source, $G_0^*$ is its image, $N$ is a smooth near-field correction, and $W$ is the wavelike term enforcing radiation.

In a commonly used normalization one writes

$$
G_0(\mathbf{x};\boldsymbol{\alpha}) = -\frac{1}{4\pi r},\qquad r = \|\mathbf{x}-\boldsymbol{\alpha}\|,
$$

and the image term as

$$
G_0^*(\mathbf{x};\boldsymbol{\alpha}) = +\frac{1}{4\pi r^*},\qquad r^* = \|\mathbf{x}-\boldsymbol{\alpha}^*\|,
$$

where $\boldsymbol{\alpha}^*$ denotes reflection across $z=0$. In the waterline setting of interest, sources lie on $z=0$ and the field is required on $z=0$; in that limiting case the Rankine source and image cancel, leaving only $(N+W)$.

$$
\boxed{\textbf{AUTHOR TODO:} \text{Replace the preceding paragraph with the exact statement you want to make about the }z\to 0^-\text{ limit for }\zeta=0\text{, consistent with Noblesse (1981).}}
$$

The development of practical representations for $N$ and $W$ spans several works (Peters 1949; Noblesse 1981; Newman 1987). In particular, Noblesse provided a widely used integral representation that organizes the forward-speed Green’s function into smooth and oscillatory components, and Newman introduced a Chebyshev surrogate for the smooth near-field term $N$ that is well suited for accelerated evaluation.

The wavelike term admits oscillatory integral representations whose numerical evaluation becomes the central difficulty near the free surface.

In dimensionless variables scaled by $\ell$, a representative form for the wavelike contribution is

$$
W(x,y,z) = 4H(-x)\int_{-\infty}^{\infty} \exp\bigl(z(1+t^2)\bigr)\,\sin\bigl(g(x,y,t)\bigr)\,dt,
$$

with phase

$$
g(x,y,t) = (x + yt)\,\sqrt{1+t^2},
$$

and $H$ denoting the Heaviside function.

The near-field term $N$ is smooth in $(x,y,z)$ for $z\le 0$ and can be accelerated by surrogate approximation. Following Newman (1987), one may represent $N$ by Chebyshev polynomials in a transformed coordinate system that resolves the near-source structure while maintaining uniform accuracy in the far field.
"""

# ╔═╡ 1e6f98ef-f9a2-4b7f-bc11-b3e7d8b16a13
md"""
## 4. Neumann–Kelvin Formulation and the Waterline Contour

In Neumann–Kelvin formulations for steady forward speed, the Kelvin Green’s function is used in Green’s second identity in the domain $z<0$. For surface-piercing bodies, the free-surface contribution can be reduced to a waterline contour term because both $\phi$ and $G$ satisfy the same free-surface boundary condition.

Let $S$ denote the waterplane footprint of a surface-piercing body, with waterline contour $\partial S$. The FSBC implies

$$
\phi G_z - G\phi_z = -\ell\,(\phi G_{xx} - G\phi_{xx}) = -\ell\,\frac{\partial}{\partial x}(\phi G_x - G\phi_x),\qquad z=0.
$$

Integrating over the exterior free surface and applying the divergence theorem reduces this to a contour integral over $\partial S$.

In the notation used by Baar and Price, the resulting waterline contribution can be written in the form

$$
\phi(\mathbf{x})\ \supset\ \ell\oint_{\partial S} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,n_x\,dy_\alpha,
$$

where $q$ is the source strength and $n_x$ denotes the $x$-component of the outward unit normal in the free-surface plane. This contour expression is consistent with the divergence-theorem reduction applied to the identity above; the present manuscript will use it as the operational form for surface-piercing configurations.

### 4.1 Submerged and Surface-Piercing Representations

For a submerged body with wetted surface $S_B$ entirely below $z=0$, one may write schematically

$$
\phi(\mathbf{x}) = \iint_{S_B} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,dS_{\alpha},
$$

with $q$ determined by enforcing the body boundary condition.

For a surface-piercing body, the same derivation yields an additional waterline term, giving the operational representation

$$
\phi(\mathbf{x}) = \iint_{S_B} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,dS_{\alpha}
\; + \; \ell\oint_{\partial S} q(\boldsymbol{\alpha})\,G(\mathbf{x};\boldsymbol{\alpha})\,n_x\,dy_{\alpha}.
$$

$$
\boxed{\textbf{AUTHOR TODO:} \text{Insert the exact boundary integral equation used to determine }q\text{ (including any principal-value terms) and define the orientation/sign conventions for }\partial S\text{.}}
$$

The central computational point for the present paper is simply that the contour contribution requires evaluation at (or arbitrarily near) $z=0$, where the wavelike term is extremely rough and where classical series-based approaches are not uniformly valid (Newman 1987). This motivates focusing on a $z\to 0^-$ evaluation strategy that remains reliable in precisely that regime.

Related reformulations have been proposed to remove the explicit waterline contribution (e.g. the Neumann–Michell theory of Noblesse, Huang, and Yang 2013). The present work does not revisit those modeling arguments; it addresses the numerical obstacle that arises when the classical contour term is retained.

$$
\boxed{\textbf{AUTHOR TODO:} \text{(i) State the NK integral equation in your final notation; (ii) decide how strongly you want to position this work relative to Neumann–Michell theory; (iii) add a 1–2 sentence summary of why you retain the classical contour term in this paper.}}
$$
"""

# ╔═╡ f5f56210-9b75-4a0c-b8b5-2412ffdbf5a1
md"""
## 5. Limiting Behavior as $z\to 0^-$ and the Role of Stationary Points

The difficulty in evaluating $W$ as $z\to 0^-$ is not merely the loss of exponential damping in $\exp(z(1+t^2))$; it is also geometric. The phase $g(x,y,t)$ contains stationary points whose location and character depend on $(x,y)$ and control the asymptotic wave field.

### 5.1 Stationary Points of the Phase

Define

$$
g(x,y,t) = (x+yt)\sqrt{1+t^2}.
$$

Then

$$
\partial_t g = \frac{x t + y(2t^2+1)}{\sqrt{1+t^2}}.
$$

The stationary condition $\partial_t g=0$ yields

$$
x t + y(2t^2+1) = 0,
$$

which has (for $y\neq 0$) the solutions

$$
t_\pm = \frac{-x \pm \sqrt{x^2-8y^2}}{4y}.
$$

Thus:

- For $|y| > |x|/\sqrt{8}$ there are no real stationary points.
- For $|y| = |x|/\sqrt{8}$ the stationary points coalesce (a degenerate saddle).
- For $|y| < |x|/\sqrt{8}$ there are two real stationary points.

Moreover, as $y\to 0$ with $x\neq 0$, the relevant saddle migrates to infinity, shifting dominant spectral content to large wavenumber. This expresses a genuine focusing mechanism in the linear Kelvin dispersion geometry and implies non-commutation of certain limiting processes.

To make this explicit, note that for $|t|\gg 1$ one has $\sqrt{1+t^2}\sim |t|$ and hence the associated longitudinal and transverse wavenumbers scale like

$$
k_x \sim |t|,\qquad k_y \sim t\,|t|,\qquad k \equiv 1+t^2 \sim t^2.
$$

Consequently, when the dominant stationary contribution satisfies $|t_0|\to\infty$ (e.g. $y\to 0$ at fixed $x\neq 0$), the surface field draws support from increasingly large $k$.

This observation has two practical consequences. First, it clarifies why $W$ may be straightforward to evaluate for $z<0$ but becomes challenging as $z\to 0^-$. Second, it explains why Neumann–Kelvin problems can become ill-conditioned near the free surface: boundary conditions and hydrodynamic outputs involve derivatives of the potential, which accentuate the large-wavenumber content generated behind sources.
"""

# ╔═╡ 8d44d2f8-1d8d-45ee-8c0c-441a3ffbc2a2
md"""
### 5.2 A Unified Oscillatory-Integral Form

To compare point-source and spanwise-smoothed (line-integrated) kernels, it is convenient to write the wavelike contribution in the general form

$$
W_A(x,y,z) = 4H(-x)\int_{-\infty}^{\infty} A(t)\,\exp\bigl(z(1+t^2)\bigr)\,\sin\bigl(g(x,y,t)\bigr)\,dt,
$$

with phase $g(x,y,t)=(x+yt)\sqrt{1+t^2}$. The amplitude $A(t)$ encodes any spanwise smoothing:

- **Point source**: $A(t)\equiv 1$.
- **Elliptic line distribution on $[-b,b]$** (Section 6):
	$$
	A_b(t)=\pi\,\frac{J_1\bigl(bk(t)\bigr)}{k(t)},\qquad k(t)=t\sqrt{1+t^2}.
	$$

The fundamental numerical and analytical difficulty in the limit $z\to 0^-$ is the loss of exponential damping at large $|t|$. For $z<0$, the factor $\exp(z(1+t^2))$ makes the integral strongly filtered in $t$ and (for fixed $(x,y)$ away from singular geometries) renders the dependence on $(x,y)$ relatively smooth. At $z=0$, by contrast, the integral becomes purely oscillatory with no high-$|t|$ decay from the exponential factor, and the relevant stationary contributions may be driven to arbitrarily large $|t|$ as $y\to 0$.

This non-uniformity in $z$ underlies the practical observation (Newman 1987) that series representations which appear to converge for $z<0$ need not remain valid on $z=0$, particularly in settings where $y\to 0$ (e.g. along waterlines or behind contour endpoints).

### 5.3 Large-$|t|$ Saddle, Roughness at $z=0$, and the Role of $z<0$

The exact stationary points are

$$
t_\pm = \frac{-x \pm \sqrt{x^2-8y^2}}{4y},
$$

when they are real. In the near-centerline regime $|y|\ll |x|$ (with $x\neq 0$), these split into a small root and a large root. Expanding for $|y/x|\ll 1$ gives

$$
t_- = -\frac{y}{x} + \mathcal O\!\left(\frac{y^3}{x^3}\right),\qquad
t_+ = -\frac{x}{2y} + \mathcal O\!\left(\frac{y}{x}\right).
$$

The large stationary point $t_+$ is the key: as $y\to 0$, $|t_+|\to\infty$ and the dominant contribution is pushed to high wavenumber $k\sim t_+^2$.

To extract the associated scaling, note that for $|t|\gg 1$ one has $\sqrt{1+t^2}=|t|\,(1+\mathcal O(t^{-2}))$, hence

$$
g(x,y,t)=(x+yt)\sqrt{1+t^2}\;=\;\operatorname{sgn}(t)\,(x t + y t^2)\; +\; \mathcal O(1).
$$

On either half-line, the leading-order stationary condition is therefore $x+2yt\approx 0$, giving

$$
t_0\;\approx\; -\frac{x}{2y}.
$$

Moreover, at an exact stationary point $t_0$ one has

$$
g''(t_0)=\frac{x+4y t_0}{\sqrt{1+t_0^2}},
$$

and for the large root $t_+\sim -x/(2y)$ this reduces to $|g''(t_+)|\sim 2|y|$. A standard stationary-phase estimate then yields the magnitude

$$
|W_A(x,y,z)|\;\sim\; C\,|A(t_+)|\,\exp\bigl(z(1+t_+^2)\bigr)\,|y|^{-1/2},
$$

where $C$ is an $\mathcal O(1)$ constant (absorbing the usual $\sqrt{2\pi}$ factor and phase shift).

Two limiting cases follow immediately.

1. **Point-source roughness on $z=0$ (downstream, $x<0$)**. With $A\equiv 1$ and $z=0$ (so that $H(-x)=1$),
	 $$
	 |W(x,y,0)|\sim |y|^{-1/2}.
	 $$
	 Since $\partial_x g=\sqrt{1+t^2}\sim |t_+|\sim |x|/(2|y|)$, differentiation amplifies the roughness; for example,
	 $$
	 |\partial_x W(x,y,0)|\ \text{typically scales like}\ \mathcal O\bigl(|y|^{-3/2}\bigr).
	 $$
	 This extreme lack of smoothness along $y=0$ is a boundary phenomenon and is one reason why representations that are numerically benign for $z<0$ can fail catastrophically as $z\to 0^-$. In particular, any putative representation on $z=0$ that implies a uniformly smooth dependence on $y$ near $y=0$ is incompatible with the large-saddle scaling above.

2. **Spectral filtering for any fixed $z<0$**. For $z<0$, the same saddle estimate carries the factor
	 $$
	 \exp\bigl(z(1+t_+^2)\bigr)\approx \exp\!\left(-\frac{|z|x^2}{4y^2}\right),
	 $$
	 which suppresses the large-$|t|$ saddle contribution as $y\to 0$.
	 Thus the interior field is comparatively smooth, while the limit $z\to 0^-$ is non-uniform: the scale $|t|\lesssim |z|^{-1/2}$ (equivalently $k\lesssim |z|^{-1}$) expands without bound.

In the remainder of the manuscript, “regularization” by line integration is understood in the specific sense that $A(t)$ decays for large $|t|$, so that $|A(t_+)|$ decreases as $y\to 0$. This reduces (but does not remove) the practical roughness of the $z=0$ field: even when the centerline blow-up is suppressed, limited smoothness remains across wake boundaries (e.g. at the cusp $|y|=|x|/\sqrt{8}$) where stationary points appear/disappear.
"""

# ╔═╡ f45c5c03-4a3d-4f96-aea5-30fa52b1e54f
md"""
## 6. Line Integrals over $[-b,b]$ and Regularization by Spanwise Smoothing

Waterline-driven configurations naturally lead to spanwise-smoothed (line-integrated) kernels. For a representative symmetric line distribution define

$$
W_b(x,y,z) = \int_{-b}^{b} W(x, y-y', z)\,dy'.
$$

In the unified notation of Section 5.2, line integration modifies the oscillatory integral only through the amplitude $A(t)$.

This is also the structural reason the flat-ship model becomes exceptionally cheap once $W_A$ can be evaluated robustly at $z\to 0^-$: after edge reduction, the entire planform appears only through two evaluations of the same spanwise-smoothed kernel (one at each chordwise edge), with the spanwise smoothing absorbed into $A(t)$.

For the elliptic strength distribution arising in the flat-ship model (Section 8), the Fourier transform yields a Bessel factor, and the wavelike contribution takes the explicit regularized form

$$
\int_{-b}^{b} \sqrt{1-(y'/b)^2}\,W(x,y-y',z)\,dy'
= 4\pi H(-x)\int_{-\infty}^{\infty}\frac{J_1(bk(t))}{k(t)}\,\exp\bigl(z(1+t^2)\bigr)\,\sin\bigl(g(x,y,t)\bigr)\,dt,
$$

where $k(t)=t\sqrt{1+t^2}$. The prefactor $J_1(bk)/k$ is finite at $t=0$ with limit $b/2$, and provides additional decay for large $|t|$.

Equivalently,

$$
W_{A_b}(x,y,z)=4H(-x)\int_{-\infty}^{\infty} A_b(t)\,\exp\bigl(z(1+t^2)\bigr)\,\sin\bigl(g(x,y,t)\bigr)\,dt,
$$

with

$$
A_b(t)=\pi\,\frac{J_1\bigl(bk(t)\bigr)}{k(t)}.
$$

### 6.1 Stationary-Point Scaling with General $A(t)$

The large-$|t|$ saddle estimate of Section 5.3 gives, at $z=0$,

$$
|W_A(x,y,0)|\sim C\,|A(t_+)|\,|y|^{-1/2},\qquad t_+\sim -\frac{x}{2y},\qquad k\sim t_+^2.
$$

Thus the high-wavenumber behavior of $A(t)$ directly controls the severity of the free-surface roughness.

For the elliptic line amplitude $A_b$, one has $k(t)\sim t^2$ as $|t|\to\infty$ and the Bessel asymptotic implies an envelope

$$
|A_b(t)|\ \text{decays like}\ \mathcal O\!\left(b^{-1/2}|t|^{-3}\right),\qquad |t|\to\infty.
$$

Substituting $|t_+|\sim |x|/(2|y|)$ and $k\sim t_+^2$ yields the qualitative scalings

$$
|W_{A_b}(x,y,0)|\sim b^{-1/2}\,k^{-3/2}\,|y|^{-1/2},
$$

and, since $\partial_x g\sim |t_+|\sim k^{1/2}$,

$$
|\partial_x W_{A_b}(x,y,0)|\sim b^{-1/2}\,k^{-1}\,|y|^{-1/2}.
$$

These relations quantify the regularization induced by spanwise smoothing: compared with the point source ($A\equiv 1$), the effective amplitude at the large saddle decreases as $y\to 0$ because the spanwise averaging suppresses high wavenumber.

The regularization is not absolute. Even when the centerline blow-up is removed, the free-surface field remains comparatively rough because the saddle structure itself changes across the cusp $|y|=|x|/\sqrt{8}$, and because the $z=0$ integral is still dominated by oscillations with large local wavenumber in substantial portions of the wake.

### 6.2 Effect of Small Negative $z$

For any fixed $z<0$, the exponential factor supplies additional damping, and the saddle contribution carries

$$
\exp\bigl(z(1+t_+^2)\bigr)\approx \exp(-|z|k).
$$

In particular, the scale of contributing wavenumbers expands like $k\lesssim |z|^{-1}$ as $z\to 0^-$, and the point-source case $A\equiv 1$ is therefore the worst case. Line-integrated amplitudes reduce the sensitivity by introducing algebraic decay in $k$.

$$
\boxed{\textbf{AUTHOR TODO:} \text{Decide how sharply you want to state the “non-uniform limit” issue (and cite Newman’s remark precisely), and whether you want to present the above as a proposition with explicit remainder estimates.}}
$$
"""

# ╔═╡ f6c4f8db-2ffd-4f56-8c93-1e0f0f55d3d5
md"""
## 7. Numerical Evaluation by Contour Deformation

### 7.1 Relation to Numerical Steepest Descent for Analytic Phases

Recent developments in numerical contour deformation and numerical steepest descent (Huybrechs 2017; Huybrechs **TODO (textbook)**; Gibbs & Huybrechs 2024) provide a systematic framework for highly oscillatory integrals. A key organizational principle is to partition the real line into neighborhoods of stationary points and complementary “tails”, and to deform each tail into the complex plane where the integrand decays.

In the ship-wave context, Iwashita and Ohkusu (1992) also employed a steepest-descent-type methodology, but in a substantially different formulation that is difficult to interpret in detail from the published description and appears to require separate treatments as $z\to 0^-$. Moreover, their published results focus on submerged configurations; the $z=0$ evaluation required by waterline contour integrals is not directly addressed.

$$
\boxed{\textbf{AUTHOR TODO:} \text{Insert the specific statements you want to make about Iwashita\&Ohkusu’s methodology (e.g. the dimensionality of their complex deformation and the “two approaches” issue), and cite any additional Japanese-language sources if you wish to mention ship results.}}
$$

In the present application, two features require additional care:

1. The phase $g(x,y,t)$ is not analytic globally because of $\sqrt{1+t^2}$.
2. For line-integrated kernels, the Bessel prefactor is best treated through a decomposition that isolates slowly varying factors.

### 7.2 Non-analytic Phase and Branch Structure

The square-root factor introduces branch points at $t=\pm i$. Any deformation must respect the chosen branch cuts. The deformation strategy therefore combines (i) finite real-line segments covering neighborhoods of stationary points and other special points, with (ii) complex deformations for the tails that remain a fixed distance from the branch points.

In computations reported here, the square-root branch is selected so that $\sqrt{1+t^2}$ remains continuous along each deformed tail contour; in practice this can be implemented by a phase-based sign selection for the complex square root that avoids branch-cut crossings along admissible paths.

$$
\boxed{\textbf{AUTHOR TODO:} \text{State the branch convention explicitly (e.g. via }\arg(1+t^2)\text{) and connect it to the practical implementation (your sign choice for the complex square root).}}
$$

### 7.3 Hankel Decomposition for Line-Integrated Kernels

To avoid evaluating Hankel functions near their singular behavior at $t=0$, the real-line partition is chosen to include a segment containing $t=0$ on which the Bessel representation is used. Away from this segment, one may represent

$$
J_1(z) = \frac{1}{2}\left(H_1^{(1)x}(z)e^{iz} + H_1^{(2)x}(z)e^{-iz}\right),
$$

where $H_1^{(1)x}$ and $H_1^{(2)x}$ denote scaled Hankel functions (slowly varying, non-oscillatory). The exponentials are absorbed into the complex phase, producing tail integrals suitable for steepest descent.

**TODO**: Present the precise pairing between the two Hankel branches and the deformed paths, and state the sufficient conditions guaranteeing exponential decay of each tail contribution.

### 7.4 Practical Partitioning by Finite Phase Ranges

The implementation used in this work does not attempt to deform contours in neighborhoods of stationary points. Instead, the real axis is partitioned into finite intervals such that each interval endpoint is separated from the nearest stationary point by a prescribed phase increment $\Delta g$ (measured in the real phase $g$). On these finite intervals, standard quadrature on the real axis is employed.

The semi-infinite tails are then evaluated using numerical stationary phase on a complex path emanating from the interval endpoints. The key numerical parameter is $\Delta g$: too small places tail evaluation too close to a stationary point (loss of accuracy), while too large increases real-line quadrature cost. In the computations motivating this manuscript, values on the order of $\Delta g\approx 5$–$6$ were effective for point-source evaluations; the line-integrated kernel requires larger separation.

Because the phase is non-analytic and highly nonlinear, the interval endpoints are determined by numerical root finding rather than by explicit local Taylor models. For each stationary point $a\in S$, one solves

$$
|g(t)-g(a)| = \Delta g
$$

to locate the interval boundaries, with safeguards for finite truncation $|t|\le R$.

Outside the wake cusp (where no real stationary point exists), a single pseudo-stationary point $t_0=-x/(4y)$ is used to organize the partition.

### 7.5 Algorithm Summary (Operational Form)

For clarity, we summarize the operational evaluation of integrals of the form

$$
I = \int_{-\infty}^{\infty} a(t)\,\sin(g(t))\,dt,
$$

where $a(t)$ includes $\exp(z(1+t^2))$ and any slowly varying prefactors. The procedure is:

1. Choose a finite truncation $R$ and phase increment $\Delta g$.
2. Form a stationary-point set $S$ for the relevant phases and include $t=0$ so that a finite real segment covers the origin.
3. Construct finite real-line intervals by solving $|g(t)-g(a)|=\Delta g$ for each $a\in S$ (numerically), clipping to $[-R,R]$.
4. Integrate on each finite interval using fixed Gauss–Legendre quadrature.
5. For flagged endpoints, evaluate the semi-infinite tails using numerical stationary phase on a complex path defined implicitly by $g(h)-g(h_0)=-ip$ at Gauss–Laguerre nodes $p$.

In the computations motivating this manuscript, four Gauss–Laguerre nodes are used per tail evaluation, and Newton’s method is used to locate the contour points with a modest tolerance; the overall accuracy is controlled primarily by the phase partitioning parameter $\Delta g$.

$$
\boxed{\textbf{AUTHOR TODO:} \text{Insert the precise parameter choices used in your reported experiments (e.g. }\Delta g\approx 5\text{–}6\text{ for point-source; }\Delta g\approx 12\text{ for the line-integrated kernel), and provide a short sensitivity discussion.}}
$$
"""

# ╔═╡ 0c83e5ef-98f6-4143-9b33-0c9d9ecb7f7b
md"""
## 8. The Horizontal Thin-Ship Problem (Flat-Ship Theory)

This section summarizes a reduced model for waterline-driven wave generation by a horizontal, surface-piercing planform. The purpose is twofold: (i) it provides a physically relevant application in which $z\to 0^-$ evaluation is intrinsic, and (ii) it yields line-integrated kernels with analytically tractable spanwise structure.

Most importantly for the present manuscript, it produces an edge-only representation in which the exterior predictions reduce to just two evaluations of a spanwise-smoothed Kelvin kernel (leading and trailing edges), once a one-dimensional closure for the spanwise source strength is adopted.

### 8.1 Baar–Price Contour Representation

Following Baar and Price (1988), represent the velocity potential for steady forward speed $U$ as a distribution of Kelvin sources. The FSBC

$$
\ell \phi_{xx} + \phi_z = 0,\qquad z=0,
$$

with $\ell=U^2/g$, is satisfied by construction when using $G$. Applying Green’s second identity in $z<0$, the contribution from the $z=0$ plane splits into the hull footprint $S$ and the exterior free surface $F$. On $F$, both $\phi$ and $G$ satisfy the FSBC, hence

(Here “satisfied by construction” is understood in the usual sense that the Green’s function enforces the FSBC on the *exterior* free-surface portion of the boundary.)

$$
\phi G_z - G\phi_z
= -\ell(\phi G_{xx} - G\phi_{xx})
= -\ell\frac{\partial}{\partial x}(\phi G_x - G\phi_x),
$$

which is an exact $x$-derivative. After integration over $F$ and application of the divergence theorem, the free-surface contribution reduces to a contour integral around the waterline $\partial S$. Side edges parallel to $x$ do not contribute, leaving leading and trailing edges.

This reduction is used here as an efficient representation of the lid/footprint influence on the *exterior* flow. It does not by itself determine the source strength; a modeling choice (closure) for the planform distribution is still required.

### 8.2 Flat Planing Hull Boundary Condition

For a flat planing hull at small angle of attack $\alpha$, the flat-ship idealization imposes a uniform downwash condition on a lid approaching the free surface. In the limiting notation used here this is written as the $z\to 0^-$ condition

$$
\phi_z = \alpha U,\qquad (x,y)\in S,\ z=0.
$$

In the present reduced model we do not attempt to solve the full lid boundary-integral equation for a two-dimensional density $q(x',y')$. Instead we adopt a leading-order closure consistent with uniform forcing and with the fact that the finite chordwise extent is already handled explicitly by the edge reduction: $q$ is taken approximately uniform in $x'$ and chosen primarily to satisfy the dominant nearfield (singular-operator) structure in the spanwise direction.

### 8.3 Reduction to Edge Integrals

With $q$ taken uniform in $x'$ and written in separable form $q(x',y')\approx q(y')$, the potential may be written schematically as

$$
\phi = \alpha U\iint_S q(y')\,G(x-x',y-y',z)\,dx'\,dy'.
$$

The same divergence-theorem argument that produces the contour integral eliminates the interior $x'$ contribution. The result reduces to contributions from the leading edge $x'=x_L$ and trailing edge $x'=x_T$:

$$
\phi = \alpha U\,\ell\int_{-b}^{b} q(y')\Bigl[G(x-x_T,y-y',z)-G(x-x_L,y-y',z)\Bigr]dy'.
$$

### 8.4 Spanwise Distribution

The remaining unknown is a one-dimensional spanwise distribution $q(y')$. For a uniform downwash forcing, the dominant nearfield operator leads to a constant-downwash singular integral equation whose leading kernel is logarithmic in span, i.e. of the schematic form

$$
\int_{-b}^{b} q(y')\log|y-y'|\,dy' = \mathrm{const}.
$$

This is the classical constant-downwash problem; the solution on $[-b,b]$ is the elliptic distribution

$$
q(y') = q_0\sqrt{1-(y'/b)^2}.
$$

This choice is also consistent with the endpoint requirement $q(\pm b)=0$, which suppresses spurious endpoint injection of high spanwise wavenumbers and therefore preserves the regularization benefits of line integration in the $z\to 0^-$ Kelvin evaluation.

### 8.5 Line-Integrated Green’s Function

For each edge, the induced potential is therefore

$$
\phi_{\mathrm{edge}} = \alpha U\,\ell\int_{-b}^{b} q_0\sqrt{1-(y'/b)^2}\,G(x-x_{\mathrm{edge}},y-y',z)\,dy'.
$$

The evaluation of the wavelike part of this expression in the limit $z\to 0^-$ is the main computational objective.

From a computational perspective, the noteworthy point is the *edge-only* structure: once a one-dimensional spanwise closure $q(y')$ is adopted, the exterior wave field associated with a shallow-draft, high-speed planform can be generated from just two spanwise line-integrals (leading and trailing edges). In practice, this means that for each field point one makes two calls to the same line-integrated Kelvin-kernel evaluator—one per edge—after which free-surface elevation and derived integrated loads can be formed from standard linear relations. This should be read as an efficiency/enabling statement (made possible by robust $z\to 0^-$ evaluation), not as a statement of flat-ship model correctness.

(Equivalently: wave-field and force/moment *predictions* in this reduced setting are based on **two function evaluations** per field point; the accuracy of those predictions is a separate modeling question tied to the chosen closure and linearization.)
"""

# ╔═╡ 4a0b4a48-98b3-4c10-9e4e-8c2175f0d18f
md"""
## 9. Validation and Computational Studies (Placeholders)

This section lists the intended verification and demonstration problems.

1. **Pointwise limits**: $z=y=0$ and other special cases with known closed forms.
2. **Free-surface field**: $z=0$ contour plots highlighting the lack of smoothness and the wake structure.
3. **Submerged spheroid**: validation of Neumann–Kelvin predictions against the classical analytic result for steady wave resistance (Farrell, **TODO full citation**), as is standard in the subsequent literature.
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
- Noblesse, F., Huang, F., and Yang, C. (2013). **TODO** (full bibliographic entry; Neumann–Michell reformulation).
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
# ╟─0c83e5ef-98f6-4143-9b33-0c9d9ecb7f7b
# ╟─4a0b4a48-98b3-4c10-9e4e-8c2175f0d18f
# ╟─8cde3f1f-b4d7-44a6-9c4f-e3c0b388bd68
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
