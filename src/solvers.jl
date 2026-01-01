using Krylov,LinearOperators
"""
    GMRESsolve!(sys, b=components(sys.panels.n,1); atol=1e-3, verbose=true)

Solve a panel system using GMRES iteration such that the normal velocity
boundary condition `∂ₙΦ(xᵢ,sys) = bᵢ` is satisfied on all panels.

**Note**: This function calls Φ on every panel center every GMRES iteration.
If Φ(xᵢ,sys) is **very** slow, this solver could be slower than a direct solve.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `atol=1e-3`: Absolute tolerance for GMRES convergence
- `verbose=true`: Print GMRES convergence statistics

# Returns
Modified `sys` with updated panel strengths in `sys.panels.q`

# References
Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual
algorithm for solving nonsymmetric linear systems. SIAM Journal on Scientific
and Statistical Computing, 7(3), 856-869.

# Example
```julia
BH = BarnesHut(panels)  # crazy fast initialization
GMRESsolve!(BH)         # fast solve
extrema(panel_cp(BH))   # very fast measure
```
"""
function GMRESsolve!(sys,b=components(sys.panels.n,1);atol=1e-3,verbose=true)
    # Make LinearOperator
    mult!(b,q) = (set_q!(sys,q); uₙ!(b,sys))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)

    # Solve with GMRES and return updated BarnesHutBEM
    q, stats = gmres(A, b, sys.panels.q; atol)
    verbose && println(stats)
    set_q!(sys,q)
end
@inline set_q!(sys,q) = (sys.panels.q .= q; sys)
@inline uₙ!(b::AbstractArray{T},sys) where T = AK.foreachindex(b) do i
    b[i] = derivative(t->Φ(sys.panels.x[i]+t*sys.panels.n[i],sys;val=zero(T)),0)
end

"""
    directsolve!(sys, b=components(sys.panels.n,1))

Solve a panel system using a direct construction and solve such that the normal
velocity boundary condition `∂ₙϕ(pᵢ,pⱼ)*qⱼ = bᵢ` is satisfied on all panels.

**Note**: This function is memory and time intensive for large number of panels N
because it constructs the full N² matrix elements and calls a linear algebra
routine that decomposes the matrix and solves for `q`.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `verbose=true`: Print direct solve statistics

# Returns
Modified `sys` with updated panel strengths in `sys.panels.q`

# Example
```julia
sys = PanelSystem()
GMRESsolve!(BH)         # fast solve
extrema(panel_cp(BH))   # very fast measure
```
"""
function directsolve!(sys::PanelSystem,b=components(sys.panels.n,1);verbose=true)
    if verbose
        @warn "This routine is memory and time intensive. See help?>directsolve!."
        @time sys.panels.q .= _direct(sys,b)
    else
        sys.panels.q .= _direct(sys,b)
    end;sys
end
function _direct((;panels,ϕ,mirrors,kwargs)::PanelSystem,b)
    ϕ_sym(x,p) = sum(ϕ(x .* m,p;kwargs...) for m in mirrors)
    ∂ₙϕ.(panels,panels';ϕ=ϕ_sym)\b
end