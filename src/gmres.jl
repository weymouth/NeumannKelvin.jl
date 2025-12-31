using Krylov,LinearOperators
"""
    GMRESSolve!(sys, b=components(sys.panels.n,1); atol=1e-3, verbose=true)

Solve a linear system using GMRES iteration such that the normal velocity
boundary condition `∂ₙΦ(xᵢ,sys) = bᵢ` is satisfied on all panels.

**Note**: This function calls Φ on every panel center every GMRES iteration.
If Φ(xᵢ,sys) is very very slow, this solver could be worse than a direct solve.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `atol=1e-3`: Absolute tolerance for GMRES convergence
- `verbose=true`: Print GMRES convergence statistics

# Returns
Modified `sys` with updated panel strengths in `sys.panels.q`

# Example
```julia
BH = BarnesHut(panels)  # crazy fast initialization
GMRESsolve!(BH)         # fast solve
extrema(panel_cp(BH))   # very fast measure
```

See also: [`BarnesHut`](@ref)
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