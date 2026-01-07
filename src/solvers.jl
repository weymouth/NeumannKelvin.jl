using Krylov,LinearOperators
"""
    gmressolve!(sys, b=components(sys.panels.n,1); atol=1e-3, verbose=true)

Approximately solves a panel system using GMRES iteration to satisfy the normal
velocity condition `Φₙ(xᵢ,sys) = bᵢ = -U⋅n`.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `atol=1e-3`: Absolute tolerance for GMRES convergence
- `verbose=true`: Print GMRES convergence statistics

# Returns
Modified `sys` with updated panel strengths `q`

# References
Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual
algorithm for solving nonsymmetric linear systems. SIAM Journal on Scientific
and Statistical Computing, 7(3), 856-869.

# Example
```julia
sys = PanelSystem(panels)  # set-up
gmressolve!(sys)           # approximate solve
extrema(cₚ(sys))           # measure
```
"""
function gmressolve!(sys,b=components(sys.panels.n,1);atol=1e-3,verbose=true,kwargs...)
    # Make LinearOperator
    mult!(b,q) = (sys.panels.q .= q; AK.foreachindex(i->b[i]=Φₙ(sys.panels[i],sys),b))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)

    # Solve with GMRES and return updated BarnesHutBEM
    q, stats = gmres(A, b, sys.panels.q; atol, kwargs...)
    verbose && println(stats)
    sys.panels.q .= q; sys
end

"""
    directsolve!(sys, b=components(sys.body.n,1))

Solve a panel system using a direct construction and solve such that the normal
velocity boundary condition `∂ₙϕ(pᵢ,pⱼ)*qⱼ = bᵢ` is satisfied on body panels.

**Note**: This function is memory (and therefore time) intensive for large number of
panels N because it constructs the full N² matrix elements and calls a linear algebra
routine that decomposes the matrix and solves for `q`.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `b`: Right-hand side vector (default: `p.n[1]`, corresponding to `U=[-1,0,0]`)
- `verbose=true`: Print warning and solve time

# Returns
Modified `sys` with updated panel strengths `q`

# Example
```julia
sys = PanelSystem(panels) # set-up
directsolve!(sys)         # solve
extrema(cₚ(sys))          # measure
```
"""
function directsolve!(sys::PanelSystem,b=components(sys.panels.n,1);verbose=true)
    if verbose
        @warn "This routine is memory intensive. See help?>directsolve!."
        @time sys.panels.q .= _direct(sys,b)
    else
        sys.panels.q .= _direct(sys,b)
    end;sys
end
function _direct((;panels,ϕ,mirrors,kwargs)::PanelSystem,b)
    ϕ_sym(x,p) = sum(ϕ(x .* m,p;kwargs...) for m in mirrors)
    A = Array{eltype(panels.q)}(undef,length(panels.q),length(panels.q))
    AK.foraxes(A,2) do j
        @simd for i in axes(A,1)
            @inbounds A[i,j] = ∂ₙϕ(panels[i],panels[j];ϕ=ϕ_sym)
        end
    end
    A\b
end