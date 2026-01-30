using Krylov,LinearOperators
"""
    gmressolve!(sys; verbose=true, atol=1e-3, kwargs...)

Approximately solves a panel system using GMRES iteration to satisfy the panel
boundary conditions.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `verbose=true`: Print GMRES convergence statistics
- `atol=1e-3, kwargs...`: Absolute tolerance and other arguments passed to gmres

# Returns
Modified `sys` with updated panel strengths `q`

# References
Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual
algorithm for solving nonsymmetric linear systems. SIAM Journal on Scientific
and Statistical Computing, 7(3), 856-869.

# Example
```julia
sys = BodyPanelSystem(panels,wrap=PanelTree)  # set-up
gmressolve!(sys)         # approximate solve
extrema(cₚ(sys))          # measure
```
"""
function gmressolve!(sys;atol=1e-3,verbose=true,kwargs...)
    # Make LinearOperators
    b,q = rhs(sys),get_q(sys)
    mult!(b,q) = (set_q!(sys,q); bc!(b,sys))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)
    M = LinearOperator(eltype(b), length(b), length(b), false, false, (z,r)->precon!(z,sys,r))

    # Solve with GMRES and return updated PanelSystem
    q, stats = gmres(A, b, q; M, atol=convert(eltype(b),atol), kwargs...)
    verbose && println(stats)
    set_q!(sys,q)
end
@inline rhs(sys) = rhs(sys.body,sys.U)
@inline rhs(panels,U) = -sum(components(panels.n) .* U)
@inline bc!(b,sys) = AK.foreachindex(i-> b[i] = Φₙ(sys.body[i],sys),b) # body: Φₙ = -Uₙ
@inline precon!(z,sys,r) = z .= r # identity
"""
    directsolve!(sys)

Solve a panel system using a direct construction and solve such that the normal
velocity boundary condition `∂ₙϕ(pᵢ,pⱼ)*qⱼ = -U⋅nᵢ` is satisfied on body panels.
This function can not be used to solve an `FSPanelSystem`.

**Note**: This function is memory (and therefore time) intensive for large number of
panels N because it constructs the full N² matrix elements. It is *not* accelerated
with a PanelTree, but *is* accelerated when Threads.nthreads()>1.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `verbose=true`: Print warning and solve time

# Returns
Modified `sys` with updated body panel strengths `q`

# Example
```julia
sys = BodyPanelSystem(panels) # set-up
directsolve!(sys)             # solve
extrema(cₚ(sys))               # measure
```
"""
function directsolve!(sys;verbose=true)
    typeof(sys)<:FSPanelSystem && throw(ArgumentError("Cannot directsolve! an FSPanelSystem"))
    q = if verbose
        @warn "This routine is memory intensive. See help?>directsolve!."
        @time influence(sys)\rhs(sys.body,sys.U)
    else
        influence(sys)\rhs(sys.body,sys.U)
    end; set_q!(sys.body,q); sys
end
influence((;body,mirrors)::AbstractPanelSystem) = influence(body,mirrors,∫G)
function influence(body,mirrors,ϕ)
    A = zeros(length(body),length(body))
    AK.foraxes(A,2) do j; for m in mirrors, i in axes(A,1)
        @inbounds A[i,j] += derivative(t->ϕ((body.x[i]+t*body.n[i]) .* m, body[j]), 0)
    end; end; A
end