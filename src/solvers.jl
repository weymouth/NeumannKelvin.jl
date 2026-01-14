using Krylov,LinearOperators
"""
    gmressolve!(sys; verbose=true, atol=1e-3, kwargs...)

Approximately solves a panel system using GMRES iteration to satisfy the panel
boundary conditions:
 - On body panels, the normal velocity condition `Φₙ(xᵢ,sys) = bᵢ = -U⋅n`
 - On free surface panels, Φₙ(xᵢ,sys)+sys.ℓ*Φₓₓ(xᵢ,sys) = 0`, where `ℓ=U²/g` is the
 Foude length.

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
BH = BarnesHut(panels)  # set-up
gmressolve!(BH)         # approximate solve
extrema(cₚ(BH))         # measure
```
"""
function gmressolve!(sys;atol=1e-3,verbose=true,kwargs...)
    # Make LinearOperators
    b = similar(panels.q); @. b = -Ref(sys.U)'components(sys.panels.n)
    p = isnothing(sys.freesurf) ? nothing : similar(b,length(sys.freesurf))
    mult!(b,q) = (set_q!(sys,q); bodybc!(b,sys); !isnothing(p) && fsbc!(b,p,sys))
    A = LinearOperator(eltype(b), length(b), length(b), false, false, mult!)
    M = LinearOperator(eltype(b), length(b), length(b), false, false, (z,r)->precon!(z,sys,r))

    # Solve with GMRES and return updated BarnesHutBEM
    q, stats = gmres(A, b, sys.panels.q; M, atol=convert(eltype(b),atol), kwargs...)
    verbose && println(stats)
    set_q!(sys,q)
end
@inline set_q!(sys,q) = (sys.panels.q .= q; sys)
@inline bodybc!(b,sys) = AK.foreachindex(i-> b[i] = Φₙ(sys.panels[i],sys),b) # body: Φₙ = -Uₙ
@inline function fsbc!(b,p,sys)   # freesurf: Φₙ-ℓ*Φₓₓ = 0
    ℓ = sys.ℓ[1]; Nᵢ,Nⱼ = sys.fssize; Nb = length(sys.body)
    AK.foreachindex(i-> p[i] = Φ(sys.freesurf.x[i],sys), p) # fill p->Φ
    AK.foreachindex(p) do j  # add ℓΦₓₓ contribution
        (j-1)%Nᵢ<3 && return     # too close to leading edge of freesurf
        i = j+Nb; h = extent(view(sys.panels.x,i-1:i))[1]
        @inbounds b[i] -=ℓ*(2p[j]-5p[j-1]+4p[j-2]-p[j-3])/h^2 # 3rd order upwind FD
    end
end
function precon!(z,sys,r)
    z .= r # identity preconditioner
    isnothing(sys.freesurf) && return

    # sweep resdual information downstream
    Nᵢ,Nⱼ = sys.fssize; Nb = length(sys.body)
    AK.foreachindex(view(z,1:Nⱼ)) do j
    for i in (2:Nᵢ) .+ (Nb+(j-1)*Nᵢ)
        @inbounds z[i] += z[i-1]
    end;end
end

"""
    directsolve!(sys)

Solve a panel system using a direct construction and solve such that the normal
velocity boundary condition `∂ₙϕ(pᵢ,pⱼ)*qⱼ = bᵢ` is satisfied on body panels.

**Warning**: This function ignores sys.freesurf!

**Note**: This function is memory (and therefore time) intensive for large number of
panels N because it constructs the full N² matrix elements.

# Arguments
- `sys`: Pre-constructed panel system (modified in-place)
- `verbose=true`: Print warning and solve time

# Returns
Modified `sys` with updated body panel strengths `q`

# Example
```julia
sys = PanelSystem(panels) # set-up
directsolve!(sys)         # solve
extrema(cₚ(sys))          # measure
```
"""
function directsolve!(sys::PanelSystem;verbose=true)
    b = similar(panels.q); @. b = -Ref(sys.U)'components(sys.panels.n)
    if verbose
        @warn "This routine ignores free surface panels and is memory intensive. See help?>directsolve!."
        @time sys.body.q .= influence(sys)\b
    else
        sys.body.q .= influence(sys)\b
    end;sys
end
function influence((;body,mirrors)::PanelSystem)
    ϕ_sym(x,p) = sum(m->∫G(x .* m,p),mirrors)
    A = Array{eltype(body.dA)}(undef,length(body),length(body))
    AK.foraxes(A,2) do j
        @simd for i in axes(A,1)
            @inbounds A[i,j] = ∂ₙϕ(body[i],body[j];ϕ=ϕ_sym)
        end
    end; A
end