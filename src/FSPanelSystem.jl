"""
    FSPanelSystem(body,freesurf::Matrix{QuadPanels}; ℓ, Umag=1, sym_axes=(), wrap=PanelTree)

A PanelSystem defined by `body` and `freesurf` source panels.

**Note**: `freesurf` must be a matrix of quadralateral panels with the first index
of the matrix aligned with -x. i.e. `freesurf[i+1,j].x-freesurf[i,j].x ≈ [-Δxᵢⱼ,0,0]`.
Similarly, the direction of the flow will always be `Û=-x̂`. Relative flow vectoring
(i.e. drift angle) must be acheived by rotating the body.

keyword arguments:
- `ℓ ≡ Umag²/g` Froude length
- `Umag` Optional background flow magnitude
- `sym_axes, wrap` see BodyPanelSystem

# Details

The convective derivative Φₓₓ in the FSBC is estimated by computing Φ at the FS panel
centers and using a 2nd-order upwind finite difference in `i`. This maintains downwind
wave propigation and adds (significant) numerical damping. A downwind flooding gmres
preconditioner is used to accelerate convergence, but it still requires O(100) iterations.
The accuracy of the solution depends on the free surface resolution; at minimum h<ℓ is required.

# Usage
```julia
sys = FSPanelSystem(body_panels,freesurf;ℓ=1/2π) # body + free surface
gmressolve!(sys, atol=1e-6)  # approximate solve
extrema(cₚ(sys))             # check solution quality
```
"""
struct FSPanelSystem{B,F,D,L,T,M} <: AbstractPanelSystem
    body::B      # body panels
    freesurf::F  # free surface panels
    fsm::D       # free-surface sized Matrix
    ℓ::L         # Froude-length
    U::SVector{3,T}
    mirrors::M
end
function FSPanelSystem(body,freesurf::AbstractMatrix; Umag=1, ℓ, sym_axes=(), wrap=PanelTree, kwargs...)
    # Lots of sanitary input checks...
    fsm = zeros(eltype(body.dA),size(freesurf)); freesurf = Table(freesurf)
    eltype(freesurf.dA) != eltype(fsm) && throw(ArgumentError("Floating point type of body and freesurf panels must match"))
    freesurf.n[1][3]>0 && throw(ArgumentError("`freesurf` panels must point down."))
    dx = freesurf.x[2]-freesurf.x[1]
    (dx[1]>-abs(dx[2]) || dx[3]≠0) && throw(ArgumentError("first index of `freesurf` must align with -x."))

    FSPanelSystem(snug(body,wrap;kwargs...), snug(freesurf,wrap;kwargs...), fsm, ℓ, SA[-abs(Umag),0,0], mirrors(sym_axes...))
end

# Pretty printing
Base.show(io::IO, sys::FSPanelSystem) = println(io, "FSPanelSystem($(length.(domains(sys))) panels, ℓ=$(sys.ℓ)")
function Base.show(io::IO, ::MIME"text/plain", sys::FSPanelSystem)
    println(io,"FSPanelSystem")
    print(  io, "  freesurf: "); show(io,sys.freesurf)
    println(io, "     size: $(size(sys.fsm))")
    println(io, "     panel type: $(eltype(sys.freesurf.kernel))")
    println(io, "  Froude length ℓ: $(sys.ℓ)")
    abstract_show(io,sys)
end

# Calculate the potential
@inline domains(sys) = (sys.body,sys.freesurf)
Φ(x,sys::FSPanelSystem) = sum(m->Φ_dom(x .* m,sys.body)+Φ_dom(x .* m,sys.freesurf),sys.mirrors)

# Set/Get the strength
@inline dviews(q,sys) = ((Nb,Nfs) = length.(domains(sys));view.(Ref(q),(1:Nb,Nb+1:Nb+Nfs)))
@inline set_q!(sys::FSPanelSystem,q) = (set_q!.(domains(sys),dviews(q,sys)); sys)
@inline get_q(sys::FSPanelSystem) = [sys.body.q; sys.freesurf.q]

# Set the rhs, solution, fsbc and preconditioner
rhs(sys::FSPanelSystem) = [rhs(sys.body,sys.U); zeros_like(sys.freesurf.dA)]
function bc!(b,sys::FSPanelSystem)
    p = sys.fsm; Nᵢ = size(p,1); Nb = length(sys.body); ℓ = sys.ℓ
    # body: Φₙ = -Uₙ
    AK.foreachindex(i-> b[i] = Φₙ(sys.body[i],sys),view(b,1:Nb))
    # freesurf: Φₙ-ℓΦₓₓ=0
    AK.foreachindex(i-> p[i] = Φ(sys.freesurf.x[i],sys), p) # fill p->Φ
    AK.foreachindex(p) do i
        b[i+Nb] = Φₙ(sys.freesurf[i],sys)
        (i-1)%Nᵢ<3 && return
        h = extent(view(sys.freesurf.x,i-1:i))[1]
        @inbounds b[i+Nb] -= ℓ*(2p[i]-5p[i-1]+4p[i-2]-p[i-3])/h^2 # 2nd order upwind FD
    end
end
function precon!(z,sys::FSPanelSystem,r)
    Nᵢ,Nⱼ = size(sys.fsm); Nb = length(sys.body)
    z .= r # identity preconditioner
    # sweep residual information downstream
    AK.foreachindex(view(z,1:Nⱼ)) do j
    for i in (2:Nᵢ) .+ (Nb+(j-1)*Nᵢ)
        @inbounds z[i] += z[i-1]
    end;end
end

"""
    ζ(x::SVector{3},sys)        # one point
    ζ(x::Vector,y::Vector,sys)  # evaluate on a regular grid (x,y') with z=0
    ζ(sys::FSPanelSystem)       # evaluate on sys.freesurf.x

Scaled linear free surface elevation `ζ/ℓ=Φₓ/|U|` induced by panel system `sys`.

See also: [`Φ`](@ref)
"""
ζ(x::SVector{3},sys) = derivative(t->Φ(x+t*SA[1,0,0],sys),0)/norm(sys.U)
function ζ(x::AbstractVector,y::AbstractVector,sys)
    z = similar(sys.body.dA,length(x),length(y))
    AK.foraxes(z,1) do i; for j in axes(z,2)
        z[i,j] = ζ(SA[x[i],y[j],0],sys)
    end; end; z
end
function ζ(sys::FSPanelSystem)
    z = sys.fsm
    AK.foreachindex(z) do i
        z[i] = ζ(sys.freesurf.x[i],sys)
    end; z
end
