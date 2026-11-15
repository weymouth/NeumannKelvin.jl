"""
    FreeSurface(panels::AbstractMatrix; ℓ, wrap=PanelTree, kwargs...)

Free-surface boundary of source `panels` applying the linear FSBC with Kelvin length `ℓ≡U²/g`.
The panels are wrapped using `wrap(panels;kwargs...)`.

**Note**: `panels` must be a matrix of quadralateral panels with the first index of the matrix
aligned with -x. i.e. `panels[i+1,j].x-panels[i,j].x ≈ [-Δxᵢⱼ,0,0]`. Similarly, the direction of
the flow must be `Û=-x̂`. Relative flow vectoring (i.e. drift angle) must be acheived by rotating the body.

# Details

The convective derivative Φₓₓ in the FSBC is estimated by computing Φ at the FS panel
centers and using a 2nd-order upwind finite difference in `i`. This maintains downwind
wave propigation and adds (significant) numerical damping. A downwind flooding gmres
preconditioner is used to accelerate convergence, but it still requires O(100) iterations.
The accuracy of the solution depends on the free surface resolution; at minimum h<ℓ is required.
"""
struct FreeSurface{T,P,D,L} <: PanelWrapper{T}
    panels::P # free surface panels
    fsm::D    # free-surface sized Matrix
    ℓ::L      # Kelvin-length
    FreeSurface(panels::P,fsm::D,ℓ::L) where {P,D,L} = new{eltype(panels),P,D,L}(panels,fsm,ℓ)
end
function FreeSurface(panels::AbstractMatrix;ℓ,wrap=PanelTree,kwargs...)
    table = Table(panels)
    table.n[1][3]>0 && throw(ArgumentError("free-surface panels must point down."))
    dx = table.x[2]-table.x[1]
    (dx[1]>-abs(dx[2]) || dx[3]≠0) && throw(ArgumentError("first index of free-surface panels must align with -x."))
    FreeSurface(wrap(Table(table;q=zeros_like(table.dA));kwargs...),zeros(eltype(table.dA),size(panels)),ℓ)
end
Base.show(io::IO,fs::FreeSurface) = println(io,"FreeSurface($(size(fs.fsm)) panels, ℓ=$(fs.ℓ))")

isbody(::FreeSurface) = false
check(::FreeSurface,sys) = checkU(sys)
@inline Φ_boundary(x,fs::FreeSurface,domain) = Φ_boundary(x,fs.panels,domain)
@inline set_q!(fs::FreeSurface,q) = set_q!(fs.panels,q)
rhs(fs::FreeSurface,sys) = zeros_like(fs.dA)
function bc!(r,(;panels,fsm,ℓ)::FreeSurface,sys) # Φₙ-ℓΦₓₓ=0
    AK.foreachindex(i-> fsm[i] = Φ(panels.x[i],sys), fsm) # fill fsm->Φ
    AK.foreachindex(fsm) do i
        r[i] = Φₙ(panels[i],sys)
        (i-1)%size(fsm,1)<3 && return
        h = extent(view(panels.x,i-1:i))[1]
        @inbounds r[i] -= ℓ*(2fsm[i]-5fsm[i-1]+4fsm[i-2]-fsm[i-3])/h^2 # 2nd order upwind FD
    end
end
function precon!(z,fs::FreeSurface) # sweep residual information downstream
    Nᵢ,Nⱼ = size(fs.fsm)
    AK.foreachindex(view(z,1:Nⱼ)) do j
    for i in (2:Nᵢ) .+ (j-1)*Nᵢ
        @inbounds z[i] += z[i-1]
    end;end
end

"""
    ζ(x::SVector{3},sys)        # one point
    ζ(x::Vector,y::Vector,sys)  # evaluate on a regular grid (x,y') with z=0
    ζ(fs::FreeSurface,sys)      # evaluate on fs.x

Scaled linear free surface elevation `ζ/ℓ=Φₓ/|U|` induced by panel system `sys`.

See also: [`Φ`](@ref)
"""
ζ(x::SVector{3},sys) = derivative(t->Φ(x+t*SA[1,0,0],sys),0)/norm(sys.U)
function ζ(x::AbstractVector,y::AbstractVector,sys)
    z = zeros(eltype(first(sys.boundaries).dA),length(x),length(y))
    AK.foraxes(z,1) do i; for j in axes(z,2)
        z[i,j] = ζ(SA[x[i],y[j],0],sys)
    end; end; z
end
ζ(fs::FreeSurface,sys) = (AK.foreachindex(i-> fs.fsm[i] = ζ(fs.x[i],sys),fs.fsm); fs.fsm)