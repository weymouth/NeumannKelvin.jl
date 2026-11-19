# Rectangular H105 hydrofoil (AR=4, α=2°) below a free surface, compared with Nicolas et al. (2023),
# "Boundary Element Method Analysis of 3D Effects and Free-Surface Proximity on Hydrofoil Lift and Drag
# Coefficients in Varied Operating Conditions", J. Sailing Technology 8, doi:10.5957/jst/2023.8.10.183
using NeumannKelvin,DelimitedFiles

# H105 section from the authors' LEModeler (github.com/hugo-nicolas/LEModeler), unit chord: TE → upper → LE → lower → TE
P = readdlm(joinpath(@__DIR__,"h105.dat"),skipstart=1); iLE = argmin(P[:,1])
upper,lower = P[iLE:-1:1,:],P[iLE:end,:]
function interp(T,x) # surface T at chordwise position x
    k = clamp(searchsortedlast(T[:,1],x),1,size(T,1)-1); t = (x-T[k,1])/(T[k+1,1]-T[k,1])
    T[k,2]+t*(T[k+1,2]-T[k,2])
end
section(n) = (x = (1 .- cospi.((0:n)/n))/2; [[(xᵢ,interp(lower,xᵢ)) for xᵢ in reverse(x)]; [(xᵢ,interp(upper,xᵢ)) for xᵢ in x[2:end]]])

# Half of a rectangular wing of span AR, nose-up α about the mid-chord at depth h. The tip is a half body of
# revolution about the camber line, radius t(x)/2, so the surface closes with no sharp edge
camber(x) = (interp(upper,x)+interp(lower,x))/2
function wing(;AR=4,α=deg2rad(2),h=1,nc=40,ns=6,nt=4)
    R = SA[cos(α) 0 -sin(α);0 1 0;sin(α) 0 cos(α)]; θ = range(π/2,0,nt+1)
    S = section(nc); y₀ = AR/2-maximum(z-camber(x) for (x,z) in S)
    tip(x,z,θ) = SA[0.5-x,y₀+abs(z-camber(x))*sin(θ),camber(x)+(z-camber(x))*cos(θ)]
    X = [R*(j≤nt+1 ? tip(x,z,θ[j]) : SA[0.5-x,y₀*cospi((j-nt-1)/2ns),z])-SA[0,0,h] for (x,z) in S, j in 1:nt+ns+1] # tip → root
    X[end,:] = X[1,:]; n,m = nt+ns,size(X,1)
    half(i) = abs(S[i][2]-camber(S[i][1])) # half-thickness at section point i
    thin(i) = π/2*max(half(i),half(i+1)) < abs(S[i][1]-S[i+1][1]) # tip arc shorter than the row
    for i in 2:m-1; thin(i-1) && thin(i) && (X[i,2:nt] .= Ref(X[i,1])); end # one panel round the tip in thin rows
    panels = Table(filter(p->p.dA>0,[measure(X[i,j],X[i+1,j],X[i+1,j+1],X[i,j+1]) for i in 1:m-1 for j in 1:n]))
    panels,[(X[1,j],X[1,j+1]) for j in nt+1:n] # the tip's trailing edge is a point
end

# Lift and drag ratios to the unbounded flow, sweeping the depth and Froude number Frc=U/√(gc), so ℓ=Frc²
CLD(sys) = (F = steadyforce(sys;S=2); SA[F[3],-F[1]]) # half-wing planform area AR/2
deep = CLD(PanelSystem(Lifting(wing()...),domain=Unbounded(sym_axes=2)) |> directsolve!)
hs,Frs = [0.5,1,1.5,2,3,4],[0.5,1.1,6.3]
ratios = [CLD(PanelSystem(Lifting(wing(;h)...),domain=Kelvin(Fr^2,sym_axes=2)) |> directsolve!) ./ deep for h in hs, Fr in Frs]

# Nicolas et al. Fig. 5 (AR=4), read by eye to ±0.01; NaN is off their plot
paper = (CL=[1.11 0.57 0.765; 1.12 0.865 0.885; 1.072 0.976 0.932; 1.044 1.01 0.955; 1.021 1.016 0.977; 1.012 1.012 0.985],
         CD=[NaN 1.79 0.98; 1.17 1.53 1.11; 0.97 1.235 1.115; 0.945 1.08 1.10; 0.955 0.98 1.075; 0.965 0.97 1.055])

using GLMakie
foil,te = wing(h=1)
sys = PanelSystem(Lifting(foil,te),domain=Kelvin(0.5^2,sym_axes=2)) |> directsolve! # Frc=0.5, h/c=1
waves = with_theme(Theme(fontsize=12)) do # 6 chords downstream, 5 to the side
    viz(sys,colorrange=(-1,1),vectors=nothing,fsargs=(x=-6:1/16:2,y=0:1/16:5))
end; resize!(waves,1600,800)
foreach(cb->(cb.width=12; cb.height=Relative(0.45)), filter(c->c isa Colorbar, waves.content)) # slim colorbars
ax3 = content(waves[1,1]); ax3.viewmode = :fitzoom; hidedecorations!(ax3); hidespines!(ax3)

forces = Figure(size=(900,450))
for (k,C) in enumerate(("L","D"))
    ax = Axis(forces[1,k],xlabel="h/c",ylabel=rich("C",subscript(C),"/C",subscript(C*"∞")),limits=(nothing,k==1 ? (0.3,1.3) : (0.5,2)))
    for (j,Fr) in enumerate(Frs)
        lines!(ax,hs,getindex.(ratios[:,j],k),color=Cycled(j),label=rich("Fr",subscript("c")," = $Fr"))
        scatter!(ax,hs,paper[k][:,j],color=Cycled(j))
    end
    k==1 && axislegend(ax,"lines: NeumannKelvin\ndots: Nicolas et al.",position=:rb)
end
forces