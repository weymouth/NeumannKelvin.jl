using NeumannKelvin,Plots,FastGaussQuadrature,HCubature
function ∫cube(ξ)
    I,e,n = hcubature_count(yz->NeumannKelvin.kelvin(ξ,SA[0.,yz[1],yz[2]]),SA[-0.5,0],SA[0.5,1],atol=1e-5,initdiv=4)
    (I=I,e=e,n=n)
end
y = range(-1,1,1000)
using JLD2,FileIO
# good = map(y->∫cube(SA[-7,y,-0.]),y)|>Table
# jldsave("hcube_panel_1em5.jld2",I=good.I,e=good.e,n=good.n)
good = load("hcube_panel_1em5.jld2")
good = Table(;:I=>good["I"],:e=>good["e"],:n=>good["n"])

plot(y,good.n,label="h-cubature, 1e-5 error",ylabel="count",xlabel="y",yscale=:log10,c=:purple)
savefig("waken.png")

using ColorSchemes
begin
    panel = measure_panel((y,z)->SA[0,y,z],0,-0.5,1,1;
                xgl=NeumannKelvin.xgl,wgl=NeumannKelvin.wgl);
    ∫gl(ξ;Fn=1)=NeumannKelvin.quadgl(x->NeumannKelvin.kelvin(ξ,x;Fn),x=reflect(panel.x₄),w=panel.w₄)
    ∫mid(ξ;Fn=1)=NeumannKelvin.kelvin(ξ,reflect(panel.x);Fn)*panel.dA
    cmap = reverse(colorschemes[:BuPu_5])
    plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=cmap[1])
    plot!(y,y->∫gl(SA[-7,y,-0]),label="Gauss 16²",c=cmap[2])
    panel = measure_panel((y,z)->SA[0,y,z],0,-0.5,1,1);
    plot!(y,y->∫gl(SA[-7,y,-0]),label="Gauss 2²",c=cmap[3])
    plot!(y,y->∫mid(SA[-7,y,-0]),label="midpoint",c=cmap[4])
end
savefig("wakeϕ.png")

begin
    panel = measure_panel((y,z)->SA[0,y,z],0,-0.5,1,1;
                Δg=NeumannKelvin.xgl,wg=NeumannKelvin.wgl);
    ∫gl(ξ;Fn=1,z_max=-0)=NeumannKelvin.quadgl(x->NeumannKelvin.kelvin(ξ,x;Fn,z_max),x=reflect(panel.x₄),w=panel.w₄)
    cmap = reverse(colorschemes[:Blues])
    plt=plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    for (z_max,c) in zip((-0,-0.03,-0.06,-0.1),cmap)
        plot!(y,y->∫gl(SA[-7,y,-0];z_max),label="Gauss 16² z_max=$z_max";c)
    end;plt
end
savefig("wakeϕ_zmax.png")

begin
    panel = measure_panel((y,z)->SA[0,y,z],0,-0.5,1,1;
                Δg=NeumannKelvin.xgl,wg=NeumannKelvin.wgl);
    ∫gl(ξ,p;λ=0.02)=NeumannKelvin.quadgl(x->NeumannKelvin.kelvin(ξ,x;sx=0,sy=λ),x=reflect(p.x₄),w=p.w₄)
    cmap = reverse(colorschemes[:Blues])
    plt=plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    for (λ,c) in zip((0,0.01,0.02,0.08),cmap)
        plot!(y,y->∫gl(SA[-7,y,-0],panel;λ),label="Gauss 16², filter λ=$λ";c)
    end;plt
end
savefig("wakeϕ_λ.png")

begin
    plt=plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    cmap = reverse(colorschemes[:Blues])
    for (dy,c) in zip((1/4,1/8,1/16),cmap)
        dz = 1/(64dy)
        panels = measure_panel.((y,z)->SA[0,y,z],-0.5(1-dy):dy:0.5,(-(1-0.5dz):dz:0)',dy,dz)|>Table
        plot!(y,y->sum(p->∫kelvin(SA[-7,y,-0],p),panels),label="$(Int(1÷dy))×$(Int(1÷dz)) grid of 2² Gauss panels";c)
    end;plt
end
savefig("wakeϕ_AR.png")

function ∫cubex(ξ)
    I,e,n = hcubature_count(xz->NeumannKelvin.kelvin(ξ,SA[xz[1],0,xz[2]]),SA[-0.5,0],SA[0.5,1],atol=1e-6,initdiv=4)
    (I=I,e=e,n=n)
end
goodx = map(y->∫cubex(SA[-7,y,-0.]),y)|>Table
jldsave("hcubex_panel_1em6.jld2",I=goodx.I,e=goodx.e,n=goodx.n)
# goodx = load("hcubex_panel_1em6.jld2")
# goodx = Table(;:I=>goodx["I"],:e=>goodx["e"],:n=>goodx["n"])

begin
    panel = measure_panel((x,z)->SA[x,0,z],0,-0.5,1,1;
                Δg=NeumannKelvin.xgl,wg=NeumannKelvin.wgl);
    ∫gl(ξ,p;λ=0.02,sx=1.,sy=0.)=NeumannKelvin.quadgl(x->NeumannKelvin.kelvin(ξ,x;λ,sx,sy),x=reflect(p.x₄),w=p.w₄)
    cmap = reverse(colorschemes[:Blues])
    plt=plot(y,goodx.I,ribbon=goodx.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    for (λ,c) in zip((0,0.01,0.02,0.08),cmap)
        plot!(y,y->∫gl(SA[-7,y,-0],panel;λ),label="Gauss 16², filter λ=$λ";c)
    end;plt
end
savefig("wakeXϕ_λ.png")

# using PlotlyBase
# plotly()
# Plots.surface(-10:0.1:2,-3:0.1:3,(x,y)->derivative(x->∫gl(SA[x,y,-0],panel),x))


# prism(Δᵤ,Δᵥ;q=0.2,Z=1,kwargs...) = measure_panel.(
#     (u,v) -> SA[0.5cos(π*u),q*0.5sin(π*u),Z*(cos(0.5π*v)-1)], # elliptical prism
#     0.5Δᵤ:Δᵤ:1,(0.5Δᵥ:Δᵥ:1)',Δᵤ,Δᵥ;kwargs...) |> Table

# function ∫surface(ξ,p;Fn)
#     p′ = reflect(p,SA[1,1,-1]) # reflect across z=0
#     ∫G(ξ,p)-∫G(ξ,p′)+NeumannKelvin.quadgl(x->NeumannKelvin.kelvin(ξ,x;Fn),x=p′.x₄,w=p′.w₄)
# end
# function ∫surface_S₂(x,p;kwargs...)  # y-symmetric potentials
#     ∫surface(x,p;kwargs...)+∫surface(x,reflect(p,SA[1,-1,1]);kwargs...)
# end

# ps = (ϕ=∫surface_S₂,Fn=0.312)
# xgl,wgl = SVector{12}.(gausslegendre(12))
# panels = prism(0.02,0.1)
# x,y,z = components(panels[1:50].x);
# A = influence(panels;ps...);
# q = A\first.(panels.n);
# begin
#     zeta = map(x->derivative(d->NeumannKelvin.Φ(x+d*SA[1,0,0],q,panels;ps...),0.),panels[1:50].x)
#     plot(x,zeta,label="centers")
#     # zeta = ζ.(x,y,Ref(q),Ref(panels);ps...)
#     # plot!(x,zeta,label="edges")
#     zeta = map(x->derivative(d->NeumannKelvin.Φ(x+d*SA[1,0,0],q,panels;ps...),0.),first.(panels[1:50].xᵤᵥ))
#     plot!(first.(first.(panels[1:50].xᵤᵥ)),zeta,label="corners")
# end