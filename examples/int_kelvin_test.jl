using NeumannKelvin,Plots,FastGaussQuadrature,HCubature,JLD2,FileIO,ColorSchemes

# HCubature directly in the wake of a single transverse panel
begin
    y = range(-1,1,1000)
    # function ∫cube(ξ)
    #     I,e,n = hcubature_count(yz->NeumannKelvin.kelvin(ξ,SA[0.,yz[1],yz[2]]),SA[-0.5,0],SA[0.5,1],atol=1e-5,initdiv=4)
    #     (I=I,e=e,n=n)
    # end
    # good = map(y->∫cube(SA[-7,y,-0.]),y)|>Table
    # jldsave("hcube_panel_1em5.jld2",I=good.I,e=good.e,n=good.n)
    good = load("examples/hcube_panel_1em5.jld2")
    good = Table(;:I=>good["I"],:e=>good["e"],:n=>good["n"])
    plot(y,good.n,label="h-cubature, 1e-5 error",ylabel="count",xlabel="y",yscale=:log10,c=:purple)
end
savefig("examples/waken.png")

# Gauss quadrature show oscilatory refinement to "good" result
begin
    cmap = reverse(colorschemes[:Blues_4])
    plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    for (n,c) in zip((32,8,2),cmap)
        Δg,wg = SVector{n}.(gausslegendre(n))
        panel = measure_panel((y,z)->SA[0,y,z],0,-0.5,1,1;Δg,wg)
        plot!(y,y->∫kelvin(SA[-7,y,-0],panel,λ=0),label="Gauss-Legendre $(n)²";c)
    end; plot!(ylims=(-2.9,-2.5))
end
savefig("examples/wakeϕ.png")

# Adding the transverse damping in the wake is great
begin
    cmap = reverse(colorschemes[:Blues_4])
    plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    for (n,c) in zip((32,8,2),cmap)
        Δg,wg = SVector{n}.(gausslegendre(n))
        panel = measure_panel((y,z)->SA[0,y,z],0,-0.5,1,1;Δg,wg)
        plot!(y,y->∫kelvin(SA[-7,y,-0],panel,λ=0.4/n),label="Gauss-Legendre $(n)²";c)
    end; plot!(ylims=(-2.9,-2.5))
end
savefig("examples/wakeϕ_λ.png")

# Same study, except we do panel refinement using a consistent 2x2 quadrature
begin
    plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    cmap = reverse(colorschemes[:Blues_5])
    for (dy,c) in zip((1/16,1/8,1/4,1/2),cmap)
        panels = measure_panel.((y,z)->SA[0,y,z],-0.5(1-dy):dy:0.5,(-(1-0.5dy):dy:0)',dy,dy)|>Table
        plot!(y,y->sum(p->∫kelvin(SA[-7,y,-0],p,λ=0),panels),label="$(Int(1÷dy))×$(Int(1÷dy)) grid of 2² Gauss panels";c)
    end; plot!(ylims=(-2.9,-2.5))
end
begin
    plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    cmap = reverse(colorschemes[:Blues_5])
    for (dy,c) in zip((1/16,1/8,1/4,1/2),cmap)
        panels = measure_panel.((y,z)->SA[0,y,z],-0.5(1-dy):dy:0.5,(-(1-0.5dy):dy:0)',dy,dy)|>Table
        plot!(y,y->sum(p->∫kelvin(SA[-7,y,-0],p),panels),label="$(Int(1÷dy))×$(Int(1÷dy)) grid of 2² Gauss panels";c)
    end; plot!(ylims=(-2.9,-2.5))
end

# Panel aspect ratio can also work, but you need very skinny panels
begin
    plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    cmap = reverse(colorschemes[:Blues_4])
    for (dy,c) in zip((1/16,1/8,1/4),cmap)
        dz = 1/(64dy)
        panels = measure_panel.((y,z)->SA[0,y,z],-0.5(1-dy):dy:0.5,(-(1-0.5dz):dz:0)',dy,dz)|>Table
        plot!(y,y->sum(p->∫kelvin(SA[-7,y,-0],p,λ=0),panels),label="$(Int(1÷dy))×$(Int(1÷dz)) grid of 2² Gauss panels";c)
    end; plot!(ylims=(-2.9,-2.5))
end
savefig("examples/wakeϕ_AR.png")
begin
    plot(y,good.I,ribbon=good.e,label="h-cubature",ylabel="ϕ",xlabel="y",c=:purple)
    cmap = reverse(colorschemes[:Blues_4])
    for (dy,c) in zip((1/16,1/8,1/4),cmap)
        dz = 1/(64dy)
        panels = measure_panel.((y,z)->SA[0,y,z],-0.5(1-dy):dy:0.5,(-(1-0.5dz):dz:0)',dy,dz)|>Table
        plot!(y,y->sum(p->∫kelvin(SA[-7,y,-0],p),panels),label="$(Int(1÷dy))×$(Int(1÷dz)) grid of 2² Gauss panels";c)
    end; plot!(ylims=(-2.9,-2.5))
end
savefig("examples/wakeϕ_λ_AR.png")

# "Thin-ship" example. 20x2 in-line panel with linear q=x
begin
    x = range(-30,20,1000)
    # ∫cubex(ξ) = hcubature(SA[-10.,0.],SA[10.,2.],atol=1e-5,initdiv=16) do (x,z)
    #     x*NeumannKelvin.kelvin(ξ,SA[x,0.,z])
    # end |> first
    # goodthin = map(x->derivative(x->∫cubex(SA[x,0.0,-0.0]),x),x)
    # save_object("examples/hcubex_thin_1em5.jld2",goodthin)
    goodthin = load_object("examples/hcubex_thin_1em5.jld2")
    plot(x,goodthin,label="h-cubature",ylabel="ζ",xlabel="x",c=:purple)
    NeumannKelvin.nearfield(1.,2.,0.)
end
# Test with square panels has a small perturbation
begin
    plot(x,goodthin,label="h-cubature",ylabel="ζ",xlabel="x",c=:purple)
    cmap = reverse(colorschemes[:Blues_5])
    for (h,c) in zip((0.25,0.5,1.,2.,),cmap)
        px = (-10+0.5h):h:10
        panels = measure_panel.((x,z)->SA[x,0,z],px,(-0.5h:-h:-2)',h,h)|>Table
        q = first.(panels.x)
        z(x) = ζ(x,0,q,panels,ϕ=(ξ,p)->∫kelvin(ξ,p))
        plot!(x,z,label="h=$h";c)
        scatter!(px,z,label="";c,markersize=3,markerstrokewidth=0)
    end; plot!()
end
# Cosine sampling isn't working
begin
    plot(x,goodthin,label="h-cubature",ylabel="ζ",xlabel="x",c=:purple)
    cmap = reverse(colorschemes[:Blues_4])
    for (n,c) in zip((40,20,10),cmap)
        x₀ = 10cos.(π.*range(0,1,n+1)); dx = diff(x₀); px = @. x₀[1:n]+0.5dx
        # z₀ = 2cos.(0.5π.*range(0,1,n+1)) .-2; dz = diff(z₀); pz = @. z₀[1:n]+0.5dz
        z₀ = range(0,-2,n+1); dz = diff(z₀); pz = @. z₀[1:n]+0.5dz
        panels = measure_panel.((x,z)->SA[x,0,z],px,pz',dx,dz')|>Table
        q = first.(panels.x)
        z(x) = ζ(x,0,q,panels,ϕ=(ξ,p)->∫kelvin(ξ,p))
        plot!(x,z,label="$n x $n cosine-sampled panels";c)
        scatter!(px,z,label="";c,markersize=3,markerstrokewidth=0)
    end; plot!()
end

using NeumannKelvin,GLMakie
# prism(Δᵤ,Δᵥ;q=0.2,Z=1,kwargs...) = measure_panel.(
#     (u,v) -> SA[0.5cos(π*u),q*0.5sin(π*u),Z*(sin(0.5π*v)-1)], # elliptical prism
#     0.5Δᵤ:Δᵤ:1,(0.5Δᵥ:Δᵥ:1)',Δᵤ,Δᵥ;kwargs...) |> Table

# function ∫surface_S₂(x,p;kwargs...)  # y-symmetric potentials
#     ∫kelvin(x,p;kwargs...)+∫kelvin(x,reflect(p,SA[1,-1,1]);kwargs...)
# end

# panels = prism(0.025,0.1)
# ps = (ϕ=∫surface_S₂,Fn=0.312)
# A = influence(panels;ps...);
# q = A\first.(panels.n);
# x,y,_ = components(panels[1:20].x)
# scatter(x,ζ.(x,y,Ref(q),Ref(panels);ps...),label="",ylims=(-1,0.5))
# u = range(0,1,160); x,y = 0.5cos.(π*u),0.1sin.(π*u)
# plot!(x,ζ.(x,y,Ref(q),Ref(panels);ps...),label="",ylims=(-1,0.5))
# # u = map(x->SA[-1,0,0]+NeumannKelvin.∇Φ(x,q,panels;ps...),panels.x)
# # viz(panels,q,vectors=0.2u)
# # x,y = range(-4,1,100),range(0,3,60)
# # zeta = ζ.(x,y',Ref(q),Ref(panels);ps...) # should multi-thread
# # GLMakie.surface(x,y,zeta)
