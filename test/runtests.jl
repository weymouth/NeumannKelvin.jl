using NeumannKelvin
using Test

using QuadGK
@testset "panels.jl" begin
    circ(u) = [4sin(u),4cos(u)]; ellip(u) = [3sin(u),cos(u)]
    @test NeumannKelvin.arcspeed(circ)(0.) == NeumannKelvin.arcspeed(circ)(0.5pi) ≈ 4
    @test NeumannKelvin.κₙ(circ,0.) ≈ NeumannKelvin.κₙ(circ,0.5pi) ≈ 4
    @test NeumannKelvin.κₙ(ellip,0.) ≈ 1
    @test NeumannKelvin.κₙ(ellip,0.5pi) ≈ 3

    for c in (1,0.1301,0.029165) # tuned s.t. S≈N-1
        S,s⁻¹ = NeumannKelvin.arclength(ellip,1,c,0,pi)
        N = 2Int(round(S/2)); u = s⁻¹(range(0,S,N))
        if c==1 # should be equal length
            l = [quadgk(NeumannKelvin.arcspeed(ellip),u[i],u[i+1])[1] for i in 1:N-1]
            @test l ≈ [sum(l)/(N-1) for i in 1:N-1] rtol=0.03
        end
        @test 3-ellip(u[N÷2])[1] ≤ 1.10c # should have bounded deviation
    end

    torus(θ₁,θ₂;r=0.3,R=1) = SA[(R+r*cos(θ₂))*cos(θ₁),(R+r*cos(θ₂))*sin(θ₁),r*sin(θ₂)]
    spheroid(θ₁,θ₂;a=1.,b=1.,c=1.) = SA[a*cos(θ₂)*sin(θ₁),b*sin(θ₂)*sin(θ₁),c*cos(θ₁)]

    # Equal areas sanity checks
    function area_checks(dA,goal)
        mdA = sum(dA)/length(dA)
        @test mdA ≈ goal rtol = 0.08
        @test maximum(abs,panels.dA .- mdA) < 0.16goal
    end
    panels = panelize(spheroid,0,pi,0,2pi,hᵤ=0.5,c=Inf)
    area_checks(panels.dA,0.5^2)
    @test sum(panels.dA) ≈ 4π rtol=1e-4
    panels = panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,2pi,hᵤ=1,hᵥ=0.5,c=Inf)
    area_checks(panels.dA,0.5)
    @test sum(panels.dA) ≈ 30.894 rtol=1e-3
    panels = panelize(torus,0,2pi,0,2pi,hᵤ=0.6,hᵥ=0.3,transpose=true,c=Inf)
    area_checks(panels.dA,0.18)
    @test sum(panels.dA) ≈ 4π^2*0.3

    # Check sign is correct when flipped
    panels_bad = panelize(torus,0,2pi,0,2pi,hᵤ=0.6,hᵥ=0.3,c=Inf)
    @test panels[1].n ⋅ panels_bad[1].n > 0.99

    # Check inputs
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,2pi,hᵤ=0)
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,0,hᵤ=0.2)
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,0,0,2pi,hᵤ=0.2)
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,2pi,hᵤ=0.1)
    @test_throws ArgumentError panelize((u,v)->[u,u,v])
end

using LinearAlgebra,BenchmarkTools
@testset "panel_method.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = measure.(S,[π/4,3π/4]',π/4:π/2:2π,π/2,π/2,cubature=true) |> Table
    @test size(panels) == (8,)
    @test panels.dA ≈ fill(π/2,8) rtol=1e-6   # cubature gives perfect areas
    @test panels.n ⋅ panels.x ≈ 4√3 rtol=1e-6 # ...and centroids

    # Check that ∫G is non-allocating, including duals
    p = panels[1]
    b = @benchmark ∫G($p.x,$p); @test minimum(b).allocs==0
    b = @benchmark gradient(x->∫G(x,$p),$p.x); @test minimum(b).allocs==0
    b = @benchmark ∂ₙϕ($p,$p); @test minimum(b).allocs==0

    A,b = ∂ₙϕ.(panels,panels'),first.(panels.n)
    @test tr(A) ≈ 8*2π
    @test minimum(A) ≈ panels[1].dA/4 rtol=0.2 # rough estimate
    @test sum(b)<8eps()

    q = A \ b
    @test A*q≈b
    @test allequal(map(x->abs(round(x,digits=14)),q))
    Ma = addedmass(panels,V=4π/3)
    @test Ma ≈ I/2 rtol=0.1 # ϵ=10% with 8 panels
    @test diag(Ma) ≈ fill(sum(diag(Ma))/3,3) rtol=1e-3 # x/y/z symmetric!
end

extreme_cₚ(sys) = collect(extrema(cₚ(sys)))
@testset "solvers.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = panelize(S,0,π,0,2π,hᵤ=0.12)
    sys = gmressolve!(PanelSystem(panels),atol=1e-6); q = copy(sys.panels.q)
    directsolve!(sys)
    @test sys.panels.q ≈ q
    @test norm(steadyforce(sys)) < 3e-5
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.015

    #check symmetry enforcement
    panels = panelize(S,0,π/2,0,π,hᵤ=0.12) # quarter plane
    sys = gmressolve!(PanelSystem(panels,sym_axes=(2,3)),atol=1e-6); q = copy(sys.panels.q)
    directsolve!(sys)
    @test sys.panels.q ≈ q
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.015
end

using ImplicitBVH
using NeumannKelvin:fill_nodes,evaluate
@testset "BarnesHutCore.jl" begin
    cen = SA[0,0,1]
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]+cen
    panels = panelize(S,0,π,0,2π,hᵤ=0.12)
    bvh = BVH(ImplicitBVH.BoundingVolume.(panels))
    nodes = fill_nodes(panels,bvh)
    @test nodes.dA[1]≈sum(panels.dA)
    @test nodes.x[1]≈sum(panels.x .* panels.dA)/sum(panels.dA)≈cen

    ρ = panels.x[length(panels)÷3]-cen
    @show length(nodes), length(panels)
    for r in 1:6
        x = r*ρ+cen
        @test evaluate(∫G,x,bvh,nodes,panels;verbose=true) ≈ sum(∫G(x,p,d²=Inf) for p in panels) rtol=0.01
        @test gradient(x->evaluate(∫G,x,bvh,nodes,panels),x) ≈ gradient(x->sum(∫G(x,p,d²=Inf) for p in panels),x) rtol=0.04
    end

    stack = Vector{Int}(undef,bvh.tree.levels)
    x = panels.x[1]
    b = @benchmark evaluate(∫G,$x,$bvh,$nodes,$panels,stack=$stack); @test minimum(b).allocs==0
    b = @benchmark gradient(x->evaluate(∫G,x,$bvh,$nodes,$panels,stack=$stack),$x); @test minimum(b).allocs==0
end

@testset "BarnesHut.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = panelize(S,0,π,0,2π,hᵤ=0.12)
    BH = gmressolve!(BarnesHut(panels))
    q = ∂ₙϕ.(panels,panels')\components(panels.n,1)
    @test norm(BH.panels.q-q)/norm(q) < 0.0032
    @test norm(steadyforce(BH)) < 4e-3
    @test extreme_cₚ(BH) ≈ [-1.25,1.0] rtol=0.01
end

@testset "freesurf" begin
    S(θ₁,θ₂,Z=-1.1) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)+Z] # just below z=0
    body = panelize(S,0,π,0,π,hᵤ=1/4)
    P(u,v; x_min = -4π, x_max = 2π, y_min = 0, y_max = 2π) = SA[u*x_min+(1-u)*x_max, v*y_max+(1-v)*y_min, 0]
    freesurf = panelize(P,hᵤ=0.3,N_max=1500)
    sys = PanelSystem(body;freesurf,sym_axes=2,ℓ=0)
    @test length(sys.panels)==length(body)+length(freesurf)
    @test sys.body.dA == body.dA
    @test sys.freesurf.dA == freesurf.dA
    @test sys.kwargs[:ℓ] == 0

    # Direct solve ignores freesurf
    directsolve!(sys)
    @test collect(extrema(cₚ(sys))) ≈ [-1.25,1.0] rtol=0.04

    # Setting ℓ=0 turns freesurf into reflection wall
    sys = gmressolve!(sys)
    sys2 = gmressolve!(BarnesHut(body;sym_axes=(2,3)))
    @test extreme_cₚ(sys)≈extreme_cₚ(sys2) rtol=0.02

    # Setting ℓ=1 turns on freesurf, but it's very slow to converge
    sys = BarnesHut(body;freesurf,sym_axes=2,ℓ=1)
    gmressolve!(sys,itmax=160) # should converge...
    @test steadyforce(sys)[1] > 0.1 # non-zero drag!
    f = ζ(sys)
    @test -2minimum(f)>maximum(f) # trough is much bigger than crest
end

using NURBS,FileIO  # or whatever triggers the extension
@testset "NURBS" begin
    sphere = load(pkgdir(NURBS) * "/test/assets/sphere.stp")
    # measure a whole patch as one panel
    panel = measure(sphere[1],0.5,0.5,1.0,1.0)
    @test panel.kernel isa NeumannKelvin.QuadKernel
    @test panel.dA ≈ 4π/6 rtol=0.03

    # panlize the whole sphere
    panels = panelize(sphere,hᵤ=0.25)
    @test all( 0.9 .< extrema(panels.dA) ./ 0.25^2 .< 4/3)

    sys = directsolve!(PanelSystem(panels))
    @test bodyarea(sys) ≈ 4π rtol=1e-5
    @test bodyvol(sys) ≈ 4π/3 rtol=0.01
    @test extreme_cₚ(sys) ≈ [-1.25,1] rtol=0.04
    @test addedmass(panels) ≈ I/2 rtol=0.01
end

using GeometryBasics,FileIO  # or whatever triggers the extension
@testset "GeometryBasics" begin
    panel = measure(SA_F32[0,0,0],SA_F32[1,0,0],SA_F32[1,1,0])
    @test panel.x ≈ SA[2,1,0]/3
    @test panel.n ≈ SA[0,0,1]
    @test panel.dA ≈ 0.5
    @test ∫G(SA[0,0,2],panel)/4π ≈ -0.01847387 rtol=1e-6
    @test ∫G(SA[0,0,-0.5],panel)/4π ≈ −0.04558955 rtol=1e-6
    @test ∫G(SA[0.5,1,0],panel)/4π ≈ −0.05806854 rtol=1e-6
    @test ∫G(SA[-0.25,0.25,0.5],panel)/4π ≈ −0.03856218 rtol=1e-6
    @test ∫G(SA[0.1,0.4,8],panel)/4π ≈ −0.00495674 rtol=1e-6
    b = @benchmark ∫G($panel.x,$panel); @test minimum(b).allocs==0
    b = @benchmark gradient(x->∫G(x,$panel),$panel.x); @test minimum(b).allocs==0

    ext = Base.get_extension(NeumannKelvin, :NeumannKelvinGeometryBasicsExt)
    panels = panelize(load("../examples/Icosahedron.stl"))
    @test length(panels)==20
    @test eltype(panels.kernel)==ext.TriKernel
    @test all([p.n'p.x>0 for p in panels]) # all outward facing

    sys = BarnesHut(panels)
    @test sys.bvh.leaves[1].volume isa ImplicitBVH.BBox # triggered correct BV
    @test bodyarea(sys) ≈ 957 rtol=1e-3
    @test bodyvol(sys) ≈ 2475 rtol=3e-2

    Ma = addedmass(panels,V=1)
    @test diag(Ma) ≈ fill(tr(Ma)/3,3) rtol=1e-4
    @test 2π/3*7.95^3 < tr(Ma)/3 < 2π/3*10^3 # between the insphere & circumsphere
end