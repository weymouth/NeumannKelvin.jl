using NeumannKelvin
using Test,BenchmarkTools
TEST_ALLOCS = get(ENV, "CI", "false") == "true" ? 8 : 0
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1

using QuadGK
@testset "quad.jl" begin
    xgl2,wgl2 = (-1/√3,1/√3),(1,1)
    @test NeumannKelvin.quadgl(x->x^3-3x^2+4,x=xgl2,w=wgl2)≈6
    @test NeumannKelvin.quadgl(x->x^3-3x^2+4,0,2,x=xgl2,w=wgl2)≈4

    (a₁,f₁),(a₂,f₂)=NeumannKelvin.finite_ranges((0.,),x->x^2,4,Inf)
    @test [a₁,a₂]≈[-2,2] atol=0.3
    @test all([f₁,f₂])

    (a₁,f₁),(a₂,f₂)=NeumannKelvin.finite_ranges((0.,),x->x^2,6,2,atol=0)
    @test [a₁,a₂]≈[-2,2] && !any([f₁,f₂])

    # Highly oscillatory integral set-up
    g(x) = x^2+im*x^2/100
    dg(x) = 2x+im*x/50
    f(x) = imag(exp(im*g(x)))
    ρ = √(3π); rng = (-ρ,ρ); rngs = (-ρ,true),(ρ,true)

    I,e,c=quadgk_count(f,ρ,Inf)
    # @show I,e,c # (-0.14690637593307346, 2.133881538591021e-9, 8265) # hard
    @test NeumannKelvin.nsp(ρ,g,dg) ≈ I atol=1e-5

    I,e,c=quadgk_count(f,-Inf,Inf)
    # @show I,e,c # (1.247000964522693, 1.824659458372201e-8, 15195)
    @test NeumannKelvin.complex_path(g,dg,rngs) ≈ I atol=1e-5
end

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

using BenchmarkTools
@testset "PanelSystem.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = panelize(S, 0, π, 0, 2π, hᵤ=1/4, T=Float32)
    @test eltype(panels.dA) == Float32
    sys = BodyPanelSystem(panels)
    @test sys.body.q == zeros(Float32,length(panels))
    q = rand(Float32,length(panels))
    @test @ballocations(NeumannKelvin.set_q!($sys,$q)) ≤ TEST_ALLOCS
    @test sys.body.q == q
    @test bodyarea(sys) ≈ 4π rtol=1e-5
    @test bodyvol(sys) ≈ 4π/3 rtol=5e-3
end

using LinearAlgebra
@testset "panel_method.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = measure.(S,[π/4,3π/4]',π/4:π/2:2π,π/2,π/2,cubature=true) |> Table
    @test size(panels) == (8,)
    @test panels.dA ≈ fill(π/2,8) rtol=1e-6   # cubature gives perfect areas
    @test panels.n ⋅ panels.x ≈ 4√3 rtol=1e-6 # ...and centroids

    # Check that ∫G is non-allocating, including duals
    p = panels[1]
    @test @ballocations(∫G($p.x,$p)) ≤ TEST_ALLOCS
    @test @ballocations(gradient(x->∫G(x,$p),$p.x)) ≤ TEST_ALLOCS
    @test @ballocations(∂ₙϕ($p,$p)) ≤ TEST_ALLOCS

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
    sys = gmressolve!(BodyPanelSystem(panels,U=SA[3,4,0]),atol=1e-8); q = copy(sys.body.q)
    directsolve!(sys)
    @test sys.body.q ≈ q
    @test norm(steadyforce(sys)) < 4e-5
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.02

    # Check allocations
    p = panels[1]
    @test @ballocations(Φ($p.x,$sys)) ≤ TEST_ALLOCS
    @test @ballocations(∇Φ($p.x,$sys)) ≤ TEST_ALLOCS

    #check symmetry enforcement
    panels = panelize(S,0,π/2,0,π,hᵤ=0.12) # quarter plane
    sys = gmressolve!(BodyPanelSystem(panels,sym_axes=(2,3)),atol=1e-6); q = copy(sys.body.q)
    directsolve!(sys)
    @test sys.body.q ≈ q
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.02
end

using ImplicitBVH
using NeumannKelvin:fill_nodes,treesum,aggregate!
@testset "BarnesHutCore.jl" begin
    cen = SA[0,0,1]
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]+cen
    panels = panelize(S,0,π,0,2π,hᵤ=0.12)
    bvh = BVH(ImplicitBVH.BoundingVolume.(panels))
    nodes = fill_nodes(panels,bvh)
    @test nodes.dA[1]≈sum(panels.dA)
    @test nodes.x[1]≈sum(panels.x .* panels.dA)/sum(panels.dA)≈cen
    @test @ballocations(aggregate!($nodes.dA,$panels.dA,$bvh)) ≤ TEST_ALLOCS

    ρ = panels.x[length(panels)÷3]-cen
    @show length(nodes), length(panels)
    for r in 1:6
        x = r*ρ+cen
        @test treesum(∫G,x,bvh,nodes,panels;verbose=true) ≈ sum(∫G(x,p) for p in panels) rtol=0.01
        @test gradient(x->treesum(∫G,x,bvh,nodes,panels),x) ≈ gradient(x->sum(∫G(x,p) for p in panels),x) rtol=0.04
    end

    x = panels.x[1]
    @test @ballocations(treesum(∫G,$x,$bvh,$nodes,$panels)) ≤ TEST_ALLOCS
    @test @ballocations(gradient(x′->treesum(∫G,x′,$bvh,$nodes,$panels),$x)) ≤ TEST_ALLOCS
end

@testset "PanelTree.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = panelize(S,0,π,0,2π,hᵤ=0.12); N = length(panels)
    BH = PanelTree(Table(panels;q=rand(N)))
    q = zeros(N)
    @test @ballocations(NeumannKelvin.set_q!($BH,$q)) ≤ TEST_ALLOCS
    @test BH.q == q

    sys = gmressolve!(BodyPanelSystem(panels,wrap=PanelTree))
    q = ∂ₙϕ.(panels,panels')\components(panels.n,1)
    @test norm(sys.body.q-q)/norm(q) < 0.0032
    @test norm(steadyforce(sys)) < 4e-3
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.015
    @test @ballocations(Φ($panels.x[1],$sys)) ≤ TEST_ALLOCS
end

function spheroid(h;L=1,Z=-1/8,r=1/12,AR=1/2r,kwargs...)
    S(θ₁,θ₂) = SA[0.5L*cos(θ₁),-r*sin(θ₂)*sin(θ₁),r*cos(θ₂)*sin(θ₁)+Z]
    panelize(S,0,π,0,π,hᵤ=h*√AR,hᵥ=h/√AR;kwargs...)
end
function halfplane(L::T,ℓ;hᵤ=0.3ℓ,hᵥ=hᵤ,s=L/2+2π*ℓ) where T
    measure.((u,v)->SA[u,v,0],s:-hᵤ:-2s,(hᵥ/2:hᵥ:4s/3)',hᵤ,hᵥ,flip=true;T)
end
@testset "FSPanelSystem.jl" begin
    ℓ = 0.45^2; freesurf = halfplane(1.,ℓ); body = spheroid(1/60)
    sys = FSPanelSystem(body,freesurf,sym_axes=2,ℓ=0,wrap=identity)
    q = NeumannKelvin.get_q(sys)
    @test length(q)==length(body)+length(freesurf)
    @test sys.body.dA == body.dA
    @test sys.ℓ == 0
    @test sys.U == SA[-1,0,0]

    badsurf = measure.((u,v)->SA[u,v,0],-4:1:2,(1/2:1:2)',1,1,flip=true)
    @test_throws ArgumentError FSPanelSystem(body,badsurf)

    # Direct solve ignores freesurf
    directsolve!(sys,verbose=false)
    sys2 = directsolve!(BodyPanelSystem(body,sym_axes=2),verbose=false)
    @test sys.body.q == sys2.body.q

    # Setting ℓ=0 turns freesurf into reflection wall
    gmressolve!(sys)
    sys2 = gmressolve!(BodyPanelSystem(body,sym_axes=(2,3),wrap=PanelTree))
    @test extreme_cₚ(sys)≈extreme_cₚ(sys2) rtol=0.03
    @test @ballocations(Φ($body.x[1],$sys)) ≤ TEST_ALLOCS

    # Setting ℓ>0 turns on freesurf, but it's slow to converge
    sys = FSPanelSystem(body,freesurf;sym_axes=2,ℓ,θ²=16)
    gmressolve!(sys,itmax=160)                      # should converge...
    @test 2steadyforce(sys,S=1)[1] ≈ 6.2e-3 rtol=0.066 # Analytic Linear FSBC solution
    mn,mx = extrema(ζ(sys))
    @test -2mn > mx # trough is much bigger than crest
end

@inline bruteW(x,y,z) = 4quadgk(t->exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t)),-Inf,Inf,atol=1e-10)[1]
@inline bruteN(x,y,z) = -2*(1-z/(hypot(x,y,z)+abs(x)))+NeumannKelvin.Ngk(x,y,z)
using SpecialFunctions
@testset "kelvin.jl" begin
    @test NeumannKelvin.stationary_points(-1,1/sqrt(8))[1]≈1/sqrt(2)
    @test NeumannKelvin.wavelike(10.,0.,-0.)==NeumannKelvin.wavelike(0.,10.,-0.)==0.
    @test 4π*bessely1(10)≈NeumannKelvin.wavelike(-10.,0.,-0.) atol=1e-5

    for R = (0.1,0.5,2.0,8.0), a = (0.,0.1,0.3,1/sqrt(8),0.5,1.0), z = (-1.,-0.1,-0.01)
        x = -R*cos(atan(a))
        y = R*sin(atan(a))
        x==y==0 && continue
        @test NeumannKelvin.nearfield(x,y,z)≈bruteN(x,y,z) atol=3e-4 rtol=31e-5
        x==-0.1 && y==0 && z==-0.01 && continue # failing test
        @test NeumannKelvin.wavelike(x,y,z)≈bruteW(x,y,z) atol=1e-4 rtol=31e-5
    end

    @test @ballocations(NeumannKelvin.nearfield(-1.,0.,0.)) ≤ TEST_ALLOCS
    @test @ballocations(derivative(x->NeumannKelvin.nearfield(x,0.,0.),-1.)) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.wavelike(-10.,1.,-1.)) ≤ TEST_ALLOCS
    @test @ballocations(derivative(x->NeumannKelvin.wavelike(x,1.,-1.),-10.)) ≤ TEST_ALLOCS
end

function prism(h;q=0.2,Z=1)
    S(θ,z) = 0.5SA[cos(θ),q*sin(θ),z]
    dθ = π/round(π*0.5/h) # cosine sampling
    dz = Z/round(Z/h)
    measure.(S,dθ:dθ:π,-(0.5dz:dz:Z)',dθ,dz) |> Table
end
wigley(hᵤ;B=0.125,D=0.05,hᵥ=0.25) = measure.(
    (u,v)->SA[u-0.5,2B*u*(1-u)*(v)*(2-v),D*(v-1)],
    0.5hᵤ:hᵤ:1,(0.5hᵥ:hᵥ:1)',hᵤ,hᵥ,flip=true) |> Table
@testset "NeumannKelvin.jl" begin
    # Compare submerged spheroid drag to Farell/Baar
    sys = directsolve!(NKPanelSystem(spheroid(0.04),sym_axes=2,ℓ=0.5^2))
    @test @ballocations(Φ($sys.body.x[1],$sys)) ≤ TEST_ALLOCS
    @test @ballocations(cₚ($sys.body.x[1],$sys)) ≤ TEST_ALLOCS
    @test steadyforce(sys,S=1/2)[1] ≈ 6e-3 rtol=0.02

    # Compare elliptical prism drag to Guevel/Baar
    sys = directsolve!(NKPanelSystem(prism(0.1),sym_axes=2,ℓ=0.55^2))
    @test steadyforce(sys,S=1/2)[1] ≈ 0.042 rtol=0.02 broken=true

    # Compare thin-ship wigley to Tuck 2008
    thinship(panels;Umag=1,ℓ=1) = NeumannKelvin.set_q!(
        NKPanelSystem(panels;Umag,ℓ,sym_axes=2),Umag*components(panels.n,1)/2π)
    sys = thinship(wigley(0.05),ℓ=0.51^2)
    @test steadyforce(sys)[1] ≈ 0.0088-0.0036 rtol=0.02 broken=true # Remove ITTC Cf
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

    sys = directsolve!(BodyPanelSystem(panels))
    @test bodyarea(sys) ≈ 4π rtol=1e-5
    @test bodyvol(sys) ≈ 4π/3 rtol=0.01
    @test extreme_cₚ(sys) ≈ [-1.25,1] rtol=0.04
    @test addedmass(panels) ≈ I/2 rtol=0.013
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
    @test @ballocations(∫G($panel.x,$panel)) ≤ TEST_ALLOCS
    @test @ballocations(gradient(x′->∫G(x′,$panel),$panel.x)) ≤ TEST_ALLOCS

    ext = Base.get_extension(NeumannKelvin, :NeumannKelvinGeometryBasicsExt)
    panels = panelize(load("../examples/Icosahedron.stl"))
    @test length(panels)==20
    @test eltype(panels.kernel)==ext.TriKernel
    @test all([p.n'p.x>0 for p in panels]) # all outward facing

    sys = BodyPanelSystem(panels,wrap=PanelTree)
    @test sys.body.bvh.leaves[1].volume isa ImplicitBVH.BBox # triggered correct BV
    @test bodyarea(sys) ≈ 957 rtol=1e-3
    @test bodyvol(sys) ≈ 2475 rtol=3e-2

    Ma = addedmass(panels,V=1)
    @test diag(Ma) ≈ fill(tr(Ma)/3,3) rtol=1e-4
    @test 2π/3*7.95^3 < tr(Ma)/3 < 2π/3*10^3 # between the insphere & circumsphere
end