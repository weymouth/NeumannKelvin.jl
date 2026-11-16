using NeumannKelvin
using Test,BenchmarkTools
TEST_ALLOCS = get(ENV, "CI", "false") == "true" ? 32 : 0
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
    @test NeumannKelvin.aₙ(circ,0.) ≈ NeumannKelvin.aₙ(circ,0.5pi) ≈ 4
    @test NeumannKelvin.aₙ(ellip,0.) ≈ 1
    @test NeumannKelvin.aₙ(ellip,0.5pi) ≈ 3

    for devlimit in (1,0.1301,0.029165) # tuned s.t. S≈N-1
        S,s⁻¹ = NeumannKelvin.arclength(ellip,1,devlimit,0,pi)
        N = 2Int(round(S/2)); u = s⁻¹(range(0,S,N))
        if devlimit==1 # should be equal length
            l = [quadgk(NeumannKelvin.arcspeed(ellip),u[i],u[i+1])[1] for i in 1:N-1]
            @test l ≈ [sum(l)/(N-1) for i in 1:N-1] rtol=0.03
        end
        @test 3-ellip(u[N÷2])[1] ≤ 1.10devlimit # should have bounded deviation
    end

    torus(θ₁,θ₂;r=0.3,R=1) = SA[(R+r*cos(θ₂))*cos(θ₁),(R+r*cos(θ₂))*sin(θ₁),r*sin(θ₂)]
    spheroid(θ₁,θ₂;a=1.,b=1.,c=3.) = SA[a*cos(θ₂)*sin(θ₁),b*sin(θ₂)*sin(θ₁),c*cos(θ₁)]
    sphere(θ₁,θ₂) = spheroid(θ₁,θ₂; c=1.)

    # Equal areas sanity checks
    function area_checks(dA,goal)
        mdA = sum(dA)/length(dA)
        @test mdA ≈ goal rtol = 0.08
        @test maximum(abs,panels.dA .- mdA) < 0.16goal
    end
    panels = panelize(sphere,0,pi,0,2pi,hᵤ=0.5,devlimit=Inf)
    area_checks(panels.dA,0.5^2)
    @test sum(panels.dA) ≈ 4π rtol=1e-4
    panels = panelize(spheroid,0,pi,0,2pi,hᵤ=1,hᵥ=0.5,devlimit=Inf)
    area_checks(panels.dA,0.5)
    @test sum(panels.dA) ≈ 30.894 rtol=1e-3
    panels = panelize(torus,0,2pi,0,2pi,hᵤ=0.6,hᵥ=0.3,transpose=true,devlimit=Inf)
    area_checks(panels.dA,0.18)
    @test sum(panels.dA) ≈ 4π^2*0.3

    # Check sign is flipped
    panels = panelize(sphere,0,pi,0,2pi,hᵤ=0.6,flip=true)
    @test all(p->p.x'p.n<-0.9,panels)

    # Check submerged: every meridian strip crosses z=0 at θ₁=π/2
    panels = panelize(spheroid,π/6,2π/3,0,2π,hᵤ=0.6,hᵥ=0.3,transpose=true,submerge=true)
    @test length(panels) > 0
    @test all(v->all(x->x[3]<0,v),panels.verts)
    @test sum(panels.dA) ≈ 2π*quadgk(θ->sin(θ)*√(9sin(θ)^2+cos(θ)^2),π/2,2π/3)[1] rtol=1e-5

    # Strips that don't cross z=0 warn: wet strips are kept whole, dry strips dropped
    deep(θ₁,θ₂) = spheroid(θ₁,θ₂)-SA[0,0,4]
    wet = @test_logs (:warn,r"cross z=0") panelize(deep,π/6,2π/3,0,2π,hᵤ=0.6,hᵥ=0.3,transpose=true,submerge=true)
    @test wet.x == panelize(deep,π/6,2π/3,0,2π,hᵤ=0.6,hᵥ=0.3,transpose=true).x
    dry = @test_logs (:warn,r"cross z=0") panelize((θ₁,θ₂)->spheroid(θ₁,θ₂)+SA[0,0,4],π/6,2π/3,0,2π,hᵤ=0.6,hᵥ=0.3,transpose=true,submerge=true)
    @test length(dry) == 0

    # A collapsed edge throws, the other strip direction works, and tiny patches are skipped
    @test_throws ArgumentError panelize(sphere,0,π/2,0,2π,hᵤ=0.3,transpose=true)
    @test sum(panelize(sphere,0,π/2,0,2π,hᵤ=0.3).dA) ≈ 2π rtol=1e-5
    @test length(panelize(sphere,0,0.01,0,0.01,hᵤ=0.5)) == 0

    # Check inputs
    @test_throws ArgumentError panelize(spheroid,0,pi,0,2pi,hᵤ=0)
    @test_throws ArgumentError panelize(spheroid,0,pi,0,0,hᵤ=0.2)
    @test_throws ArgumentError panelize(spheroid,0,0,0,2pi,hᵤ=0.2)
    @test_throws ArgumentError panelize(spheroid,0,pi,0,2pi,hᵤ=0.1)
    @test_throws ArgumentError panelize((u,v)->[u,u,v])
end

using BenchmarkTools
@testset "PanelSystem.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = panelize(S, 0, 1f0π, 0, 2f0π, hᵤ=1/4f0)
    @test eltype(panels.dA) == Float32
    sys = PanelSystem(panels)
    @test sys.boundaries[1].q == zeros(Float32,length(panels))
    q = rand(Float32,length(panels))
    @test @ballocations(NeumannKelvin.set_q!($sys,$q)) ≤ TEST_ALLOCS
    @test sys.boundaries[1].q == q
    @test bodyarea(sys) ≈ 4π rtol=1e-5
    @test bodyvol(sys) ≈ 4π/3 rtol=5e-3
end

using LinearAlgebra
using NeumannKelvin:∂ₙϕ
@testset "kernels.jl" begin
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

@testset "PolyKernel" begin
    @testset "self-influence" begin
        pϕ(a,b) = -4*(a*asinh(b/a) + b*asinh(a/b)) # rectangle [±a,±b] centroid potential
        v = (SA[-1/2,-1,0],SA[1/2,-1,0],SA[1/2,1,0],SA[-1/2,1,0])
        tri = measure(v[1:3]...)
        quad = measure(v...) # centroid sits exactly on the fan-triangulation diagonal
        pent = measure(SA[0.,0,0], SA[1.,0,0], SA[1.5,1,0], SA[0.5,1.5,0], SA[-0.5,1,0])
        for p in (tri, quad, pent)
            @test ∂ₙϕ(p,p) ≈ 2π # self-jump regardless of shape/symmetry
        end
        @test ∫G(quad.x,quad) ≈ pϕ(1/2,1)
    end

    @testset "triangle" begin
        panel = measure(SA_F32[0,0,0],SA_F32[1,0,0],SA_F32[1,1,0])
        @test panel.x ≈ SA[2,1,0]/3
        @test panel.n ≈ SA[0,0,1]
        @test panel.dA ≈ 0.5
        @test ∫G(SA[0,0,2],panel)/4π ≈ -0.01847387 rtol=1e-6
        @test ∫G(SA[0,0,-0.5],panel)/4π ≈ -0.04558955 rtol=1e-6
        @test ∫G(SA[0.5,1,0],panel)/4π ≈ -0.05806854 rtol=1e-6
        @test ∫G(SA[-0.25,0.25,0.5],panel)/4π ≈ -0.03856218 rtol=1e-6
        @test ∫G(SA[0.1,0.4,8],panel)/4π ≈ -0.00495674 rtol=1e-6
        @test @ballocations(∫G($panel.x,$panel)) ≤ TEST_ALLOCS
        @test @ballocations(gradient(x′->∫G(x′,$panel),$panel.x)) ≤ TEST_ALLOCS
    end

    @testset "square" begin
        panel = measure(SA_F32[-1/2,-1/2,0],SA_F32[1/2,-1/2,0],SA_F32[1/2,1/2,0],SA_F32[-1/2,1/2,0])
        @test panel.x ≈ SA[0,0,0]
        @test panel.n ≈ SA[0,0,1]
        @test panel.dA ≈ 1
        @test ∫G(SA[0,0,2],panel)/4π ≈ -0.03899412 rtol=1e-6
        @test ∫G(SA[0,0,-0.5],panel)/4π ≈ -0.12626703 rtol=1e-6
        @test ∫G(SA[0.5,1,0],panel)/4π ≈ -0.07396339 rtol=1e-6
        @test ∫G(SA[-0.25,0.25,0.5],panel)/4π ≈ -0.11549639 rtol=1e-6
        @test ∫G(SA[0.1,0.4,8],panel)/4π ≈ -0.00992118 rtol=1e-6
        @test @ballocations(∫G($panel.x,$panel)) ≤ TEST_ALLOCS
        @test @ballocations(gradient(x′->∫G(x′,$panel),$panel.x)) ≤ TEST_ALLOCS
    end
end

@testset "Dipole" begin
    ring(x,V) = sum(eachindex(V)) do i # Biot-Savart of a unit-circulation vortex ring
        A,B = x-V[i], x-V[i%length(V)+1]; c = A×B
        c/(c'c)*((B-A)'*(A/norm(A)-B/norm(B)))/4π
    end
    v = (SA[0.,0,0],SA[1.,0,0],SA[1.,2,0],SA[0.,2,0])
    p = merge(measure(v...),(;kernel=NeumannKelvin.Dipole(NeumannKelvin.PolyKernel())))
    for x in (SA[0.3,0.4,0.5],SA[1.5,1.0,0.0],SA[0.5,3.0,0.0],SA[4.,-3,1]) # off-plane, in-plane, far
        @test gradient(x->∫G(x,p),x) ≈ -4π*ring(x,v) atol=1e-12 # constant-strength dipole panel ≡ vortex ring
    end
    @test NeumannKelvin.∂ₙϕ(p,p) ≈ -4π*ring(p.x,v)[3] # self-influence

    # in a Kelvin domain, the dipole is still the derivative of the source WRT its position
    sink(t) = measure((v .+ Ref(SA[0,0,t-1]))...) # source panel moved by t*n from z=-1
    p,d,x,h = merge(sink(0),(;kernel=p.kernel)),Kelvin(0.5),SA[0.3,-0.2,-0.5],1e-4
    ϕ = NeumannKelvin.ϕ
    @test ϕ(x,p,d) ≈ (ϕ(x,sink(h),d)-ϕ(x,sink(-h),d))/2h rtol=1e-6
end

extreme_cₚ(sys) = collect(extrema(cₚ(sys)))
@testset "solvers.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = panelize(S,0,π,0,2π,hᵤ=0.12)
    sys = gmressolve!(PanelSystem(panels,U=SA[3,4,0]),atol=1e-8); q = NeumannKelvin.get_q(sys)
    directsolve!(sys)
    @test NeumannKelvin.get_q(sys) ≈ q
    @test norm(steadyforce(sys)) < 4e-5
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.02

    # Check allocations
    p = panels[1]
    @test @ballocations(Φ($p.x,$sys)) ≤ TEST_ALLOCS
    @test @ballocations(∇Φ($p.x,$sys)) ≤ TEST_ALLOCS

    #check symmetry enforcement
    panels = panelize(S,0,π/2,0,π,hᵤ=0.12) # quarter plane
    sys = gmressolve!(PanelSystem(panels,domain=Unbounded(sym_axes=(2,3))),atol=1e-6); q = NeumannKelvin.get_q(sys)
    directsolve!(sys)
    @test NeumannKelvin.get_q(sys) ≈ q
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.02
end

using GeometryBasics,FileIO
function icosphere(n) # Icosahedron.stl with each triangle split in four n times, vertices on the unit sphere
    T = [map(v->normalize(SVector{3,Float64}(v)),Tuple(p.verts)) for p in panelize(load(joinpath(@__DIR__,"..","examples","Icosahedron.stl")))]
    split((a,b,c)) = (ab = normalize(a+b); bc = normalize(b+c); ca = normalize(c+a); [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
    for _ in 1:n; T = reduce(vcat,split.(T)); end
    Table(map(t->(p = measure(t...); p.n'p.x>0 ? p : measure(reverse(t)...)),T))
end
const noTE = Tuple{SVector{3,Float64},SVector{3,Float64}}[]

@testset "surfacegradient.jl" begin
    w = NeumannKelvin.loopweights(icosphere(2),noTE,Unbounded().mirrors)
    @test maximum(wᵢ->norm(sum(last,wᵢ)),w) < 1e-10 # uniform q induces no surface velocity
end

@testset "Lifting.jl" begin
    # a trailing strip = a dipole panel of length L + the strip trailing from its far end
    v₁,v₂,L = SA[0.,0,0],SA[0.,1,0],10
    strip(a,b) = (;x=(a+b)/2,n=SA[0.,0,1],verts=SA[a,b],kernel=NeumannKelvin.Wake())
    panel = merge(measure(v₁,v₂,v₂-SA[L,0,0],v₁-SA[L,0,0]),(;kernel=NeumannKelvin.Dipole(NeumannKelvin.PolyKernel())))
    for x in (SA[0.3,0.4,0.5],SA[0.2,0.5,-0.05],SA[-12.,1.5,0.3])
        ϕ(x) = ∫G(x,panel)+∫G(x,strip(v₁-SA[L,0,0],v₂-SA[L,0,0]))
        @test ∫G(x,strip(v₁,v₂)) ≈ ϕ(x)
        @test gradient(x->∫G(x,strip(v₁,v₂)),x) ≈ gradient(ϕ,x)
    end
    # ...and for the Kelvin part, the strip minus the strip L downstream = small dipole panels covering [0,L]
    ∫k(x,p,k) = NeumannKelvin.∫kelvin(x,p,k,0.25,-0.0)
    a,b,X,h,w = SA[0.,0,-0.3],SA[0.,0.2,-0.3],SA[1.,0,0],0.005,0.025
    for x in (SA[0.5,0.1,-0.2],SA[-1.,0.05,-0.25]) # upstream and downstream of the strip edge
        chain = sum(Iterators.product(0:h:1-h/2,0:w:0.2-w/2)) do (t,y)
            ∫k(x,measure(a+SA[-t,y,0],a+SA[-t,y+w,0],a+SA[-t-h,y+w,0],a+SA[-t-h,y,0]),NeumannKelvin.Dipole(NeumannKelvin.PolyKernel()))
        end
        @test ∫k(x,strip(a,b),NeumannKelvin.Wake())-∫k(x,strip(a-X,b-X),NeumannKelvin.Wake()) ≈ chain rtol=1e-3
    end

    # closed sphere without a wake: cₚ converges faster than first order
    function cₚerror(n)
        sys = directsolve!(PanelSystem(Lifting(icosphere(n),noTE)),verbose=false); x = sys.boundaries[1].x
        sum(abs.(cₚ(sys) .- map(x->1-9/4*(1-x[1]^2/x'x),x)))/length(x)
    end
    @test cₚerror(2) < cₚerror(1)/2

    # elliptic wing with a closed NACA0012 section: unit root chord, aspect ratio AR, nose up by α
    naca(s) = 0.6*(0.2969√s-0.1260s-0.3516s^2+0.2843s^3-0.1036s^4)
    AR,α = 6,deg2rad(5); b = π*AR/4; area = π*b/4; R = SA[cos(α) 0 -sin(α);0 1 0;sin(α) 0 cos(α)]
    W(θ,φ,depth) = (s=(1+cos(θ))/2; c=sinpi(φ/π); R*SA[c*(1/4-s), b/2*cos(φ), -sign(sin(θ))*c*naca(s)]-SA[0,0,depth])
    function wing(φ;depth=0)
        X = [W(θ,φ,depth) for θ in range(0,2π,33), φ in φ]; X[end,:] = X[1,:] # θ=0,2π: trailing edge
        Table([measure(X[i,j],X[i+1,j],X[i+1,j+1],X[i,j+1]) for i in 1:32 for j in 1:length(φ)-1]),[(X[1,j],X[1,j+1]) for j in 1:length(φ)-1]
    end
    CLΓ(sys,S) = (w = sys.boundaries[1].wake; sum(k->4π*w.q[k]*(sys.U×(w.verts[k][2]-w.verts[k][1]))[3],eachindex(w))/(S/2)) # Kutta–Joukowski
    sys = directsolve!(PanelSystem(Lifting(wing(range(0,π,25))...)),verbose=false)
    @test CLΓ(sys,area)/α ≈ 2π/(1+2/AR) rtol=0.01 # thin-foil lifting line
    @test steadyforce(sys;S=area)[3]/α ≈ 2π/(1+2/AR) rtol=0.02

    half = directsolve!(PanelSystem(Lifting(wing(range(0,π/2,13))...),domain=Unbounded(sym_axes=2)),verbose=false)
    @test half.boundaries[1].wake.q ≈ sys.boundaries[1].wake.q[1:12] rtol=1e-8 # symmetric domain = full wing
    @test CLΓ(half,area/2) ≈ CLΓ(sys,area) rtol=1e-8

    q = randn(length(sys.boundaries[1])); NeumannKelvin.set_q!(sys,q); r = similar(q); NeumannKelvin.bc!(r,sys)
    @test NeumannKelvin.influence(sys)*q ≈ r # direct and matrix-free operators agree

    # below a free surface: ℓ→0 is a rigid wall at z=0, and a deep foil makes no waves
    solve(depth,domain) = directsolve!(PanelSystem(Lifting(wing(range(0,π/2,13);depth)...);domain),verbose=false)
    @test CLΓ(solve(1,Kelvin(0.01,sym_axes=2)),area/2) ≈ CLΓ(solve(1,Unbounded(sym_axes=(2,3))),area/2) rtol=2e-3
    deep,wall = solve(4,Kelvin(0.5,sym_axes=2)),solve(4,Unbounded(sym_axes=(2,3)))
    @test CLΓ(deep,area/2) ≈ CLΓ(wall,area/2) rtol=1e-3
    @test steadyforce(deep;S=area/2)[1] ≈ steadyforce(wall;S=area/2)[1] atol=1e-5
    v = icosphere(1).verts[1]
    @test_throws ArgumentError Lifting(icosphere(1),[(v[1],(v[1]+v[2])/2)]) # half a panel edge
end

using ImplicitBVH
using NeumannKelvin:fill_nodes,treesum,aggregate!
@testset "barneshut.jl" begin
    cen = SA[0,0,1]
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]+cen
    panels = panelize(S,0,π,0,2π,hᵤ=0.12)
    bvh = BVH(map(ImplicitBVH.BoundingVolume,panels.verts))
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

    sys = gmressolve!(PanelSystem(PanelTree(panels)))
    q = ∂ₙϕ.(panels,panels')\components(panels.n,1)
    @test norm(NeumannKelvin.get_q(sys)-q)/norm(q) < 0.0032
    @test norm(steadyforce(sys)) < 4e-3
    @test extreme_cₚ(sys) ≈ [-1.25,1.0] rtol=0.015
    @test @ballocations(Φ($panels.x[1],$sys)) ≤ TEST_ALLOCS
end

function spheroid(h;L=1,Z=-1/8,r=1/12,AR=1/2r,kwargs...)
    S(θ₁,θ₂) = SA[0.5L*cos(θ₁),-r*sin(θ₂)*sin(θ₁),r*cos(θ₂)*sin(θ₁)+Z]
    panelize(S,0,π,0,π,hᵤ=h*√AR,hᵥ=h/√AR;kwargs...)
end
halfplane(L,ℓ;hᵤ=0.3f0ℓ,hᵥ=hᵤ,s=L/2+2f0π*ℓ) = measure.((x,y)->SA[x,y,0],s:-hᵤ:-2s,(hᵥ/2:hᵥ:4s/3)',hᵤ,hᵥ,flip=true)
@testset "FreeSurface.jl" begin
    ℓ = 0.45^2; freesurf = halfplane(1.,ℓ); body = spheroid(1/60)
    fs = FreeSurface(freesurf,ℓ=0,wrap=identity)
    sys = PanelSystem((body,fs),domain=Unbounded(sym_axes=2))
    q = NeumannKelvin.get_q(sys)
    @test length(q)==length(body)+length(freesurf)
    @test sys.boundaries[1].dA == body.dA
    @test fs.ℓ == 0
    @test sys.U == SA[-1,0,0]

    badsurf = measure.((u,v)->SA[u,v,0],-4:1:2,(1/2:1:2)',1,1,flip=true)
    @test_throws ArgumentError FreeSurface(badsurf;ℓ)

    # Setting ℓ=0 turns freesurf into reflection wall
    gmressolve!(sys)
    sys2 = gmressolve!(PanelSystem(PanelTree(body),domain=Unbounded(sym_axes=(2,3))))
    @test extreme_cₚ(sys)≈extreme_cₚ(sys2) rtol=0.03
    @test @ballocations(Φ($body.x[1],$sys)) ≤ TEST_ALLOCS

    # Setting ℓ>0 turns on freesurf, but it's slow to converge
    fs = FreeSurface(freesurf;ℓ,θ²=16)
    sys = PanelSystem((PanelTree(body;θ²=16),fs),domain=Unbounded(sym_axes=2))
    gmressolve!(sys,itmax=160)                      # should converge...
    @test -2steadyforce(sys,S=1)[1] ≈ 6.2e-3 rtol=0.066 # Analytic Linear FSBC solution
    mn,mx = extrema(ζ(fs,sys))
    @test -2mn > mx # trough is much bigger than crest
end

@inline bruteW(x,y,z) = 4quadgk(t->exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t)),-Inf,Inf,atol=1e-10)[1]
@inline bruteN(x,y,z) = -2*(1-z/(hypot(x,y,z)+abs(x)))+NeumannKelvin.Ngk(x,y,z)
using SpecialFunctions
using ForwardDiff: Dual
@testset "kelvin.jl" begin
    @test NeumannKelvin.stationary_points(-1.,1/sqrt(8))[1]≈1/sqrt(2)
    @test NeumannKelvin.wavelike(10.,0.,-0.)==NeumannKelvin.wavelike(0.,10.,-0.)==0.
    @test 4π*bessely1(10)≈NeumannKelvin.wavelike(-10.,0.,-0.) atol=1e-5

    for R = (0.1,0.5,2.0,8.0), a = (0.,0.1,0.3,1/sqrt(8),0.5,1.0), z = (-1.,-0.1,-0.01)
        x = -R*cos(atan(a))
        y = R*sin(atan(a))
        x==y==0 && continue
        @test NeumannKelvin.nearfield(x,y,z)≈bruteN(x,y,z) atol=3e-4 rtol=31e-5
        @test NeumannKelvin.wavelike(x,y,z)≈bruteW(x,y,z) atol=1e-4 rtol=2e-4
    end
    @test NeumannKelvin.Ngk(0.,0.6,-0.6)≈NeumannKelvin.Ngk(1e-8,0.6,-0.6) atol=1e-8 # Re(ζ)=0 inside (-1,1) at x=0
    jump(y) = (d(x) = derivative(x->NeumannKelvin.nearfield(x,y,-0.),x); d(1e-7)-d(-1e-7)) # ∂ₓN jumps at x=0
    for y in (0.25,0.1,0.05,0.02)
        @test jump(y)≈(-3bruteN(0.,y,0.)+4bruteN(1e-4,y,0.)-bruteN(2e-4,y,0.))/1e-4 rtol=5e-5
    end

    # ∇N is smooth across the Chebyshev zone boundaries, so the one-sided gradients must agree
    ∇N(X) = gradient(X->NeumannKelvin.nearfield(X...),X)
    for (Rb,bound) in ((1,3e-4),(4,1.5e-3),(10,4e-3)), θ in (0.05,0.3,0.7,1.2,1.5), α in (0.,0.4,0.9,1.3,π/2)
        X = NeumannKelvin.S2X(SA[Rb,θ,α^2]); g⁺ = ∇N(X*(1+1e-8))
        @test norm(g⁺-∇N(X*(1-1e-8))) ≤ bound*norm(g⁺)
    end

    @test @ballocations(NeumannKelvin.nearfield(-0.3,0.2,-0.1)) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.nearfield(Dual(-0.3,0),Dual(0.2,0),Dual(-0.1,0))) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.nearfield(Dual(-0.3f0,0),Dual(0.2f0,0),Dual(-0.1f0,0))) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.nearfield(-1.,0.,0.)) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.nearfield(Dual(-1.,0),Dual(0.,0),Dual(0.,0))) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.nearfield(Dual(-1f0,0),Dual(0f0,0),Dual(0f0,0))) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.wavelike(-10.,1.,-1.)) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.wavelike(Dual(-10.,1),Dual(1.,1),Dual(-1.,1))) ≤ TEST_ALLOCS
    @test @ballocations(NeumannKelvin.wavelike(Dual(-10f0,1),Dual(1f0,1),Dual(-1f0,1))) ≤ TEST_ALLOCS
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
    sys = PanelSystem(spheroid(0.04),domain=Kelvin(0.5^2,sym_axes=2)) |> directsolve!
    x = sys.boundaries[1].x[1]
    @test @ballocations(Φ($x,$sys)) ≤ TEST_ALLOCS
    @test @ballocations(cₚ($x,$sys)) ≤ TEST_ALLOCS
    @test -steadyforce(sys,S=1/2)[1] ≈ 6e-3 rtol=0.023

    # Compare elliptical prism drag to Guevel/Baar (no WL contour)
    sys = PanelSystem(prism(0.1),domain=Kelvin(0.55^2,sym_axes=2)) |> directsolve!
    @test -steadyforce(sys,S=1/2)[1] ≈ 0.062 rtol=0.03

    # Compare thin-ship wigley to Tuck 2008 (no WL contour)
    sys = PanelSystem(wigley(0.05),domain=Kelvin(0.51^2,sym_axes=2)) |> directsolve!
    @test -steadyforce(sys)[1] ≈ 0.0088-0.003 rtol=0.03 # Remove ITTC Cf
end

using NURBS,FileIO
@testset "NURBS" begin
    sphere = load(pkgdir(NURBS) * "/test/assets/sphere.stp")
    
    # Test that gNURBS gives same results as NURBS.jl
    patch = sphere[1]
    ext = Base.get_extension(NeumannKelvin, :NeumannKelvinNURBSExt)
    gpatch = ext.gNURBS(patch)
    u,v = collect(0.1:0.2:0.9), collect(0.1:0.4:0.9)
    nurbs_result = patch(u, v)
    gnurbs_result = gpatch.(u, v')
    @test gnurbs_result ≈ nurbs_result
    # @btime $patch($u, $v)
    # @btime $gpatch.($u, $v')    
    # @btime $patch(0.45, 0.55)[1]
    # @btime $gpatch(0.45, 0.55)

    # Test AD w.r.t control points
    using ForwardDiff
    pnts_flat = reduce(vcat, [collect(p) for p in vec(gpatch.pnts)])
    cu, cv = size(gpatch.pnts)
    make_matrix(x) = reshape(reinterpret(SVector{3,eltype(x)}, x), cu, cv)
    test_fn(x) = begin
        g = ext.gNURBS(make_matrix(x), gpatch.uknots, gpatch.vknots, gpatch.wgts)
        norm(g(0.3, 0.4))
    end
    grad = ForwardDiff.gradient(test_fn, pnts_flat)
    @test grad isa Vector  # Should compute without error
    @test !all(iszero, grad)  # Should have non-zero gradients

    # measure a whole patch as one panel
    sphere = loadNURBS(pkgdir(NURBS) * "/test/assets/sphere.stp")
    patch = sphere[1]
    panel = measure(patch,0.5,0.5,1.0,1.0)
    @test panel.kernel isa NeumannKelvin.QuadKernel
    @test panel.dA ≈ 4π/6 rtol=0.03

    # panlize the whole sphere
    panels = panelize(sphere,hᵤ=0.25)
    @test all( 0.9 .< extrema(panels.dA) ./ 0.25^2 .< 4/3)

    sys = directsolve!(PanelSystem(panels))
    @test bodyarea(sys) ≈ 4π rtol=1e-5
    @test bodyvol(sys) ≈ 4π/3 rtol=0.01
    @test extreme_cₚ(sys) ≈ [-1.25,1] rtol=0.04
    @test addedmass(panels) ≈ I/2 rtol=0.013
end

using GeometryBasics,FileIO
@testset "GeometryBasics" begin
    panels = panelize(load(joinpath(@__DIR__,"..","examples","Icosahedron.stl")))
    @test length(panels)==20
    @test eltype(panels.kernel)==NeumannKelvin.PolyKernel
    @test all([p.n'p.x>0 for p in panels]) # all outward facing

    sys = PanelSystem(PanelTree(panels))
    @test sys.boundaries[1].bvh.leaves[1].volume isa ImplicitBVH.BBox # triggered correct BV
    @test bodyarea(sys) ≈ 957 rtol=1e-3
    @test bodyvol(sys) ≈ 2475 rtol=3e-2

    Ma = addedmass(panels,V=1)
    @test diag(Ma) ≈ fill(tr(Ma)/3,3) rtol=1e-4
    @test 2π/3*7.95^3 < tr(Ma)/3 < 2π/3*10^3 # between the insphere & circumsphere
end