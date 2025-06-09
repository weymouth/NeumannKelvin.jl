using NeumannKelvin
using Test

using QuadGK
@testset "quad.jl" begin
    xgl2,wgl2 = (-1/√3,1/√3),(1,1)
    @test NeumannKelvin.quadgl(x->x^3-3x^2+4,x=xgl2,w=wgl2)≈6
    @test NeumannKelvin.quadgl(x->x^3-3x^2+4,0,2,x=xgl2,w=wgl2)≈4

    rngs=NeumannKelvin.finite_ranges((0.,),x->x^2,4,Inf)
    ((a₁,f₁),(a₂,f₂)),((a₃,f₃),(a₄,f₄))=NeumannKelvin.finite_ranges((0.,),x->x^2,4,Inf)
    @test [a₁,a₂,a₃,a₄]≈[-2,0,0,2] atol=0.3
    @test [f₁,f₂,f₃,f₄]==[true,false,false,true]

    ((a₁,f₁),(a₂,f₂)),((a₃,f₃),(a₄,f₄))=NeumannKelvin.finite_ranges((0.,),x->x^2,6,2,atol=0)
    @test [a₁,a₂,a₃,a₄]≈[-2,0,0,2] && !any([f₁,f₂,f₃,f₄])

    # Highly oscillatory integral set-up
    g(x) = x^2+im*x^2/100
    dg(x) = 2x+im*x/50
    f(x) = imag(exp(im*g(x)))
    ρ = √(3π); rng = (-ρ,ρ); rngs = (((-ρ,true),(ρ,true)),)

    I,e,c=quadgk_count(f,rng...)
    # @show I,e,c # (1.5408137136825548, 1.0477053419277738e-8, 165) # easy
    @test NeumannKelvin.quadgl(f,rng...) ≈ I atol=1e-5

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

    # Deviation checks
    dev_checks(panels,goal) = @test maximum(NeumannKelvin.deviation,panels) ≤ goal
    dev_checks(panelize(spheroid,0,pi,0,2pi,hᵤ=0.5),0.05)
    dev_checks(panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,2pi,hᵤ=1,hᵥ=0.5),0.075)
    dev_checks(panelize(torus,0,2pi,0,2pi,hᵤ=0.6,hᵥ=0.3,transpose=true),0.045)

    # Check inputs
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,2pi,hᵤ=0)
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,0,hᵤ=0.2)
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,0,0,2pi,hᵤ=0.2)
    @test_throws ArgumentError panelize((u,v)->spheroid(u,v;c=3.),0,pi,0,2pi,hᵤ=0.1)
end

using LinearAlgebra
@testset "panel_method.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = measure_panel.(S,[π/4,3π/4]',π/4:π/2:2π,π/2,π/2,cubature=true) |> Table
    @test size(panels) == (8,)
    @test panels.dA ≈ fill(π/2,8) rtol=1e-6   # cubature gives perfect areas
    @test panels.n ⋅ panels.x ≈ 4√3 rtol=1e-6 # ...and centroids

    A,b = ∂ₙϕ.(panels,panels'),first.(panels.n)
    @test A ≈ influence(panels)
    @test tr(A) ≈ 8*2π
    @test minimum(A) ≈ panels[1].dA/4 rtol=0.2 # rough estimate
    @test sum(b)<8eps()

    q = A \ b
    @test A*q≈b
    @test allequal(map(x->abs(round(x,digits=14)),q))
    Ma = added_mass(panels)
    @test Ma ≈ 2π/3*I rtol=0.1 # ϵ=10% with 8 panels
    @test diag(Ma) ≈ fill(sum(diag(Ma))/3,3) rtol=1e-3 # x/y/z symmetric!
end

@inline bruteW(x,y,z) = 4quadgk(t->exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t)),-Inf,Inf,atol=1e-10)[1]
@inline bruteN(x,y,z) = -2*(1-z/(hypot(x,y,z)+abs(x)))+NeumannKelvin.Ngk(x,y,z)
using SpecialFunctions
@testset "green.jl" begin
    @test NeumannKelvin.stationary_points(-1,1/sqrt(8))[1]≈1/sqrt(2)

    @test NeumannKelvin.wavelike(10.,0.,-0.)==NeumannKelvin.wavelike(0.,10.,-0.)==0.

    @test 4π*bessely1(10)≈NeumannKelvin.wavelike(-10.,0.,-0.) atol=1e-5

    @test @allocated(NeumannKelvin.wavelike(-10.,0.,-0.))==0

    for R = (0.0,0.1,0.5,2.0,8.0), a = (0.,0.1,0.3,1/sqrt(8),0.5,1.0), z = (-1.,-0.1,-0.01)
        x = -R*cos(atan(a))
        y = R*sin(atan(a))
        x==y==0 && continue
        @test NeumannKelvin.nearfield(x,y,z)≈bruteN(x,y,z) atol=3e-4 rtol=31e-5
        @test NeumannKelvin.wavelike(x,y,z)≈bruteW(x,y,z) atol=1e-5 rtol=1e-4
    end
end

Cw(panels;kwargs...) = steady_force(influence(panels;kwargs...)\first.(panels.n),panels;kwargs...)[1]
function spheroid(h;L=1,Z=-1/8,r=1/12,AR=2)
    S(θ₁,θ₂) = SA[0.5L*cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)+Z]
    panelize(S,0,π,0,2π,hᵤ=h*√AR,hᵥ=h/√AR)
end
function prism(h;q=0.2,Z=1)
    S(θ,z) = 0.5SA[cos(θ),q*sin(θ),z]
    dθ = π/round(π*0.5/h) # cosine sampling
    dz = Z/round(Z/h)
    measure_panel.(S,dθ:dθ:2π,-(0.5dz:dz:Z)',dθ,dz) |> Table
end
@testset "NeumannKelvin.jl" begin
    # Compare submerged spheroid drag to Farell/Baar
    d = Cw(spheroid(0.06);ϕ=∫kelvin,Fn=0.5)
    @test d ≈ 6e-3 rtol=0.02
    # Compared elliptical prism drag to Guevel/Baar
    d = Cw(prism(0.1);ϕ=∫kelvin,Fn=0.55)
    @test d ≈ 0.053 rtol=0.02
end