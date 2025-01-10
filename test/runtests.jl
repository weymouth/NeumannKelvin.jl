using NeumannKelvin
using Test

using QuadGK
@testset "util.jl" begin
    using NeumannKelvin: xgl2,wgl2
    @test NeumannKelvin.quadgl(x->x^3-3x^2+4,x=xgl2,w=wgl2)≈6
    @test NeumannKelvin.quadgl(x->x^3-3x^2+4,0,2,x=xgl2,w=wgl2)≈4

    rngs=NeumannKelvin.finite_ranges((0.,),x->x^2,4,Inf) 
    @test all( isapprox.(rngs[1],(-2,0),atol=4*0.3) .&& isapprox.(rngs[2],(0,2),atol=4*0.3) )

    rngs=NeumannKelvin.finite_ranges((0.,),x->x^2,6,2,atol=0)
    @test all(rngs[1] .≈ (-2,0) .&& rngs[2] .≈ (0,2))

    # Highly oscillatory integral set-up
    g(x) = x^2+im*x^2/100
    dg(x) = 2x+im*x/50
    f(x) = imag(exp(im*g(x)))
    ρ = √(3π); rng = (-ρ,ρ)

    I,e,c=quadgk_count(f,rng...)
    # @show I,e,c # (1.5408137136825548, 1.0477053419277738e-8, 165) # easy
    @test NeumannKelvin.quadgl(f,rng...) ≈ I atol=1e-5

    I,e,c=quadgk_count(f,ρ,Inf)
    # @show I,e,c # (-0.14690637593307346, 2.133881538591021e-9, 8265) # hard
    @test NeumannKelvin.nsp(ρ,g,dg) ≈ I atol=1e-5

    I,e,c=quadgk_count(f,-Inf,Inf)
    # @show I,e,c # (1.247000964522693, 1.824659458372201e-8, 15195)
    @test NeumannKelvin.complex_path(g,dg,(rng,)) ≈ I atol=1e-5
end

@testset "panel_method.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = param_props.(S,[π/4,3π/4]',π/4:π/2:2π,π/2,π/2) |> Table
    @test size(panels) == (8,)
    @test panels.n ⋅ panels.x == 8
    @test sum(panels.dA) ≈ 4π rtol=0.1

    A,b = ∂ₙϕ.(panels,panels'),-Uₙ.(panels)
    @test A≈influence(panels)
    @test tr(A) == 8*2π
    @test 4minimum(A) ≈ panels[1].dA  rtol=1/8
    @test sum(b)<8eps()

    q = A \ b
    @test A*q≈b
    @test allequal(map(x->abs(round(x,digits=5)),q))
    # m_zz is wrong! Must have to do with the choice of parameterization
    @test added_mass(panels)≈[2π/3 0 0;0 2π/3 0; 0 0 5] rtol=0.055 # ϵ=5.5% with 8 panels
end

function bruteW(x,y,z)
	Wi(t) = exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t))
	4quadgk(Wi,-Inf,Inf,atol=1e-10)[1]
end
using SpecialFunctions
@testset "green.jl" begin
    @test NeumannKelvin.stationary_points(-1,1/sqrt(8))[1]≈1/sqrt(2)

    @test NeumannKelvin.wavelike(10.,0.,-0.)==NeumannKelvin.wavelike(0.,10.,-0.)==0.

    @test 4π*bessely1(10)≈NeumannKelvin.wavelike(-10.,0.,-0.) atol=1e-5

    for R = (0.0,0.1,0.5,2.0,8.0), a = (0.,0.1,0.3,1/sqrt(8),0.5,1.0), z = (-1.,-0.1,-0.01)
        x = -R*cos(atan(a))
        y = R*sin(atan(a))
        x==y==0 && continue
        @test NeumannKelvin.nearfield(x,y,z)≈NeumannKelvin.bruteN(x,y,z) atol=6e-4
        @test NeumannKelvin.wavelike(x,y,z)≈bruteW(x,y,z) atol=1e-5 rtol=2.1e-5
    end
end

function solve_drag(panels,U=SVector(-1,0,0);kwargs...)
	q = influence(panels;kwargs...)\(-Uₙ.(panels;U))
	steady_force(q,panels;U,kwargs...)[1]
end
function spheroid(h;L=1,Z=-1/8,r=1/12,AR=2)
    S(θ₁,θ₂) = SA[0.5L*cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)+Z]
    dθ₁ = π/round(π*0.5L/√AR/h) # cosine sampling increases density at ends 
    mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
        dx = dθ₁*hypot(r*cos(θ₁),0.5L*sin(θ₁))
        dθ₂ = π/round(π*r*sin(θ₁)*AR/dx) # polar step size at this azimuth
        param_props.(S,θ₁,0.5dθ₂:dθ₂:2π,dθ₁,dθ₂)
    end |> Table
end
function prism(h;q=0.2,Z=1,r=1.2)
    S(θ,z) = 0.5SA[cos(θ),q*sin(θ),z]
    dθ = π/round(π*0.5/h) # cosine sampling
    mapreduce(vcat,0.5dθ:dθ:2π) do θ
        dx = dθ*hypot(q*cos(θ),sin(θ))
        i = round(log(1+2Z/dx*(r-1))/log(r)) # geometric growth
        mapreduce(vcat,1:i) do j
            z,dz = -dx*(1-r^j)/(1-r),dx*r^j
            param_props.(S,θ,z+0.5dz,dθ,dz)
        end
    end |> Table
end
@testset "NeumannKelvin.jl" begin
    h = 0.06
    # Compare submerged spheroid drag to Farell/Baar
    d = solve_drag(spheroid(h);G=kelvin,Fn=0.5)
    @test d ≈ 6e-3 rtol=0.02
    # Compared elliptical prism drag to Guevel/Baar
    d = solve_drag(prism(h);G=kelvin,Fn=0.55)
    @test d ≈ 0.1 rtol=0.01
end