using NeumannKelvin
using Test

using QuadGK
@testset "util.jl" begin
    @test NeumannKelvin.quadgl(x->x^3-3x^2+4)≈6
    @test NeumannKelvin.quadgl_ab(x->x^3-3x^2+4,0,2)≈4
    @test NeumannKelvin.quadgl_ab(x->sin(x^2/pi),0,2π)≈0.86185 atol=1e-5

    g(x) = x^2+im*x^2/100
    dg(x) = 2x+im*x/50
    I,e,c=quadgk_count(x->imag(exp(im*g(x))),2.,Inf)
    # @show I,e,c # (-0.17247612166701298, 2.5164550897175237e-9, 8355)
    @test NeumannKelvin.nsp(2.,g,dg) ≈ I atol=1e-5

    I,e,c=quadgk_count(x->imag(exp(im*g(x))),-Inf,Inf)
    # @show I,e,c # (1.247000964522693, 1.824659458372201e-8, 15195)
    @test NeumannKelvin.complex_path(g,dg,[(-2.,2.)]) ≈ I atol=1e-5
end

@testset "panel_method.jl" begin
    S(θ₁,θ₂) = SA[cos(θ₂)*sin(θ₁),sin(θ₂)*sin(θ₁),cos(θ₁)]
    panels = param_props.(S,[π/4,3π/4]',π/4:π/2:2π,π/2,π/2) |> Table
    @test size(panels) == (8,)
    @test panels.n ⋅ panels.x == 8
    @test sum(panels.dA)/4π-1 < 0.12

    A,b = ∂ₙϕ.(panels,panels'),-Uₙ.(panels)
    @test tr(A) == 8*2π
    @test 1-minimum(A)/0.25panels[1].dA < 0.12
    @test sum(b)<8eps()

    q = A \ b
    @test A*q≈b
end

using SpecialFunctions
function bruteN(x,y,z)
    Ni(t) = imag(expintx((z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)))
    2/π*quadgk(Ni,-1,1)[1]
end
function bruteW(x,y,z)
	Wi(t) = exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t))
	4quadgk(Wi,-Inf,Inf)[1]
end
@testset "green.jl" begin
    @test NeumannKelvin.stationary_points(-1,1/sqrt(8))[1]≈1/sqrt(2)

    @test abs(NeumannKelvin.wavelike(10.,0.,-0.))==NeumannKelvin.wavelike(0.,10.,-0.)==0.

    @test abs(4π*bessely1(10)-NeumannKelvin.wavelike(-10.,0.,-0.))<1e-5

    for R = (0.0,0.1,0.5,2.0,8.0), a = (0.,0.1,0.3,1/sqrt(8),0.5,1.0), z = (-1.,-0.1,-0.01)
        x = -R*cos(atan(a))
        y = R*sin(atan(a))
        x==y==0 && continue
        @test abs(NeumannKelvin.nearfield(x,y,z)/bruteN(x,y,z)-1)<1e-3
        @test abs(NeumannKelvin.wavelike(x,y,z)-bruteW(x,y,z))<5e-5
    end
end

u²(x,q,panels;kwargs...) = sum(abs2,SA[-1,0,0]+∇φ(x,q,panels;kwargs...))
drag(q,panels;kwargs...) = sum(panels) do pᵢ
	cₚ = 1-u²(pᵢ.x,q,panels;kwargs...)
	cₚ*pᵢ.n[1]*pᵢ.dA
end
function solve_drag(panels;kwargs...)
	A,b = ∂ₙϕ.(panels,panels';kwargs...),-Uₙ.(panels, U= [-1.,0.,0.])
	q = A\b; @assert A*q ≈ b
	drag(q,panels;kwargs...)
end
function submarine(h;Z=-0.5,L=1,r=0.25)
    S(θ₁,θ₂) = SA[0.5L*cos(θ₁),r*cos(θ₂)*sin(θ₁),r*sin(θ₂)*sin(θ₁)+Z]
    dθ₁ = π/round(π*0.5L/h) # azimuth step size
    mapreduce(vcat,0.5dθ₁:dθ₁:π) do θ₁
        dθ₂ = π/round(π*0.25L*sin(θ₁)/h) # polar step size at this azimuth
        param_props.(S,θ₁,0.5dθ₂:dθ₂:2π,dθ₁,dθ₂)
    end |> Table
end
@testset "NeumannKelvin.jl" begin
    # Compare to Baar_1982_prolate_spheroid
    d = solve_drag(submarine(0.05;Z=-1/8,r=1/12);G=kelvin,Fn = 0.5)
    @test d ≈ 6e-3 atol=3e-4 # 5%
end