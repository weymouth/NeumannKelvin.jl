using NeumannKelvin
using Test

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

using QuadGK
using SpecialFunctions
function bruteN(x,y,z)
    r = hypot(x,y,z)
    Ni(t) = imag(expintx((z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)))
    1/r+2/π*quadgk(Ni,-1,1)[1]
end
function bruteW(x,y,z)
	Wi(t) = exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t))
	4quadgk(Wi,-Inf,Inf)[1]
end
@testset "green.jl" begin
    @test NeumannKelvin.stat_points(-1,1/sqrt(8),Inf)≈[1/sqrt(2)+0im]
    x,y,z = -10,1,-1
    @test abs(bruteN(x,y,z)-NeumannKelvin.nearfield(x,y,z))<1e-3
    @test abs(bruteW(x,y,z)-NeumannKelvin.wavelike(x,y,z))<1e-3
end
