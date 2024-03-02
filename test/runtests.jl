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
    Ni(t) = imag(expintx((z*sqrt(1-t^2)+y*t+im*abs(x))*sqrt(1-t^2)))
    2/π*quadgk(Ni,-1,1)[1]
end
function bruteW(x,y,z)
	Wi(t) = exp(z*(1+t^2))*sin((x+y*t)*hypot(1,t))
	4quadgk(Wi,-Inf,Inf)[1]
end
@testset "green.jl" begin
    @test NeumannKelvin.wavelike(10,0,0)==NeumannKelvin.wavelike(10,0,0)==0

    for x = (-1.,-4.,-16.), y = (0.,1.,4.,16.), z = (-1.,-0.1,-0.01)
        @test abs(NeumannKelvin.nearfield(x,y,z)/bruteN(x,y,z)-1)<0.01
        r = abs(NeumannKelvin.wavelike(x,y,z)/bruteW(x,y,z)-1)
        r>0.01 && @show x,y,z,r
        @test r<0.03
    end
end