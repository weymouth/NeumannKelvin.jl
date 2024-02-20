using NeumannKelvin
using Test

@testset "NeumannKelvin.jl" begin
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
