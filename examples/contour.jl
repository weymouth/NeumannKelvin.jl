using NeumannKelvin,Plots,ColorSchemes
using NeumannKelvin: kelvin

kelvin_contour(θ;x=-5,y=10/√8,z=-1,R=5) = kelvin(SA[x,y,z],SA[R*cos(θ),R*sin(θ),0])
plot(range(0,2π,2^9),θ->kelvin_contour(θ,z=-0.1))

N = 2^11
θ = range(0,2π,N+1)[1:end-1]
plt=plot(xlabel="θ",ylabel="G");
for (z,c) in zip((-0.05,-0.1,-0.2),reverse(colorschemes[:Reds_4]))
    G = kelvin_contour.(θ;z)
    plot!(θ,G,label="z=$z";c)
end;plot!(xlims=(0.9,1.1))
savefig("contour_G.png")

dk = 1/5
plt=plot(ylims=(1e-7,2),yscale=:log10,ylabel="|Ĝ|",xlabel="k");
for (z,c) in zip((-0.05,-0.1,-0.2),reverse(colorschemes[:Reds_4]))
    G = kelvin_contour.(θ;z)
    Ĝ = fft(G)
    plot!(0:dk:(N÷2-1)*dk,abs.(Ĝ[1:N÷2])/N,label="z=$z";c)
    plot!([-π/2z,-3π/z],[1e-2,1e-7]; line = (3,:black), label=nothing)
    plot!([-π/2z,-3π/z],[1e-2,1e-7]; line = (2,:dash,c), label=nothing)
    scatter!([-π/z],[1e-3]; c, label=nothing)
end;plt
savefig("contour_spectral.png")

panel = measure_panel((θ,z)->SA[5cos(θ),5sin(θ),z],1,-0.05,0.1,0.1)
∫kelvin(SA[-5,10/√8,-0.05],panel)/10panel.dA
using QuadGK,FastGaussQuadrature
quadgk_count(θ->kelvin_contour(θ,z=-0.05),0.95,1.05,atol=2e-5) # x,y are too easy
x,w = gausslegendre(ceil(Int,5*0.1/0.05))
NeumannKelvin.quadgl(θ->kelvin_contour(θ,z=-0.05),0.95,1.05;x,w)