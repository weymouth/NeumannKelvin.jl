using NeumannKelvin,Plots,ColorSchemes
using NeumannKelvin: kelvin

R = 15
# kelvin_contour(θ;x=-2R,y=R/√2,z=-R/100,R=R) = kelvin(SA[x,y,z],SA[R*cos(θ),R*sin(θ),0],ltol=-8log(10))
kelvin_contour(θᵢ,θⱼ=3.;x=R*cos(θⱼ),y=R*sin(θⱼ),z=-R/100,R=R) = kelvin(SA[x,y,z],SA[R*cos(θᵢ),R*sin(θᵢ),0],ltol=-8log(10))
using QuadGK,FastGaussQuadrature
x,w = gausslegendre(10)
plot(range(0,2π,2^12),kelvin_contour)
θᵢ = 0.3; plot(range(θᵢ-0.1,θᵢ+0.1,1000),kelvin_contour)
quadgk_count(kelvin_contour,θᵢ-0.05,θᵢ+0.05)
NeumannKelvin.quadgl(kelvin_contour,θᵢ-0.05,θᵢ+0.05;x,w)

N = 2^12
θ = range(0,2π,N+1)[1:end-1]
begin; plt=plot(xlabel="θ",ylabel="G")
for (z,c) in zip((-0.01,-0.02,-0.04),reverse(colorschemes[:Reds_4]))
    G = kelvin_contour.(θ;z=z*R)
    plot!(θ,G,label="z/R=$z";c)
end;end;plt
savefig("contour_G.png")

using FFTW
begin;plt=plot(ylims=(1e-8,1),yscale=:log10,ylabel="|FFT(G)|",
               xlims=(1e-1,100),xscale=:log10,xlabel="-kz");
for (z,c) in zip((-0.01,-0.02,-0.04),reverse(colorschemes[:Reds_4]))
    G = kelvin_contour.(θ;z=z*R)
    Ĝ = fft(G)
    dk = -z
    plot!(0:dk:(N÷2-1)*dk,abs.(Ĝ[1:N÷2])/N,label="z/R=$z";c)
end;end;plt
savefig("contour_spectral.png")