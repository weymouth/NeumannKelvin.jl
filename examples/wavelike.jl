using NeumannKelvin,StaticArrays
using Plots

x = -20:0.01:-1
plot(x,x->NeumannKelvin.wavelike(x,0.,-0.),xlabel="x",ylabel="W(x,0,0)",label="")
plot!(x,x->4π*bessely1(-x),label="",lc=:grey,ls=:dash)
savefig("wavelike_centerline.png")

x = -40:0.01:-1
plot(x,x->NeumannKelvin.wavelike(x,abs(x/(2√2)),-0.),label="")
plot!(x,x->11hypot(x,x/(2√2))^(-1/3),lc=:grey,label="",xlabel="x")
plot!(x,x->-11hypot(x,x/(2√2))^(-1/3),lc=:grey,label="",ylabel="W(x,x/2√2,0)")
savefig("wavelike_kelvin_angle.png")

z = log.(0.01:0.01:1)
plot(z,z->4exp(z)*√(-π*z),lc=:grey,xlabel="z",ylabel="zW(x,0,z)",label="")
plot!(z,z .*NeumannKelvin.wavelike.(-1.2,0.,z),lc=:darkgreen,label="x=-1.2")
plot!(z,z .*NeumannKelvin.wavelike.(-1,0.,z),lc=:green,label="x=-1")
plot!(z,z .*NeumannKelvin.wavelike.(-0.8,0.,z),lc=:seagreen,label="x=-0.8")
savefig("wavelike_vertical.png")