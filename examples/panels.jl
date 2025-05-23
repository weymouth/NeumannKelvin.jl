using NeumannKelvin,Plots,ApproxFun
gr()
ellip(u) = [3sin(u),cos(u)]
ep(u) = ellip.(u)|> x->(first.(x),last.(x))

plot(ep(0:0.01:pi),label="exact",c=:black)
colors = [:darkblue,:mediumblue,:lightblue]
cs = (0.01,0.1,1.0)
for (color,c) in zip(colors,cs)
    speed = NeumannKelvin.pseudospeed(ellip,1,c,0..pi)
    S = sum(speed); N = 2Int(round(S/2)); @show N,c
    u = NeumannKelvin.dist⁻¹.(speed,range(0,S,N))
    plot!(ep(u),line=:dash, marker=:circle,c=color,label="deviation≤$c")
end
plot!()

plot(ep(0:0.01:pi),label="exact",c=:black)
colors = [:forestgreen,:limegreen,:lightgreen]
types = [:dashdot,:dash,:solid]
hs = (0.25,0.5,1.0)
for (color,h,lt) in zip(colors,hs,types)
    speed = NeumannKelvin.pseudospeed(ellip,h,0.1,0..pi)
    S = sum(speed); N = 2Int(round(S/2h)); @show N,h
    u = NeumannKelvin.dist⁻¹.(speed,range(0,S,N))
    plot!(ep(u),line=lt, marker=:circle,c=color,label="h≤$h")
end
plot!()

using NeumannKelvin,Plots,Colors
plotlyjs()
spheroid(θ₁,θ₂;a=1.,b=1.,c=1.) = SA[a*cos(θ₂)*sin(θ₁),b*sin(θ₂)*sin(θ₁),c*cos(θ₁)]
grid = spheroid.(range(0,π,20),range(0,2π,30)')
X,Y,Z = ntuple(i->getindex.(grid,i),3)
panels = panelize(spheroid,0,π,0,2π,hᵤ=0.5)
q = influence(panels)\first.(panels.n)

function plot_panel!(panel; show_center=true, show_normal=true, show_gauss=false, color=:orange, nscale=0.3)
    surface!(ntuple(i->getindex.(panel.c₄,i),3); color, alpha=0.6)
    show_center && scatter3d!(ntuple(i->[panel.x[i]],3); color, markersize=2,label=nothing)
    show_normal && quiver!(ntuple(i->[panel.x[i]],3), quiver=ntuple(i->[nscale*panel.n[i]],3), color=:black, linewidth=1)
    show_gauss && scatter3d!(ntuple(i->vec(getindex.(panel.x₄,i)),3), color=:red, markersize=2,label=nothing)
end
normalize(q) = ((h,l)=extrema(q); (q .- l) ./(h-l))

plt=surface(X,Y,Z, color=:lightblue, alpha=0.5, label="Spheroid")
foreach(i->plot_panel!(panels[i],color=cgrad(:plasma)[q[i]]),eachindex(panels,normalize(q)))
scatter3d(eachrow(stack(panels.x))...,color=q,colormap=:plasma,colorbar=true)
