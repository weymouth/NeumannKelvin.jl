module NeumannKelvinPlotsExt
using NeumannKelvin,Plots
using Plots

function NeumannKelvin.viz(panels::Table, values=panels.dA; vectors=panels.n,
                           clims=extrema(values), colormap=:viridis, kwargs...)
    # Get colors from colormap
    cmap = cgrad(colormap)
    color = [cmap[(v - clims[1]) / (clims[2] - clims[1])] for v in values]
    tricolor = eltype(panels.kernel)==NeumannKelvin.QuadKernel ? mapreduce(c->[c;c],vcat,color) : color
    
    # Collect all vertices
    xs = Float32[]
    ys = Float32[]
    zs = Float32[]
    connections_i = Int[]
    connections_j = Int[]
    connections_k = Int[]
    
    for panel in panels
        addtri!(xs,ys,zs,connections_i,connections_j,connections_k,panel.verts,panel.kernel)
    end
    
    # Main 3D plot
    cx,cy,cz = components(panels.x)
    p1 = scatter3d(cx,cy,cz;markersize=0,fill_z=values,clims,c=colormap,colorbar=true,legend=false,kwargs...)
    mesh3d!(p1, xs, ys, zs; color = tricolor, linewidth=0,
            connections=(connections_i, connections_j, connections_k))
            
    # Add normal vectors if provided
    if !isnothing(vectors)
        # Extract panel centers and normal vectors
        nx,ny,nz = components(vectors)
        quiver!(p1, cx, cy, cz; color,
            quiver=(nx, ny, nz), arrow=:closed, linewidth=1.5)
    end
    plot(p1)
end
addtri!(x,y,z,i,j,k,verts,::NeumannKelvin.QuadKernel) = (
    addtri!(x,y,z,i,j,k,verts[1:3]); addtri!(x,y,z,i,j,k,verts[[1,3,4]]))
function addtri!(x,y,z,i,j,k,verts,args...)
    for v in verts
        push!(x, v[1]); push!(y, v[2]); push!(z, v[3])
    end
        
    # Connections for this triangle (0-indexed)
    base = length(x)-3
    push!(i, base); push!(j, base + 1); push!(k, base + 2)
end
end