import numpy as np
ndim = 2
order = 4
nbary = ndim + 1
nodes = []

# Generate Vertices
for i in range(0, nbary):
    vertex = np.zeros(nbary)
    vertex[i] = order
    nodes.append(vertex)

# Free index sets
def gen_free_index_set(nfree, maxdim):
    free_index_set = []
    # base case nfree = 2
    if nfree == 2:
        for idim in range(nfree-1, maxdim+1):
            # generate all the free indices added in the idim-th dimension
            top_list = 2 * [0]
            top_list[1] = idim;
            for jdim in range(idim):
                free_indices = top_list.copy()
                free_indices[0] = jdim
                free_index_set.append(free_indices)
                
            # Manual fix for NASA CGNS shenanigans (for the 2D triangle we do ccw and then abandon that convention)
            if(len(free_index_set) > 2):
                free_index_set[1] = [1, 2]
                free_index_set[2] = [0, 2]
    else:
        for idim in range(nfree-1, maxdim+1):
            # generate all the free indices recursively
            # [[nfree = ndim-1, maxdim = idim - 1], idim]
            sublist = gen_free_index_set(nfree-1, idim-1)
            for free_indices_base in sublist:
                free_indices = free_indices_base.copy()
                free_indices.append(idim)
                free_index_set.append(free_indices)
    
    return free_index_set

# Generate Edges
# Generate all the free indices chosing two free indices
nfree = 2
free_index_set = gen_free_index_set(nfree, ndim)
    
print("list of free indices: ")
for free_indices in free_index_set:
    print(free_indices)
    
print("list of edge nodes:")
for free_indices in free_index_set:
    
    if(free_indices[0] == 0 and free_indices[1] == 2): # Flip it for the 3-1 edge adjacent objects to match CGNS
        for ibary in range(1, order): # note not looping backwards
            point = np.zeros(nbary)
            point[free_indices[0]] = ibary
            point[free_indices[1]] = order - ibary
            print(point)
            nodes.append(point)
    else:
        for ibary in reversed(range(1, order)):
            point = np.zeros(nbary)
            point[free_indices[0]] = ibary
            point[free_indices[1]] = order - ibary
            print(point)
            nodes.append(point)

# General: Face nodes, Volume nodes, ...
def gen_vertices(nfree, free_indices, order_sum):
    points = []
    if(nfree == 2): #base case the two indices sum up to order_sum
        for ibary in reversed(range(1, order_sum)):
            point_base = np.zeros(nbary)
            point = point_base.copy()
            point[free_indices[0]] = ibary
            point[free_indices[1]] = order_sum - ibary
            points.append(point)
    else:
        # whats the highest you can get in the index 
        # while putting ones in all the other free indices
        max_bary = order_sum - nfree + 1 
        for ibary in range(1, max_bary+1): # 1 to max_bary inclusive
            pts_recursive = gen_vertices(nfree-1, free_indices[:(nfree-1)], order_sum - ibary)
            for pt in pts_recursive:
                pt[free_indices[nfree-1]] = ibary
                points.append(pt.copy())
    return points

# flipped version for CGNS shenanigans
def gen_vertices2(nfree, free_indices, order_sum): 
    points = []
    if(nfree == 2): #base case the two indices sum up to order_sum
        for ibary in range(1, order_sum):
            point_base = np.zeros(nbary)
            point = point_base.copy()
            point[free_indices[0]] = ibary
            point[free_indices[1]] = order_sum - ibary
            points.append(point)
    else:
        # whats the highest you can get in the index 
        # while putting ones in all the other free indices
        max_bary = order_sum - nfree + 1 
        for ibary in range(1, max_bary+1): # 1 to max_bary inclusive
            pts_recursive = gen_vertices2(nfree-1, free_indices[:(nfree-1)], order_sum - ibary)
            for pt in pts_recursive:
                pt[free_indices[nfree-1]] = ibary
                points.append(pt.copy())
    return points
            
        

for nfree in range(3, nbary + 1):
    free_index_set = gen_free_index_set(nfree, ndim)

    for free_indices in free_index_set:

        if(free_indices[0] == 0 and free_indices[1] == 2): # Flip it for the 3-1 edge adjacent objects to match CGNS
            print(gen_vertices2(nfree, free_indices, order))
            for node in gen_vertices2(nfree, free_indices, order):
                nodes.append(node)
        else:
            print(gen_vertices(nfree, free_indices, order))
            for node in gen_vertices(nfree, free_indices, order):
                nodes.append(node)

# print out the nodes!
i = 1
for node in nodes:
    print(i, ". ", node);
    i = i + 1
