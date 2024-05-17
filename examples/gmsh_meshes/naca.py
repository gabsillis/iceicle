import gmsh
import math
import sys
import numpy as np 
import matplotlib.pyplot as plt

# parameters 

# number of points on one of the airfoil surfaces
npoints = 300

# naca airfoil specification
naca = [0, 0, 1, 2]

# maximum camber
m = naca[0] / 100.0;

# position of maximum camber 
p = naca[1] / 10.0 

# maximum thickness in percentage of chord
t = naca[2] / 10.0 + naca[3] / 100.0

# mesh size parameter
lc_airfoil = 0.02
lc_freestream = 5.0

# ====================
# = make the airfoil =
# ====================

x = np.linspace(0.0, 1.0, npoints)

yc = np.piecewise(x, [x < p, x >=p], [lambda x: m / p**2 * (2 * p * x - np.power(x, 2)), lambda x: m / (1 - p)**2 * ((1 - 2 * p) + 2*p*x - np.power(x, 2))])
dycdx = np.piecewise(x, [x < p, x >= p], [lambda x: 2 * m / p**2 * (p - x), lambda x: 2 * m / (1 - p)**2 * (p - x)])
theta = np.arctan(dycdx)
yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * np.power(x, 2) + 0.2843 * np.power(x, 3) - 0.1036 * np.power(x, 4))

xu = x - yt * np.sin(theta)
yu = yc + yt * np.cos(theta)

xl = x + yt * np.sin(theta)
yl = yc - yt * np.cos(theta)

# plot the airfoil
fig, ax = plt.subplots()
ax.plot(xu, yu, linewidth=2.0)
ax.plot(xl, yl, linewidth=2.0)
#plt.show()

# setup the gmsh model
gmsh.initialize()
gmsh.model.add("naca")

# make the endpoints
gmsh.model.geo.addPoint(1, 0, 0, lc_airfoil, 1)
# make the upper surface in reverse
for i in range(1, npoints):
    gmsh.model.geo.addPoint(xu[i], yu[i], 0, lc_airfoil, npoints - i + 1);
gmsh.model.geo.addPoint(0, 0, 0, lc_airfoil, npoints + 1)
# make the lower surface
for i in range(1, npoints):
    gmsh.model.geo.addPoint(xl[i], yl[i], 0, lc_airfoil, npoints + i + 1)
airfoil_spline = gmsh.model.geo.addSpline(list(range(1, 2*npoints+1)) + [1])
airfoil_loop = gmsh.model.geo.addCurveLoop([airfoil_spline]);


end_pt_airfoil = 2 * npoints+1

# =========================
# = make the bounding box =
# =========================
chord_multx = 4
thick_multy = 40
bp1 = gmsh.model.geo.addPoint(-chord_multx, -thick_multy*t, 0, lc_freestream)
bp2 = gmsh.model.geo.addPoint(1 + chord_multx, -thick_multy*t, 0, lc_freestream)
bp3 = gmsh.model.geo.addPoint(1 + chord_multx, thick_multy*t, 0, lc_freestream)
bp4 = gmsh.model.geo.addPoint(-chord_multx, thick_multy*t, 0, lc_freestream)
bl1 = gmsh.model.geo.addLine(bp1, bp2);
bl2 = gmsh.model.geo.addLine(bp2, bp3);
bl3 = gmsh.model.geo.addLine(bp3, bp4);
bl4 = gmsh.model.geo.addLine(bp4, bp1);
bbox_loop = gmsh.model.geo.addCurveLoop([bl1, bl2,  bl3, bl4])
domain = gmsh.model.geo.addPlaneSurface([bbox_loop, airfoil_loop])

# =======================
# = Boundary Conditions =
# =======================
gmsh.model.addPhysicalGroup(1, [airfoil_spline], 1)
gmsh.model.addPhysicalGroup(1, [bl1, bl3], 2)
gmsh.model.addPhysicalGroup(1, [bl2], 3)
gmsh.model.addPhysicalGroup(1, [bl4], 4)
gmsh.model.addPhysicalGroup(2, [domain], 4)

gmsh.model.geo.synchronize()
gmsh.option.setNumber("Mesh.Algorithm", 11)  
gmsh.model.mesh.generate(2)
gmsh.write("naca.msh");

gmsh.fltk.run()

gmsh.finalize()


