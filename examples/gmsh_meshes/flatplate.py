# =============================================================================
# = Flatplate Mesh Generation 
# = Author: Gianni Absillis   
#
#
# 
# =============================================================================

import gmsh
import math
import sys
import numpy as np

# length of slip wall before flat plate
l_slip = 0.5

# length of flatplate
l_plate = 1.0

# height of the domain
height = 0.25

# mesh size parameters
lc_freestream = 0.080
lc_wall = 0.050
lc_plate_start = 0.050


# setup the gmsh model
gmsh.initialize()
gmsh.model.add("flatplate")

# =========================
# = make the bounding box =
# =========================
#
#  5               4
#  +---------------+
#  |              |
#  +-----+--------+
#  1     2        3

bp1 = gmsh.model.geo.addPoint(-l_slip, 0, 0, lc_wall)
bp2 = gmsh.model.geo.addPoint(0, 0, 0, lc_plate_start)
bp3 = gmsh.model.geo.addPoint(l_plate, 0, 0, lc_wall)
bp4 = gmsh.model.geo.addPoint(l_plate, height, 0, lc_freestream)
bp5 = gmsh.model.geo.addPoint(-l_slip, height, 0, lc_freestream)

bl1 = gmsh.model.geo.addLine(bp1, bp2);
bl2 = gmsh.model.geo.addLine(bp2, bp3);
bl3 = gmsh.model.geo.addLine(bp3, bp4);
bl4 = gmsh.model.geo.addLine(bp4, bp5);
bl5 = gmsh.model.geo.addLine(bp5, bp1);
bbox_loop = gmsh.model.geo.addCurveLoop([bl1, bl2,  bl3, bl4, bl5])
domain = gmsh.model.geo.addPlaneSurface([bbox_loop])

# =======================
# = Boundary Conditions =
# =======================
gmsh.model.addPhysicalGroup(1, [bl1], 1) # slip_wall
gmsh.model.addPhysicalGroup(1, [bl2], 2) # no-slip wall 
gmsh.model.addPhysicalGroup(1, [bl3, bl4, bl5], 3) # freestream
gmsh.model.addPhysicalGroup(2, [domain], 4)

gmsh.model.geo.synchronize()
#gmsh.option.setNumber("Mesh.Algorithm", 5)  # delunay
gmsh.option.setNumber("Mesh.Algorithm", 11) # quads 
gmsh.model.mesh.generate(2)
gmsh.write("flatplate.msh");

gmsh.fltk.run()

gmsh.finalize()
