import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.colors as mcolors

ddg_standard = np.loadtxt("ddg_standard.dat", skiprows=1)
ddgic = np.loadtxt("ddgic.dat", skiprows=1)
ip = np.loadtxt("ip.dat", skiprows=1)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('ndof')
plt.ylabel('L2 error norm')
plt.grid()

color_map = [
    mcolors.TABLEAU_COLORS["tab:blue"],
    mcolors.TABLEAU_COLORS["tab:orange"],
    mcolors.TABLEAU_COLORS["tab:green"],
    mcolors.TABLEAU_COLORS["tab:red"],
    mcolors.TABLEAU_COLORS["tab:purple"],
]

plot_lines = []
for order in range(1, 5):
    ndof = (order + 1) * ddg_standard[:, 0]
    l1, = plt.plot(ndof, ddg_standard[:, order], color=color_map[order-1], ls='-')
    l2, = plt.plot(ndof, ddgic[:, order], color=color_map[order-1], ls='--')

    # ip has less data 
    ndof = (order + 1) * ip[:, 0]
    l3, = plt.plot(ndof, ip[:, order], color=color_map[order-1], ls='-.')

    plot_lines.append([l1, l2, l3])

legend1 = plt.legend(plot_lines[0], ["ddg-standard", "ddgic", "interior penalty"], loc=1)
plt.legend([l[0] for l in plot_lines], ["DGP1", "DGP2", "DGP3", "DGP4"], loc=3)
plt.gca().add_artist(legend1)
plt.show()
