import matplotlib.pyplot as plt
import matplotlib
import csv

file = open("SimplexTransformTest_ReferenceNodes.dat")
x = []
y = []
for line in file:
    coords = line.split()
    x.append(float(coords[0]))
    y.append(float(coords[1]))

file2 = open("SimplexTransformTest_DiscretizedVis.dat");
x2 = []
y2 = []
for line in file2:
    coords = line.split()
    print(coords)
    x2.append(float(coords[0]))
    y2.append(float(coords[1]))

fig, axs = plt.subplots(2)
axs[0].scatter(x, y)
for i in range(0, len(x)):
    axs[0].annotate(i, (x[i], y[i]))

axs[1].scatter(x2, y2)

plt.show()
