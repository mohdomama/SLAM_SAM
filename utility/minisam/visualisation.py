import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math

from minisam import *

# plot SE2 with covariance
def plotSE2WithCov(pose, cov, vehicle_size=0.5, line_color='k', vehicle_color='r'):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size
    line = plt.Polygon([p1, p2, p3], closed=True, fill=True, edgecolor=line_color, facecolor=vehicle_color)
    plt.gca().add_line(line)
    # plot cov
    ps = []
    circle_count = 50
    for i in range(circle_count):
        t = float(i) / float(circle_count) * math.pi * 2.0
        cp = pose.translation() + np.matmul(cov[0:2, 0:2], np.array([math.cos(t), math.sin(t)]))
        ps.append(cp)
    line = plt.Polygon(ps, closed=True, fill=False, edgecolor=line_color)
    plt.gca().add_line(line)


# plot 2D pose graph
def plot2DPoseGraphResult(ax, graph, variables, color, linewidth=1):
    lines = []
    for factor in graph:
        # only plot between factor
        if factor.__class__.__name__ == BetweenFactor_SE2_.__name__:
            keys = factor.keys()
            p1 = variables.at_SE2_(keys[0]).translation()
            p2 = variables.at_SE2_(keys[1]).translation()
            lines.append([p1, p2])
    lc = matplotlib.collections.LineCollection(lines, colors=color, linewidths=linewidth)
    ax.add_collection(lc)
    plt.axis('equal')