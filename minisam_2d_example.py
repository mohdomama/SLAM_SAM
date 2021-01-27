'''
This file is almost identical to: https://minisam.readthedocs.io/pose_graph_2d.html
'''

import minisam
import numpy as np
import math 

from minisam import FactorGraph, Variables
from minisam import DiagonalLoss, PriorFactor, BetweenFactor
from minisam import SE2, SO2, key
from minisam import LevenbergMarquardtOptimizer, LevenbergMarquardtOptimizerParams, NonlinearOptimizationStatus
from minisam import MarginalCovarianceSolver, MarginalCovarianceSolverStatus
from minisam import sophus

import matplotlib.pyplot as plt


def setup_graph():
    '''
    Here we will setup our factor graph
    We have 3 factors in this example:
        - Prior Factor: Anchors the inital position
        - Odometry Factor: Between every pose
        - Loop Closure Factor: It is what the name suggests
    '''
    graph = FactorGraph()

    # We will now losses for each factor. This is basically a loss(Mahalanobis distance) 
    # specified by a covariance matrix. 
    priorLoss = DiagonalLoss.Sigmas(np.array([0.1, 0.1, 0.01]))
    odomLoss = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))
    loopLoss = DiagonalLoss.Sigmas(np.array([0.5, 0.5, 0.1]))

    # We will now define factors between nodes

    # Prior Factor
    graph.add(PriorFactor(key('x', 1), SE2(SO2(0), np.array([0, 0])), priorLoss))

    # Odom Factor
    graph.add(BetweenFactor(key('x', 1), key('x', 2), SE2(SO2(0), np.array([5, 0])), odomLoss))
    graph.add(BetweenFactor(key('x', 2), key('x', 3), SE2(SO2(-1.57), np.array([5, 0])), odomLoss))
    graph.add(BetweenFactor(key('x', 3), key('x', 4), SE2(SO2(-1.57), np.array([5, 0])), odomLoss))
    graph.add(BetweenFactor(key('x', 4), key('x', 5), SE2(SO2(-1.57), np.array([5, 0])), odomLoss))

    # Loop Closure Factor
    graph.add(BetweenFactor(key('x', 5), key('x', 2), SE2(SO2(-1.57), np.array([5, 0])), loopLoss))

    return graph

    
def initialise_variables():
    '''
    We will give an initial value to our pose variables
    '''

    initials = Variables()
    initials.add(key('x', 1), SE2(SO2(0.2), np.array([0.2, -0.3])))
    initials.add(key('x', 2), SE2(SO2(-0.1), np.array([5.1, 0.3])))
    initials.add(key('x', 3), SE2(SO2(-1.57 - 0.2), np.array([9.9, -0.1])))
    initials.add(key('x', 4), SE2(SO2(-3.14 + 0.1), np.array([10.2, -5.0])))
    initials.add(key('x', 5), SE2(SO2(1.57 - 0.1), np.array([5.1, -5.1])))

    return initials


def optimize(graph, initials):
    '''
    Here we apply smoothing on the entire pose graph
    We also calculate covariance of poses
    '''

    # Setting Up the Optimizer
    opt_param = LevenbergMarquardtOptimizerParams()
    opt = LevenbergMarquardtOptimizer(opt_param)

    # Placeholder for output
    results = Variables()

    status = opt.optimize(graph, initials, results)

    if status != NonlinearOptimizationStatus.SUCCESS:
        print("optimization error: ", status)


    # Calculating Covariances
    mcov_solver = MarginalCovarianceSolver()

    status = mcov_solver.initialize(graph, results)
    if status != MarginalCovarianceSolverStatus.SUCCESS:
        print("maginal covariance error", status)

    # Example usage
    # cov1 = mcov_solver.marginalCovariance(key('x', 1))

    return results, mcov_solver


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


def visualise(results, mcov_solver, nvert):
    fig, ax = plt.subplots()
    

    for i in range(nvert):
        pose = results.at(key('x', i+1))
        cov = mcov_solver.marginalCovariance(key('x', i+1))
        plotSE2WithCov(pose, cov)

    plt.axis('equal')
    plt.show()


def main():
    graph = setup_graph()
    initials = initialise_variables()

    results, mcov_solver = optimize(graph, initials)

    print(results[key('x', 1)].translation())
    print(results[key('x', 1)].so2())

    visualise(results, mcov_solver, 5)


if __name__=='__main__':
    main()