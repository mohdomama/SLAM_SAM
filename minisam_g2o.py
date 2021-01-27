'''
Minisam provides a function to load g2o data 
This makes optimising g2o data very simple
Also, this example is from the minisam github examples folder
'''

import sys
import numpy as np
from matplotlib import pyplot as plt

import minisam as sam
from utility.minisam.visualisation import plot2DPoseGraphResult


def get_g2o_data(filename):
    graph = sam.FactorGraph()
    initials = sam.Variables()

    _ = sam.loadG2O(filename, graph, initials)

    return graph, initials


def optimize(graph, initials):
    """
    Choose an solver from
    CHOLESKY,              // Eigen Direct LDLt factorization
    CHOLMOD,               // SuiteSparse CHOLMOD
    QR,                    // SuiteSparse SPQR
    CG,                    // Eigen Classical Conjugate Gradient Method
    CUDA_CHOLESKY,         // cuSolverSP Cholesky factorization
    """

    # optimize by GN
    opt_param = sam.GaussNewtonOptimizerParams()
    opt_param.max_iterations = 1000
    opt_param.min_rel_err_decrease = 1e-10
    opt_param.min_abs_err_decrease = 1e-10
    opt_param.linear_solver_type = sam.LinearSolverType.CHOLMOD
    # opt_param.verbosity_level = sam.NonlinearOptimizerVerbosityLevel.SUBITERATION
    print(opt_param)
    opt = sam.GaussNewtonOptimizer(opt_param)

    all_timer = sam.global_timer().getTimer("Pose graph all")
    all_timer.tic()

    results = sam.Variables()
    status = opt.optimize(graph, initials, results)

    all_timer.toc()

    if status != sam.NonlinearOptimizationStatus.SUCCESS:
        print("optimization error: ", status)

    sam.global_timer().print()

    return results


def visualise(graph, initials, results):
    fig, ax = plt.subplots()
    plot2DPoseGraphResult(ax, graph, initials, 'r', linewidth=1)
    plot2DPoseGraphResult(ax, graph, results, 'b', linewidth=1)
    ax.set_title('Pose graph, blue is optimized and red is non-optimized')
    plt.show()


def main():
    filename = sys.argv[1]
    graph, initials = get_g2o_data(filename)

    # add a prior factor to first pose to fix the whole system
    priorloss = sam.ScaleLoss.Scale(1) # value will be broadcasted to the dimension of pose. I guess. 
    graph.add(sam.PriorFactor(sam.key('x', 0), initials.at(sam.key('x', 0)), priorloss))

    results = optimize(graph, initials)

    # print(graph)
    visualise(graph, initials, results)


if __name__=='__main__':
    main()