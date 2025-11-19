#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pointcloud_random_walker_seg.h>


namespace py = pybind11;


PYBIND11_MODULE(_pc_rwalker, m) {
    m.def("random_walker_segmentation",
        &pc_rwalker::randomWalkerSegmentation, 
        "Random walker segmentation of point clouds",
        py::arg("xyz"),
        py::arg("seed_indices"),
        py::arg("n_neighbors"),
        py::arg("sigma1") = 1.0,
        py::arg("sigma2") = 1.0, 
        py::arg("min_weight") = 0.0001,
        py::arg("n_proc") = -1
    );
}
