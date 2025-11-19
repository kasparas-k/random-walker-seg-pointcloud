#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>	
#include <omp.h>

#include "nanoflann.hpp"


namespace pc_rwalker {

using Points3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;

namespace detail {
    using SparseMatrixd = Eigen::SparseMatrix<double>;
    using Edges = std::vector<std::tuple<int, int>>;
    using KdTree = nanoflann::KDTreeEigenMatrixAdaptor<Points3d, 3, nanoflann::metric_L2>;

    const double EPS = 1e-12;

    inline std::tuple<Edges, Eigen::VectorXd> computeEdgesAndWeights(
        const Points3d &xyz,
        const int n_neighbors,
        const int n_threads,
        const double sigma1 = 1.0, // smoothing of distance weight
        const double sigma2 = 1.0, // smoothing of normal weight
        const double min_weight = 0.0001
    ) {
        
        int n_vertices = xyz.rows();
        int n_edges = n_neighbors * n_vertices;
        Edges edges(n_edges);
        Eigen::MatrixXd sq_distsances(n_vertices, n_neighbors);
        Eigen::MatrixX3d normals(n_vertices, 3);

        { // kdtree scope
            std::vector<int> num_didnt_find_self(n_threads);

            KdTree kdtree(3 /*dim*/, std::cref(xyz), 10 /* max leaf */);
            #pragma omp parallel for schedule(dynamic) num_threads(n_threads) if(n_threads > 1)
            for (int point_idx = 0; point_idx < n_vertices; ++point_idx) {
                Eigen::Vector3d pt = xyz.row(point_idx);
                
                // will return n_neighbors + query point
                // need to check matches to remove the query point later
                std::vector<long> neighbor_idxs(n_neighbors + 1);
                std::vector<double> neighbor_sq_dists(n_neighbors + 1);

                std::vector<double> search_pt({pt[0], pt[1], pt[2]});
                kdtree.index_->knnSearch(pt.data(), n_neighbors + 1, neighbor_idxs.data(), neighbor_sq_dists.data());
                
                bool found_self = false;

                // Estimate normals from the neighborhood's covariance matrix
                Points3d neighborhood(n_neighbors + 1, 3);
                int nonquery_point_idx = 0;
                for (int local_neighbor_idx = 0; local_neighbor_idx < neighbor_idxs.size(); ++local_neighbor_idx) {
                    int global_neighbor_idx = neighbor_idxs[local_neighbor_idx];
                    neighborhood.row(local_neighbor_idx) = xyz.row(global_neighbor_idx);

                    // While we're at it, also build the graph
                    if (global_neighbor_idx == point_idx) {
                        found_self = true;
                    } else if (nonquery_point_idx < n_neighbors) {
                        sq_distsances(point_idx, nonquery_point_idx) = neighbor_sq_dists[local_neighbor_idx];
                        edges[point_idx * n_neighbors + nonquery_point_idx] = {point_idx, global_neighbor_idx};
                        ++nonquery_point_idx;
                    }
                }
                neighborhood = neighborhood.rowwise() - neighborhood.colwise().mean();
                Eigen::Matrix3d cov = (neighborhood.transpose() * neighborhood) / static_cast<double>(neighbor_idxs.size());
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(cov, Eigen::ComputeEigenvectors);

                Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
                int argmin;
                eigenvalues.minCoeff(&argmin);               

                Eigen::Vector3d normal = eigensolver.eigenvectors().col(argmin);
                normal = normal / (normal.norm() + EPS);
                normals.row(point_idx) = normal;

                if (!found_self) {
                    ++num_didnt_find_self[omp_get_thread_num()];
                }
            }

            int total_didnt_find_self = 0;
            for (int n : num_didnt_find_self) {
                total_didnt_find_self += n;
            }
            if (total_didnt_find_self > 0) {
                std::cout << total_didnt_find_self << " points were not in their own " << n_neighbors
                << " nearest neighbors. The given point cloud may have excessive duplicate points,"
                " leading to a singular Laplacian. Remove duplicate points of increase number of neighbors."
                << std::endl;
            } 
        } // exit kdtree scope

        // Lai et al. (2009), Eqn. (12)
        Eigen::VectorXd dist_weights(n_edges);
        Eigen::VectorXd normal_weights(n_edges);

        Eigen::VectorXd average_sq_dists = sq_distsances.rowwise().mean();
        #pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
        for (int edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
            const auto [i, j] = edges[edge_idx];
            Eigen::Vector3d pt_i, pt_j, norm_i, norm_j;
            pt_i = xyz.row(i);
            pt_j = xyz.row(j);
            norm_i = normals.row(i);
            norm_j = normals.row(j);
            
            double dist_weight = std::exp(
                -sq_distsances(i, edge_idx % n_neighbors) / (average_sq_dists[i] * sigma1 + EPS)
            );
            dist_weights[edge_idx] = dist_weight;
            
            // Lei et al. (2009), sec. 3.1.1 -- the norm_dist_scale is set to 1.0 if the
            // edge is convex, and set to 0.2 if the edge is not convex.
            // Convexity is checked with a heuristic defined in eqn. (11) 
            bool is_convex = (
                (pt_j - pt_i) - (pt_j - pt_i).dot(norm_i) * norm_i
            ).dot(norm_j) > 0;
            double norm_dist_scale = is_convex ? 1.0 : 0.2; 
            double normal_dist = norm_dist_scale / 2.0 * (norm_i - norm_j).squaredNorm();

            normal_weights[edge_idx] = normal_dist;
        }
        
        // free memory
        normals.resize(0, 3);
        sq_distsances.resize(0, 0);
        average_sq_dists.resize(0, 1);

        // normal_weights is normal distances at the start of calculation
        normal_weights = (-normal_weights / (sigma2 * normal_weights.mean() + EPS)).array().exp();
        
        Eigen::VectorXd weights = dist_weights.array() * normal_weights.array();
        if (min_weight > 0.0) {
            weights = weights.cwiseMax(min_weight);
        }

        // free memory
        dist_weights.resize(0, 1);
        normal_weights.resize(0, 1);

        // Normalize so all the weights of a vertex add to 1, as per Lai et al. (2009)
        // NOTE: this means that the graph is directed, since w_ij != w_ji
        // although this condition was broken even before the nrmalization due to 
        // local neighborhood-dependent division of the distance weight
        for (int vertex_idx = 0; vertex_idx < n_vertices; ++vertex_idx) {
            weights.block(vertex_idx * n_neighbors, 0, n_neighbors, 1) /= 
                weights.block(vertex_idx * n_neighbors, 0, n_neighbors, 1).sum() + EPS;
        }
        
        return {edges, weights};
    }

    inline SparseMatrixd buildConstitutiveMatrix(const Eigen::VectorXd &weights) {
        SparseMatrixd C(weights.rows(), weights.rows());
        
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(weights.rows());

        for (int edge_idx = 0; edge_idx < weights.rows(); ++edge_idx) {
            triplets.emplace_back(edge_idx, edge_idx,  weights[edge_idx]);
        }

        C.setFromTriplets(triplets.begin(), triplets.end());
        return C;
    }

    // Grady (2006), Eqn. (5)
    inline SparseMatrixd buildIncidenceMatrix(const Edges &edges, const int n_vertices) {
        SparseMatrixd A(edges.size(), n_vertices);
        
        std::vector<Eigen::Triplet<double>> triplets;
        
        // two non-zero elements per edge
        triplets.reserve(edges.size() * 2);

        for (int edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
            int i = std::get<0>(edges[edge_idx]);
            int j = std::get<1>(edges[edge_idx]);
            triplets.emplace_back(edge_idx, i,  1.0);
            triplets.emplace_back(edge_idx, j, -1.0);
        }

        A.setFromTriplets(triplets.begin(), triplets.end());

        return A;
    }

    // Grady (2006), Eqn. (9)
    inline SparseMatrixd buildLabelMatrix(const std::vector<std::vector<int>> &seed_idxs, const int n_marked) {
        SparseMatrixd M(n_marked, seed_idxs.size());

        int vertex_idx = 0;
        for (int class_idx = 0; class_idx < seed_idxs.size(); ++class_idx) {
            for (int _ : seed_idxs[class_idx]) {
                M.coeffRef(vertex_idx, class_idx) = 1.0;
                vertex_idx++;
            }
        }
        return M;
    }
    
    // Decompose the Laplacian like in Grady (2006), Eqn. (7)
    inline std::tuple<SparseMatrixd, SparseMatrixd> buildLinearSystem(
        const SparseMatrixd &A, // incidence matrix
        const SparseMatrixd &C, // constitutive matrix
        const int n_marked
    ) {
        int n_unmarked = A.cols() - n_marked;
        auto BT_Lu = A.transpose().bottomRows(n_unmarked) * C * A;
        return {BT_Lu.leftCols(n_marked), BT_Lu.rightCols(n_unmarked)};
    }
} // namespace detail

inline std::vector<std::vector<int>> randomWalkerSegmentation(
    const Points3d &xyz,
    const std::vector<std::vector<int>> &seed_indices,
    const int n_neighbors,
    const double sigma1 = 1.0,        // smoothing of distance weight
    const double sigma2 = 1.0,        // smoothing of normal weight
    const double min_weight = 0.0001, // to avoid singular Laplacians
    const int n_proc = -1             // -1 for all processors, 0 or 1 for no OMP
) {
    const int n_threads = (n_proc < 0) ? omp_get_max_threads()
        : std::max(1, std::min(omp_get_max_threads(), n_proc));
    
    Points3d xyz_marked_first(xyz.rows(), xyz.cols());
    std::vector<int> original_to_marked_first(xyz.rows(), -1);
    int remapped_idx = 0;
    for (std::vector<int> idxs_for_seed : seed_indices) {
        for (int marked_idx : idxs_for_seed) {
            xyz_marked_first.row(remapped_idx) = xyz.row(marked_idx);
            original_to_marked_first[marked_idx] = remapped_idx;
            remapped_idx++;
        }
    }
    int n_marked = remapped_idx;

    // for all remaining unmarked points, insert them sequentially
    for (int original_idx = 0; original_idx < xyz.rows(); ++original_idx) {
        if (original_to_marked_first[original_idx] < 0) {
            xyz_marked_first.row(remapped_idx) = xyz.row(original_idx);
            original_to_marked_first[original_idx] = remapped_idx;
            remapped_idx++;
        }
    }

    auto [edges, weights] = detail::computeEdgesAndWeights(
        xyz_marked_first, n_neighbors, n_threads, sigma1, sigma2, min_weight
    );

    // Constitutive matrix is a diagonal matrix with edge weights on the diagonal
    detail::SparseMatrixd C = detail::buildConstitutiveMatrix(weights);
    detail::SparseMatrixd A = detail::buildIncidenceMatrix(edges, xyz_marked_first.rows());
    detail::SparseMatrixd M = detail::buildLabelMatrix(seed_indices, n_marked);

    auto [BT, Lu] = detail::buildLinearSystem(A, C, n_marked);

    // solve equation
    detail::SparseMatrixd RHS = -(BT * M);
    BT.resize(0, 0);
    M.resize(0, 0);
    Eigen::SimplicialLDLT<detail::SparseMatrixd> solver;
    solver.compute(Lu);

    std::vector<std::vector<int>> final_segments = seed_indices;
    if (solver.info() != Eigen::Success) {
        std::cout << "Failed to invert the graph Laplacian." << std::endl;
        return final_segments;
    }

    Eigen::MatrixXd X = solver.solve(RHS);

    std::vector<int> argmax_x(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        X.row(i).maxCoeff(&argmax_x[i]);
    } 

    for (int global_point_idx = 0; global_point_idx < xyz.rows(); ++global_point_idx) {
        int idx_x = original_to_marked_first[global_point_idx];
        if (idx_x < n_marked) continue;
        
        idx_x -= n_marked;
        int class_idx = argmax_x[idx_x];
        
        final_segments[class_idx].push_back(global_point_idx);        
    }

    return final_segments;
}

} // namespace pc_rwalker
