#include "delaunator-cpp/include/delaunator.hpp"
#include "happly.h"
#include "kernels.hpp"
#include "neighbor-search-engine.hpp"
#include "particle.hpp"
#include "particles-alembic-manager.hpp"
#include "types.hpp"
#include <filesystem>
#include <iostream>
#include <parallel-util.hpp>
#include <timer.hpp>
#include <vector>

constexpr auto calcKernel      = calcCubicSplineKernel;
constexpr auto calcGradKernel  = calcGradCubicSplineKernel;
constexpr int  scale_          = 10;
int            frame_id        = 0;
int            total_particles = 25600;

Scalar e_rho = 600;
Scalar e_d   = 40;
Scalar e_a   = 0.2;

using NeighborSearchEngine = HashGridNeighborSearchEngine;

// Scalar radius    = 0.05 * 0.25; // of particle
Scalar particle_radius = 0.05 * 1.00;
Scalar kernel_radius   = 6 * particle_radius;  // of kernel
Scalar screen_radius   = 12 * particle_radius; // of kernel
Scalar rest_density    = 1000;

inline Vec3 crossProduct(const Vec3& a, const Vec3& b)
{
    Scalar u1 = a.x(), u2 = a.y(), u3 = a.z();
    Scalar v1 = b.x(), v2 = b.y(), v3 = b.z();
    return Vec3(u2 * v3 - u3 * v2, u3 * v1 - u1 * v3, u1 * v2 - u2 * v1);
}

Scalar calcDensity(const int                    target_index,
                   const std::vector<Particle>& particles,
                   const NeighborSearchEngine&  neighbor_search_engine,
                   const Scalar                 radius);
Scalar calcDensityConstraint(const int                    target_index,
                             const std::vector<Particle>& particles,
                             const NeighborSearchEngine&  neighbor_search_engine,
                             const Scalar                 rest_density,
                             const Scalar                 radius);
Vec3   calcGradConstraint(const int                    target_index,
                          const int                    var_index,
                          const std::vector<Particle>& particles,
                          const NeighborSearchEngine&  neighbor_search_engine,
                          const Scalar                 rest_density,
                          const Scalar                 radius);
void   printAverageNumNeighbors(const NeighborSearchEngine& neighbor_search_engine);
Scalar printAverageDensity(const std::vector<Particle>& particles,
                           const NeighborSearchEngine&  neighbor_search_engine,
                           const Scalar                 radius);
void   step(const Scalar dt, std::vector<Particle>& particles, int num_iters = 100);
void   saveIntermediateResult(const std::vector<Particle>& particles,
                              const std::vector<int>&      sp_ids_array,
                              const std::vector<int>&      sp_triangles);

int main()
{
    std::vector<Particle> particles{};
    std::vector<Particle> save_particles{};

    constexpr Scalar dt         = 1.0 / 30;
    constexpr int    num_frames = 100;
    constexpr Scalar pi         = 3.1415926;

    // Generate and initialize particles
    particles.clear();

    // Instantiate an alembic manager and submit the initial status
    save_particles.resize(320 * num_frames);
    ParticlesAlembicManager alembic_manager("./tap2_V50_stf30.abc", 1 / 24.0, "Fluid", &save_particles);

    // Write the current status
    alembic_manager.submitCurrentStatus();

    // Simulate particles
    constexpr int    num_substeps = 1;
    constexpr Scalar sub_dt       = dt / static_cast<Scalar>(num_substeps);
    for (frame_id = 0; frame_id < num_frames; ++frame_id) {
        // Instantiate timer
        const auto timer = timer::Timer("Frame #" + std::to_string(frame_id));

        int cur_particles = 0;
        do {
            Scalar scale         = 1 / 2.7144;
            Vec3   sampled_point = Vec3::Random();
            if (sampled_point.squaredNorm() - sampled_point.z() * sampled_point.z() > scale * scale)
                continue;
            Particle p;
            p.i = particles.size();
            p.m = 1.0;
            p.x = sampled_point * scale + Vec3(0.0, 0.0, 1.25);
            p.p = p.x;
            p.v = 0 * dt * Vec3(0.0, -9.8, 0.0);
            particles.push_back(p);
            cur_particles += 1;
        } while (cur_particles < 320);

        // Step the simulation time
        for (int k = 0; k < num_substeps; ++k) {
            step(sub_dt, particles, 100);
        }

        parallelutil::parallel_for(particles.size(), [&](const int i) {
            save_particles[i] = particles[i];
        });

        // Write the current status
        alembic_manager.submitCurrentStatus();
    }

    return 0;
}

Scalar calcDensity(const int                    target_index,
                   const std::vector<Particle>& particles,
                   const NeighborSearchEngine&  neighbor_search_engine,
                   const Scalar                 radius)
{
    const auto& p_target = particles[target_index];

    Scalar density = 0.0;
    for (int neighbor_index : neighbor_search_engine.retrieveNeighbors(target_index)) {
        const auto& p = particles[neighbor_index];

        density += p.m * calcKernel(p_target.p - p.p, radius);
    }

    return density;
}

Scalar calcDensityConstraint(const int                    target_index,
                             const std::vector<Particle>& particles,
                             const NeighborSearchEngine&  neighbor_search_engine,
                             const Scalar                 rest_density,
                             const Scalar                 radius)
{
    const Scalar density = calcDensity(target_index, particles, neighbor_search_engine, radius);

    return (density / rest_density) - 1.0;
}

Vec3 calcGradConstraint(const int                    target_index,
                        const int                    var_index,
                        const std::vector<Particle>& particles,
                        const NeighborSearchEngine&  neighbor_search_engine,
                        const Scalar                 rest_density,
                        const Scalar                 radius)
{
    const auto& p_target = particles[target_index];

    if (target_index == var_index) {
        Vec3 sum = Vec3::Zero();
        for (int neighbor_index : neighbor_search_engine.retrieveNeighbors(target_index)) {
            const auto& p = particles[neighbor_index];

            sum += p.m * calcGradKernel(p_target.p - p.p, radius);
        }

        return sum / rest_density;
    } else {
        const auto& p = particles[var_index];

        return -p.m * calcGradKernel(p_target.p - p.p, radius) / rest_density;
    }
}

void printAverageNumNeighbors(const NeighborSearchEngine& neighbor_search_engine)
{
    const int num_particles = neighbor_search_engine.getNumParticles();

    VecX nums(num_particles);
    for (int i = 0; i < num_particles; ++i) {
        nums[i] = neighbor_search_engine.retrieveNeighbors(i).size();
    }
    std::cout << "Average(#neighbors): " << nums.mean() << std::endl;
}

Scalar printAverageDensity(const std::vector<Particle>& particles,
                           const NeighborSearchEngine&  neighbor_search_engine,
                           const Scalar                 radius)
{
    VecX buffer(particles.size());
    for (int i = 0; i < particles.size(); ++i) {
        buffer[i] = calcDensity(i, particles, neighbor_search_engine, radius);
    }
    std::cout << "Average(density): " << buffer.mean() << std::endl;
    return buffer.mean();
}

void step(const Scalar dt, std::vector<Particle>& particles, int num_iters)
{
    constexpr int    num_reset_iters = 10;
    constexpr Scalar damping         = 1.0;
    constexpr Scalar viscosity_coeff = 0.25;
    constexpr bool   verbose         = false;

    const int num_particles = particles.size();
    std::cout << num_particles << std::endl << std::endl;

    // Predict positions using the semi-implicit Euler integration
    for (int i = 0; i < num_particles; ++i) {
        particles[i].v = particles[i].v + dt * Vec3(0.0, 0.0, -9.8);
        particles[i].p = particles[i].x + dt * particles[i].v;
    }

    // Prepare a neighborhood search engine
    NeighborSearchEngine neighbor_search_engine(kernel_radius, particles);

    // Find neighborhoods of every particle
    neighbor_search_engine.searchNeighbors();

    if constexpr (verbose) {
        printAverageNumNeighbors(neighbor_search_engine);
        printAverageDensity(particles, neighbor_search_engine, kernel_radius);
    }

    std::vector<int>              sp_ids_array{}; // surface point
    std::vector<std::vector<int>> sp_neighboring_triangles{};
    std::vector<int>              sp_triangles{};
    NeighborSearchEngine          screen_search_engine(screen_radius, particles);

    for (int k = 0; k < num_iters; ++k) {
        if (k % num_reset_iters == 0 && frame_id >= 0) {
            std::cout << "rebuild begin " << k << std::endl;
            // ============================================
            // detect surface particles
            // ============================================
            constexpr Scalar    pi      = 3.14159265358979323;
            constexpr Scalar    rad2deg = 180.0 / pi;
            std::vector<Scalar> rillu(num_particles);
            screen_search_engine.searchNeighbors();

            sp_ids_array.clear();
            sp_neighboring_triangles.clear();
            sp_triangles.clear();

            std::cout << "detect surface particles" << std::endl;
            std::vector<int> n_nei(num_particles);
            parallelutil::parallel_for(num_particles, [&](const int i) {
                MatX fk3x3_pass(3 * 180 / scale_, 3 * 360 / scale_);
                fk3x3_pass.setConstant(1.0);

                auto& p_i = particles[i];
                p_i.n     = Vec3::Zero();
                n_nei[i]  = screen_search_engine.retrieveNeighbors(i).size();
                for (int neighbor_index : screen_search_engine.retrieveNeighbors(i)) {
                    const auto& p_j  = particles[neighbor_index];
                    const auto  diff = p_j.p - p_i.p;
                    if (diff.norm() <= particle_radius)
                        continue;
                    const Scalar theta =
                        rad2deg * std::atan(diff.y() / std::sqrt(diff.x() * diff.x() + diff.z() * diff.z()));
                    const Scalar phi = rad2deg * std::atan2(diff.x(), diff.z());
                    const Scalar dtheta =
                        rad2deg *
                        std::atan(particle_radius / std::sqrt(diff.squaredNorm() - particle_radius * particle_radius));
                    const Scalar dphi = dtheta;
                    int          tmin = (theta - dtheta + 270) / scale_;
                    int          tmax = (theta + dtheta + 270) / scale_;
                    int          pmin = (phi - dphi + 540) / scale_;
                    int          pmax = (phi + dphi + 540) / scale_;
                    fk3x3_pass.block(tmin, pmin, tmax - tmin, pmax - pmin).setConstant(0.0);
                }

                auto rows = 180 / scale_, cols = 360 / scale_;
                MatX fk_pass(180 / scale_, 360 / scale_);
                fk_pass.setConstant(0.0);
                for (int a : {0, 1, 2})
                    for (int b : {0, 1, 2}) {
                        auto block = fk3x3_pass.block(a * rows, b * cols, rows, cols);
                        if (a != 1) {
                            auto block2 = block.colwise().reverse();
                            fk_pass += block2;
                        } else {
                            fk_pass += block;
                        }
                    }
                fk_pass /= 9.0;
                int illu_count = (fk_pass.array() > 0.95).count();

                rillu[i] = illu_count / (180.0 * 360.0 / scale_ / scale_);
                if (rillu[i] > 0.22)
                    p_i.s = true;
                else
                    p_i.s = false;
            });

            for (int i = 0; i < num_particles; i++) {
                if (particles[i].s)
                    sp_ids_array.push_back(i);
            }

            // ============================================
            // calculate normal of surface points
            // ============================================
            std::cout << "calculate densities" << num_particles << std::endl;
            VecX densities(num_particles);
            parallelutil::parallel_for(num_particles, [&](const int i) {
                densities[i] = calcDensity(i, particles, neighbor_search_engine, kernel_radius);
                if (densities[i] == 0) {
                    std::cout << "zero density!" << std::endl;
                    exit(0);
                }
            });

            std::cout << "calculate normal of surface points" << std::endl;
            parallelutil::parallel_for(sp_ids_array.size(), [&](const int _i) {
                int   i   = sp_ids_array[_i];
                auto& p_i = particles[i];
                Vec3  n   = Vec3::Zero();
                for (int neighbor_index : neighbor_search_engine.retrieveNeighbors(i)) {
                    const auto& p_j = particles[neighbor_index];
                    n += -calcGradKernel(p_i.p - p_j.p, kernel_radius) / densities[neighbor_index];
                }
                n     = n.normalized();
                p_i.n = n;
            });

            // ============================================
            // project to tangent plane and build local mesh
            // ============================================
            std::cout << "[Local Mesh] project and build" << frame_id << std::endl;
            sp_neighboring_triangles.resize(sp_ids_array.size());
            for (int _i = 0; _i < sp_ids_array.size(); _i++) {
                int                i           = sp_ids_array[_i];
                const auto&        p_i         = particles[i];
                std::vector<float> coords      = {0.0, 0.0};
                std::vector<int>   coords_pids = {i};
                Scalar             min_dist_2d = std::numeric_limits<double>::max();
                Vec3               proj_z_axis = p_i.n;
                if (proj_z_axis.squaredNorm() > 0.95) {
                    // build local frame and set +z axis to normal
                    Vec3 proj_x_axis = Vec3::UnitZ(); // initialize (0, 0, 1) as +x axis of local frame
                    // if normal is parallel with current +x axis of local frame,
                    // initialize (0, 1, 0) as +x axis of local frame
                    if ((proj_z_axis - proj_x_axis).squaredNorm() < 1e-3 ||
                        (proj_z_axis + proj_x_axis).squaredNorm() < 1e-3)
                        proj_x_axis = Vec3::UnitY();
                    // make it a valid orthogonal frame
                    auto proj_y_axis = crossProduct(proj_z_axis, proj_x_axis).normalized();
                    proj_x_axis      = crossProduct(proj_y_axis, proj_z_axis).normalized();
                    for (int neighbor_index : neighbor_search_engine.retrieveNeighbors(i)) {
                        const auto& p_j = particles[neighbor_index];
                        if (!p_j.s) // not surface point
                            continue;
                        if (std::acos(p_i.n.dot(p_j.n)) > pi / 4)
                            continue;
                        if (neighbor_index == i)
                            continue;
                        // project 3d point to 2d coordinate in this local frame
                        const Vec3 diff    = p_j.p - p_i.p;
                        Scalar     x_coord = diff.dot(proj_x_axis);
                        Scalar     y_coord = diff.dot(proj_y_axis);
                        Scalar     dist    = x_coord * x_coord + y_coord * y_coord;
                        if (neighbor_index != i && dist < min_dist_2d)
                            min_dist_2d = dist;
                        coords.push_back(x_coord);
                        coords.push_back(y_coord);
                        // record index of 3d point that 2d coordinate belongs to
                        coords_pids.push_back(neighbor_index);
                    }
                    sp_neighboring_triangles[_i] = std::vector<int>{};
                    // triangulation should take at least 3 points,
                    // and these points should not be too close to each other
                    if (coords.size() >= 2 * 3 && min_dist_2d > 0.0) {
                        try {
                            delaunator::Delaunator d(coords);
                            for (std::size_t triangle_id = 0; triangle_id < d.triangles.size(); triangle_id += 3) {
                                size_t v0 = coords_pids[d.triangles[triangle_id + 0]];
                                size_t v1 = coords_pids[d.triangles[triangle_id + 1]];
                                size_t v2 = coords_pids[d.triangles[triangle_id + 2]];
                                if (v0 == i || v1 == i || v2 == i) {
                                    // first-ring triangle that directly links to point i
                                    sp_neighboring_triangles[_i].push_back(v0);
                                    sp_neighboring_triangles[_i].push_back(v1);
                                    sp_neighboring_triangles[_i].push_back(v2);
                                }
                            }
                        } catch (...) {
                        }
                    }
                }
            }
            for (auto& s : sp_neighboring_triangles) {
                for (auto tid : s) {
                    sp_triangles.push_back(tid);
                }
            }
        }

        if (k == num_iters - 1)
            saveIntermediateResult(particles, sp_ids_array, sp_triangles);

        // ============================================
        // projection
        // ============================================
        // Calculate delta p in the Jacobi style
        MatX delta_p(3, num_particles);

        // ============================================
        // density constraints projection
        // ============================================
        VecX lambda_rho(num_particles);
        delta_p.setConstant(0.0);
        parallelutil::parallel_for(num_particles, [&](const int i) {
            const auto&  p = particles[i];
            const Scalar numerator =
                calcDensityConstraint(i, particles, neighbor_search_engine, rest_density, kernel_radius);

            Scalar denominator = 0.0;
            for (int neighbor_index : neighbor_search_engine.retrieveNeighbors(i)) {
                const Vec3 grad = calcGradConstraint(i,
                                                     neighbor_index,
                                                     particles,
                                                     neighbor_search_engine,
                                                     rest_density,
                                                     kernel_radius);

                denominator += (1.0 / particles[neighbor_index].m) * grad.squaredNorm();
            }
            denominator += e_rho;

            lambda_rho[i] = -numerator / denominator;
        });

        parallelutil::parallel_for(num_particles, [&](const int i) {
            const auto& p             = particles[i];
            const auto& neighbors     = neighbor_search_engine.retrieveNeighbors(i);
            const int   num_neighbors = neighbors.size();

            // Calculate the sum of pressure effect (Eq.12)
            MatX buffer(3, num_neighbors);
            for (int j = 0; j < num_neighbors; ++j) {
                const int neighbor_index = neighbors[j];
                // Calculate the artificial tensile pressure correction
                const Scalar kernel_val = calcKernel(p.p - particles[neighbor_index].p, kernel_radius);
                const Scalar coeff      = particles[neighbor_index].m * (lambda_rho[i] + lambda_rho[neighbor_index]);
                buffer.col(j)           = coeff * calcGradKernel(p.p - particles[neighbor_index].p, kernel_radius);
            }
            delta_p.col(i) = (1.0 / p.m) * (1.0 / rest_density) * buffer.rowwise().sum();
        });
        parallelutil::parallel_for(num_particles, [&](const int i) {
            particles[i].p += delta_p.col(i);
        });

        // ============================================
        // distance projection
        // ============================================
        delta_p.setConstant(0.0);
        const Scalar d0 = 2.0 * particle_radius;
        for (int i = 0; i < num_particles; i++) {
            const auto& p_i = particles[i];
            for (int j : neighbor_search_engine.retrieveNeighbors(i)) {
                const auto& p_j            = particles[j];
                auto        diff           = p_i.p - p_j.p;
                Scalar      diff_norm      = diff.norm();
                Scalar      constraint_val = std::min(0.0, diff_norm - d0);
                auto        grad_i         = diff.normalized();
                auto        grad_j         = (-diff).normalized();
                if (diff_norm > d0) {
                    grad_i.setConstant(0.0);
                    grad_j.setConstant(0.0);
                }
                Scalar numerator   = constraint_val;
                Scalar denominator = (1.0 / p_i.m) * grad_i.squaredNorm() + (1.0 / p_j.m) * grad_j.squaredNorm();
                denominator += e_d;
                Scalar lambda_dist = -numerator / std::max(denominator, 1e-24);
                delta_p.col(i) += (1.0 / p_i.m) * lambda_dist * grad_i;
                delta_p.col(j) += (1.0 / p_j.m) * lambda_dist * grad_j;
            }
        }
        parallelutil::parallel_for(num_particles, [&](const int i) {
            particles[i].p += delta_p.col(i);
        });

        // ============================================
        // area constraints projection
        // ============================================
        Scalar total_area = 0.0;
        delta_p.setConstant(0.0);
        int num_triangles = sp_triangles.size();
        for (int t = 0; t < num_triangles; t += 3) {
            int    t1        = sp_triangles[t + 0];
            int    t2        = sp_triangles[t + 1];
            int    t3        = sp_triangles[t + 2];
            auto   pt1       = particles[t1];
            auto   pt2       = particles[t2];
            auto   pt3       = particles[t3];
            auto   pt21      = pt2.p - pt1.p;
            auto   pt31      = pt3.p - pt1.p;
            auto   pt32      = pt3.p - pt2.p;
            Scalar numerator = 0.5 * crossProduct(pt21, pt31).norm();
            total_area += numerator;
            auto   grad_t1     = 0.5 * crossProduct(crossProduct(pt21, pt31).normalized(), pt32);
            auto   grad_t2     = 0.5 * crossProduct(crossProduct(pt32, -pt21).normalized(), -pt31);
            auto   grad_t3     = 0.5 * crossProduct(crossProduct(-pt31, -pt32).normalized(), pt21);
            Scalar denominator = (1.0 / pt1.m) * grad_t1.squaredNorm() + (1.0 / pt2.m) * grad_t2.squaredNorm() +
                                 (1.0 / pt3.m) * grad_t3.squaredNorm();
            denominator += e_a;
            Scalar lambda_area = -numerator / std::max(denominator, 1e-24);
            delta_p.col(t1) += (1.0 / pt1.m) * lambda_area * grad_t1;
            delta_p.col(t2) += (1.0 / pt2.m) * lambda_area * grad_t2;
            delta_p.col(t3) += (1.0 / pt3.m) * lambda_area * grad_t3;
        }
        parallelutil::parallel_for(num_particles, [&](const int i) {
            particles[i].p += delta_p.col(i);
        });
        std::cout << "Area: " << total_area << std::endl;

        parallelutil::parallel_for(num_particles, [&](const int i) {
            auto& p = particles[i];
            if (p.p.z() >= 1.25) {
                Scalar r = std::sqrt(p.p.x() * p.p.x() + p.p.y() * p.p.y());
                if (r >= 1 / 2.7144) {
                    p.p.x() = p.p.x() / r * 1 / 2.7144;
                    p.p.y() = p.p.y() / r * 1 / 2.7144;
                }
            }
        });
        // std::cout << "Solve collision constraints " << k << std::endl;
    }

    // Update positions and velocities
    parallelutil::parallel_for(num_particles, [&](const int i) {
        particles[i].v = damping * (particles[i].p - particles[i].x) / dt;
        particles[i].x = particles[i].p;
    });
    // std::cout << "Update" << std::endl;

    // Apply the XSPH viscosity effect [Schechter+, SIGGRAPH 2012]
    VecX densities(num_particles);
    MatX delta_v(3, num_particles);

    parallelutil::parallel_for(num_particles, [&](const int i) {
        densities[i] = calcDensity(i, particles, neighbor_search_engine, kernel_radius);
    });

    Scalar total_vol = 0.0;
    Scalar init_vol  = 0.0;
    for (int i = 0; i < num_particles; i++) {
        total_vol += particles[i].m / densities[i];
        init_vol += particles[i].m / rest_density;
    }
    std::cout << "=======================" << std::endl;
    std::cout << "Volume: " << total_vol << ", Initial Volume: " << init_vol << std::endl;
    std::cout << "=======================" << std::endl;

    parallelutil::parallel_for(num_particles, [&](const int i) {
        const auto& p             = particles[i];
        const auto& neighbors     = neighbor_search_engine.retrieveNeighbors(i);
        const int   num_neighbors = neighbors.size();

        MatX buffer(3, num_neighbors);
        for (int j = 0; j < num_neighbors; ++j) {
            const int    neighbor_index = neighbors[j];
            const Scalar kernel_val     = calcKernel(p.x - particles[neighbor_index].x, kernel_radius);
            const auto   rel_velocity   = particles[neighbor_index].v - p.v;

            buffer.col(j) = (p.m / densities[neighbor_index]) * kernel_val * rel_velocity;
        }
        const auto sum = buffer.rowwise().sum();

        delta_v.col(i) = viscosity_coeff * sum;
    });
    // std::cout << "XSPH viscosity done" << std::endl;

    parallelutil::parallel_for(num_particles, [&](const int i) {
        particles[i].v += delta_v.col(i);
    });

    // TODO: Apply vorticity confinement
}

void saveIntermediateResult(const std::vector<Particle>& particles,
                            const std::vector<int>&      sp_ids_array,
                            const std::vector<int>&      sp_triangles)
{
    int num_particles = particles.size();
    std::filesystem::create_directory("./vis");
    std::filesystem::create_directory("./out");

    std::vector<float> px{}, py{}, pz{};
    std::vector<float> nx{}, ny{}, nz{};
    for (auto i : sp_ids_array) {
        auto& p = particles[i];
        px.emplace_back(p.p.x());
        py.emplace_back(p.p.y());
        pz.emplace_back(p.p.z());
        nx.emplace_back(p.n.x());
        ny.emplace_back(p.n.y());
        nz.emplace_back(p.n.z());
    }

    // Create an empty object
    happly::PLYData plyOut;
    // Add elements
    plyOut.addElement("vertex", sp_ids_array.size());
    plyOut.getElement("vertex").addProperty<float>("x", px);
    plyOut.getElement("vertex").addProperty<float>("y", py);
    plyOut.getElement("vertex").addProperty<float>("z", pz);
    plyOut.getElement("vertex").addProperty<float>("nx", nx);
    plyOut.getElement("vertex").addProperty<float>("ny", ny);
    plyOut.getElement("vertex").addProperty<float>("nz", nz);

    // Write the object to file
    char file_name[100];
    sprintf(file_name, "vis/surface_frame%04d.ply", frame_id);
    plyOut.write(file_name, happly::DataFormat::Binary);

    happly::PLYData      plyOut2;
    std::vector<float>   px2{}, py2{}, pz2{};
    std::vector<uint8_t> r{}, g{}, b{};

    px2.resize(num_particles);
    py2.resize(num_particles);
    pz2.resize(num_particles);
    r.resize(num_particles);
    g.resize(num_particles);
    b.resize(num_particles);
    parallelutil::parallel_for(num_particles, [&](const int i) {
        auto& p = particles[i];
        px2[i]  = p.p.x();
        py2[i]  = p.p.y();
        pz2[i]  = p.p.z();
        if (p.s) { // red for surface point
            r[i] = uint8_t(255);
            g[i] = uint8_t(0);
            b[i] = uint8_t(0);
        } else {
            r[i] = uint8_t(0);
            g[i] = uint8_t(0);
            b[i] = uint8_t(255);
        }
    });

    int                  num_edges = sp_triangles.size();
    std::vector<int>     e0{}, e1{};
    std::vector<uint8_t> er{}, eg{}, eb{};
    e0.resize(num_edges);
    e1.resize(num_edges);
    er.resize(num_edges);
    eg.resize(num_edges);
    eb.resize(num_edges);

    parallelutil::parallel_for(num_edges / 3, [&](const int triangle_id) {
        int v0 = sp_triangles[3 * triangle_id + 0];
        int v1 = sp_triangles[3 * triangle_id + 1];
        int v2 = sp_triangles[3 * triangle_id + 2];

        e0[3 * triangle_id + 0] = v0;
        e1[3 * triangle_id + 0] = v1;
        e0[3 * triangle_id + 1] = v1;
        e1[3 * triangle_id + 1] = v2;
        e0[3 * triangle_id + 2] = v2;
        e1[3 * triangle_id + 2] = v0;

        Vec3 color = 127.5 * Vec3::Random();
        color += Vec3(127.5, 127.5, 127.5);
        int R = color.x();
        int G = color.y();
        int B = color.z();

        er[3 * triangle_id + 0] = uint8_t(R);
        eg[3 * triangle_id + 0] = uint8_t(G);
        eb[3 * triangle_id + 0] = uint8_t(B);
        er[3 * triangle_id + 1] = uint8_t(R);
        eg[3 * triangle_id + 1] = uint8_t(G);
        eb[3 * triangle_id + 1] = uint8_t(B);
        er[3 * triangle_id + 2] = uint8_t(R);
        eg[3 * triangle_id + 2] = uint8_t(G);
        eb[3 * triangle_id + 2] = uint8_t(B);
    });

    plyOut2.addElement("vertex", num_particles);
    plyOut2.getElement("vertex").addProperty<float>("x", px2);
    plyOut2.getElement("vertex").addProperty<float>("y", py2);
    plyOut2.getElement("vertex").addProperty<float>("z", pz2);
    plyOut2.getElement("vertex").addProperty<uint8_t>("red", r);
    plyOut2.getElement("vertex").addProperty<uint8_t>("green", g);
    plyOut2.getElement("vertex").addProperty<uint8_t>("blue", b);

    plyOut2.addElement("edge", num_edges);
    plyOut2.getElement("edge").addProperty<int>("vertex1", e0);
    plyOut2.getElement("edge").addProperty<int>("vertex2", e1);
    plyOut2.getElement("edge").addProperty<uint8_t>("red", er);
    plyOut2.getElement("edge").addProperty<uint8_t>("green", eg);
    plyOut2.getElement("edge").addProperty<uint8_t>("blue", eb);

    sprintf(file_name, "vis/points_frame%04d.ply", frame_id);
    plyOut2.write(file_name, happly::DataFormat::Binary);
}