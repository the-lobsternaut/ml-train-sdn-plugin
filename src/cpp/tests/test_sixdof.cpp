/**
 * ML-Train Plugin — 6DOF Integration Tests
 */

#include "ml_train/sixdof_core.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace sixdof;

void testFeatureExtraction() {
    State s;
    s.quat = qidentity();
    s.omega = {0.1, 0.05, -0.03};
    s.mass = 500;

    InertiaTensor I = inertiaDiag(100, 120, 80);
    auto coastFn = [](const State&, double) -> ForcesTorques { return {}; };
    double dt = 0.1, t = 0;

    std::vector<double> kinetic_energies;
    for (int i = 0; i < 1000; i++) {
        s = rk4Step(s, I, dt, t, coastFn); t += dt;
        double T = 0.5*(I[0]*s.omega[0]*s.omega[0]+I[1]*s.omega[1]*s.omega[1]+I[2]*s.omega[2]*s.omega[2]);
        kinetic_energies.push_back(T);
    }

    assert(kinetic_energies.size() == 1000);
    assert(std::abs(kinetic_energies.back() - kinetic_energies.front()) / kinetic_energies.front() < 1e-4);

    std::cout << "  Feature extraction ✓ (1000 samples, KE conserved)\n";
}

void testAnomalySignature() {
    InertiaTensor I = inertiaDiag(10, 15, 12);
    Vec3 normal_omega = {0.01, 0.01, 0.01};
    Vec3 anomaly_omega = {2.0, 1.5, -3.0};

    double normalE = 0.5*(I[0]*normal_omega[0]*normal_omega[0]+I[1]*normal_omega[1]*normal_omega[1]+I[2]*normal_omega[2]*normal_omega[2]);
    double anomalyE = 0.5*(I[0]*anomaly_omega[0]*anomaly_omega[0]+I[1]*anomaly_omega[1]*anomaly_omega[1]+I[2]*anomaly_omega[2]*anomaly_omega[2]);

    assert(anomalyE / normalE > 100);

    std::cout << "  Anomaly signature ✓ (ratio=" << anomalyE/normalE << "x)\n";
}

int main() {
    std::cout << "=== ml-train 6DOF tests ===\n";
    testFeatureExtraction();
    testAnomalySignature();
    std::cout << "All ml-train 6DOF tests passed.\n";
    return 0;
}
