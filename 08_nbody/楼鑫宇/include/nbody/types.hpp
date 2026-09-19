#pragma once

#include <cmath>

namespace nbody {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }
};

inline Vec3 operator+(Vec3 lhs, const Vec3& rhs) {
    lhs += rhs;
    return lhs;
}

inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vec3 operator*(const Vec3& value, double scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline double dot(const Vec3& lhs, const Vec3& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

inline double norm(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

struct Particle {
    Vec3 position;
    Vec3 velocity;
    double mass = 0.0;
};

struct Diagnostics {
    double kinetic = 0.0;
    double potential = 0.0;
    double total_energy = 0.0;
    Vec3 momentum;
    Vec3 angular_momentum;
    Vec3 center_of_mass;
};

}  // namespace nbody

