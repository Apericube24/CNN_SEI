#ifndef VIRGULE_FIXE_H
#define VIRGULE_FIXE_H

#include <cstdint>
#include <iostream>

template <typename BaseType, int FractionalBits>
class FixedPoint {
private:
    BaseType value;

public:
    // Constructeurs
    FixedPoint() : value(0) {}
    FixedPoint(float f) : value(static_cast<BaseType>(f * (1 << FractionalBits))) {}
    FixedPoint(double d) : value(static_cast<BaseType>(d * (1 << FractionalBits))) {}
    FixedPoint(int i) : value(static_cast<BaseType>(i * (1 << FractionalBits))) {}

    // Accès au "raw value" (non-shifté)
    BaseType raw() const { return value; }

    // Conversion vers float
    float toFloat() const { return static_cast<float>(value) / (1 << FractionalBits); }

    // Conversion vers int
    int toInt() const { return static_cast<int>(value >> FractionalBits); }

    // Opérateurs arithmétiques
    FixedPoint operator+(const FixedPoint& other) const { return FixedPoint::fromRaw(value + other.value); }
    FixedPoint operator-(const FixedPoint& other) const { return FixedPoint::fromRaw(value - other.value); }
    FixedPoint operator*(const FixedPoint& other) const {
        return FixedPoint::fromRaw((value * other.raw()) >> FractionalBits);
    }
    FixedPoint operator/(const FixedPoint& other) const {
        return FixedPoint::fromRaw((value << FractionalBits) / other.raw());
    }

    // Opérateurs de comparaison
    bool operator<(const FixedPoint& other) const { return value < other.value; }
    bool operator<=(const FixedPoint& other) const { return value <= other.value; }
    bool operator>(const FixedPoint& other) const { return value > other.value; }
    bool operator>=(const FixedPoint& other) const { return value >= other.value; }
    bool operator==(const FixedPoint& other) const { return value == other.value; }
    bool operator!=(const FixedPoint& other) const { return value != other.value; }

    // Affichage pour debug
    friend std::ostream& operator<<(std::ostream& os, const FixedPoint& fp) {
        os << fp.toFloat();
        return os;
    }

    // Constructeur "raw"
    static FixedPoint fromRaw(BaseType rawValue) {
        FixedPoint fp;
        fp.value = rawValue;
        return fp;
    }
};

#endif // VIRGULE_FIXE_H
