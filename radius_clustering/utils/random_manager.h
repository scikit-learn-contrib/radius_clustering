#ifndef RANDOM_MANAGER_H
#define RANDOM_MANAGER_H

#include <random>
#include <vector>

class RandomManager {
private:
    static std::mt19937 rng;
    static std::vector<std::mt19937> parallelRng;

public:
    static void setSeed(long seed);
    static std::mt19937& getRandom();
    static void initParallel(int nRandoms, long initSeed);
    static std::mt19937& getRandom(int i);
    static int nextInt(int max);  // Add this line
};

#endif // RANDOM_MANAGER_H