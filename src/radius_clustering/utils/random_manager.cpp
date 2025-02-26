#include "random_manager.h"
#include <chrono>
#include <limits>

std::mt19937 RandomManager::rng(std::chrono::system_clock::now().time_since_epoch().count());
std::vector<std::mt19937> RandomManager::parallelRng;

void RandomManager::setSeed(long seed) {
    rng.seed(seed);
}

std::mt19937& RandomManager::getRandom() {
    return rng;
}

void RandomManager::initParallel(int nRandoms, long initSeed) {
    parallelRng.resize(nRandoms);
    std::mt19937 rndStart(initSeed);
    for (int i = 0; i < nRandoms; ++i) {
        int seed = std::uniform_int_distribution<>(0, std::numeric_limits<int>::max())(rndStart);
        parallelRng[i].seed(seed);
    }
}

std::mt19937& RandomManager::getRandom(int i) {
    return parallelRng[i];
}

int RandomManager::nextInt(int max) {
    return std::uniform_int_distribution<>(0, max - 1)(rng);
}
