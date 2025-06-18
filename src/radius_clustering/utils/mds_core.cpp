/**
 * @file mds_core.cpp
 * @brief Core implementation of the Minimum Dominating Set (MDS) algorithm.
 *
 * This file contains the C++ implementation of the MDS algorithm,
 * including the iterated greedy approach and supporting data structures.
 * It provides the main computational logic for solving MDS problems.
 */
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <csignal>
#include <limits>
#include "random_manager.h"


class Result {
public:
    Result() {} // Add this line
    Result(std::string instanceName) : instanceName(instanceName) {}

    void add(std::string key, float value) {
        map.push_back(Tuple(key, value));
    }

    float get(int pos) {
        return map[pos].value;
    }

    std::vector<std::string> getKeys() {
        std::vector<std::string> keys;
        for (auto& tuple : map) {
            keys.push_back(tuple.name);
        }
        return keys;
    }

    std::string getInstanceName() {
        return instanceName;
    }

    std::unordered_set<int> getSolutionSet() {
        return solutionSet;
    }

    void setSolutionSet(std::unordered_set<int> solutionSet) {
        this->solutionSet = solutionSet;
    }


private:
    class Tuple {
        public:

        std::string name;
        float value;
        
        Tuple(std::string name, float value) : name(name), value(value) {}
    };
    std::string instanceName;
    std::vector<Tuple> map;
    std::unordered_set<int> solutionSet;
};

class Instance {
public:
    Instance(int n, const std::vector<int>& edges_list, int nb_edges, std::string name) 
        : name(name), numNodes(n), adjacencyList(n) {
        for (int i = 0; i < numNodes; ++i) {
            unSelectedNodes.insert(i);
        }
        constructAdjacencyList(edges_list, nb_edges);
        setSupportNodes();
    }

    const std::vector<std::vector<int>>& getAdjacencyList() const { return adjacencyList; }
    const std::unordered_set<int>& getSupportNodes() const { return supportNodes; }
    const std::unordered_set<int>& getLeavesNodes() const { return leavesNodes; }
    const std::unordered_set<int>& getUnSelectedNodes() const { return unSelectedNodes; }
    int getNumNodes() const { return numNodes; }
    std::string getName() const { return name; }

private:
    std::string name;
    int numNodes;
    std::vector<std::vector<int>> adjacencyList;
    std::unordered_set<int> supportNodes;
    std::unordered_set<int> leavesNodes;
    std::unordered_set<int> unSelectedNodes;
    const bool supportAndLeafNodes = true;

    void constructAdjacencyList(const std::vector<int>& edge_list, int nb_edges) {
        for (int i = 0; i < 2 * nb_edges; i+=2) {
            int u = edge_list[i];
            int v = edge_list[i+1];
            adjacencyList[u].push_back(v);
            adjacencyList[v].push_back(u);
        }
    }

    void setSupportNodes() {
        for (int i = 0; i < numNodes; ++i) {
            if (adjacencyList[i].size() == 1 && supportAndLeafNodes) {
                int neighbour = adjacencyList[i][0];
                if (leavesNodes.find(neighbour) == leavesNodes.end()) {
                    leavesNodes.insert(i);
                    supportNodes.insert(neighbour);
                }
                unSelectedNodes.erase(neighbour);
                unSelectedNodes.erase(i);
            } else if (adjacencyList[i].empty() && supportAndLeafNodes) {
                supportNodes.insert(i);
            }
        }
    }
};

class Solution {
public:
    Solution(const Instance& inst) 
        : instance(&inst), numCovered(0), watchers(inst.getNumNodes()) {
        unSelectedNodes = inst.getUnSelectedNodes();
    }

    Solution(const Solution& other) = default;
    Solution& operator=(const Solution& other) = default;

    bool isFeasible() const { return numCovered == instance->getNumNodes(); }
    bool checking() {
        bool removed = false;
        std::vector<int> selectedList(selectedNotSupportNodes.begin(), selectedNotSupportNodes.end());
        for (int select : selectedList) {
            if (watchers[select].size() > 1) {
                bool remove = true;
                for (int elem : instance->getAdjacencyList()[select]) {
                    if (watchers[elem].size() == 1) {
                        remove = false;
                        break;
                    }
                }
                if (remove) {
                    removed = true;
                    removeNode(select);
                }
            }
        }
        return removed;
    }

    void addNode(int node) {
        selectedNodes.insert(node);
        unSelectedNodes.erase(node);
        addWatcher(node);
        if (instance->getSupportNodes().find(node) == instance->getSupportNodes().end()) {
            selectedNotSupportNodes.insert(node);
        }
    }

    void removeNode(int node) {
        selectedNodes.erase(node);
        unSelectedNodes.insert(node);
        removeWatcher(node);
        selectedNotSupportNodes.erase(node);
    }

    int getBestNextNode() const {
        int bestCount = -1;
        int bestNode = -1;

        for (int i : unSelectedNodes) {
            int count = 0;
            for (int neighbour : instance->getAdjacencyList()[i]) {
                if (watchers[neighbour].empty()) {
                    count++;
                }
            }
            if (bestCount < count && instance->getLeavesNodes().find(i) == instance->getLeavesNodes().end()) {
                bestCount = count;
                bestNode = i;
            }
        }
        return bestNode;
    }

    int getWorstNodeNew() const {
        int worstNode = -1;
        int totalMaxWatchers = 0;

        for (int i : selectedNotSupportNodes) {
            int minWatchers = std::numeric_limits<int>::max();
            for (int neighbour : instance->getAdjacencyList()[i]) {
                if (minWatchers > static_cast<int>(watchers[neighbour].size())) {
                    minWatchers = watchers[neighbour].size();
                }
            }
            if (totalMaxWatchers < minWatchers) {
                worstNode = i;
                totalMaxWatchers = minWatchers;
            }
        }

        return worstNode;
    }

    int evaluate() const { return selectedNodes.size(); }
    const std::unordered_set<int>& getSelectedNodes() const { return selectedNodes; }
    const std::unordered_set<int>& getSelectedNotSupportNodes() const { return selectedNotSupportNodes; }
    const std::unordered_set<int>& getUnSelectedNodes() const { return unSelectedNodes; }
    const std::vector<std::unordered_set<int>>& getWatchers() const { return watchers; }
    int getNumNodes() const { return instance->getNumNodes(); }

private:
    const Instance* instance;
    std::unordered_set<int> selectedNodes;
    std::unordered_set<int> selectedNotSupportNodes;
    std::unordered_set<int> unSelectedNodes;
    int numCovered;
    std::vector<std::unordered_set<int>> watchers;

    void addWatcher(int selectedNode) {
        if (watchers[selectedNode].empty()) {
            numCovered++;
        }
        watchers[selectedNode].insert(selectedNode);

        for (int neighbour : instance->getAdjacencyList()[selectedNode]) {
            if (watchers[neighbour].empty()) {
                numCovered++;
            }
            watchers[neighbour].insert(selectedNode);
        }
    }

    void removeWatcher(int selectedNode) {
        watchers[selectedNode].erase(selectedNode);
        if (watchers[selectedNode].empty()) {
            numCovered--;
        }

        for (int neighbour : instance->getAdjacencyList()[selectedNode]) {
            watchers[neighbour].erase(selectedNode);
            if (watchers[neighbour].empty()) {
                numCovered--;
            }
        }
    }
};

class GIP {
public:
    Solution construct(const Instance& instance) {
        Solution solution(instance);
        for (int supportNode : instance.getSupportNodes()) {
            solution.addNode(supportNode);
        }
        while (!solution.isFeasible()) {
            int selectedNode = solution.getBestNextNode();
            solution.addNode(selectedNode);
        }
        return solution;
    }
};

class LocalSearch {
public:
    static Solution execute(Solution& sol, const Instance& instance) {
        bool improve = true;
        while (improve) {
            improve = checkImprove(sol, instance);
        }
        return sol;
    }

private:
    static bool checkImprove(Solution& sol, const Instance& instance) {
        std::vector<int> copySelected(sol.getSelectedNotSupportNodes().begin(), sol.getSelectedNotSupportNodes().end());
        std::shuffle(copySelected.begin(), copySelected.end(), RandomManager::getRandom());

        for (int nodeRem : copySelected) {
            int nodeNew = selectElemToAdd(nodeRem, instance, sol);
            if (nodeNew != -1) {
                int of = sol.evaluate();
                sol.removeNode(nodeRem);
                sol.addNode(nodeNew);
                sol.checking();
                if (sol.evaluate() < of) {
                    return true;
                }
            }
        }
        return false;
    }

    static int selectElemToAdd(int node, const Instance& instance, const Solution& solution) {
        std::unordered_set<int> neighbours;
    bool neighboursInitialized = false;

    if (solution.getWatchers()[node].size() == 1) {
        neighbours = std::unordered_set<int>(instance.getAdjacencyList()[node].begin(), instance.getAdjacencyList()[node].end());
        neighboursInitialized = true;
    }

    for (int neighbour : instance.getAdjacencyList()[node]) {
        if (solution.getWatchers()[neighbour].size() == 1) {
            if (!neighboursInitialized) {
                neighbours = std::unordered_set<int>(instance.getAdjacencyList()[neighbour].begin(), instance.getAdjacencyList()[neighbour].end());
                neighboursInitialized = true;
            } else {
                std::unordered_set<int> temp;
                for (int n : instance.getAdjacencyList()[neighbour]) {
                    if (neighbours.find(n) != neighbours.end()) {
                        temp.insert(n);
                    }
                }
                neighbours = std::move(temp);
            }
        }
    }

    if (neighboursInitialized) {
        neighbours.erase(node);
    }

        return !neighboursInitialized || neighbours.empty() ? -1 : *neighbours.begin();
    }
};

class IG {
private:
    GIP constructive;
    LocalSearch localSearch;
    int maxItersWithoutImprove = 200;
    float beta = 0.2f;
    bool randomDestruct = true;
    bool randomConstruct = false;

public:
    IG(GIP& constructive, LocalSearch& localSearch) 
        : constructive(constructive), localSearch(localSearch) {}

    Result execute(const Instance& instance) {
        long initialTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        long totalTime = 0;
        float secs = 0.0f;
        Result result(instance.getName());

        Solution solution = firstSol(instance);
        int numElemsToDestruct = std::ceil(beta * solution.getSelectedNotSupportNodes().size());

        int numItersWithoutImprove = 0;
        int bestOF = solution.evaluate();
        while (numItersWithoutImprove < maxItersWithoutImprove && secs <= 600) {
            Solution current_solution = solution;
            destruct(current_solution, numElemsToDestruct);
            construct(current_solution);
            executeLocalSearch(current_solution, instance);
            if (current_solution.evaluate() >= bestOF) {
                numItersWithoutImprove++;
            } else {
                numItersWithoutImprove = 0;
                bestOF = current_solution.evaluate();
                solution = std::move(current_solution);
            }

            totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count() - initialTime;
            secs = totalTime / 1000.0f;
        }

        result.setSolutionSet(solution.getSelectedNodes());
        result.add("Time", secs);
        result.add("OF", static_cast<float>(bestOF));
        return result;
    }

private:
    Solution firstSol(const Instance& instance) {
        Solution solution = constructive.construct(instance);
        executeLocalSearch(solution, instance);
        return solution;
    }

    void destruct(Solution& solution, int numElemsToDestruct) {
        if (randomDestruct) {
            destructRandom(solution, numElemsToDestruct);
        } else {
            destructGreedy(solution, numElemsToDestruct);
        }
    }

    void construct(Solution& solution) {
        if (randomConstruct) {
            constructRandom(solution);
        } else {
            constructGreedy(solution);
        }
    }

    void destructRandom(Solution& solution, int numElemsToDestruct) {
        std::vector<int> selectedList(solution.getSelectedNotSupportNodes().begin(), solution.getSelectedNotSupportNodes().end());
        std::shuffle(selectedList.begin(), selectedList.end(), RandomManager::getRandom());
        for (int i = 0; i < numElemsToDestruct; i++) {
            solution.removeNode(selectedList[i]);
        }
    }

    void destructGreedy(Solution& solution, int numElemsToDestruct) {
        for (int i = 0; i < numElemsToDestruct; i++) {
            int worstNode = solution.getWorstNodeNew();
            solution.removeNode(worstNode);
        }
    }

    void constructRandom(Solution& solution) {
        while (!solution.isFeasible()) {
            int randomNode = RandomManager::nextInt(solution.getNumNodes());
            solution.addNode(randomNode);
        }
    }

    void constructGreedy(Solution& solution) {
        while (!solution.isFeasible() && !solution.getUnSelectedNodes().empty()) {
            int bestNode = solution.getBestNextNode();
            solution.addNode(bestNode);
        }
    }

    void executeLocalSearch(Solution& solution, const Instance& instance) {
        solution = localSearch.execute(solution, instance);
    }
};

class Main {
private:
    GIP constructive;
    LocalSearch localSearch;
    IG algorithm;

    static void signal_handler(int sig) {
      exit(sig);
    }

public:
    Main() : algorithm(constructive, localSearch) {}

    Result execute(int numNodes, const std::vector<int>& edges_list, int nb_edges, long seed) {
        Instance instance(numNodes, edges_list, nb_edges, "name");
        RandomManager::setSeed(seed);
        signal(SIGINT, signal_handler);
        return algorithm.execute(instance);
    }
};

extern "C" {
    inline Result iterated_greedy_wrapper(int numNodes, const std::vector<int>& edges_list, int nb_edges, long seed) {
        static Main main;  // Create a single static instance

        return main.execute(numNodes, edges_list, nb_edges, seed);
    }
}