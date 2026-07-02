#pragma once
// ============================================================================
// data_gen.hpp
// Shared synthetic dataset generator with automated Excel (CSV) export.
// ============================================================================
#include <vector>
#include <random>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

// Global constants describing the data range
constexpr double DATA_MIN = 0.0;
constexpr double DATA_MAX = 10000.0;

// Helper function to export the generated vectors into an Excel-friendly CSV
inline void export_dataset_to_excel(const std::vector<double>& colA, const std::vector<double>& colB, uint64_t n) {
    std::string filename = "dataset_custom.csv";
    if (n == 1'000'000ULL) filename = "dataset_small.csv";
    else if (n == 10'000'000ULL) filename = "dataset_medium.csv";
    else if (n == 100'000'000ULL) filename = "dataset_large.csv";

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create excel data file: " << filename << "\n";
        return;
    }

    // Write header rows for Excel columns
    file << "Index,Column A,Column B\n";

    // Write the raw array rows (limits to first 1M lines to prevent Excel crash on huge files)
    uint64_t linesToPrint = (n > 1'000'000ULL) ? 1'000'000ULL : n;
    
    for (uint64_t i = 0; i < linesToPrint; ++i) {
        file << i << "," << colA[i] << "," << colB[i] << "\n";
    }

    file.close();
    std::cout << "--> Raw data samples successfully saved to Excel file: '" << filename << "'\n";
}

// Generates two correlated columns of doubles.
inline void generate_dataset(std::vector<double>& colA,
                              std::vector<double>& colB,
                              uint64_t n,
                              uint64_t seed = 42) {
    colA.resize(n);
    colB.resize(n);

    std::mt19937_64 rngA(seed);
    std::uniform_real_distribution<double> dist(DATA_MIN, DATA_MAX);
    for (uint64_t i = 0; i < n; ++i) {
        colA[i] = dist(rngA);
    }

    std::mt19937_64 rngB(seed + 1);
    std::normal_distribution<double> noise(0.0, 500.0);
    for (uint64_t i = 0; i < n; ++i) {
        colB[i] = colA[i] * 0.7 + noise(rngB);
    }

    // Explicit outlier injections so the statistical logic triggers properly
    if (n >= 200) {
        for (int i = 0; i < 100; ++i) {
            colA[i * 10] = -40000.0; 
            colA[i * 10 + 1] = 50000.0; 
        }
    }

    // Automatically call the Excel exporter function
    export_dataset_to_excel(colA, colB, n);
}

// Slice generation fallback logic for standalone partition tasks
inline void generate_dataset_slice(std::vector<double>& colA,
                                    std::vector<double>& colB,
                                    uint64_t start,
                                    uint64_t count,
                                    uint64_t seed = 42) {
    colA.resize(count);
    colB.resize(count);

    std::mt19937_64 rngA(seed);
    std::uniform_real_distribution<double> distA(DATA_MIN, DATA_MAX);
    std::mt19937_64 rngB(seed + 1);
    std::normal_distribution<double> noise(0.0, 500.0);

    for (uint64_t i = 0; i < start + count; ++i) {
        double a = distA(rngA);
        double b = a * 0.7 + noise(rngB);
        if (i >= start) {
            uint64_t idx = i - start;
            colA[idx] = a;
            colB[idx] = b;
        }
    }
}