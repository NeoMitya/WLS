//
// Created by neodima on 18.11.24.
//

#ifndef WLS_READCSV_H
#define WLS_READCSV_H

#include <fstream>
#include <string>
#include <optional>

namespace Utils {
    std::optional<std::pair<size_t, size_t>> getSizeTab(const std::string &filename) {
        std::ifstream file(std::string(DATA_DIR) + "/" + filename);
        if (!file.is_open()) {
            std::cerr << "Не удалось открыть файл: " << filename << "\n";
            return std::nullopt;
        }

        size_t rows = 0, cols = 0;
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            size_t currentCols = 0;

            while (std::getline(ss, cell, ',')) {
                ++currentCols;
            }

            if (cols == 0) {
                cols = currentCols;
            } else if (currentCols != cols) {
                std::cerr << "Ошибка: Неконсистентное число столбцов в строке файла " << filename << "\n";
                return std::nullopt;
            }

            ++rows;
        }

        return std::make_pair(rows, cols);
    }


    Eigen::VectorXd loadColumnFromCSV(const std::string &filename, size_t columnIndex) {
        std::ifstream file(std::string(DATA_DIR) + "/" + filename);
        if (!file.is_open()) {
            std::cout << filename << std::endl;
            throw std::runtime_error("Не удалось открыть файл: " + filename);
        }

        std::vector<double> columnValues;
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            size_t currentCol = 0;

            while (std::getline(ss, cell, ',')) {
                if (currentCol == columnIndex) {
                    try {
                        columnValues.push_back(std::stod(cell));
                    } catch (const std::invalid_argument &) {
                        std::cerr << "Ошибка преобразования значения: " << cell << "\n";
                    }
                }
                ++currentCol;
            }
        }

        Eigen::VectorXd columnVector(columnValues.size());
        for (size_t i = 0; i < columnValues.size(); ++i) {
            columnVector[i] = columnValues[i];
        }

        return columnVector;
    }


    Eigen::MatrixXd loadMatrixFromCSV(const std::string &filename, size_t numRows, size_t numCols) {
        std::ifstream file(std::string(DATA_DIR) + "/" + filename);
        if (!file.is_open()) {
            throw std::runtime_error("Не удалось открыть файл: " + filename);
        }

        Eigen::MatrixXd matrix(numRows, numCols + 1);
        std::string line;
        size_t rowIndex = 0;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            size_t colIndex = 1;

            matrix(rowIndex, 0) = 1.0;

            while (std::getline(ss, cell, ',')) {
                if (colIndex > numCols) {
                    throw std::runtime_error("Ошибка: Строка содержит больше столбцов, чем ожидалось.");
                }
                try {
                    matrix(rowIndex, colIndex) = std::stod(cell);
                } catch (const std::invalid_argument &) {
                    std::cerr << "Ошибка преобразования значения: " << cell << "\n";
                    matrix(rowIndex, colIndex) = 0.0;
                }
                ++colIndex;
            }

            if (colIndex - 1 != numCols) {
                throw std::runtime_error("Ошибка: Несоответствие числа столбцов в строке.");
            }
            ++rowIndex;
        }

        return matrix;
    }

    struct InputParameters {
        Eigen::VectorXd Y;
        Eigen::MatrixXd X;
        Eigen::VectorXd beta;
        Eigen::VectorXd W;
    };


    std::optional<InputParameters> loadInputData(const std::string &yFile, const std::string &xFile,
                                                 const std::string &betaFile, const std::string &weightsFile) {
        const auto ySize = getSizeTab(yFile);
        if (!ySize || ySize->second != 1) {
            std::cerr << "Файл Y некорректен (должен содержать один столбец).\n";
            return std::nullopt;
        }

        const auto xSize = getSizeTab(xFile);
        if (!xSize) {
            std::cerr << "Файл X некорректен.\n";
            return std::nullopt;
        }

        if (xSize->first != ySize->first) {
            std::cerr << "Размеры X и Y несовместимы (разное число строк).\n";
            return std::nullopt;
        }

        Eigen::VectorXd Y = loadColumnFromCSV(yFile, 0);
        Eigen::MatrixXd X = loadMatrixFromCSV(xFile, xSize->first, xSize->second);

        Eigen::VectorXd beta;
        const auto betaSize = getSizeTab(betaFile);
        if (!betaSize || betaSize->first != xSize->second + 1) {
            std::cerr << "Файл beta некорректен (инициализация нулями).\n";
            beta = Eigen::VectorXd::Zero(xSize->second + 1);
        } else {
            beta = loadColumnFromCSV(betaFile, 0);
        }

        Eigen::VectorXd W;
        const auto weightsSize = getSizeTab(weightsFile);
        if (!weightsSize || weightsSize->first != xSize->first) {
            std::cerr << "Файл весов некорректен (инициализация единицами).\n";
            W = Eigen::VectorXd::Ones(xSize->first);
        } else {
            W = loadColumnFromCSV(weightsFile, 0);
        }

        return InputParameters{Y, X, beta, W};
    }


}

#endif //WLS_READCSV_H
