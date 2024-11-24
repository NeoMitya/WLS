#include <iostream>

#include "Eigen/Dense"
#include "utils/readCSV.h"
#include "utils/WLS.h"
//#include "chrono"

int main() {

    const auto INP = Utils::loadInputData("y.csv", "data.csv", "beta.csv", "weight.csv");

    if (INP.has_value()) {

//        const auto start{std::chrono::steady_clock::now()};

        const auto result = WLS(*INP, 1e-4, 10000, 0.1);
//        const auto end{std::chrono::steady_clock::now()};

        {
//            const auto startTable{std::chrono::steady_clock::now()};
//            const auto resultTable = (INP->X.transpose() * INP->X).inverse() * INP->X.transpose() * INP->Y;
//            const auto endTable{std::chrono::steady_clock::now()};

//            const std::chrono::duration<double> elapsed_seconds{end - start};
//            const std::chrono::duration<double> elapsed_secondsT{endTable - startTable};

//            std::cout << elapsed_seconds.count() << "<----my----" << std::endl;
//            std::cout << elapsed_secondsT.count() << "<----eq----" << std::endl;

            if (result.has_value()) {
//                std::cout << (INP->Y - INP->X * result->beta).norm() << "   " << (INP->Y - INP->X * resultTable).norm()
//                          << std::endl;
                std::cout << *result << std::endl;

            }
        }



    }
    return 0;
}
