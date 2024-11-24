//
// Created by neodima on 18.11.24.
//

#ifndef WLS_WLS_H
#define WLS_WLS_H

#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/fisher_f.hpp>

#include "Eigen/Dense"

struct solution {
    Eigen::VectorXd beta;
    Eigen::VectorXd sigma_beta;
    Eigen::VectorXd diagW;
    double cofDet;
    double MAE;
    double MSE;
    Eigen::VectorXi tTest;
    bool fTest;
};

std::ostream &operator<<(std::ostream &os, const solution &sol) {


    os << std::fixed << std::setprecision(4);

    os << std::setw(5) << "tTest"
       << std::setw(10) << "Beta"
       << std::setw(10) << "Sigma\n";

    for (int i = 0; i < sol.beta.size(); ++i) {
        std::string significance = sol.tTest[i] ? "+" : "-";
        os << std::setw(5) << significance
           << std::setw(10) << sol.beta[i]
           << std::setw(10) << sol.sigma_beta[i]
           << "\n";
    }

    os << "\nПрочие параметры:\n";
    os << "Коэффициент детерминации (cofDet): " << sol.cofDet << "\n";
    os << "Средняя абсолютная ошибка (MAE): " << sol.MAE << "\n";
    os << "Среднеквадратичная ошибка (MSE): " << sol.MSE << "\n";
    os << "F-тест: " << (sol.fTest ? "Пройден" : "Не пройден") << "\n";
    os << "Веса: ";

    for (unsigned int j = 0; j < sol.diagW.size(); ++j){
        os << std::setprecision(4) << j << ": " << sol.diagW[j] << "; ";
    }
    os << "\n";
    return os;
}

double calculate_t_crit(const double alpha, const int df) {
    boost::math::students_t dist(df);
    return boost::math::quantile(dist, 1 - alpha);
}

double calculate_f_crit(const double alpha, const int df1,const int df2) {
    boost::math::fisher_f dist(df1, df2);
    return boost::math::quantile(dist, 1 - alpha);
}

inline double MAE(const Eigen::VectorXd &Error) noexcept {
    return Error.cwiseAbs().sum() / static_cast<double>(Error.size());
}

inline double MSE(const Eigen::VectorXd &Error) noexcept {
    return Error.squaredNorm() / static_cast<double>(Error.size());
}

double doubleR(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &beta) {
    const double ss_total = (y.array() - y.mean()).square().sum();
    const double ss_residual = (y - X * beta).squaredNorm();
    return 1.0 - (ss_residual / ss_total);
}

double TSSj(const Eigen::MatrixXd &X, const unsigned int j) {
    double tss_j = 0.0;
    for (unsigned int i = 0; i < X.rows(); ++i) {
        const double row_mean = X.row(i).mean();
        const double diff = X(i, j) - row_mean;

        tss_j += diff * diff;
    }
    return tss_j;
}

double goodStep(const Eigen::MatrixXd &X, const Eigen::VectorXd &W_diag,
                const Eigen::VectorXd &grad, const Eigen::VectorXd &beta,
                const Eigen::VectorXd &Y, double initialStep, double gamma, double c) {
    double step = initialStep;
    const Eigen::MatrixXd W = W_diag.asDiagonal();

    const double initialError = (Y - X * beta).transpose() * W * (Y - X * beta);

    while (step > 1e-8) {
        const Eigen::VectorXd candidateBeta = beta - step * grad;
        const Eigen::VectorXd Err = Y - X * (beta - step * grad);
        const double candidateError = Err.transpose() * W * Err;

        if (candidateError <= initialError - c * step * grad.squaredNorm()) {
            break;
        }
        step *= gamma;
    }
    return step;
}


std::optional<solution> WLS(const Utils::InputParameters &inp, const double tol,
                            const unsigned int maxIter, const double alpha) {

    Eigen::MatrixXd W = inp.W.asDiagonal();
    const Eigen::VectorXd &beta = inp.beta;
    const Eigen::VectorXd &Y = inp.Y;
    const Eigen::MatrixXd &X = inp.X;
//    Eigen::VectorXd beta = Eigen::VectorXd::Constant(X.cols(), Y.mean() / X.cols());

    const unsigned int N = X.rows();
    const unsigned int p = X.cols();

    const double t_crit = calculate_t_crit(alpha, N - p);
    const double f_crtu = calculate_f_crit(alpha, p - 1, N - p);

    Eigen::VectorXd newBeta = beta;
    Eigen::VectorXd Err = Y - X * beta;
    Eigen::VectorXd newErr = Y - X * beta;

    for (unsigned int iter = 0; iter < maxIter; ++iter) {
        Eigen::VectorXd gradient = -2 * X.transpose() * (W * newErr);
        const double bestStep = goodStep(inp.X, W.diagonal(), gradient, newBeta, inp.Y, 1.0, 0.5, 1e-4);;

        newBeta -= bestStep * gradient;

        newErr = Y - X * newBeta;
        const auto deltaErr = newErr.cwiseAbs();
        const double minErr = deltaErr.minCoeff();
        const double maxErr = deltaErr.maxCoeff();

        const double gamma = 0.8; // для ограничения весов снизу [gamma; 1]
//        const double RR = doubleR(X, Y, newBeta);
//        const double errNorm = newErr.norm();

        for (unsigned int i = 0; i < N; ++i) {
            const double w = 1 - (1 - gamma) / std::abs(maxErr - minErr) * std::abs(newErr[i] - minErr);
            W(i, i) = std::abs(w);
        }

        if ((newErr - Err).norm() < tol) {
//            std::cout << "MAX ITER: " << iter << std::endl;
//            std::cout << newErr.transpose() * newErr << " c весом " << newErr.transpose() * W * newErr << std::endl;

            const double mse = MSE(newErr);
            const double rss = mse * N;
            const double sigma_2 = rss / (N - p);
            Eigen::VectorXd sigma_beta(p);
            Eigen::VectorXi tTest(p);
            for (int i = 0; i < p; ++i) {
                sigma_beta[i] = std::sqrt(sigma_2 / TSSj(X, i));
                tTest[i] = static_cast<int>(t_crit < std::abs(newBeta[i] / sigma_beta[i]));
            }
            const double R = doubleR(X, Y, newBeta);
            const double F = static_cast<double>(N - p) * R / (1 - R) / (p - 1);

            return solution{
                    .beta = newBeta,
                    .sigma_beta = sigma_beta,
                    .diagW = W.diagonal(),
                    .cofDet = R,
                    .MAE = MAE(newErr),
                    .MSE = mse,
                    .tTest = tTest,
                    .fTest = (F > f_crtu),
            };
        }
        Err = newErr;
    }

    return std::nullopt;
}


#endif //WLS_WLS_H
