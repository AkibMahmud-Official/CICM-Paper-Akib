#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <fstream>
#include <string>
#include <time.h>
#include <nlopt.hpp>
using std::ws;
using std::string;

int equi_sdof(double* M1star, double* k1star, double* omega_1st)
{
    constexpr int n = 10;    //Number of stories in the Building
    double m = 360e3;       //mass of each story in kg
    double k = 650000e3;    //stiffness of each story in N/m
    double ks = 1;          // bottom soft storey stiffness as percentage of upper floors. Type 0.6 for 60%

    // Define the stiffness and mass matrices

    Eigen::MatrixXd massmatrix1;
    massmatrix1.setIdentity(n, n);
    Eigen::MatrixXd M = m * massmatrix1;

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);
    K(0, 0) = k + ks * k;
    K(0, 1) = -k;
    K(n - 1, n - 2) = -k;
    K(n - 1, n - 1) = k;
    for (int i = 1; i < n - 1; i++)
    {
        K(i, i - 1) = -k;
        K(i, i) = 2 * k;
        K(i, i + 1) = -k;
    }

    // Solve for eigenvalues and eigenvectors
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(K, M);
    if (es.info() != Eigen::Success) {
        std::cerr << "Eigenvalue solver failed to converge!" << std::endl;
        return 1;
    }

    // Compute the natural frequencies and periods
    Eigen::VectorXd omega = es.eigenvalues().cwiseSqrt();                   // natural frequencies
    Eigen::VectorXd T = 2 * M.diagonal().cwiseSqrt().cwiseQuotient(omega);  // natural time periods  
    Eigen::MatrixXd V = es.eigenvectors();                                  // mode shapes  

    // Calculate the mode frequencies
    double h = 3.048;                                                       // height of each story
    Eigen::VectorXd imega = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; i++) {
        imega(i) = omega(i) * sqrt(1 + pow(h * V(0, i) / V(n - 1, i), 2));
    }

    *omega_1st = omega(0);                                                  // mode frequencies

    /*********************************************************************************************/
    // Effective Modal Mass Determination

    Eigen::Matrix<double, n, 1> modeshape_1st = Eigen::Matrix<double, n, 1>::Map(V.col(0).data());

    Eigen::MatrixXd influence_vector(n, 1);
    influence_vector.setOnes();

    Eigen::MatrixXd L1h = modeshape_1st.transpose() * M * influence_vector;

    Eigen::MatrixXd M1 = modeshape_1st.transpose() * M * modeshape_1st;

    *M1star = (L1h * L1h * M1).value();
    *k1star = *M1star * *omega_1st * *omega_1st;

    return 0;
}

int top_deflection(const double& M1star, const double& omega_1st, const double& k1star,
    double& mbar, double& md, double& kd, double& cd, int& response, double* topdef_struc, double* topdef_tmd)
{
    double fopt, Xidopt;

    //Initial value of mbar:
    //
    //double H1opt = 7, H2opt = 7;    //Equal for very small mbar. Assume value as 7. 
    //b = 2 - 0.5 * H2opt * H2opt;
    //mbar = std::min((-b + sqrt(b * b - 4)) / 2, (-b - sqrt(b * b - 4)) / 2);
    //calculating shows mbar =0.0445326. Used in main()

    md = mbar * M1star;
    fopt = sqrt(1 - 0.5 * mbar) / (1 + mbar);
    kd = mbar * k1star * fopt * fopt;
    Xidopt = sqrt((mbar * (3 - sqrt(0.5 * mbar))) / (8 * (1 + mbar) * (1 - 0.5 * mbar)));
    cd = mbar * fopt * (2 * Xidopt * omega_1st * M1star);

    //.....................////   Top Deflection    ///.................................

    double omegan1, omegan2;		//Natural circular frequency for 1st mode
    double T1, T2;				//Natural time period for 1st
    double D1, D2;				//Dn determination from pseusdo-acceleration An==Cs
    double L1h, L2h, M1, M2;	    //for gamma n determination
    double gamma1, gamma2;		    //gamma n
    constexpr double pi = 3.14159265;

    // Define matrcies for 2DOF (struc + tmd)

    Eigen::MatrixXd M(2, 2), K(2, 2), C(2, 2);

    M << M1star, 0.0,
        md, md;
    K << k1star, -kd,
        0, kd;
    C << 0, -cd,
        0, cd;

    // Solve for eigenvalues and eigenvectors using generalized eigenvalue problem
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(K, M);
    if (es.info() != Eigen::Success) {
        std::cout << "Failed to compute eigenvalues." << std::endl;
        return -1;
    }
    Eigen::VectorXd freq = es.eigenvalues();

    Eigen::VectorXd omega = freq.array().sqrt();

    omegan1 = omega[0];              //natural omega for 1st mode
    omegan2 = omega[1];              //natural omega for 2nd mode

    Eigen::MatrixXd mode = es.eigenvectors();

    double phi11 = mode(0);			//Deflected shape phi at (1) prim struc for 1st Mode
    double phi12 = mode(1);			//Deflected shape phi at (1) prim struc for 2nd Mode 
    double phi21 = mode(2);			//Deflected shape phi at (2) tmd for 1st Mode
    double phi22 = mode(3);			//Deflected shape phi at (2) tmd for 2nd Mode 

    // Tn value determination
    T1 = (2 * pi) / omegan1;
    T2 = (2 * pi) / omegan2;

    // Cs value determination
    double Cs1{}, Cs2{}, Tb, Tc, Td;

    double xin = 0.05; //assume

    double eta = sqrt(10 / (5 + xin * 100));
    if (eta < 0.55) { eta = 0.55; }

    if (response == 1) {  //Using El Centro EQ

        Tb = 0.12;
        Tc = 0.55;
        Td = 2;

        // Cs determination for T1
        if (0 <= T1 && T1 < Tb)
            Cs1 = (4.0325 * T1 + 2.695) * 9.81;

        else if (Tb <= T1 && T1 < Tc)
            Cs1 = 0.8 * 9.81;

        else if (Tc <= T1 && T1 < Td)
            Cs1 = (0.3093 / std::pow(T1, 0.786)) * 9.81;

        else if (Td <= T1 && T1 < 4)
            Cs1 = (0.785 / std::pow(T1, 1.837)) * 9.81;

        // Cs determination for T2
        if (0 <= T2 && T2 < Tb)
            Cs2 = (4.0325 * T2 + 2.695) * 9.81;

        else if (Tb <= T2 && T2 < Tc)
            Cs2 = 0.8 * 9.81;

        else if (Tc <= T2 && T2 < Td)
            Cs2 = (0.3093 / std::pow(T2, 0.786)) * 9.81;

        else if (Td <= T2 && T2 < 4)
            Cs2 = (0.785 / std::pow(T2, 1.837)) * 9.81;
    }
    else if (response == 2) {

        // Let (soiltype == "SC")
        double S = 1.15;
        Tb = 0.15;
        Tc = 0.4;
        Td = 2;
        double Z = 0.2;  // For zone 2, Dhaka

        // Cs determination for T1
        if (0 <= T1 && T1 < Tb)
            Cs1 = S * (1 + (T1 / Tb) * (2.5 * eta - 1)) * Z * 9.81;

        else if (Tb <= T1 && T1 < Tc)
            Cs1 = 2.5 * S * eta * Z * 9.81;

        else if (Tc <= T1 && T1 < Td)
            Cs1 = 2.5 * S * eta * (Tc / T1) * Z * 9.81;

        else if (Td <= T1 && T1 < 4)
            Cs1 = 2.5 * S * eta * Tc * (Td / (T1 * T1)) * Z * 9.81;

        // Cs determination for T2
        if (0 <= T2 && T2 < Tb)
            Cs2 = S * (1 + (T2 / Tb) * (2.5 * eta - 1)) * Z * 9.81;

        else if (Tb <= T2 && T2 < Tc)
            Cs2 = 2.5 * S * eta * Z * 9.81;

        else if (Tc <= T2 && T2 < Td)
            Cs2 = 2.5 * S * eta * (Tc / T2) * Z * 9.81;

        else if (Td <= T2 && T2 < 4)
            Cs2 = 2.5 * S * eta * Tc * Td / (T2 * T2) * Z * 9.81;
    }
    else {

        std::cout << "Invalid input. response should be either 1 or 2." << std::endl;
        return 1;
    }

    //Dn determination from pseusdo-acceleration An==Cs
    D1 = Cs1 / (omegan1 * omegan1);
    D2 = Cs2 / (omegan2 * omegan2);

    //Ln determination
    L1h = M1star * phi11 + 2 * md * phi21;
    L2h = M1star * phi12 + 2 * md * phi22;

    //Mn determination
    M1 = M1star * (phi11 * phi11) + md * (phi11 * phi12 + phi21 * phi21);
    M2 = M1star * (phi12 * phi12) + md * (phi12 * phi22 + phi21 * phi22);

    //Gamma n determination
    gamma1 = L1h / M1;
    gamma2 = L2h / M2;

    //top deflection determination
    *topdef_struc = gamma1 * phi11 * D1 + gamma2 * phi12 * D2;     //top deflection for structure
    *topdef_tmd = gamma1 * phi21 * D1 + gamma2 * phi22 * D2;     //top deflection for tmd


    return 0;
}


double obj_func(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    double M1star, k1star, omega_1st, md, cd, kd, topdef_struc, topdef_tmd;
    int response = *static_cast<int*>(data);
    double mbar = x[0];

    equi_sdof(&M1star, &k1star, &omega_1st);
    top_deflection(M1star, omega_1st, k1star, mbar, md, kd, cd, response, &topdef_struc, &topdef_tmd);

    double gx = topdef_struc;

    static int iteration = 0;
    std::cout << "Iteration = " << iteration++ << "   topdef_struc = " << gx << std::endl;

    return gx;
}

// Define a single constraint function that checks all three explicit constraints
void explicit_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data)
{
    double M1star, k1star, omega_1st, md, cd, kd, topdef_struc, topdef_tmd;
    int response = *static_cast<int*>(data);
    double mbar = x[0];
    equi_sdof(&M1star, &k1star, &omega_1st);
    top_deflection(M1star, omega_1st, k1star, mbar, md, kd, cd, response, &topdef_struc, &topdef_tmd);


    double acc_reduc = 0.2; // 20~40 % .Designated acceleration reduction level γ refers to the target level
                            // of reduction in acceleration that the TMD is designed to achieve
    double cost = (16.1 * acc_reduc * acc_reduc - 6.8 * acc_reduc + 1.5) * (M1star + md) + (1.9 * acc_reduc * acc_reduc - 1.7 * acc_reduc + 2.2);

    // Calculate the constraint values
    result[0] = md - 0.05 * M1star;
    result[1] = kd - 5e6;
    result[2] = cd - 150e3;
    result[3] = cost - 2500000;

}


double obj_func1(const std::vector<double>& x1, std::vector<double>& grad, void* data)
{
    double M1star, k1star, omega_1st, md, cd, kd, topdef_struc, topdef_tmd;
    int response = *static_cast<int*>(data);
    double mbar = x1[0];

    equi_sdof(&M1star, &k1star, &omega_1st);
    top_deflection(M1star, omega_1st, k1star, mbar, md, kd, cd, response, &topdef_struc, &topdef_tmd);


    double acc_reduc = 0.2; // 20~40 % .Designated acceleration reduction level γ refers to the target level
    // of reduction in acceleration that the TMD is designed to achieve
    double cost = (16.1 * acc_reduc * acc_reduc - 6.8 * acc_reduc + 1.5) * (M1star + md) + (1.9 * acc_reduc * acc_reduc - 1.7 * acc_reduc + 2.2);

    double fx = cost*1e-4;

    static int iteration = 0;
    std::cout << "iteration = "<<iteration++ <<  "  mbar = "<<mbar << "   cost = " << cost << std::endl;

    return fx;
}

// Define a single constraint function that checks all three explicit constraints
void explicit_constraints1(unsigned m, double* result, unsigned n, const double* x1, double* grad, void* data)
{
    double M1star, k1star, omega_1st, md, cd, kd, topdef_struc, topdef_tmd;
    int response = *static_cast<int*>(data);
    double mbar = x1[0];
    equi_sdof(&M1star, &k1star, &omega_1st);
    top_deflection(M1star, omega_1st, k1star, mbar, md, kd, cd, response, &topdef_struc, &topdef_tmd);


    // Calculate the constraint values
    result[0] = md - 0.05 * M1star;
    result[1] = kd - 5e6;
    result[2] = cd - 150e3;
    result[3] = topdef_struc - 0.0762;

}

int main()
{
    double M1star, k1star, omega_1st, md, cd, kd, topdef_struc, topdef_tmd;
    int response;
    std::cout << "Enter a value for response (1 for El Centro 1940 NS seismic motion  or 2 for BNBC 2020): ";
    std::cin >> response;
    equi_sdof(&M1star, &k1star, &omega_1st);
    std::cout << "M1star: " << M1star << std::endl;
    std::cout << "k1star: " << k1star << std::endl;
    std::cout << "omega_1st: " << omega_1st << std::endl;

    // Define the upper and lower bounds of the optimization variables
    std::vector<double> lb{ 0.0 };
    std::vector<double> ub{ 0.05 };

    // Create the optimization problem for obj1 i.e. top deflection
    nlopt::opt opt(nlopt::GN_ISRES, 1);

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_maxeval(5000); // Stop after 500 iterations
    opt.set_min_objective(obj_func, &response);

    // Add a single inequality constraint that checks all three explicit constraints
    opt.add_inequality_mconstraint(explicit_constraints, &response, std::vector<double>(4, 1e-8));

    // Solve the optimization problem
    std::vector<double> x{ 0.0445326 }; // Initial guess for mbar
    double minf = 0.0;
   
        opt.optimize(x, minf);
        std::cout << "Optimal value of mbar: " << x[0] << std::endl;
        std::cout << "Minimum value of the objective functions: " << minf << std::endl;
        std::cout << "Found minimum after " << opt.get_numevals() << " function evaluations\n";


        
        double mbar = x[0];

        equi_sdof(&M1star, &k1star, &omega_1st);
        top_deflection(M1star, omega_1st, k1star, mbar, md, kd, cd, response, &topdef_struc, &topdef_tmd);

        std::cout << "mbar= " << mbar << std::endl;
        std::cout << "md= " << md << std::endl;
        std::cout << "cd= " << cd << std::endl;
        std::cout << "kd= " << kd << std::endl;
        std::cout << "topdef_struc= " << topdef_struc << std::endl;
        std::cout << "Response= " << response << std::endl;

        double mbar1 = mbar;    // optimal mbar from top deflection minimization
    // Create the optimization problem for obj2 i.e. cost
   

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_maxeval(15000); // Stop after 500 iterations
    opt.set_min_objective(obj_func1, &response);

    // Add a single inequality constraint that checks all three explicit constraints
    opt.add_inequality_mconstraint(explicit_constraints1, &response, std::vector<double>(4, 1e-8));

    // Solve the optimization problem
    std::vector<double> x1{ mbar1 }; // Initial guess for mbar used as the optimal value obtained from top deflection minimization
    double minf1 = 0.0;
   
        opt.optimize(x1, minf1);
        std::cout << "Optimal value of mbar: " << x1[0] << std::endl;
        std::cout << "Minimum value of the objective functions: " << minf << std::endl;
        std::cout << "Found minimum after " << opt.get_numevals() << " function evaluations\n";


        mbar = x1[0];

        equi_sdof(&M1star, &k1star, &omega_1st);
        top_deflection(M1star, omega_1st, k1star, mbar, md, kd, cd, response, &topdef_struc, &topdef_tmd);

        std::cout << "mbar= " << mbar << std::endl;
        std::cout << "md= " << md << std::endl;
        std::cout << "cd= " << cd << std::endl;
        std::cout << "kd= " << kd << std::endl;
        std::cout << "topdef_struc= " << topdef_struc << std::endl;
        std::cout << "Response= " << response << std::endl;


   
    return 0;
}