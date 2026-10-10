/* friedmann_eq.txt */
#define FRIEDMANN_H2(a, rho_tot, K) ((rho_tot) - (K)/((a)*(a)))
#define FRIEDMANN_H_PRIME(a, rho_tot, p_tot, K) ( - (3.0/2.0) * ((rho_tot) + (p_tot)) * (a) + (K)/(a) )
