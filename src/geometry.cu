#include <math.h>
#include "geometry.h"

CUDA_CALLABLE_MEMBER void eigen3(double* matrix, double* eig_val, double* eig_vec) {

    // test if matrix is symmetric
    assert(matrix[1] == matrix[3]);
    assert(matrix[2] == matrix[6]);
    assert(matrix[5] == matrix[7]);

    // test if matrix is diagonal
    double p1 = matrix[1] * matrix[1] + matrix[2] * matrix[2] + matrix[5] * matrix[5];
    if (p1 == 0) {

        eig_val[0] = matrix[0];
        eig_val[1] = matrix[4];
        eig_val[2] = matrix[8];
    } else {    // computation
        double trace = matrix[0] + matrix[4] + matrix[8];
        double q = trace / 3.0;
        double p2 = (matrix[0] - q) * (matrix[0] - q)
                    + (matrix[4] - q) * (matrix[4] - q)
                    + (matrix[8] - q) * (matrix[8] - q)
                    + 2.0 * p1;
        double p = sqrt(p2 / 6.0);
        double I[] = {1.,0.,0.,0.,1.,0.,0.,0.,1};
        double B[9];
        for (int i = 0; i < 9; i++) {
            B[i] = (1/p) * (matrix[i] - q * I[i]);
        }
        double r = det3(B) / 2.0;
        double phi;
        if (r <= -1) {
            phi = M_PI / 3;
        } else {
            if (r >= 1) {
                phi = 0;
            } else {
                phi = acos(r) / 3.0;
            }
        }

        eig_val[0] = q + 2 * p * cos(phi);
        eig_val[2] = q + 2 * p * cos(phi + (2*M_PI/3.0));
        eig_val[1] = 3 * q - eig_val[0] - eig_val[2];
    }

    assert(eig_val[0]!=eig_val[1]);
    assert(eig_val[1]!=eig_val[2]);
    assert(eig_val[0]!=eig_val[2]);

    for (int k = 0; k < 3; k++) {
        double I[] = {1.,0.,0.,0.,1.,0.,0.,0.,1};
        double C[9];
        for (int i = 0; i < 9; i++) {
            C[i] = (matrix[i] - eig_val[k] * I[i]);
        }

        //compute the cross product of two rows
        eig_vec[3*k+0] = C[1]*C[5] - C[2]*C[4];
        eig_vec[3*k+1] = C[2]*C[3] - C[0]*C[5];
        eig_vec[3*k+2] = C[0]*C[4] - C[1]*C[3];

        norm(&eig_vec[3*k], 3);
    }

}

CUDA_CALLABLE_MEMBER void norm(double *vector, int size) {

    double norm = 0;

    for (int i = 0; i < size; i++) {
        norm += vector[i] * vector[i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < size; i++) {
        vector[i] /= norm;
    }

}

CUDA_CALLABLE_MEMBER double det3(double *matrix) {

    return matrix[0] * ((matrix[4]*matrix[8])-(matrix[7]*matrix[5]))
            - matrix[1] * ((matrix[3]*matrix[8])-(matrix[6]*matrix[5]))
            + matrix[2] * ((matrix[3]*matrix[7])-(matrix[6]*matrix[4]));

}