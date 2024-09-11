#include <stdio.h>     
#include <stdlib.h>
#include <complex.h>   
#include <fftw3.h>
#include <math.h>
#define PI 3.14159265358979323846

void read_complex_matrix_from_text(const char* file_name, double** real_part, double** imag_part, int m, int n) {
    FILE *file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "Error opening file \"%s\"!\n", file_name);
        exit(1);
    }

    *real_part = (double*)malloc(sizeof(double) * m * n);
    *imag_part = (double*)malloc(sizeof(double) * m * n);
    if (!(*real_part) || !(*imag_part)) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    for (int i = 0; i < m * n; i++) {
        fscanf(file, "%lf %lf", &(*real_part)[i], &(*imag_part)[i]);
    }

    fclose(file);
}

void save_complex_matrix_to_text(const char* file_name, fftw_complex *data, int m, int n) {
    FILE *file = fopen(file_name, "w");
    if (!file) {
        fprintf(stderr, "Error creating file \"%s\"!\n", file_name);
        return;
    }

    for (int i = 0; i < m * n; i++) {
        fprintf(file, "%f %f\n", creal(data[i]), cimag(data[i]));
    }

    fclose(file);
}

// Function to initialize FFT plans and perform forward FFT
void perform_fft(fftw_complex *u0, fftw_complex *h, int mm, int nn, fftw_plan *p_fwd_u0, fftw_plan *p_fwd_h) {
    *p_fwd_u0 = fftw_plan_dft_2d(mm, nn, u0, u0, FFTW_FORWARD, FFTW_ESTIMATE);
    *p_fwd_h = fftw_plan_dft_2d(mm, nn, h, h, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(*p_fwd_u0);
    fftw_execute(*p_fwd_h);
}

// Function to perform inverse FFT and normalize the result
void perform_ifft_and_normalize(fftw_complex *C, int mm, int nn, fftw_plan *p_bwd) {
    *p_bwd = fftw_plan_dft_2d(mm, nn, C, C, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(*p_bwd);
    for (int i = 0; i < mm * nn; i++) {
        C[i] /= (mm * nn);
    }
}

void fftshift(fftw_complex* data, int width, int height) {
    int half_width = (width + 1) / 2;
    int half_height = (height + 1) / 2;
    
    fftw_complex* temp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * width * height);
    if (temp == NULL) {
        fprintf(stderr, "Memory allocation failed for fftshift temporary buffer.\n");
        return;
    }

    // Rearrange the quadrants of the array
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int new_i = (i + half_height) % height;
            int new_j = (j + half_width) % width;
            temp[new_i * width + new_j] = data[i * width + j];
        }
    }

    // Copy the temporary array back to the original data array
    for (int i = 0; i < width * height; i++) {
        data[i] = temp[i];
    }

    fftw_free(temp);
}

void generate_coords(double* coordsx, double* coordsy, int M, double pixel) {
    int half_M = M / 2;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            coordsx[i * M + j] = (j - half_M) * pixel;
            coordsy[i * M + j] = (i - half_M) * pixel;
        }
    }
}

void create_h_matrix(fftw_complex* h, double* coordsx, double* coordsy, int M, double lambda, double z) {
    double k = 2 * PI / lambda;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int index = i * M + j;
            double rSq = coordsx[index] * coordsx[index] + coordsy[index] * coordsy[index] + z * z;
            h[index] = z * cexp(I * k * sqrt(rSq)) / (I * lambda * rSq);
        }
    }
}

// Function to propagate the signal
void propagate(fftw_complex *u0, fftw_complex *h, fftw_complex *C, int m, int n) {
    fftw_plan p_fwd_u0, p_fwd_h, p_bwd;

    fftshift(u0, m, n);
    fftshift(h, m, n);
    perform_fft(u0, h, m, n, &p_fwd_u0, &p_fwd_h); //FFT forward
    fftshift(u0, m, n);
    fftshift(h, m, n);

    // multiplication in frequency domain
    for (int i = 0; i < m * n; i++) {
        C[i] = u0[i] * h[i];
    }

    fftshift(C, m, n);
    perform_ifft_and_normalize(C, m, n, &p_bwd); // IFFT
    fftshift(C, m, n);

    // Clean up
    fftw_destroy_plan(p_fwd_u0);
    fftw_destroy_plan(p_fwd_h);
    fftw_destroy_plan(p_bwd);
}

void init_fft_u(fftw_complex *u0, double *u0_real_part, double *u0_imag_part, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            u0[index] = u0_real_part[i * n + j] + u0_imag_part[i * n + j] * I;
        }
    }
}

int main() {
    const char* file_name = "source_500.txt";
    const char* slm_name = "spatial_modulation_500_linear.txt";
    const char* output_file = "out_500_test.txt";

    double lambda = 0.6e-6;   // wavelength m
    double z = 1.0;           // propagation distance m
    double L1 = 8e-3;         // real size of array: 1 cm
    int M = 500;             // number of pixels
    double pixel = L1 / M;

    int m = M; // Number of rows
    int n = M; // Number of columns

    double* u_real;
    double* u_imag;
    double* sp_real;
    double* sp_imag;

    // Read the source matrix from the text file
    read_complex_matrix_from_text(file_name, &u_real, &u_imag, m, n);
    read_complex_matrix_from_text(slm_name, &sp_real, &sp_imag, m, n);

    // Prepare complex arrays for FFT
    fftw_complex *u0 = fftw_malloc(sizeof(fftw_complex) * m * n);
    fftw_complex *h = fftw_malloc(sizeof(fftw_complex) * m * n);
    fftw_complex *C = fftw_malloc(sizeof(fftw_complex) * m * n);
    fftw_complex *spatial = fftw_malloc(sizeof(fftw_complex) * m * n);
    fftw_complex *C1 = fftw_malloc(sizeof(fftw_complex) * m * n);
    fftw_complex *C2 = fftw_malloc(sizeof(fftw_complex) * m * n);

    // Generate impulse response matrix h
    double* coordsx = (double*)malloc(sizeof(double) * M * M);
    double* coordsy = (double*)malloc(sizeof(double) * M * M);
    generate_coords(coordsx, coordsy, M, pixel);
    create_h_matrix(h, coordsx, coordsy, M, lambda, z);

    // Initialize u0 
    init_fft_u(u0, u_real, u_imag, m, n);
    init_fft_u(spatial, sp_real, sp_imag, m, n);

    // Propagate the signal
    propagate(u0, h, C, m, n);

    // Phase modulation
    for (int i = 0; i < m * n; i++) {
        C1[i] = C[i] * spatial[i];
    }

    propagate(C1, h, C2, m, n);

    // Save the result 
    save_complex_matrix_to_text(output_file, C2, m, n);

    // Clean up
    fftw_free(u0);
    fftw_free(h);
    fftw_free(C);
    fftw_free(spatial);
    fftw_free(C1);
    fftw_free(C2);
    free(u_real);
    free(u_imag);
    free(sp_real);
    free(sp_imag);
    free(coordsx);
    free(coordsy);

    return 0;
}
