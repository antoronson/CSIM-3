/*
# hermitian matrix of nxn dimension is created using the matlab script
# 
*/

#include <iostream>
#include <cassert>
#include <fstream>
#include <iterator>
#include <tuple>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <mpi.h>

using namespace std;

tuple<double*, int, int, int> getData(string file1, int size, int rank) {
  
  ifstream    fp_mat;
  int         m, n;
  int         block;
  double      *matA;
  MPI_Status  status;
  
  fp_mat.open(file1);   assert(fp_mat);
  fp_mat >> m >> n;
  assert ( m % size == 0);
  block = m / size;
  matA     = new double[n * block];
  if (rank > 0)
    MPI_Recv(matA, n * block, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
   
  else {
    for (int k = 0; k < n * block; ++k)
      fp_mat >> matA[k];
    double  *temp = new double[n * block];
    for (int dest = 1; dest < size; ++dest) {
      for (int k = 0; k < n * block; ++k)
        fp_mat >> temp[k];
      MPI_Send(temp, n * block, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    }
    delete[] temp;
  }
  fp_mat.close();
  return make_tuple(matA, m, n, block);
}

double dotprod(const double vec_1[], const double vec_2[], int n, int block = 0){
  double y_local = 0;
  for (int j = 0; j < n; ++j)
    y_local += vec_1[block * n + j] * vec_2[j];
  return y_local;
}

void randomGenerator(double x_0[], int n) {
  
  for (int i = 0; i < n; ++i)
      x_0[i] = drand48();
}

int main(int argc, char** argv) {
  assert(argc == 2);

  int         rank , size;
  int         m, n, block;
  double      loc_alpha, alpha;
  double      loc_beta,  beta = 0;
  double      *matA;
  double      *y_local;
  double      *x_0, *x_1, *x_1_local;
  double      vecNorm;
  double      tstart_total, tend_total, time_tot;
  double      tstart_calc, tend_calc, time_calc;
  double      tstart_comm, tend_comm, time_comm; 
  ofstream    fp_out, fp_time;
  
  MPI_Init(&argc, &argv);
  
  tstart_total = MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  tstart_comm      = MPI_Wtime();
  matA             = new double;
  tie(matA, m, n, block) = getData(argv[1], size, rank);
  MPI_Barrier(MPI_COMM_WORLD);
  tend_comm        = MPI_Wtime();
  x_0              = new double[n];
  x_1              = new double[n];
  x_1_local        = new double[block];
  y_local          = new double[block];
  tstart_calc      = MPI_Wtime();
  srand48(time(0));
  do {
    randomGenerator(x_0, n);
    vecNorm = sqrt(dotprod(x_0, x_0, n));
  }while( vecNorm == 0 );
  
  for (int i = 0; i < n; ++i)      x_0[i] /= vecNorm;
  for (int i = 0; i < block; ++i)
    y_local[i] = dotprod(matA, x_0, n, i);

  loc_alpha = dotprod(&x_0[rank * (block - 1)], y_local, block);
  MPI_Allreduce(&loc_alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    fp_out.open("mat_prod_out.dat");
    fp_out << setw(15) << "alpha" << setw(15) << "beta" << endl;
    fp_out << setw(15) << alpha << setw(15) << beta << endl;
  }
  for (int i = 0; i < block; ++i)
    x_1_local[i] = y_local[i] - alpha * x_0[rank * (block - 1)  + i];
  loc_beta = sqrt(dotprod(x_1_local, x_1_local, block));  beta = 0;
  MPI_Allreduce(&loc_beta, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
  assert(beta != 0 || beta != beta);
  
  for (int i = 0; i < n - 1; ++i) {
    
    for (int j = 0; j < block; ++j)   x_1_local[j] /= beta;
    MPI_Allgather(x_1_local, block, MPI_DOUBLE, x_1, block, MPI_DOUBLE, MPI_COMM_WORLD);
    
    for (int j = 0; j < block; ++j)
      y_local[j] = dotprod(matA, x_1, n, j) - beta * x_0[rank * (block - 1) + j];
    
    loc_alpha = dotprod(&x_1[rank * (block - 1)], y_local, block);  alpha = 0;
    MPI_Allreduce(&loc_alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)   fp_out << setw(15) << alpha << setw(15) << beta << endl;
    
    for (int j = 0; j < block; ++j)   x_0[j] = x_1[j];
    
    for (int j = 0; j < block; ++j)
      x_1_local[j] = y_local[j] - alpha * x_1[rank * (block - 1)  + i];
    
    loc_beta = sqrt(dotprod(x_1_local, x_1_local, block));  beta = 0;
    MPI_Allreduce(&loc_beta, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
    assert(beta != 0);
  }
  if (rank == 0)  fp_out.close();
  tend_calc        = MPI_Wtime();
  tend_total = MPI_Wtime();
  
  time_tot  = tend_total - tstart_total;
  time_comm = tend_comm - tstart_comm;
  time_calc = tend_calc - tstart_calc;
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  delete[] x_0;
  delete[] x_1;
  delete[] matA;
  delete[] y_local;
  
  if (rank == 0) {
    fp_time.open("time_ouput.dat", ofstream::app);
    fp_time << time_tot << setw(15) << time_comm << setw(15) << time_calc << endl;
    fp_time.close();
  }
 
  MPI_Finalize();
  return 0;
}
