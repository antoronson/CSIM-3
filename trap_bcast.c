#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#define PI_VAL 3.14159265

void get_Data(FILE *fs, float *a, float *b, int *n, int size, int rank) {
 
  fscanf(fs, "%f %f %d", a, b, n);
  assert(*n % size == 0);
}

float f(float x) {
  return 1 / (1 + x*x );
}

float integral(float a, float b, float h, float end) {
  float I;
  if ( b > end )	b = end;
  I = 0.5 * h * (f(a) + f(b));
  return I;
} 

double r_err(float PI_calc) {
  return PI_VAL - PI_calc;
}

double abs_err(float PI_calc) {
  return abs(100 * (PI_VAL - PI_calc) / PI_VAL);
}

int main(int argc, char* argv[]) {
  
  float a, b, h;
  int n, i;
  int rank;
  int size;
  double time1, time2;
  FILE *fp, *output;
  float Pi = 0.0, result = 0;
  
  fp = fopen(argv[1], "r");
  assert (fp != NULL);
  output = fopen("error.txt", "a");

  MPI_Init(&argc, &argv);
  time1 = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    get_Data(fp, &a, &b, &n, size, rank);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&a, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  
  h = (b - a) / n;
  for (i = 0; i < n; i = i + size)
    Pi = Pi + integral(a + (rank + i) * h, a + (rank + 1 + i) * h, h, b);
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&Pi, &result, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  time2 = MPI_Wtime();
  if (rank == 0) {
    fclose(fp);
    printf(" This is Process %d \n The value of Pi is: %f\n", rank, 4 * result);
    printf(" For %d trapezoidal steps:\n Relative error = %f \n Absolute Error = %f\n", n, r_err(4 * result), abs_err(4 * result));
    
    fprintf(output, " This is Process %d \n The value of Pi is: %f\n", rank, 4 * result);
    fprintf(output, " For %d trapezoidal steps:\n Relative error = %lf \n Absolute Error = %lf \n", n, r_err(4 * result), abs_err(4 * result));
    fprintf(output, " For %d process time taken = %lf\n", size, time2 - time1);
    fclose(output);
  } 
 
  MPI_Finalize();
  return 0;
}
