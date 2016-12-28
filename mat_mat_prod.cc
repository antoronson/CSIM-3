/* Matrix multiplication using openMPI 
 * Author : Anto Ronson
 * 
 * Input --> 
 *       # Two files containing the matrices with row column format_spacing 
 *         as the first line information.
 *       # All process does I/O function
 * Limitations --> 
 *       # The program fails if incorrect information is provided in the 
 *         first line of the input files.
 *       # The program fails if the input files are not formatted using
 *         uniform spacing, since the program heavily relies on it.
 *       
 * Description -->
 *       # > In the first section, the data from the files
 *         are stored in the array of each process.
 *         > An offset for the file pointer is set for 
 *         each process, to read data from file by all process.
 *         > The data from matrixA is stored as row blocks.
 *         > The data from matrixB is stored as column block.
 *         > Since c++ is a row major memory access type, this
 *         can have significant effect on the process speed.
 *         > Each process now holds a row block of matrixA and 
 *         a column block of matrixB. 
 *       # > In the second section the row x column multiplication
 *         is carried out.
 *         > Once every process completes its first cycle, the column
 *         data of each process is passed in a ring formation. 
 *         > This cylcle is repeated until all process completes 
 *         multiplication with all columns of matrix B.
 *       # > In the final section the calculated data is printed
 *         into a file.
 *         > An offset is again set for each process to printed 
 *         its own copy of matrix C block into the file, in the 
 *         allocated space.
 * 
 * Output -->
 *       # Resultant matrix (Matrix C) 
 *       # Time -> Total_time  Computation_time  MPI_Send-Recv-time
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <cassert>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {
  assert(argc == 3);
  
  int          rank;
  int          size;
  int          row_A, column_A, width_A;
  int          row_B, column_B, width_B;
  int          m_block, n_block;
  double*      mat_A;
  double*      mat_B;
  double*      mat_C;
  double       st_tot_time, end_tot_time;
  double       st_calc_time, end_calc_time;
  double       st_sendrecv_time, end_sendrecv_time;
  double       sendRecv_time = 0, calc_time = 0;
  ifstream     ifp_matA, ifp_matB; 
  ofstream     ofp_matC, ofp_time;
  MPI_Status   status;

  MPI_Init(&argc, &argv);
  
  st_tot_time = MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  /****************    Reading Data from File  **************/
  ifp_matA.open(argv[1]);  assert(ifp_matA);
  ifp_matB.open(argv[2]);  assert(ifp_matB);
  
  ifp_matA >> row_A >> column_A >> width_A;
  ifp_matB >> row_B >> column_B >> width_B;
  assert(column_A == row_B);
  assert(row_A % size == 0 && column_B % size == 0);
  
  m_block = row_A / size;
  n_block = column_B / size;
  
  mat_A = new double[m_block * column_A];
  mat_B = new double[row_B * n_block];
  mat_C = new double[m_block * column_B];
  
  // sets offset for the file pointer on each process
  ifp_matA.seekg(width_A * (rank * m_block * column_A + 3)
                         + 1 + rank * m_block, ios_base::beg);
  // Each process reads in its own copy of matrix A block
  for (int i = 0; i < m_block * column_A; ++i)
    ifp_matA >> mat_A[i];
  
  // Each process reads in its own copy of matrix B block
  // The array is filled with column data rather than row data
  for (int i = 0, k = 0; i < n_block; ++i) {
    for (int j = 0; j < row_B; ++j, ++k) {
      ifp_matB.seekg(width_B * (rank * n_block + 3 + j * column_B + i)
                             + j + 1, ios_base::beg);
      ifp_matB >> mat_B[k]; 
    }
  }
  /*********************************************************************/
  /*************** Matrix Multiplication *******************************/  
  
  // multiplication & cyclic exchange of matrixB block
  
  for (int cycle = 0; cycle < size; ++cycle) {
    st_calc_time = MPI_Wtime();
    for (int i = 0; i < m_block; ++i) {
      for (int j = 0; j < n_block; ++j) {
        int cIndex = i * column_B + j + n_block * ((rank + cycle) % size); 
        mat_C[cIndex] = 0;
        for (int k = 0; k < row_B; ++k)
          mat_C[cIndex] += mat_A[i * column_A + k] * mat_B[j * row_B + k];
      }
    }
    
    end_calc_time = MPI_Wtime();
    calc_time += end_calc_time - st_calc_time;
    
    st_sendrecv_time = MPI_Wtime();
    int left = (rank == 0)? size - 1 : rank - 1;
    int right = (rank + 1) % size;
    if (rank % 2 == 0) {
      MPI_Send(mat_B, row_B * n_block, MPI_DOUBLE, left, 0, MPI_COMM_WORLD);
      MPI_Recv(mat_B, row_B * n_block, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &status);
    }
    else {
      MPI_Recv(mat_B, row_B * n_block, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &status);
      MPI_Send(mat_B, row_B * n_block, MPI_DOUBLE, left, 1, MPI_COMM_WORLD);
    }
    end_sendrecv_time = MPI_Wtime();
    sendRecv_time += end_sendrecv_time - st_sendrecv_time;
  }
  /*************************************************************************/
  /********************** Output *******************************************/
  
  ofp_matC.open("matrixC.dat", ios_base::out);  assert(ofp_matC);
  
  if (rank == 0) {
    ofp_matC << setw(width_A) << m_block * size << setw(width_A) << column_B;
    ofp_matC << setw(width_A) << width_A << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // output of A * B into matixC.dat Each process is assigned an offset for the file pointer
  ofp_matC.seekp(width_A * (rank * m_block * column_B + 3) + 1 + rank * m_block, ios_base::beg);
  for (int i = 0; i < m_block * column_B; ++i) {
    ofp_matC <<setw(width_A)<<mat_C[i];
    if ((i+1) % column_B == 0)
      ofp_matC << endl;
  }
  end_tot_time = MPI_Wtime();
  if (rank == 0) {
    ofp_time.open("time_out.dat",ios_base::app);  assert(ofp_time);
    ofp_time << setw(width_A) << end_tot_time - st_tot_time << setw(width_A) << calc_time;
    ofp_time << setw(width_A) << sendRecv_time << endl;
    ofp_time.close();
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  ifp_matA.close();
  ifp_matB.close();
  ofp_matC.close();
  delete[] mat_A;
  delete[] mat_B;
  delete[] mat_C;
  MPI_Finalize();
}
  

