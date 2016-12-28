#include <iomanip>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <limits>

using namespace std;

int main(int argc, char** argv) {
  int m = 2048, n = 2048, l = 2048;
  const int width = 15;
  ofstream fp;
  fp.open("matrixA.dat");
  srand48(time(0));
  fp << setw(width) << m << setw(width) << n;
  fp << setw(width) << width << endl;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      fp << setw(15) << drand48();
    fp << endl;
  }
  fp.close();
  fp.open("matrixB.dat");
  fp << setw(width) << n << setw(width) << l;
  fp << setw(width) << width << endl;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      fp << setw(15) << drand48();
    fp << endl;
  }
  fp.close();
  return 0;
}

/*
  
class matrix {
private:
  string fname;
  int m = 2048;
  int n = 2048;
public:
  matrix(string fname_);
};

class vec {
private:
  string fname;
  int n = 2048;
public:
  vec(string fname_);
};
 
*/