// #include diffusion solver header
#include "mfem.hpp"
#include "src/integrate.hpp"
#include <iostream>


// new line

using namespace std;
using namespace mfem;

real_t inflow_function(const Vector &x){
      return 0.0;



real_t FC_opacity(const Vector &T ){
    return 0.0;
}

real_t Planck_opacity(const Vector &T){
    return
}

struct group_structure {
    vector<real_t> bounds ;
};

vector<real_t> group_bounds (group_structure groups, int index) {
    vector<real_t> gb;
    gb.push_back (groups.bounds[index]  );
    gb.push_back (groups.bounds[index+1]);
    return gb;
}


int main() {

    FunctionCoefficient sigma(inflow_function);

    cout << integrate()<<endl;
    group_structure fc17g = {{0.0, 1.0, 11, 15}};
    vector<real_t> slice = group_bounds(fc17g, 1);
    
    for (auto i : slice) {
        cout << i << endl;
    }

    // set up material data

    // define group structure


    // input file should include:
        // 

    // initialize solver (ICs, guesses)

    // diffusion.solve()

    // read output

    cout << "working"<< endl;
    return 0;
}