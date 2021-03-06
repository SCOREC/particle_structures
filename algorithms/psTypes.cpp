#include "psTypes.h"

namespace particle_structs {

elemCoords::elemCoords(int ne, int np, int s) {
  num_elems = ne;
  verts_per_elem = np;
  size = s*np;
  x = new fp_t[s*np];
  y = new fp_t[s*np];
  z = new fp_t[s*np];
}

elemCoords::~elemCoords() {
  delete [] x;
  delete [] y;
  delete [] z;
}

}
