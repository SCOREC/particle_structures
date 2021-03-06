#ifndef PSTYPES_H_
#define PSTYPES_H_

#include <MemberTypes.h>

namespace particle_structs {

#ifdef FP64
typedef double fp_t;
#endif
#ifdef FP32
typedef float fp_t;
#endif

typedef int lid_t;

typedef fp_t Vector3d[3];

//Particle = <current position vector, pushed position vector>
typedef MemberTypes<Vector3d, Vector3d> Particle;

class elemCoords {
  public:
  int num_elems;
  int verts_per_elem;
  int size;
  fp_t* x;
  fp_t* y;
  fp_t* z;
  elemCoords(int ne, int np, int size);
  ~elemCoords();
  private:
    elemCoords() {};
};

}

#endif
