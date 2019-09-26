#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>

#include <psAssert.h>
#include <Distribute.h>

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_elements;
using particle_structs::getLastValue;
using particle_structs::lid_t;

typedef MemberTypes<int> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef SellCSigma<Type,exe_space> SCS;

bool resortElementsTest(const int ne, const int sigma, const int sliceSz) {
  int np = 20;
  particle_structs::gid_t* gids = new particle_structs::gid_t[ne];
  distribute_elements(ne, 0, 0, 1, gids);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
 
  const auto chunkSz=32; 
  Kokkos::TeamPolicy<exe_space> po(1024, chunkSz);
  SCS::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
  SCS::kkGidView element_gids_v("element_gids_v", ne);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  particle_structs::hostToDevice(element_gids_v, gids);

  SCS* scs = new SCS(po, sigma, sliceSz, ne, np, ptcls_per_elem_v, element_gids_v);
  delete [] ptcls_per_elem;
  delete [] ids;
  delete [] gids;

  auto values = scs->get<0>();

  SCS::kkLidView new_element("new_element", scs->capacity());
  //Remove all particles from first element
  auto moveParticles = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    if (mask) {
      values(ptcl_id) = elm_id;
      if (ptcl_id % 4 == 0 && ptcl_id < 8)
        new_element(ptcl_id) = -1;
      else
        new_element(ptcl_id) = elm_id;
    }
  };
  scs->parallel_for(moveParticles);
  scs->rebuild(new_element);


  values = scs->get<0>();
  SCS::kkLidView fail("", 1);
  auto checkParticles = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
    if (mask) {
      if (values(ptcl_id) != elm_id) {
        fail(0) = 1;
      }
    }
  };
  scs->parallel_for(checkParticles);

  if (getLastValue<lid_t>(fail) == 1) {
    printf("Value mismatch on some particles\n");
    return false;
  }
  delete scs;
  return true;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  
  bool passed = true;
  bool sortOn[2] = {0,1};
  auto sliceSz=64;
  for(int i=0; i<2; i++) {
    int sigma = sortOn[i] ? INT_MAX : 1;
    for(int ne=1024; ne<=(1*1024*1024); ne*=2) {
      fprintf(stderr,"ne %d sigma %d sliceSz %d\n", ne, sigma, sliceSz);
      if (!resortElementsTest(ne,sigma,sliceSz)) {
        passed = false;
        printf("[ERROR] resortElementsTest() failed\n");
      }
    }
  }

  Kokkos::finalize();
  MPI_Finalize();
  if (passed)
    printf("All tests passed\n");
  return 0;
}
