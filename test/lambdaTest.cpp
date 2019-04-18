#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>
#include <SCS_Macros.h>

#include <psAssert.h>
#include <Distribute.h>

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;


typedef MemberTypes<int> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  Kokkos::TeamPolicy<exe_space> po(4, 4);
  SellCSigma<Type, exe_space>* scs =
    new SellCSigma<Type, exe_space>(po, 5, 2, ne, np, ptcls_per_elem, ids, true);

  scs->transferToDevice();

  auto lamb = SCS_LAMBDA(const int& eid, const int& pid, const int& mask) {
    if (mask > 0)
      printf("SECOND: %d %d\n", eid, pid);
  };

  scs->parallel_for(lamb);

  delete scs;
  Kokkos::finalize();
  printf("All tests passed\n");
  return 0;
}
