#pragma once

#include "MemberTypeArray.h"
#include "SCS_Macros.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdlib>

namespace particle_structs {

  template <typename T> struct MpiType;
#define CREATE_MPITYPE(type, mpi_type) \
  template <> struct MpiType<type> { \
    static constexpr int mpitype = mpi_type; \
  }
  CREATE_MPITYPE(char, MPI_CHAR);
  CREATE_MPITYPE(short, MPI_SHORT);
  CREATE_MPITYPE(int, MPI_INT);
  CREATE_MPITYPE(long, MPI_LONG);
  CREATE_MPITYPE(unsigned char, MPI_UNSIGNED_CHAR);
  CREATE_MPITYPE(unsigned short, MPI_UNSIGNED_SHORT);
  CREATE_MPITYPE(unsigned int, MPI_UNSIGNED);
  CREATE_MPITYPE(unsigned long int, MPI_UNSIGNED_LONG);
  CREATE_MPITYPE(float, MPI_FLOAT);
  CREATE_MPITYPE(double, MPI_DOUBLE);
  CREATE_MPITYPE(long double, MPI_LONG_DOUBLE);
  CREATE_MPITYPE(long long int, MPI_LONG_LONG_INT);

#undef CREATE_MPI_TYPE

  //This type represents an array of views for each type of the given DataTypes
  template <typename DataTypes> using MemberTypeViews = void*[DataTypes::size];
  template <typename DataTypes> using MemberTypeViewsConst = void* const*;
  //TODO don't use default execution space
  template <typename T> using MemberTypeView = 
    Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>;

  template <typename... Types> struct CreateViewsImpl;
  template <> struct CreateViewsImpl<> {
    CreateViewsImpl(MemberTypeViews<MemberTypes<void> >, int) {}
  };
  template <typename T, typename... Types> struct CreateViewsImpl<T, Types...> {
    CreateViewsImpl(MemberTypeViews<MemberTypes<T, Types...> > views, int size) {
      
      views[0] = new MemberTypeView<T>("datatype_view", size);
      MemberTypeView<T> view = *static_cast<MemberTypeView<T>*>(views[0]);
      CreateViewsImpl<Types...>(views+1, size);
    }
  };

  template <typename... Types> struct CreateViews;
  template <typename... Types> struct CreateViews<MemberTypes<Types...> > {
    CreateViews(MemberTypeViews<MemberTypes<Types...> >& views, int size) {
      CreateViewsImpl<Types...>(views, size);
    }
  };

  //TODO Make these functions device&host functions
  template <class T> struct CopyArrayToViewActual {
    CopyArrayToViewActual(MemberTypeView<T> v, int view_index, T* array, int array_index) {
      v(view_index) = array[array_index];
    }
  };
  template <class T, int N> struct CopyArrayToViewActual<T[N]> {
    CopyArrayToViewActual(MemberTypeView<T[N]> v, int view_index, T (*array)[N], int array_index) {
      for (int i = 0; i < N; ++i)
        v(view_index, i) = array[array_index][i];
    }
  };
  template <class T, int N, int M> struct CopyArrayToViewActual<T[N][M]> {
    CopyArrayToViewActual(MemberTypeView<T[N][M]> v, int view_index, T (*array)[N][M], int array_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          v(view_index, i,j) = array[array_index][i][j];
    }
  };
  template <class T, int N, int M, int P> struct CopyArrayToViewActual<T[N][M][P]> {
    CopyArrayToViewActual(MemberTypeView<T[N][M][P]> v, int view_index, T (*array)[N][M][P], 
                          int array_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          for (int k = 0; k < P; ++k)
            v(view_index, i, j, k) = array[array_index][i][j][k];
    }
  };
 
  template <typename... Types> struct CopyArrayToViewImpl;
  template <> struct CopyArrayToViewImpl<> {
    CopyArrayToViewImpl(MemberTypeViewsConst<MemberTypes<void> > views, int view_index, 
                    MemberTypeArray<MemberTypes<void> > arrays, int array_index) {}
  };
  template <typename T, typename... Types> struct CopyArrayToViewImpl<T,Types...> {
    CopyArrayToViewImpl(MemberTypeViewsConst<MemberTypes<T,Types...> > views, int view_index, 
                        MemberTypeArray<MemberTypes<T, Types...> > arrays, int array_index) {
      MemberTypeView<T> v = *static_cast<MemberTypeView<T> const*>(views[0]);
      T* arr = static_cast<T*>(arrays[0]);
      CopyArrayToViewActual<T>(v, view_index, arr, array_index);
      CopyArrayToViewImpl<Types...>(views+1, view_index, arrays+1, array_index);
    }
  };
  template <typename... Types> struct CopyArrayToView;
  template <typename... Types> struct CopyArrayToView<MemberTypes<Types...> > {
    CopyArrayToView(MemberTypeViewsConst<MemberTypes<Types...> > views, int view_index, 
                    MemberTypeArray<MemberTypes<Types...> > arrays, int array_index) {
      CopyArrayToViewImpl<Types...>(views, view_index, arrays, array_index);
    }
  };

  template <typename... Types> struct SendViewsImpl;
  template <> struct SendViewsImpl<> {
    SendViewsImpl(MemberTypeViews<MemberTypes<void> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {}
  };
  template <typename T, typename... Types> struct SendViewsImpl<T, Types... > {
    SendViewsImpl(MemberTypeViews<MemberTypes<T, Types...> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {
      MemberTypeView<T> v = *static_cast<MemberTypeView<T>*>(views[0]);
      typename BaseType<T>::type* data = v.data();
      int size_per_entry = BaseType<T>::size;
      MPI_Datatype mpi_type = MpiType<typename BaseType<T>::type>::mpitype;
      MPI_Isend(data + offset, size_per_entry * size, mpi_type, dest, tag, MPI_COMM_WORLD, reqs);
      SendViewsImpl<Types...>(views+1, offset, size, dest, tag + 1, reqs + 1);
    }
  };

  template <typename... Types> struct SendViews;
  template <typename... Types> struct SendViews<MemberTypes<Types...> > {
    SendViews(MemberTypeViews<MemberTypes<Types...> > views, int offset, int size, 
              int dest, int start_tag, MPI_Request* reqs) {
      SendViewsImpl<Types...>(views, offset, size, dest, start_tag, reqs);
    }
  };

  template <typename... Types> struct RecvViewsImpl;
  template <> struct RecvViewsImpl<> {
    RecvViewsImpl(MemberTypeViews<MemberTypes<void> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {}
  };
  template <typename T, typename... Types> struct RecvViewsImpl<T, Types... > {
    RecvViewsImpl(MemberTypeViews<MemberTypes<T, Types...> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {
      MemberTypeView<T> v = *static_cast<MemberTypeView<T>*>(views[0]);
      typename BaseType<T>::type* data = v.data();
      int size_per_entry = BaseType<T>::size;
      MPI_Datatype mpi_type = MpiType<typename BaseType<T>::type>::mpitype;
      MPI_Irecv(data + offset, size_per_entry * size, mpi_type, dest, tag, MPI_COMM_WORLD, reqs);
      RecvViewsImpl<Types...>(views+1, offset, size, dest, tag + 1, reqs + 1);
    }
  };

  template <typename... Types> struct RecvViews;
  template <typename... Types> struct RecvViews<MemberTypes<Types...> > {
    RecvViews(MemberTypeViews<MemberTypes<Types...> > views, int offset, int size, 
              int dest, int start_tag, MPI_Request* reqs) {
      RecvViewsImpl<Types...>(views, offset, size, dest, start_tag, reqs);
    }
  };

  //Implementation to deallocate views of different types
  template <typename... Types> struct DestroyViewsImpl;
  template <> struct DestroyViewsImpl<> {
    DestroyViewsImpl(MemberTypeView<MemberTypes<void> >) {}
  };
  template <typename T, typename... Types> struct DestroyViewsImpl<T,Types...> {
    DestroyViewsImpl(MemberTypeView<MemberTypes<T,Types...> > data) {
      delete static_cast<MemberTypeView<T>*>(data[0]);
      DestroyViewsImpl<Types...>(data+1);
    }
  };

  //Call to deallocate arrays of different types
  template <typename... Types> struct DestroyViews;
  template <typename... Types> struct DestroyViews<MemberTypes<Types...> > {
    DestroyViews(MemberTypeView<MemberTypes<Types...> > data) {
      DestroyViewsImpl<Types...>({data});
    }
  };

}
