#Setting Version Number, Project Name
cmake_minimum_required(VERSION 3.7.0...${CMAKE_VERSION})
project(particleStructs VERSION 0.0.1 LANGUAGES CXX)

include(cmake/bob.cmake)
bob_begin_package()

set(CMAKE_MODULE_PATH 
   ${CMAKE_MODULE_PATH} 
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#Settings options for testing
enable_testing()
include(CTest)


option(PS_DEBUG "Turn on for debug output" OFF)
option(PS_OPT "Turn on for debug output" OFF)

if (PS_DEBUG_PREPROC)
   add_definitions(-DDEBUG)
endif()

set(debug_flag)
if (PS_ENABLE_DEBUG_SYMBOLS)
  set(debug_flag "-g")
endif()

set(opt_flag)
if (PS_ENABLE_OPT)
  set(opt_flag "-O3")
endif()

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

if (ENABLE_KOKKOS)
  set(KOKKOS_ENABLED true)
  set(particleStructs_USE_Kokkos_DEFAULT ON)
  find_package(Kokkos 3.0 CONFIG REQUIRED)
  #bob_add_dependency(PUBLIC NAME Kokkos TARGETS Kokkos::kokkos)
  #get_target_property(kokkosCompileOptions Kokkos::kokkoscore INTERFACE_COMPILE_OPTIONS)
  #message(kokkosCompileOptions ${kokkosCompileOptions})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${debug_flag} ${opt_flag}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

include_directories(support)
include_directories(algorithms)

set(PS_SOURCES 
  algorithms/Distribute.cpp
  algorithms/psTypes.cpp
  support/psAssert.cpp
  support/SellCSigma.cpp
  support/ViewComm.cpp
)

set(PS_HEADERS
  support/SCS_Macros.h
  support/SCS_Types.h
  support/SupportKK.h
  support/ViewComm.h
  support/MemberTypes.h
  support/MemberTypeArray.h
  support/MemberTypeLibraries.h
  support/SCSPair.h
  support/SellCSigma.h
  support/Segment.h
  support/psAssert.h
  algorithms/psParams.h
  algorithms/psTypes.h
  algorithms/Distribute.h
)

add_library(particleStructs ${PS_SOURCES})
install(FILES ${PS_HEADERS} DESTINATION include)
bob_export_target(particleStructs)

if (ENABLE_KOKKOS)
  add_definitions(-DKOKKOS_ENABLED)
  target_compile_definitions(particleStructs INTERFACE -DKOKKOS_ENABLED)
  target_include_directories(particleStructs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/support>
    $<INSTALL_INTERFACE:include>)
  target_include_directories(particleStructs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/algorithms>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(particleStructs Kokkos::kokkos)
  if ("CUDA" IN_LIST Kokkos_DEVICES)
    add_definitions(-DSCS_USE_CUDA)
    target_compile_definitions(particleStructs INTERFACE -DSCS_USE_CUDA)
  endif()
  if ("OPENMP" IN_LIST Kokkos_DEVICES)
    add_definitions(-DSCS_USE_OPENMP)
    target_compile_definitions(particleStructs INTERFACE -DSCS_USE_OPENMP)
  endif()
endif()

option(FP64 "Use 64bits for floating point values" ON)
option(FP32 "Use 32bits for floating point values" OFF)
message(STATUS "FP64: ${FP64}")
message(STATUS "FP32: ${FP32}")
if( (FP64 AND FP32) OR (NOT FP64 AND NOT FP32) )
  message(FATAL_ERROR "Enable either FP64 or FP32, but not both")
endif()
if( FP64 )
  add_definitions(-DFP64)
  target_compile_definitions(particleStructs INTERFACE -DFP64)
endif()
if( FP32 )
  add_definitions(-DFP32)
  target_compile_definitions(particleStructs INTERFACE -DFP32)
endif()

add_subdirectory(test)

bob_end_package()
