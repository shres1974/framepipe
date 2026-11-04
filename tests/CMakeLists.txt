find_package(GTest 1.15 REQUIRED)
include(GoogleTest)

# ---------------------------------------------------------------- unit tests (no CUDA)
add_executable(framepipe_unit_tests
  unit/test_grayscale.cpp
  unit/test_tables.cpp
  unit/test_resize.cpp
  unit/test_gaussian.cpp
  unit/test_sobel.cpp
  unit/test_image_view.cpp
  unit/test_config.cpp
  unit/test_pipeline.cpp
  unit/test_lifecycle.cpp
  unit/test_compare.cpp
  unit/test_parallel.cpp
  unit/test_sha256.cpp
  unit/test_random_matrix.cpp
)
target_include_directories(framepipe_unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(framepipe_unit_tests PRIVATE framepipe_core GTest::gtest_main framepipe_warnings framepipe_sanitize)
target_compile_definitions(framepipe_unit_tests PRIVATE FRAMEPIPE_HAVE_OPENMP=$<BOOL:${FRAMEPIPE_HAVE_OPENMP}>)
gtest_discover_tests(framepipe_unit_tests PROPERTIES LABELS "unit;cpu" DISCOVERY_TIMEOUT 60)

# ---------------------------------------------------------------- integration (I/O, manifest, batch, harness)
add_executable(framepipe_integration_tests
  integration/test_image_io.cpp
  integration/test_manifest_batch.cpp
  integration/test_harness.cpp
  integration/test_stats.cpp
)
target_include_directories(framepipe_integration_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(framepipe_integration_tests PRIVATE framepipe_app_lib GTest::gtest_main framepipe_warnings framepipe_sanitize)
target_compile_definitions(framepipe_integration_tests PRIVATE
  FRAMEPIPE_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
gtest_discover_tests(framepipe_integration_tests PROPERTIES LABELS "integration;cpu" DISCOVERY_TIMEOUT 60)

# ---------------------------------------------------------------- CLI tests (shell driven)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME cli_integration
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/integration/test_cli.py
            $<TARGET_FILE:framepipe> ${CMAKE_SOURCE_DIR})
  set_tests_properties(cli_integration PROPERTIES LABELS "integration;cli;cpu" TIMEOUT 600)

  # Independent oracle: requires NumPy. Visibly skipped (exit 77) if absent.
  add_test(NAME oracle_vs_cpp
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/run_oracle_suite.py
            --framepipe $<TARGET_FILE:framepipe> --work-dir ${CMAKE_BINARY_DIR}/oracle_work)
  set_tests_properties(oracle_vs_cpp PROPERTIES LABELS "oracle;cpu" SKIP_RETURN_CODE 77 TIMEOUT 900)

  add_test(NAME oracle_self_tests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_oracle.py)
  set_tests_properties(oracle_self_tests PROPERTIES LABELS "oracle;cpu" SKIP_RETURN_CODE 77 TIMEOUT 300)

  add_test(NAME results_schema
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/integration/test_results_schema.py
            $<TARGET_FILE:framepipe> ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/schema_work)
  set_tests_properties(results_schema PROPERTIES LABELS "integration;benchmark;cpu" TIMEOUT 600)
endif()

# ---------------------------------------------------------------- GPU tests
if(FRAMEPIPE_HAVE_CUDA)
  add_executable(framepipe_cuda_tests gpu/test_cuda_parity.cpp)
  target_include_directories(framepipe_cuda_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(framepipe_cuda_tests PRIVATE framepipe_core CUDA::cudart GTest::gtest_main framepipe_warnings)
  gtest_discover_tests(framepipe_cuda_tests PROPERTIES LABELS "gpu;cuda" SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
                       DISCOVERY_TIMEOUT 120)
endif()
