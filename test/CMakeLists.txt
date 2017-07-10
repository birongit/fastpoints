add_executable(test_geometry test_geometry.cpp testing.cpp)

target_link_libraries(
		test_geometry
		fastpoints
)

add_test(test_geometry_all ${PROJECT_SOURCE_DIR}/build/test_geometry)

add_executable(test_kernels test_kernels.cpp testing.cpp)

target_link_libraries(
		test_kernels
		fastpoints
)

add_test(test_core_all ${PROJECT_SOURCE_DIR}/build/test_kernels)

add_executable(test_utils test_utils.cpp testing.cpp)

target_link_libraries(
		test_utils
		fastpoints
)

add_test(test_utils_all ${PROJECT_SOURCE_DIR}/build/test_utils)