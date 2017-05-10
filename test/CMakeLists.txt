add_executable(test_geometry test_geometry.cpp)

target_link_libraries(
		test_geometry
		fastpoints
)

add_test(test_geometryy ${PROJECT_SOURCE_DIR}/build/test_geometry)