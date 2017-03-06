
cuda_add_executable(simple test.cpp)

target_link_libraries(
		simple
		fastpoints
)
