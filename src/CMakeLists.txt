add_subdirectory(io)
add_subdirectory(common)
add_subdirectory(geometry)
add_subdirectory(utils)

cuda_add_library(
		fastpoints ${io} ${common} ${geometry} ${utils})
