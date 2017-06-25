add_subdirectory(io)
add_subdirectory(common)
add_subdirectory(geometry)
add_subdirectory(utils)
add_subdirectory(core)

cuda_add_library(
		fastpoints ${common} ${core} ${io} ${geometry} ${utils})
