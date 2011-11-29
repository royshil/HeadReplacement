find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

IF(APPLE)
	set( COCOA_LIBS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Cocoa.framework )
#	message(STATUS "overriding OpenGL includes: ${OpenGL_INCLUDE_DIRS}")
#	set( OpenGL_INCLUDE_DIRS /usr/X11R6/include)
#	message(STATUS "overriding OpenGL libs: ${OpenGL_LIBRARY}")
#	set( OpenGL_LIBRARY /usr/X11R6/lib/libGL.dylib /usr/X11R6/lib/libGLU.dylib )
#	set( Freeglut_LIBRARY /usr/local/lib/libglut.3.dylib )
ENDIF(APPLE)

if(OPENMP_FOUND)
	message(STATUS "CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	message(STATUS "CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(OpenCV_FOUND)
	list(REMOVE_ITEM OpenCV_LIBS opencv_contrib opencv_legacy opencv_stitching opencv_gpu opencv_calib3d )
endif(OpenCV_FOUND)



message(STATUS "OpenCV_INCLUDE_DIRS " "${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS " "${OpenCV_LIBS}")

######## Relighting #######
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../../fltk-3.0/include)
include_directories(${OpenGL_INCLUDE_DIRS})
add_library(VirtualSurgeon_Relighting 
	../HeadReplacement/glm.cpp 
	../HeadReplacement/spherical_harmonics_analysis.cpp
	../HeadReplacement/LaplacianBlending.cpp
	../HeadReplacement/SpharmonicsUI.cpp
	../HeadReplacement/OGL_OCV_common.cpp
	)

install(TARGETS VirtualSurgeon_Relighting 
  # IMPORTANT: Add the library to the "export-set"
  EXPORT HeadReplacementLibraryDepends
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT stlib
    COMPONENT dev)



include_directories(matting/matting)
add_subdirectory(matting/matting)
include_directories(VirtualSurgeon_HeadExtractor)
add_subdirectory(VirtualSurgeon_HeadExtractor)
include_directories(VirtualSurgeon_Recoloring)
add_subdirectory(VirtualSurgeon_Recoloring)
include_directories(VirtualSurgeon_Utils)
add_subdirectory(VirtualSurgeon_Utils)
include_directories(libjson)
add_subdirectory(libjson)

