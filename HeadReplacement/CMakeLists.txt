find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)

IF(APPLE)
	set( COCOA_LIBS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Cocoa.framework )
#	set( OpenGL_INCLUDE_DIRS /usr/X11R6/include)
#	set( OpenGL_LIBRARY /usr/X11R6/lib/libGL.dylib /usr/X11R6/lib/libGLU.dylib )
#	set( Freeglut_LIBRARY /usr/local/lib/libglut.3.dylib )
ENDIF(APPLE)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(OpenCV_FOUND)
	list(REMOVE_ITEM OpenCV_LIBS opencv_contrib opencv_legacy opencv_stitching opencv_gpu  )
endif(OpenCV_FOUND)

#### Head Replacement ####
link_directories(../../../fltk-3.0/lib/)

if(APPLE)
	add_executable( HeadReplacement MACOSX_BUNDLE main.cpp )
else(APPLE)
	add_executable( HeadReplacement main.cpp )
endif(APPLE)

target_link_libraries(HeadReplacement 
	VirtualSurgeon_HeadExtractor
	VirtualSurgeon_matting
	VirtualSurgeon_Utils
	VirtualSurgeon_Recoloring
	VirtualSurgeon_Relighting
	${OpenCV_LIBS}
#	${OpenGL_LIBRARY}
	fltk3 fltk3gl
	libjson
	)

IF(APPLE)
	# must add AGL and Cocoa frameworks...
	target_link_libraries(HeadReplacement ${COCOA_LIBS}	"-framework AGL" "-framework OpenGL")
ELSE(APPLE)
	target_link_libraries(HeadReplacement ${OpenGL_LIBRARY})
ENDIF(APPLE)

install(TARGETS HeadReplacement
  EXPORT HeadReplacementLibraryDepends
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  BUNDLE DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  )
  
foreach(ocvlib ${OpenCV_LIBS})
	file(GLOB ocvlib_path	"${OpenCV_LIB_DIR}/lib${ocvlib}.dylib"
							"${OpenCV_LIB_DIR}/lib${ocvlib}.${OpenCV_VERSION_MAJOR}.dylib"
							"${OpenCV_LIB_DIR}/lib${ocvlib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.dylib"
							"${OpenCV_LIB_DIR}/lib${ocvlib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH}.dylib"
	)
	set(ocvlib_paths ${ocvlib_paths} ${ocvlib_path})
endforeach(ocvlib)
#message(STATUS "install opencv libs: ${ocvlib_paths}")
set(HeadReplacement_EXE_DIR ${CMAKE_INSTALL_PREFIX}/HeadReplacement/HeadReplacement.app/Contents/MacOS/)
install(FILES ${ocvlib_paths} DESTINATION "${HeadReplacement_EXE_DIR}")
install(FILES ${ocvlib_paths} DESTINATION "${INSTALL_LIB_DIR}")

IF(APPLE)
	file(GLOB_RECURSE libogg_LIBS FOLLOW_SYMLINKS /usr/local/Cellar/libogg/1.3.0/lib/libogg*dylib )
	INSTALL(FILES ${libogg_LIBS} DESTINATION "${HeadReplacement_EXE_DIR}")
	message(STATUS "install libogg: ${libogg_LIBS}")
ENDIF(APPLE)