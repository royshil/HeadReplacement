find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)
find_library(GLEW glew32)

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
	if(APPLE)
		list(REMOVE_ITEM OpenCV_LIBS  opencv_contrib opencv_legacy opencv_stitching opencv_gpu  )
	endif()
	if(WIN32)
		set(OCV_POSTFIX "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
		list(REMOVE_ITEM OpenCV_LIBS 
			debug 
			optimized 
			opencv_contrib${OCV_POSTFIX} 
			opencv_legacy${OCV_POSTFIX} 
			opencv_stitching${OCV_POSTFIX} 
			opencv_gpu${OCV_POSTFIX}
			opencv_contrib${OCV_POSTFIX}d 
			opencv_legacy${OCV_POSTFIX}d 
			opencv_stitching${OCV_POSTFIX}d 
			opencv_gpu${OCV_POSTFIX}d
			)
		foreach(ocvlib ${OpenCV_LIBS})
			list(APPEND TO_REMOVE_FROM_OCV_LIBS "${ocvlib}d")
		endforeach(ocvlib)
		
		foreach(ocvlib ${TO_REMOVE_FROM_OCV_LIBS})
#			message(STATUS "Remove: ${ocvlib}")
			list(REMOVE_ITEM OpenCV_LIBS ${ocvlib})
		endforeach()
	endif()
#	message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
endif(OpenCV_FOUND)


#### Head Replacement ####
link_directories(${FLTK_LIB_DIR})

if(APPLE)
	add_executable( HeadReplacement MACOSX_BUNDLE main.cpp )
endif(APPLE)
if(WIN32)
	add_executable( HeadReplacement WIN32 main.cpp )
endif(WIN32)

target_link_libraries(HeadReplacement 
	VirtualSurgeon_HeadExtractor
	VirtualSurgeon_matting
	VirtualSurgeon_Utils
	VirtualSurgeon_Recoloring
	VirtualSurgeon_Relighting
	${OpenCV_LIBS}
	fltk3 fltk3gl
	libjson
	GLEE
	)

IF(APPLE)
	# must add AGL and Cocoa frameworks...
	target_link_libraries(HeadReplacement ${COCOA_LIBS}	"-framework AGL" "-framework OpenGL")
ENDIF(APPLE)
IF(WIN32)
#	message(STATUS "linking opengl: ${OPENGL_LIBRARY} ${OPENGL_LIBRARIES}")
	target_link_libraries(HeadReplacement ${OPENGL_LIBRARY} ${OPENGL_LIBRARIES})
	target_link_libraries(HeadReplacement comctl32)  #fltk3 needs it...
ENDIF(WIN32)

install(TARGETS HeadReplacement
  EXPORT HeadReplacementLibraryDepends
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  BUNDLE DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  )
  
#message(STATUS "looking for OpenCV shared libraries in ${OpenCV_LIB_DIR} and ${OpenCV_LIB_DIR}/../bin")
foreach(ocvlib ${OpenCV_LIBS})
	if(APPLE)
		file(GLOB ocvlib_path	"${OpenCV_LIB_DIR}/lib${ocvlib}.dylib"
								"${OpenCV_LIB_DIR}/lib${ocvlib}.${OpenCV_VERSION_MAJOR}.dylib"
								"${OpenCV_LIB_DIR}/lib${ocvlib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.dylib"
								"${OpenCV_LIB_DIR}/lib${ocvlib}.${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.${OpenCV_VERSION_PATCH}.dylib"
		)
	endif()
	if(WIN32)
#		message(STATUS "lookfor ${OpenCV_LIB_DIR}/../bin/lib${ocvlib}.dll")
		file(GLOB ocvlib_path "${OpenCV_LIB_DIR}/../bin/lib${ocvlib}.dll")
	endif()
	list(APPEND ocvlib_paths "${ocvlib_path}")
endforeach(ocvlib)
#message(STATUS "install opencv libs: ${ocvlib_paths}")
install(FILES ${ocvlib_paths} DESTINATION ${EXECUTABLE_DIR})
install(FILES ${ocvlib_paths} DESTINATION "${INSTALL_LIB_DIR}")

IF(APPLE)
	file(GLOB_RECURSE libogg_LIBS FOLLOW_SYMLINKS /usr/local/Cellar/libogg/1.3.0/lib/libogg*dylib )
	INSTALL(FILES ${libogg_LIBS} DESTINATION "${EXECUTABLE_DIR}")
	message(STATUS "install libogg: ${libogg_LIBS}")
ENDIF(APPLE)