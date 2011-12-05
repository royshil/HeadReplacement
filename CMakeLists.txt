project(HeadReplacement)
cmake_minimum_required(VERSION 2.8)

set(HEADREPLACEMENT_MAJOR_VERSION 0)
set(HEADREPLACEMENT_MINOR_VERSION 1)
set(HEADREPLACEMENT_PATCH_VERSION 0)
set(HEADREPLACEMENT_VERSION
  ${HEADREPLACEMENT_MAJOR_VERSION}.${HEADREPLACEMENT_MINOR_VERSION}.${HEADREPLACEMENT_PATCH_VERSION})
 
set(CMAKE_BUILD_TYPE Release)

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/HEADREPLACEMENT)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")
 
# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

if(APPLE)
	set(EXECUTABLE_DIR "${CMAKE_INSTALL_PREFIX}/HeadReplacement/HeadReplacement.app/Contents/MacOS/")
endif()
if(WIN32)
	set(EXECUTABLE_DIR "${CMAKE_INSTALL_PREFIX}/HeadReplacement/output/")
endif()

if(DO_DEBUG) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVIRTUAL_SURGEON_DEBUG")
endif(DO_DEBUG)
 
# set up include-directories
include_directories(
  "${PROJECT_SOURCE_DIR}" 
  "${PROJECT_BINARY_DIR}") 



######## Find and add GLEE ########
file(GLOB_RECURSE GLEE_PATH "${CMAKE_SOURCE_DIR}/GLee.c")
if(GLEE_PATH STREQUAL GLEE_PATH-NOTFOUND)
	message(STATUS "GLEE was not found")
else()
	file(RELATIVE_PATH GLEE_PATH ${CMAKE_SOURCE_DIR} ${GLEE_PATH})
	list(LENGTH GLEE_PATH GLEE_PATH_LEN)
	if(GLEE_PATH_LEN GREATER 1)
		list(GET GLEE_PATH 1 GLEE_PATH)
	endif()
	get_filename_component(GLEE_PATH ${GLEE_PATH} REALPATH)
	get_filename_component(GLEE_PATH ${GLEE_PATH} PATH)
	message(STATUS "Found GLEE at ${GLEE_PATH}")
	add_library(GLEE ${GLEE_PATH}/GLee.c)
endif()

############ Find FLTK ############
if(NOT DEFINED FLTK_PATH)
	file(GLOB_RECURSE FLTK_PATH "${CMAKE_SOURCE_DIR}/Widget.h")
	if(FLTK_PATH STREQUAL FLTK_PATH-NOTFOUND   OR   FLTK_PATH STREQUAL "")
		message(STATUS "FLTK was not found !!!!!")
	else()
		file(RELATIVE_PATH FLTK_PATH ${CMAKE_SOURCE_DIR} ${FLTK_PATH})
		list(LENGTH FLTK_PATH FLTK_PATH_LEN)
		if(FLTK_PATH_LEN GREATER 1)
			list(GET FLTK_PATH 1 FLTK_PATH)
		endif()
		get_filename_component(FLTK_PATH ${FLTK_PATH} REALPATH)
		get_filename_component(FLTK_PATH ${FLTK_PATH} PATH)
		message(STATUS "Found FLTK at ${FLTK_PATH}")
	endif()
else()
	get_filename_component(FLTK_PATH ${FLTK_PATH} REALPATH)
	message(STATUS "FLTK path set to ${FLTK_PATH}")
	set(FLTK_INCLUDE_DIR ${FLTK_PATH}/include)
	set(FLTK_LIB_DIR ${FLTK_PATH}/lib)
endif()

########### Find TCLAP ###########
file(GLOB_RECURSE TCLAP_PATH "${CMAKE_SOURCE_DIR}/CmdLine.h")
if(TCLAP_PATH STREQUAL TCLAP_PATH-NOTFOUND)
	message(STATUS "TCLAP was not found")
else()
	file(RELATIVE_PATH TCLAP_PATH ${CMAKE_SOURCE_DIR} ${TCLAP_PATH})
	list(LENGTH TCLAP_PATH TCLAP_PATH_LEN)
	if(TCLAP_PATH_LEN GREATER 1)
		list(GET TCLAP_PATH 1 TCLAP_PATH)
	endif()
	get_filename_component(TCLAP_PATH ${TCLAP_PATH} REALPATH)
	get_filename_component(TCLAP_PATH ${TCLAP_PATH} PATH)
	set(TCLAP_PATH "${TCLAP_PATH}/..")
	message(STATUS "Found TCLAP at ${TCLAP_PATH}")
endif()

########### Find GMM ###########
file(GLOB_RECURSE GMM_PATH "${CMAKE_SOURCE_DIR}/gmm_solver_Schwarz_additive.h")
if(GMM_PATH STREQUAL GMM_PATH-NOTFOUND)
	message(STATUS "GMM was not found")
else()
	file(RELATIVE_PATH GMM_PATH ${CMAKE_SOURCE_DIR} ${GMM_PATH})
	list(LENGTH GMM_PATH GMM_PATH_LEN)
	if(GMM_PATH_LEN GREATER 1)
		list(GET GMM_PATH 1 GMM_PATH)
	endif()
	get_filename_component(GMM_PATH ${GMM_PATH} REALPATH)
	get_filename_component(GMM_PATH ${GMM_PATH} PATH)
	message(STATUS "Found GMM at ${GMM_PATH}")
endif()


#### Find and add GCOptimization ####
if(NOT DEFINED GCO_PATH)
	file(GLOB_RECURSE GCO_PATH "${CMAKE_SOURCE_DIR}/GCoptimization.cpp")
	if(GCO_PATH STREQUAL GCO_PATH-NOTFOUND)
		message(STATUS "GCOptimization was not found")
	else()
		file(RELATIVE_PATH GCO_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${GCO_PATH})
		list(LENGTH GCO_PATH GCO_PATH_LEN)
		if(GCO_PATH_LEN GREATER 1)
			list(GET GCO_PATH 1 GCO_PATH)
		endif()
		get_filename_component(GCO_PATH ${GCO_PATH} REALPATH)
		get_filename_component(GCO_PATH ${GCO_PATH} PATH)
		message(STATUS "Found GCOptimization at ${GCO_PATH}")
	endif()
else()
	get_filename_component(GCO_PATH ${GCO_PATH} REALPATH)
	message(STATUS "GCOptimization path set to ${GCO_PATH}")
endif()

add_subdirectory(VirtualSurgeon) # LIBRARY
add_subdirectory(HeadReplacement) # EXECUTABLE

############# Install stuff ##############
# Add all targets to the build-tree export set
export(TARGETS HeadReplacement VirtualSurgeon_HeadExtractor VirtualSurgeon_matting VirtualSurgeon_Utils VirtualSurgeon_Recoloring VirtualSurgeon_Relighting libjson GCOptimization
  FILE "${PROJECT_BINARY_DIR}/HeadReplacementLibraryDepends.cmake")
 
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE HeadReplacement)
 
# Create a HeadReplacementBuildTreeSettings.cmake file for the use from the build tree
configure_file(HeadReplacementBuildTreeSettings.cmake.in
  "${PROJECT_BINARY_DIR}/HeadReplacementBuildTreeSettings.cmake" @ONLY)
 
# Create the HeadReplacementConfig.cmake and HeadReplacementConfigVersion files
file(RELATIVE_PATH CONF_REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
configure_file(HeadReplacementConfig.cmake.in
  "${PROJECT_BINARY_DIR}/HeadReplacementConfig.cmake" @ONLY)
configure_file(HeadReplacementConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/HeadReplacementConfigVersion.cmake" @ONLY)
 
# Install the export set for use with the install-tree
install(EXPORT HeadReplacementLibraryDepends DESTINATION
  "${INSTALL_DATA_DIR}/HeadReplacement/CMake"
  COMPONENT dev)


install(FILES 
	HeadReplacement/haarcascade_frontalface_alt.xml 
	HeadReplacement/haarcascade_eye_tree_eyeglasses.xml 
	HeadReplacement/3280223114_e244fca104_b_d.dummy.png 
	HeadReplacement/3281508985_668495be2a_b_d.dummy.png
	DESTINATION ${EXECUTABLE_DIR})
install(DIRECTORY HeadReplacement/reshaping HeadReplacement/reshaping1
	DESTINATION ${EXECUTABLE_DIR})

if(APPLE)
	SET(APPS "\${CMAKE_INSTALL_PREFIX}/HeadReplacement/HeadReplacement.app")  # paths to executables
	set(DIRS ${CMAKE_INSTALL_PREFIX}/HeadReplacement.app/Contents/MacOS ${INSTALL_LIB_DIR})   # directories to search for prerequisites
	INSTALL(CODE "
		include(BundleUtilities)
		fixup_bundle(\"${APPS}\"   \"${OpenCV_LIBS}\"   \"${DIRS}\")
		" COMPONENT Runtime)
	
	set(CPACK_BINARY_DRAGNDROP ON)
endif()
if(WIN32)
	install(FILES "${INSTALL_BIN_DIR}/HeadReplacement.exe" DESTINATION ${EXECUTABLE_DIR})

	set(CPACK_GENERATOR "ZIP")
endif()

INCLUDE(CPack)
