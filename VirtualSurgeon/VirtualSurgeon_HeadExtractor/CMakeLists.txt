add_library(GCOptimization 
	${GCO_PATH}/GCoptimization.cpp 
	${GCO_PATH}/graph.cpp 
	${GCO_PATH}/LinkedBlockList.cpp 
	${GCO_PATH}/maxflow.cpp)

include_directories(${GCO_PATH})
include_directories(${VirtualSurgeon_matting_INCLUDE_DIRS})
include_directories(${FLTK_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIRS})
include_directories(${TCLAP_PATH})

add_definitions(-DHEAD_EXTRACTOR_MAIN)

add_library(VirtualSurgeon_HeadExtractor 
	graphcut.cpp 
	head_extractor.cpp 
	head_extractor_main.cpp 
	StdAfx.cpp
	CurveWindow.cpp
	LandmarksWindow.cpp
	SegmentationWindow.cpp
	MainWindow.cpp
	)
target_link_libraries(VirtualSurgeon_HeadExtractor GCOptimization)

install(TARGETS VirtualSurgeon_HeadExtractor GCOptimization
  # IMPORTANT: Add the library to the "export-set"
  EXPORT HeadReplacementLibraryDepends
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT stlib
    COMPONENT dev)
