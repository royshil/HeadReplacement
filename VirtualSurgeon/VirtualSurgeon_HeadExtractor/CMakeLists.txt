add_library(GCOptimization 
	../../gco-v3.0/GCoptimization.cpp 
	../../gco-v3.0/graph.cpp 
	../../gco-v3.0/LinkedBlockList.cpp 
	../../gco-v3.0/maxflow.cpp)


include_directories(../../gco-v3.0)
include_directories(${VirtualSurgeon_matting_INCLUDE_DIRS})
include_directories(../../fltk-3.0/include)
include_directories(${OpenGL_INCLUDE_DIRS})
include_directories(../../tclap-1.2.1/include)

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
