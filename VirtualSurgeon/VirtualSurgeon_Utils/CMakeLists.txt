include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../../gmm-4.1/include/gmm)
include_directories(../curl-7.19.3/include)
include_directories(../../tclap-1.2.1/include)

add_library(VirtualSurgeon_Utils
	curl_get.cpp
	ImageEditingUtils.cpp
	OpenCV2ImageWrapper.cpp
	stdafx.cpp
	VirtualSurgeon_Utils.cpp
	)

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SCL_SECURE_NO_DEPRECATE")
endif(MSVC)	

install(TARGETS VirtualSurgeon_Utils
  # IMPORTANT: Add the library to the "export-set"
  EXPORT HeadReplacementLibraryDepends
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT stlib
    COMPONENT dev)
