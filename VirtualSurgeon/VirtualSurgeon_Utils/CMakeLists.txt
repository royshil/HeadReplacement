include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GMM_PATH})
include_directories(../curl-7.19.3/include)
include_directories(${TCLAP_PATH})

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
