include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(../../../gmm-4.1/include/gmm)

add_library(VirtualSurgeon_Utils
	curl_get.cpp
	ImageEditingUtils.cpp
	OpenCV2ImageWrapper.cpp
	stdafx.cpp
	VirtualSurgeon_Utils.cpp
	)
	
install(TARGETS VirtualSurgeon_Utils
  # IMPORTANT: Add the library to the "export-set"
  EXPORT HeadReplacementLibraryDepends
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT stlib
    COMPONENT dev)
