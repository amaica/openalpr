enable_testing()

add_definitions( -DOPENALPR_TESTING_CONFIG_PATH="${CMAKE_SOURCE_DIR}/../config/openalpr.conf.defaults")
add_definitions( -DOPENALPR_TESTING_RUNTIME_DIR="${CMAKE_SOURCE_DIR}/../runtime_data/")

ADD_EXECUTABLE( unittests 
  test_api.cpp 
  test_utility.cpp
  test_config.cpp
  test_regex.cpp
)

TARGET_LINK_LIBRARIES(unittests

	openalpr

  )

add_test(unittests unittests)

if(UNIX AND NOT WIN32)
  add_test(
    NAME garage_alpr_contract
    COMMAND "${OPENALPR_REPO_ROOT}/scripts/test_garage_alpr_contract.sh"
    WORKING_DIRECTORY "${OPENALPR_REPO_ROOT}"
  )
  set_tests_properties(garage_alpr_contract PROPERTIES LABELS "integration;script")
endif()

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})