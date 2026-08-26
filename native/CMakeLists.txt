cmake_minimum_required(VERSION 3.24)
project(shadow_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(shadow
    src/main.cpp
    src/model.cpp
    src/archive.cpp
)
target_include_directories(shadow PRIVATE include)
target_compile_options(shadow PRIVATE -Wall -Wextra -Wpedantic)

if(APPLE)
    enable_language(OBJCXX)
    target_sources(shadow PRIVATE src/metal_archive.mm)
    target_compile_definitions(shadow PRIVATE SHADOW_APPLE=1)
    target_compile_options(shadow PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:OBJCXX>:-O3>)
    target_link_libraries(shadow PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
        "-framework Metal"
    )
    set_target_properties(shadow PROPERTIES OSX_DEPLOYMENT_TARGET 14.0)
    add_custom_command(TARGET shadow POST_BUILD
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_metallib.sh
                ${CMAKE_CURRENT_SOURCE_DIR}/shaders/archive.metal $<TARGET_FILE:shadow>.metallib
        VERBATIM)
else()
    target_sources(shadow PRIVATE src/metal_archive_stub.cpp)
    target_compile_options(shadow PRIVATE -O3)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(shadow_native_tests tests/native_tests.cpp src/model.cpp src/archive.cpp)
    target_include_directories(shadow_native_tests PRIVATE include)
    target_compile_options(shadow_native_tests PRIVATE -O2 -Wall -Wextra -Wpedantic)
    target_sources(shadow_native_tests PRIVATE src/metal_archive_stub.cpp)
    add_test(NAME shadow_native_tests COMMAND shadow_native_tests)
endif()
