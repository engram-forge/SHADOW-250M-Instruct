cmake_minimum_required(VERSION 3.24)
project(shadow_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)

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
    target_link_libraries(shadow PRIVATE Threads::Threads)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    option(SHADOW_H618 "Restrict code generation to Allwinner H618 Cortex-A53" ON)
    if(SHADOW_H618)
        target_compile_options(shadow PRIVATE -mcpu=cortex-a53)
        target_compile_definitions(shadow PRIVATE SHADOW_H618=1)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(shadow_native_tests tests/native_tests.cpp src/model.cpp src/archive.cpp)
    target_include_directories(shadow_native_tests PRIVATE include)
    target_compile_options(shadow_native_tests PRIVATE -O2 -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(shadow_native_tests PRIVATE -UNDEBUG)
    endif()
    target_link_libraries(shadow_native_tests PRIVATE Threads::Threads)
    if(SHADOW_H618)
        target_compile_options(shadow_native_tests PRIVATE -mcpu=cortex-a53)
    endif()
    target_sources(shadow_native_tests PRIVATE src/metal_archive_stub.cpp)
    add_test(NAME shadow_native_tests COMMAND shadow_native_tests)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        add_executable(shadow_rvq_microbench tests/rvq_microbench.cpp)
        target_compile_options(shadow_rvq_microbench PRIVATE -O3 -Wall -Wextra -Wpedantic)
        if(SHADOW_H618)
            target_compile_options(shadow_rvq_microbench PRIVATE -mcpu=cortex-a53)
        endif()
        add_executable(shadow_ternary_microbench tests/ternary_microbench.cpp)
        target_compile_options(shadow_ternary_microbench PRIVATE -O3 -Wall -Wextra -Wpedantic)
        add_executable(shadow_ternary_int8_activation_microbench tests/ternary_int8_activation_microbench.cpp)
        target_compile_options(shadow_ternary_int8_activation_microbench PRIVATE -O3 -Wall -Wextra -Wpedantic)
        if(SHADOW_H618)
            target_compile_options(shadow_ternary_microbench PRIVATE -mcpu=cortex-a53)
            target_compile_options(shadow_ternary_int8_activation_microbench PRIVATE -mcpu=cortex-a53)
        endif()
        add_executable(shadow_worker_microbench tests/worker_microbench.cpp)
        target_compile_options(shadow_worker_microbench PRIVATE -O3 -Wall -Wextra -Wpedantic)
        target_link_libraries(shadow_worker_microbench PRIVATE Threads::Threads)
        if(SHADOW_H618)
            target_compile_options(shadow_worker_microbench PRIVATE -mcpu=cortex-a53)
        endif()
    endif()
endif()
