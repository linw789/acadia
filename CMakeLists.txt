cmake_minimum_required(VERSION 3.26)

# Configure FetchContent_Declare to set the timestamp of the extracted files to be the time
# of extraction, instead of the time when the files were archived.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

project(YOSEMITE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(third_party)

add_executable(yosemite
    src/main.cpp
)

target_link_libraries(yosemite
    PRIVATE
        glfw
        glm
)
