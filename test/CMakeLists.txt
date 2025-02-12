
#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Werror=return-type -Werror=unused-variable -Werror=unused-parameter")
enable_testing()
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/torchscripts
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/datasets
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${PROJECT_SOURCE_DIR}/src/CL
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
macro(add_catch_test appName SOURCE_FILES SOURCE_LIBS)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
    add_executable(${appName} ${SOURCE_FILES})
    target_link_libraries(${appName} ${SOURCE_LIBS})
    add_test(
            NAME catch_${appName}
            COMMAND $<TARGET_FILE:${appName}> --success
    )
endmacro()
add_catch_test(cpp_test SystemTest/SimpleTest.cpp LibAMM)
add_catch_test(sketch_test SystemTest/SketchTest.cpp LibAMM)
add_catch_test(crs_test SystemTest/CRSTest.cpp LibAMM)
add_catch_test(ews_test SystemTest/EWSTest.cpp LibAMM)
add_catch_test(weighted_cr_test SystemTest/WeightedCRTest.cpp LibAMM)
add_catch_test(smp_pca_test SystemTest/SMPPCATest.cpp LibAMM)
add_catch_test(Block_LRA_test SystemTest/BlockLRATest.cpp LibAMM)
add_catch_test(block_partition_test SystemTest/BlockPartitionTest.cpp LibAMM)
add_catch_test(rip_test SystemTest/RIPTest.cpp LibAMM)
add_catch_test(tug_of_war_test SystemTest/TugOfWarTest.cpp LibAMM)
add_catch_test(int8_test SystemTest/INT8Test.cpp LibAMM)

add_catch_test(pq_test SystemTest/PQTest.cpp LibAMM)
add_catch_test(streaming_test SystemTest/StreamingTest.cpp LibAMM)
add_catch_test(mtxMatrixLoader_test SystemTest/MtxMatrixLoaderTest.cpp LibAMM)
add_catch_test(zeroMaskedMatrixLoader_test SystemTest/ZeroMaskedMatrixLoaderTest.cpp LibAMM)
add_catch_test(zipfMatrixLoader_test SystemTest/ZipfMatrixLoaderTest.cpp LibAMM)
if (ENABLE_OPENCL)
    add_catch_test(cl_test SystemTest/CLTest.cpp LibAMM)
endif ()


