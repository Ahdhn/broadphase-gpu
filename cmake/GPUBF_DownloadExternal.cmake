include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(GPUBF_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(GPUBF_EXTRA_OPTIONS "")
endif()

function(gpubf_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${GPUBF_EXTERNAL}/${name}
        DOWNLOAD_DIR ${GPUBF_EXTERNAL}/.cache/${name}
        QUIET
        ${GPUBF_EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

################################################################################

function(gpubf_download_libigl)
  gpubf_download_project(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        v2.3.0
  )
endfunction()

function(gpubf_download_nlohmann_json)
  gpubf_download_project(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.9.1
)
endfunction()

# Eigen
# function(gpubf_download_eigen)
#     ccd_download_project(eigen
# 	GIT_REPOSITORY           https://gitlab.com/libeigen/eigen.git
# 	GIT_TAG       3.3.7
#     )
# endfunction()


# # Sampled CCD Queries
# function(ticcd_download_sample_queries)
#     ccd_download_project(Sample-Queries
#     GIT_REPOSITORY https://github.com/Continuous-Collision-Detection/Sample-Queries.git
#     GIT_TAG        4d6cce33477d8d5c666c31c8ea23e1aea97be371
#   )
# endfunction()

## tbb
function(gpubf_download_tbb)
    gpubf_download_project(tbb
    # GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    # GIT_TAG v2021.4.0
    GIT_REPOSITORY https://github.com/wjakob/tbb.git
      GIT_TAG        9e219e24fe223b299783200f217e9d27790a87b0 #ddbe45cd3ad89df9a84cd77013d5898fc48b8e89
    )
endfunction()
