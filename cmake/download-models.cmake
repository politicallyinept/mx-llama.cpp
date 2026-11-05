# Offline-friendly short-circuit: if the destination already exists and matches
# the expected hash, skip the network fetch. Rationale: hosts without egress to
# huggingface.co (this test box) can vendor the fixture by copying a
# hash-valid file into place; the hash check keeps the vendored copy honest.
# (R2 close-out 2026-11-04: test-download-model/test-thread-safety failed here
# purely on DNS, not on a broken upstream file.)
if(EXISTS "${DEST}")
    file(SHA256 "${DEST}" _dest_sha)
    string(REGEX REPLACE "^SHA256=" "" _want_sha "${HASH}")
    if(_dest_sha STREQUAL _want_sha)
        message(STATUS "Skipping download of ${NAME}: ${DEST} already present with matching hash")
        return()
    endif()
endif()
get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ggml-org/models...")
endif()

file(DOWNLOAD
    "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true"
    "${DEST}"
    TLS_VERIFY ON
    EXPECTED_HASH ${HASH}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()
