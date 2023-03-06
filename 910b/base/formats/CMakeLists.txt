set(SRC_FORMAT_TRANSFERS
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwcn_fracz3D.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_dhwnc_fracz3D_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_nz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_zz.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fractal_z_tbe.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_fz_c04.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "${AIR_CODE_DIR}/base/formats/format_transfers/format_transfer_transpose.cc"
    "${AIR_CODE_DIR}/base/formats/formats.cc"
    "${AIR_CODE_DIR}/base/formats/ge_format_util.cc"
    "${AIR_CODE_DIR}/base/formats/register_format_transfer.cc"
    "${AIR_CODE_DIR}/base/formats/utils/formats_trans_utils.cc"
)

############ libformats.so ############
add_library(formats SHARED
    ${SRC_FORMAT_TRANSFERS}
)

target_compile_definitions(formats PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    FUNC_VISIBILITY
)

target_compile_options(formats PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_include_directories(formats PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/base
    ${CMAKE_BINARY_DIR}
)

target_link_options(formats PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(formats PRIVATE
    intf_pub
    -Wl,--no-as-needed
    graph
    graph_base
    c_sec
    error_manager
    slog
    -Wl,--as-needed
    ascend_protobuf
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

install(TARGETS formats OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
