set (TOOLS_FILES 
     pytools.cpp
     ../tools/hilbert.cpp
     ../tools/config_parser.cpp
     ../tools/system.cpp
     )

set (EMF_FILES
     pyemf.cpp
     ../core/emf/tile.cpp
     ../core/emf/yee_lattice.cpp
     ../core/emf/yee_lattice_FDTD2.cpp
     ../core/emf/yee_lattice_interpolate_linear_1st.cpp
     ../core/emf/yee_lattice_current_filter_binomial2.cpp
     #../core/emf/propagators/fdtd2.cpp 
     #../core/emf/propagators/fdtd2_pml.cpp 
     #../core/emf/propagators/fdtd4.cpp 
     #../core/emf/propagators/fdtd_general.cpp 
     #../core/emf/filters/binomial2.cpp
     #../core/emf/filters/compensator.cpp
     #../core/emf/filters/general_binomial.cpp
     #../core/emf/filters/strided_binomial.cpp
     #../core/emf/filters/sweeping_binomial.cpp
     #../core/emf/boundaries/damping_tile.cpp
     #../core/emf/boundaries/conductor.cpp
     )

#set (VLV_FILES
#      pyvlv.cpp
#      ../core/vlv/tile.cpp
#      ../core/vlv/boundaries/piston.cpp
#      ../core/vlv/boundaries/outflow.cpp
#      ../core/vlv/momentum-solvers/amr_momentum_solver.cpp
#      ../core/vlv/momentum-solvers/bwd_lagrangian.cpp
#     )


set (PIC_FILES
     pypic.cpp
     ../core/pic/tile.cpp
     ../core/pic/tile_communication.cpp
     ../core/pic/particle.cpp
     ../core/pic/particle_boris.cpp
     ../core/pic/particle_current_zigzag_1st.cpp
     #../core/pic/boundaries/wall.cpp
     #../core/pic/boundaries/piston.cpp
     #../core/pic/boundaries/piston_z.cpp
     #../core/pic/boundaries/star_surface_injector.cpp
     #../core/pic/boundaries/gap.cpp
     #../core/pic/pushers/boris.cpp
     #../core/pic/pushers/boris_drag.cpp
     #../core/pic/pushers/boris_rad.cpp
     #../core/pic/pushers/boris_grav.cpp
     #../core/pic/pushers/vay.cpp
     #../core/pic/pushers/higuera_cary.cpp
     #../core/pic/pushers/rgca.cpp
     #../core/pic/pushers/photon.cpp
     #../core/pic/pushers/pulsar.cpp
     #../core/pic/interpolators/linear_1st.cpp
     #../core/pic/interpolators/quadratic_2nd.cpp
     #../core/pic/interpolators/cubic_3rd.cpp
     #../core/pic/interpolators/quartic_4th.cpp
     #../core/pic/depositers/zigzag.cpp
     #../core/pic/depositers/zigzag_2nd.cpp
     #../core/pic/depositers/zigzag_3rd.cpp
     #../core/pic/depositers/zigzag_4th.cpp
     #../core/pic/depositers/esikerpov_2nd.cpp
     #../core/pic/depositers/esikerpov_4th.cpp
     #../core/pic/depositers/esikerpov_4th_vec.cpp
     )

#set (QED_FILES 
#     pyqed.cpp
#     ../core/qed/interactions/pair_ann.cpp
#     ../core/qed/interactions/phot_ann.cpp
#     ../core/qed/interactions/compton.cpp
#     ../core/qed/interactions/synchrotron.cpp
#     ../core/qed/interactions/multi_phot_ann.cpp
#     ../core/qed/pairing.cpp
#     ../core/qed/pairing_all2all.cpp
#     )


#set (FFE_FILES 
#     pyffe.cpp
#     ../core/ffe/tile.cpp
#     ../core/ffe/slim_grids.cpp
#     ../core/ffe/currents/rffe2.cpp
#     ../core/ffe/currents/rffe4.cpp
#     ../core/ffe/currents/ffe2.cpp
#     ../core/ffe/currents/ffe4.cpp
#     )



set (IO_FILES 
     ../io/snapshots/fields.cpp
    )


pybind11_add_module(runko_cpp_bindings
    runko_cpp_bindings.cpp
    ${TOOLS_FILES}
    ${EMF_FILES}
    #${VLV_FILES}
    ${PIC_FILES}
    #${QED_FILES}
    #${FFE_FILES}
    ${IO_FILES}
)

target_link_libraries(runko_cpp_bindings
    PRIVATE
    runko_includes
    runko_warnings
    corgi
    tyvi::tyvi
    hdf5::hdf5
    # OpenMP::OpenMP_CXX
)

if(RUNKO_CODE_COVERAGE)
    target_link_libraries(runko_cpp_bindings PUBLIC runko_coverage_config)
endif()

set_target_properties(runko_cpp_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
