npu_op_kernel_options(ascendc_kernels ALL OPTIONS
  "-I${PTO_ISA_ROOT}/include" "-O3" "-DMEMORY_BASE"
  "-mllvm" "-cce-aicore-stack-size=0x8000"
  "-mllvm" "-cce-aicore-function-stack-size=0x8000"
  "-mllvm" "-cce-aicore-record-overflow=false")
npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
npu_op_kernel_library(ascendc_kernels SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/ TILING_LIBRARY cust_optiling)
npu_op_package_add(${package_name} LIBRARY ascendc_kernels)
