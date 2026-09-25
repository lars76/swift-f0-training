function(swiftf0_optimize_math)
  target_compile_options(onnxruntime_mlas PRIVATE -O3)
endfunction()
cmake_language(DEFER CALL swiftf0_optimize_math)
