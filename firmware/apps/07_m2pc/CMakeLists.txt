# 07_m2pc — FCS-M²PC de corriente en lazo cerrado (primer cierre de lazo).
# Compilado con -O2: el paso completo (sensado + 66 candidatos + SVM) debe
# terminar antes del valle del PWM (10 µs) para que el modelo de medio
# retardo sea exacto; m2pc_overruns() lo vigila.
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(m2pc.elf
    src/main.c
    ${STARTUP_SRC}
)

target_compile_options(m2pc.elf PRIVATE -O2)
option(M2PC_ADALINE "Usar ADALINE experimental en el predictor de 07" OFF)
target_compile_definitions(m2pc.elf PRIVATE ADALINE=$<BOOL:${M2PC_ADALINE}>)
target_link_libraries(m2pc.elf PRIVATE stm32g4_lib)

target_link_options(m2pc.elf PRIVATE -Wl,-Map=m2pc.map)

add_custom_command(TARGET m2pc.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:m2pc.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:m2pc.elf>" "m2pc.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:m2pc.elf>" "m2pc.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
