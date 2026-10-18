# 08_bemf_oc — BEMF en circuito abierto con la red de sensado del 6-step de la
# placa (PA4/PC4/PB11), puente apagado, rotor girado a mano. PLANNING §11 Exp. 2b.
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(bemf_oc.elf
    src/main.c
    ${STARTUP_SRC}
)

target_link_libraries(bemf_oc.elf PRIVATE stm32g4_lib)
target_compile_options(bemf_oc.elf PRIVATE -O2)
target_link_options(bemf_oc.elf PRIVATE -Wl,-Map=bemf_oc.map)

add_custom_command(TARGET bemf_oc.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:bemf_oc.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:bemf_oc.elf>" "bemf_oc.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:bemf_oc.elf>" "bemf_oc.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
