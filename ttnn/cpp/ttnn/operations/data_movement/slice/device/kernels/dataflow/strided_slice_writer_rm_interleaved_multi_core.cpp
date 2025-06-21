// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);

    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t total_sticks = get_arg_val<uint32_t>(1);      // total rows to write (can be > num_rows_for_core if needed)
    uint32_t start_row = get_arg_val<uint32_t>(2);          // global starting row index
    uint32_t num_rows_for_core = get_arg_val<uint32_t>(3);  // number of rows this core will write

    // DEBUG
    DPRINT << "DEBUG:Write Kernel " <<  ENDL();
    DPRINT << "dst_addr = " << dst_addr << ENDL();
    DPRINT << "total_sticks = " << total_sticks << ENDL();
    DPRINT << "start_row = " << start_row << ENDL();
    DPRINT << "num_rows_for_core = " << num_rows_for_core << ENDL();

    // Address generator for interleaved layout
    const InterleavedAddrGen<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = page_size
    };

    constexpr uint32_t cb_id_out = 24;  

    for (uint32_t i = 0; i < num_rows_for_core; ++i) {
        uint32_t logical_row = start_row + i;

        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint64_t dst_noc_addr = get_noc_addr(logical_row, s);

        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, 1);
    }
}

