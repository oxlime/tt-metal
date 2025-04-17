#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t dims = get_compile_time_arg_val(2);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows_assigned = get_arg_val<uint32_t>(2);

    // Offsets:
    // [0] - input base addr
    // [1] - start row
    // [2] - number of rows this core should handle
    // [3...3+dims) - shape
    // [3+dims...3+2*dims) - starts
    // [3+2*dims...3+3*dims) - ends
    // [3+3*dims...3+4*dims) - strides

    uint32_t offset = 3;
    uint32_t shape[dims], starts[dims], ends[dims], strides[dims];

    for (uint32_t i = 0; i < dims; i++) {
        shape[i]   = get_arg_val<uint32_t>(offset + i);
        starts[i]  = get_arg_val<uint32_t>(offset + dims + i);
        ends[i]    = get_arg_val<uint32_t>(offset + 2*dims + i);
        strides[i] = get_arg_val<uint32_t>(offset + 3*dims + i);
    }

    uint32_t prod[dims];
    for (uint32_t i = 0; i < dims - 1; i++) {
        prod[i] = 1;
        for (uint32_t j = i + 1; j < dims - 1; j++) {
            prod[i] *= shape[j];
        }
    }
    prod[dims - 1] = 1;

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src_addr, .page_size = page_size};

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_out0 = 24;
    uint32_t src_buffer_l1_addr = get_write_ptr(cb_id_in0);
    volatile tt_l1_ptr uint16_t* in_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src_buffer_l1_addr);

    // Preload index for all but the last dimension (row dimension is dims - 2)
    uint32_t index[dims - 1];

    for (uint32_t rel_row = 0; rel_row < num_rows_assigned; rel_row++) {
        // Compute the actual output row this core is handling
        index[dims - 2] = start_row + rel_row;

        // Compute input row via strides
        uint32_t input_row = starts[dims - 2] + index[dims - 2] * strides[dims - 2];

        // Set up the rest of index[] from starts[]
        for (uint32_t i = 0; i < dims - 2; i++) {
            index[i] = starts[i];
        }

        // Compute base linear index of input row
        uint32_t base_linear_index = 0;
        for (uint32_t i = 0; i < dims - 2; i++) {
            base_linear_index += index[i] * prod[i];
        }
        base_linear_index += input_row * prod[dims - 2];

        // Read the page from DRAM/SRAM
        noc_async_read_page(base_linear_index, s0, src_buffer_l1_addr);
        cb_reserve_back(cb_id_out0, 1);
        noc_async_read_barrier();

        // Apply slice in last dimension
        volatile tt_l1_ptr uint16_t* out_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id_out0));
        uint32_t out_idx = 0;

        for (uint32_t l = starts[dims - 1]; l < ends[dims - 1]; l += strides[dims - 1]) {
            out_stick[out_idx++] = in_stick[l];
        }

        cb_push_back(cb_id_out0, 1);
    }
}
