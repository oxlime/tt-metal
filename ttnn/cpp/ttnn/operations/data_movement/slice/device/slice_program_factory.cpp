// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optional"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

#include "slice_op.hpp"
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t unpadded_row_size_bytes_offset = output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM
                                                  ? tt::round_up(unpadded_row_size_bytes, TILE_WIDTH)
                                                  : tt::round_up(unpadded_row_size_bytes, TILE_WIDTH / 2);

    std::vector<uint32_t> common_reader_kernel_args = {
        input_tensor.buffer()->address() + output_tensor_start[-1] * output_tensor.element_size(),
        padded_row_size_bytes,
        unpadded_row_size_bytes,
        unpadded_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + (32 - num_sticks_per_core % 32) % 32;
            num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(
                num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        //
        uint32_t addr_offset = 5;  // input buffer addr, padded_row_size_bytes, unpadded_row_size_bytes, num_dims
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = num_sticks_per_core;
        reader_kernel_args[addr_offset++] = num_sticks_per_core_read;
        reader_kernel_args[addr_offset] = num_read_per_barrier;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> writer_kernel_args = {
            output_buffer->address(),
            unpadded_row_size_bytes,
            unpadded_row_size_bytes_offset,
            num_sticks_per_core,
            num_sticks_per_core_read,
            num_read_per_barrier,
            num_sticks_written,
            0};
        num_sticks_written += num_sticks_per_core;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks slice_rm_multi_core(
    const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    const ttnn::Shape output_shape = output.get_padded_shape();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    uint32_t num_unpadded_sticks = output.volume() / output.get_padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t padded_row_size_bytes = a.get_padded_shape()[-1] * a.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * a.element_size();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t src_stick_size = padded_row_size_bytes;
    uint32_t dst_stick_size = unpadded_row_size_bytes;

    uint32_t src0_cb_index = 0;
    uint32_t max_read_size = 4096;
    uint32_t cb_page_size = dst_is_dram ? tt::round_up(unpadded_row_size_bytes, TILE_WIDTH)
                                        : tt::round_up(unpadded_row_size_bytes, TILE_WIDTH / 2);
    uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2 ? num_sticks_per_core_group_1
                                                                                         : num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_input_pages != 0) {
        auto num_sticks_per_core_pad32 = num_input_pages + (32 - num_input_pages % 32) % 32;
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, cb_page_size, max_read_size);
        num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
    }
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_read_per_barrier * 2 * cb_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    auto all_runtime_args = get_slice_runtime_args_rm(
        a,
        output,
        output_tensor_start,
        num_cores_total,
        num_cores,
        num_cores_y,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        max_read_size);

    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);

        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
    }

    auto override_runtime_args_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, compute_with_storage_grid_size, max_read_size](
            const void* operation,
            const Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            auto src_tensor = input_tensors.at(0);
            auto dst_tensor = output_tensors.at(0);
            uint32_t num_cores_x = compute_with_storage_grid_size.x;
            uint32_t num_cores_y = compute_with_storage_grid_size.y;
            uint32_t num_cores_total = num_cores_x * num_cores_y;
            uint32_t num_unpadded_sticks = dst_tensor.volume() / dst_tensor.get_padded_shape()[-1];
            auto
                [num_cores,
                 all_cores,
                 core_group_1,
                 core_group_2,
                 num_sticks_per_core_group_1,
                 num_sticks_per_core_group_2] =
                    tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

            const auto tensor_start =
                static_cast<const ttnn::operations::data_movement::SliceDeviceOperation*>(operation)->slice_start;
            auto all_runtime_args = get_slice_runtime_args_rm(
                src_tensor,
                dst_tensor,
                tensor_start,
                num_cores_total,
                num_cores,
                num_cores_y,
                core_group_1,
                core_group_2,
                num_sticks_per_core_group_1,
                num_sticks_per_core_group_2,
                max_read_size);

            for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; i++) {
                CoreCoord core = {i / num_cores_y, i % num_cores_y};

                { SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first); }

                { SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second); }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

operation::ProgramWithCallbacks slice_rm_strided_single_core_n_dims(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {
    // TODO: multi core implementation - work division is not trivial as we need to determine the N/C/H/W start and end
    // points for each split, and base that off stride
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto output_shape = output.get_padded_shape();
    const auto input_shape = a.get_padded_shape();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t src_is_dram = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t page_size_output = dst_is_dram ? tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH)
                                            : tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH / 2);
    uint32_t page_size_input = src_is_dram ? tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH)
                                           : tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH / 2);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(1 * page_size_input, {{tt::CBIndex::c_0, cb_data_format}})
            .set_page_size(tt::CBIndex::c_0, page_size_input);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size_output, {{tt::CBIndex::c_24, cb_data_format}})
            .set_page_size(tt::CBIndex::c_24, page_size_output);

    CoreRange core({0, 0}, {0, 0});
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    auto cb_output_tensor = tt::tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "strided_slice_reader_rm_interleaved_nd.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig({
            src_is_dram,
            (uint32_t)page_size_input,
            (uint32_t)input_shape.rank(),
        }

                                               ));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig({
            dst_is_dram,
            (uint32_t)page_size_output,
        }));

    std::vector<uint32_t> reader_runtime_args;
    reader_runtime_args.reserve(1 + (4 * input_shape.rank()));
    reader_runtime_args.push_back(a.buffer()->address());

    reader_runtime_args.insert(reader_runtime_args.end(), input_shape.cbegin(), input_shape.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), output_tensor_start.cbegin(), output_tensor_start.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), output_tensor_end.cbegin(), output_tensor_end.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), step.cbegin(), step.cend());

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

    uint32_t pages = output.volume() / output_shape[-1];
    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            output.buffer()->address(),
            pages,
        });

    auto override_address_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
                                         const Program& program,
                                         const std::vector<Buffer*>& input_buffers,
                                         const std::vector<Buffer*>& output_buffers) {
        auto input_buffer = input_buffers.at(0);
        auto output_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            writer_runtime_args[0] = output_buffer->address();
        }
    };

    return {.program = std::move(program), .override_addresses_callback = override_address_callback};
}

operation::ProgramWithCallbacks slice_rm_strided_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto output_shape = output.get_padded_shape();
    const auto input_shape = a.get_padded_shape();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    // Device and compute grid information
    tt::tt_metal::IDevice* device = a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    // Calculate output rows (second-to-last dimension)
    uint32_t num_output_rows = 0;
    for (uint32_t i = output_tensor_start[-2]; i < output_tensor_end[-2]; i += step[-2]) {
        num_output_rows++;
    }
    // Distribute work across cores
    auto [num_cores, all_cores, core_group_1, core_group_2, 
          num_rows_per_core_group_1, num_rows_per_core_group_2] = 
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_rows);
// DEBUG: Print all cores immediately after creation
std::cout << "=== DEBUG: All cores in the range set ===" << std::endl;
std::cout << "num_cores_x = " << num_cores_x << ", num_cores_y = " << num_cores_y << std::endl;
std::cout << "num_cores = " << num_cores << std::endl;
for (uint32_t i = 0; i < num_cores_y; i++) {
    for (uint32_t j = 0; j < num_cores_x; j++) {
        CoreCoord core = {j, i};
        if (all_cores.contains(core)) {
            std::cout << "Valid core: (" << core.x << ", " << core.y << ")" << std::endl;
        }
    }
}
std::cout << "=== End core listing ===" << std::endl;
    uint32_t src_is_dram = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    // Calculate page sizes with proper padding
    uint32_t page_size_output = dst_is_dram ? 
        tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH) :
        tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH / 2);
    uint32_t page_size_input = src_is_dram ? 
        tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH) :
        tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH / 2);
    // Create circular buffers - fix CB indices to match kernel expectations
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            2 * page_size_input, 
            {{tt::CBIndex::c_0, cb_data_format}}
        ).set_page_size(tt::CBIndex::c_0, page_size_input);
    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(
            2 * page_size_output, 
            {{tt::CBIndex::c_24, cb_data_format}}  // Changed from c_16 to c_24
        ).set_page_size(tt::CBIndex::c_24, page_size_output);
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    auto cb_output_tensor = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_dst0_config);
    // Create reader and writer kernels
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "strided_slice_reader_rm_interleaved_nd_multi_core.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig({
            src_is_dram,
            (uint32_t)page_size_input,
            (uint32_t)input_shape.rank()
        }));
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "strided_slice_writer_rm_interleaved_multi_core.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig({
            dst_is_dram,
            (uint32_t)page_size_output
        }));
//DEBUG assertions
const auto rank = input_shape.rank();
TT_ASSERT(output_tensor_start.size() == rank);
TT_ASSERT(output_tensor_end.size() == rank);
TT_ASSERT(step.size() == rank);
std::cout << "input_shape.rank() = " << input_shape.rank() << std::endl;
std::cout << "start.size() = " << output_tensor_start.size() << std::endl;
std::cout << "end.size() = " << output_tensor_end.size() << std::endl;
std::cout << "step.size() = " << step.size() << std::endl;
    // Distribute runtime arguments across cores
    uint32_t core_idx = 0;
    for (uint32_t i = 0; i < num_cores_y; i++) {
//DEBUG
std::cout << "num_cores_y = " << num_cores_y << std::endl;
std::cout << "i = " << i << std::endl;
        for (uint32_t j = 0; j < num_cores_x; j++) {
            //DEBUG
            std::cout << "num_cores_x = " << num_cores_x << std::endl;
            std::cout << "j = " << j << std::endl;
            if (core_idx >= num_cores) break;

            CoreCoord core = {j, i};
    // Debug Check immediately after creating the core
    if (!all_cores.contains(core)) {
        std::cout << "Skipping core not in kernel: (" << core.x << "," << core.y << ")" << std::endl;
        continue;  // Skip this core entirely
    }
//DEBUG            
std::cout << "Currently assigning to core: (" << j << "," << i << ")" << std::endl;
            // Determine row range for this core
            uint32_t start_row, num_rows_for_core;
            if (core_idx < core_group_1.num_cores()) {
                start_row = core_idx * num_rows_per_core_group_1;
                num_rows_for_core = num_rows_per_core_group_1;
            } else {
                start_row = core_group_1.num_cores() * num_rows_per_core_group_1 + 
                            (core_idx - core_group_1.num_cores()) * num_rows_per_core_group_2;
                num_rows_for_core = num_rows_per_core_group_2;
            }
            if (num_rows_for_core == 0) {
                core_idx++;
                continue;
            }
            // Prepare reader runtime arguments - match kernel expected order
            std::vector<uint32_t> reader_runtime_args;
            reader_runtime_args.push_back(a.buffer()->address());      // [0] src_addr
            reader_runtime_args.push_back(start_row);                  // [1] start_row  
            reader_runtime_args.push_back(num_rows_for_core);          // [2] num_rows_assigned

            // [3...3+dims) - shape
            for (uint32_t dim = 0; dim < input_shape.rank(); dim++) {
                //DEBUG
                std::cout << "    Adding shape[" << dim << "] = " << input_shape[dim] << std::endl;
                reader_runtime_args.push_back(input_shape[dim]);
            }

            // [3+dims...3+2*dims) - starts
            for (uint32_t dim = 0; dim < output_tensor_start.size(); dim++) {
        //DEBUG
        std::cout << "start[" << dim << "] = " << output_tensor_start[dim] << std::endl;
                reader_runtime_args.push_back(output_tensor_start[dim]);
            }

            // [3+2dims...3+3dims) - ends  
            for (uint32_t dim = 0; dim < output_tensor_end.size(); dim++) {
        //DEBUG
        std::cout << "end[" << dim << "] = " << output_tensor_end[dim] << std::endl;
                reader_runtime_args.push_back(output_tensor_end[dim]);
            }

            // [3+3dims...3+4dims) - strides
            for (uint32_t dim = 0; dim < step.size(); dim++) {
        //DEBUG
        std::cout << "step[" << dim << "] = " << step[dim] << std::endl;
                reader_runtime_args.push_back(step[dim]);
            }
//DEBUG
//CoreCoord core_coord = {j, i};
//if (!all_cores.contains(core_coord)) {
//    std::cout << "Skipping core not in kernel: (" << core_coord.x << "," << core_coord.y << ")" << std::endl;
//    continue;
//}
//DEBUG
std::cout << "core_idx = " << core_idx << ", num_cores = " << num_cores << std::endl;
std::cout << "Total reader_runtime_args size: " << reader_runtime_args.size() << std::endl;
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
//DEBUG
std::cout << "SetRuntimeArgs successful for core: (" << j << "," << i << ")" << std::endl;
            // Prepare writer runtime arguments - match kernel expected order
            std::vector<uint32_t> writer_runtime_args = {
                output.buffer()->address(),  // [0] dst_addr
                num_output_rows,             // [1] total_sticks (total output rows)
                start_row,                   // [2] start_row (global starting row index)
                num_rows_for_core            // [3] num_rows_for_core
            };
//DEBUG
std::cout << "Args for core (" << core.x << "," << core.y << "): ";
for (auto v : reader_runtime_args) std::cout << v << " ";
std::cout << std::endl;
std::cout << "Setting runtime args for core " << core.x << "," << core.y << std::endl;
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
std::cout << "Args set." << std::endl;
            core_idx++;
        }
    }
    // Address override callback to handle buffer changes
//    auto override_address_callback = [unary_reader_kernel_id, unary_writer_kernel_id, num_cores_x, num_cores_y, num_cores](
//                                         const Program& program,
//                                         const std::vector<Buffer>& input_buffers,
//                                         const std::vector<Buffer>& output_buffers) {
//        auto input_buffer = input_buffers.at(0);
//        auto output_buffer = output_buffers.at(0);
//        uint32_t core_idx = 0;
//        for (uint32_t i = 0; i < num_cores_y; i++) {
//            for (uint32_t j = 0; j < num_cores_x; j++) {
//                if (core_idx >= num_cores) break;
//
//                CoreCoord core = {j, i};
//                auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
//                reader_runtime_args[0] = input_buffer->address();
//                auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
//                writer_runtime_args[0] = output_buffer->address();
//
//                core_idx++;
//            }
//        }
//    };
    // Address override callback to handle buffer changes
    auto override_address_callback = [unary_reader_kernel_id, unary_writer_kernel_id, num_cores_x, num_cores_y, num_cores](
                                         const Program& program,
                                         const std::vector<Buffer*>& input_buffers,
                                         const std::vector<Buffer*>& output_buffers) {
        auto input_buffer = input_buffers.at(0);
        auto output_buffer = output_buffers.at(0);
        uint32_t core_idx = 0;
        for (uint32_t i = 0; i < num_cores_y; i++) {
            for (uint32_t j = 0; j < num_cores_x; j++) {
                if (core_idx >= num_cores) break;

                CoreCoord core = {j, i};
                auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                reader_runtime_args[0] = input_buffer->address();
                auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                writer_runtime_args[0] = output_buffer->address();

                core_idx++;
            }
        }
    };
    return {
        .program = std::move(program), 
        .override_addresses_callback = override_address_callback
    };
}

operation::ProgramWithCallbacks Gslice_rm_strided_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto output_shape = output.get_padded_shape();
    const auto input_shape = a.get_padded_shape();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    // Device and compute grid information
    tt::tt_metal::IDevice* device = a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Determine work distribution strategy
    uint32_t num_rows = output_shape[-2];  // Assume dividing work by rows
    auto [num_cores, all_cores, core_group_1, core_group_2, 
          num_rows_per_core_group_1, num_rows_per_core_group_2] = 
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    uint32_t src_is_dram = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    // Calculate page sizes with proper padding
    uint32_t page_size_output = dst_is_dram ? 
        tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH) :
        tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH / 2);
    uint32_t page_size_input = src_is_dram ? 
        tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH) :
        tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH / 2);

    // Create circular buffers for input and output
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            2 * page_size_input, 
            {{tt::CBIndex::c_0, cb_data_format}}
        ).set_page_size(tt::CBIndex::c_0, page_size_input);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(
            2 * page_size_output, 
            {{tt::CBIndex::c_24, cb_data_format}}
        ).set_page_size(tt::CBIndex::c_24, page_size_output);

    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    auto cb_output_tensor = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_dst0_config);

    // Create reader and writer kernels
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "strided_slice_reader_rm_interleaved_nd_multi_core.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig({
            src_is_dram,
            (uint32_t)page_size_input,
            (uint32_t)input_shape.rank(),
            num_cores
        }));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "strided_slice_writer_rm_interleaved_multi_core.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig({
            dst_is_dram,
            (uint32_t)page_size_output,
            num_cores
        }));

    // Distribute runtime arguments across cores
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        
        // Determine row range for this core
        uint32_t start_row, num_rows_for_core;
        if (i < core_group_1.num_cores()) {
            start_row = i * num_rows_per_core_group_1;
            num_rows_for_core = num_rows_per_core_group_1;
        } else {
            start_row = core_group_1.num_cores() * num_rows_per_core_group_1 + 
                        (i - core_group_1.num_cores()) * num_rows_per_core_group_2;
            num_rows_for_core = num_rows_per_core_group_2;
        }

        // Prepare reader runtime arguments
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(1 + (4 * input_shape.rank()));
        reader_runtime_args.push_back(a.buffer()->address());

        // Modify start and end points for this core's subset of rows
        auto modified_output_start = output_tensor_start;
        modified_output_start[-2] += start_row;

        auto modified_output_end = output_tensor_end;
        modified_output_end[-2] = std::min(
            output_tensor_end[-2], 
            modified_output_start[-2] + num_rows_for_core
        );

        reader_runtime_args.insert(reader_runtime_args.end(), input_shape.cbegin(), input_shape.cend());
        reader_runtime_args.insert(reader_runtime_args.end(), modified_output_start.cbegin(), modified_output_start.cend());
        reader_runtime_args.insert(reader_runtime_args.end(), modified_output_end.cbegin(), modified_output_end.cend());
        reader_runtime_args.insert(reader_runtime_args.end(), step.cbegin(), step.cend());

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

        // Prepare writer runtime arguments
        uint32_t pages = output.volume() / output_shape[-1];
        tt::tt_metal::SetRuntimeArgs(
            program, 
            unary_writer_kernel_id, 
            core,
            {
                output.buffer()->address(),
                pages,
                start_row,
                num_rows_for_core
            }
        );
    }

    // Address override callback to handle buffer changes
    auto override_address_callback = [unary_reader_kernel_id, unary_writer_kernel_id, num_cores_x](
                                         const Program& program,
                                         const std::vector<Buffer*>& input_buffers,
                                         const std::vector<Buffer*>& output_buffers) {
        auto input_buffer = input_buffers.at(0);
        auto output_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < input_buffers.size(); i++) {
            CoreCoord core = {i % num_cores_x, i / num_cores_x};

            auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            writer_runtime_args[0] = output_buffer->address();
        }
    };

    return {
        .program = std::move(program), 
        .override_addresses_callback = override_address_callback
    };
}

inline std::vector<std::vector<uint32_t>> group_contiguous_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.push_back(values[0]);

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1] + 1) {
            current_chunk.push_back(values[i]);
        } else {
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_chunk.push_back(values[i]);
        }
    }
    // Add the last chunk
    chunks.push_back(current_chunk);
    return chunks;
}

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores_unpadded,
    bool row_major,
    uint32_t num_cores_x_unpadded,
    uint32_t num_cores_y_unpadded,
    uint32_t shard_height_unpadded,
    uint32_t shard_height_padded,
    uint32_t num_cores_x_padded,
    uint32_t num_cores_y_padded) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, TILE_WIDTH / 2);

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_unpadded);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // figure out the start read stick id for each core, and the start id for each dim
        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        num_sticks_written += num_sticks_per_core_unpadded;

        // stores all sticks id for a core
        std::vector<uint32_t> stick_ids_per_core;
        uint32_t src_stick_id = start_id;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            stick_ids_per_core.push_back(src_stick_id);
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks_per_dim[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks_per_dim[j];
                } else {
                    break;
                }
            }
        }

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            uint32_t stick_id = stick_ids_per_core[i];
            uint32_t shard_id = stick_id / num_sticks_per_core_padded;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_padded);

            uint32_t shard_grid_inner_dim = row_major ? num_cores_x_padded : num_cores_y_padded;
            uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
            uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

            uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
            uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

            if (worker_x_logical < num_cores_x_padded and worker_y_logical < num_cores_y_padded) {
                auto core_physical =
                    device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});
                // save stick id in a shard, and core coord into a map
                std::pair<uint32_t, uint32_t> xy_pair = row_major ? std::make_pair(core_physical.y, core_physical.x)
                                                                  : std::make_pair(core_physical.x, core_physical.y);
                core_stick_map[xy_pair].push_back(stick_id_in_shard);
            }
        }

        // reader rt args
        std::vector<uint32_t> reader_kernel_args;
        reader_kernel_args.push_back(core_stick_map.size());  // num_cores

        tt::log_debug("num_cores: {}", core_stick_map.size());

        for (const auto& core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_kernel_args.push_back(xy_pair.second);  // noc x
                reader_kernel_args.push_back(xy_pair.first);   // noc y
            } else {
                reader_kernel_args.push_back(xy_pair.first);   // noc x
                reader_kernel_args.push_back(xy_pair.second);  // noc y
            }

            tt::log_debug("xy_pair.first: {}", xy_pair.first);
            tt::log_debug("xy_pair.second: {}", xy_pair.second);
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_values(core_stick_pair.second);
            stick_chunks_per_core.push_back(stick_chunks);

            reader_kernel_args.push_back(stick_chunks.size());  // num_chunks for current core
            tt::log_debug("chunk_size: {}", stick_chunks.size());
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]);  // start id of a chunk
                tt::log_debug("chunk_start_id: {}", chunk[0]);

                reader_kernel_args.push_back(chunk.size());  // length of a chunk
                tt::log_debug("chunk_length: {}", chunk.size());
            }
        }

        std::vector<uint32_t> writer_kernel_args;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

operation::ProgramWithCallbacks slice_rm_multi_core_sharded(
    const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    const ttnn::Shape output_shape = output.get_padded_shape();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    uint32_t num_padded_sticks = a.volume() / a.get_padded_shape()[-1];
    uint32_t num_unpadded_sticks = output.volume() / output.get_padded_shape()[-1];

    // stick sizes
    uint32_t W_padded = a.get_logical_shape()[-1];
    uint32_t W_unpadded = output.get_logical_shape()[-1];
    auto stick_size_padded = W_padded * a.element_size();
    auto stick_size_unpadded = W_unpadded * output.element_size();

    // input shard spec
    auto shard_spec_padded = a.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];
    uint32_t shard_width_padded = shard_spec_padded.shape[1];

    auto& all_cores_padded = shard_spec_padded.grid;
    uint32_t num_cores_padded = shard_spec_padded.num_cores();
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    tt::log_debug("num_padded_sticks: {}", num_padded_sticks);
    tt::log_debug("shard_height_padded: {}", shard_height_padded);
    tt::log_debug("all_cores_padded: {}", all_cores_padded);
    tt::log_debug("num_cores_padded: {}", num_cores_padded);

    // output shard spec
    auto shard_spec_unpadded = output.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    uint32_t shard_width_unpadded = shard_spec_unpadded.shape[1];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = shard_spec_unpadded.grid.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y + 1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    tt::log_debug("num_unpadded_sticks: {}", num_unpadded_sticks);
    tt::log_debug("shard_height_unpadded: {}", shard_height_unpadded);
    tt::log_debug("all_cores_unpadded: {}", all_cores_unpadded);
    tt::log_debug("num_cores_unpadded: {}", num_cores_unpadded);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(shard_height_padded * stick_size_padded, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_padded)
            .set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_unpadded * stick_size_unpadded, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, stick_size_unpadded)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)stick_size_padded, (std::uint32_t)stick_size_unpadded, (std::uint32_t)shard_height_unpadded};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_sharded.cpp",
        all_cores_unpadded,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    auto all_runtime_args = get_slice_runtime_args_rm_sharded(
        a,
        output,
        output_tensor_start,
        num_cores_unpadded,
        row_major,
        num_cores_x_unpadded,
        num_cores_y_unpadded,
        shard_height_unpadded,
        shard_height_padded,
        num_cores_x_padded,
        num_cores_y_padded);

    for (uint32_t i = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
    }

    auto override_runtime_args_callback = [cb_src0, cb_output](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer_a = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

template <bool initialize_args>
inline __attribute__((always_inline)) void set_slice_runtime_args_tile(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const uint32_t& num_cores_total,
    const uint32_t& num_cores,
    const std::vector<CoreCoord>& cores,
    const uint32_t& num_cores_group_1,
    const uint32_t& num_cores_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const Program& program,
    const tt::tt_metal::KernelHandle& unary_reader_kernel_id,
    const tt::tt_metal::KernelHandle& unary_writer_kernel_id,
    std::vector<uint32_t>& accumulated_total_per_dim) {
    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();
    const auto& input_shape = input_tensor.get_padded_shape();
    const auto& output_shape = output_tensor.get_padded_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    const auto set_common_reader_args = [&](
        uint32_t * reader_common_args, uint32_t * num_unpadded_tiles_per_dim, uint32_t * num_padded_tiles_per_dim)
        __attribute__((always_inline)) {
        reader_common_args[0] = input_buffer->address();
        num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
        num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
        num_padded_tiles_per_dim[0] = num_padded_Xt;
        num_padded_tiles_per_dim[1] = num_padded_Yt;
        accumulated_total_per_dim[0] = num_total_Xt;
        accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;
        for (int32_t i = 2; i < num_dims; ++i) {
            uint32_t num_unpadded_dim = output_shape[-(i + 1)];
            uint32_t num_total_dim = input_shape[-(i + 1)];
            uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
            num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
            num_padded_tiles_per_dim[i] = num_padded_dim;
            accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        }
    };

    const auto set_reader_rt_args = [&](
        uint32_t * reader_rt_args,
        const uint32_t* num_unpadded_tiles_per_dim,
        const uint32_t* num_padded_tiles_per_dim,
        const uint32_t& num_tiles_per_core,
        const uint32_t& start_offset,
        const uint32_t& num_tiles_written) __attribute__((always_inline)) {
        reader_rt_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_rt_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_rt_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_rt_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_rt_args[0] = start_id;
        reader_rt_args[1] = num_tiles_per_core;
    };

    if constexpr (initialize_args) {
        std::vector<uint32_t> reader_common_args(1 + num_dims * 2);
        uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
        uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
        SetCommonRuntimeArgs(program, unary_reader_kernel_id, reader_common_args);
    }
    auto& reader_common_args = GetCommonRuntimeArgs(program, unary_reader_kernel_id);
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    if constexpr (!initialize_args) {
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
    }

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, output_tensor_start);

    auto& reader_kernel_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
    auto& writer_kernel_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
    const uint32_t num_used_cores = num_cores_group_1 + num_cores_group_2;
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_tiles_per_core;
        if (i < num_cores_group_1) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (i < num_used_cores) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            if constexpr (initialize_args) {
                std::vector<uint32_t> reader_kernel_args(2 + num_dims, 0);
                std::vector<uint32_t> writer_kernel_args(3, 0);
                tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
                tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
            } else {
                auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
                reader_kernel_args[1] = 0;
                auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
                writer_kernel_args[1] = 0;
            }
            continue;
        }

        if constexpr (initialize_args) {
            std::vector<uint32_t> reader_kernel_args(2 + num_dims);
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
        } else {
            auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
        }

        if constexpr (initialize_args) {
            const std::array writer_kernel_args = {output_buffer->address(), num_tiles_per_core, num_tiles_written};
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
        } else {
            auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
            writer_kernel_args[0] = output_buffer->address();
            writer_kernel_args[1] = num_tiles_per_core;
            writer_kernel_args[2] = num_tiles_written;
        }
        num_tiles_written += num_tiles_per_core;
    }
}

operation::ProgramWithCallbacks slice_tile_multi_core(
    const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    const auto output_shape = output.get_padded_shape();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    uint32_t num_unpadded_tiles = output.volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    std::uint32_t num_dims = static_cast<std::uint32_t>(a.get_padded_shape().rank());

    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(num_dims),
        static_cast<uint32_t>(src0_is_dram),
    };
    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(src0_cb_index), static_cast<uint32_t>(dst_is_dram)};

    // Tilized reader
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, false);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    set_slice_runtime_args_tile<true>(
        a,
        output,
        output_tensor_start,
        num_cores_total,
        num_cores,
        cores,
        core_group_1.num_cores(),
        core_group_2.num_cores(),
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        program,
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        accumulated_total_per_dim);

    auto override_runtime_args_callback = [unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           compute_with_storage_grid_size,
                                           cores,
                                           accumulated_total_per_dim](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) mutable {
        const Tensor& src_tensor = input_tensors[0];
        const Tensor& dst_tensor = output_tensors[0];
        uint32_t num_unpadded_tiles = dst_tensor.volume() / TILE_HW;

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores_total = cores.size();

        auto
            [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

        const auto& tensor_start =
            static_cast<const ttnn::operations::data_movement::SliceDeviceOperation*>(operation)->slice_start;
        set_slice_runtime_args_tile<false>(
            src_tensor,
            dst_tensor,
            tensor_start,
            num_cores_total,
            num_cores,
            cores,
            core_group_1.num_cores(),
            core_group_2.num_cores(),
            num_tiles_per_core_group_1,
            num_tiles_per_core_group_2,
            program,
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            accumulated_total_per_dim);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

//operation::ProgramWithCallbacks slice_multi_core(
//    const Tensor& a,
//    Tensor& output,
//    const ttnn::Shape& output_tensor_start,
//    const ttnn::Shape& output_tensor_end,
//    const ttnn::Shape& step) {
//    bool has_step = false;
//    for (int i = 0; i < step.size(); i++) {
//        if (step[i] != 1) {
//            has_step = true;
//            break;
//        }
//    }
//    switch (a.get_layout()) {
//        case Layout::ROW_MAJOR:
//            return a.is_sharded() ? slice_rm_multi_core_sharded(a, output, output_tensor_start, output_tensor_end)
//                                  : (has_step ? slice_rm_strided_single_core_n_dims(
//                                                    a, output, output_tensor_start, output_tensor_end, step)
//                                              : slice_rm_multi_core(a, output, output_tensor_start, output_tensor_end));
//        case Layout::TILE: return slice_tile_multi_core(a, output, output_tensor_start, output_tensor_end);
//        default: TT_ASSERT(false, "Unsupported Layout");
//    }
//    return {};
//}

operation::ProgramWithCallbacks slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {
    
    bool has_step = false;
    for (int i = 0; i < step.size(); i++) {
        if (step[i] != 1) {
            has_step = true;
            break;
        }
    }

    switch (a.get_layout()) {
        case Layout::ROW_MAJOR:
            return a.is_sharded() 
                ? slice_rm_multi_core_sharded(a, output, output_tensor_start, output_tensor_end)
                : (has_step 
                    ? slice_rm_strided_multi_core(a, output, output_tensor_start, output_tensor_end, step)
                    : slice_rm_multi_core(a, output, output_tensor_start, output_tensor_end));
        
        case Layout::TILE: 
            return slice_tile_multi_core(a, output, output_tensor_start, output_tensor_end);
        
        default: 
            TT_ASSERT(false, "Unsupported Layout");
    }

    return {};
}

}  // namespace ttnn::operations::data_movement::detail
