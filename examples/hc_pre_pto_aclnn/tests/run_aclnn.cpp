// SPDX-License-Identifier: Apache-2.0
#include "acl/acl.h"
#include "aclnn_hc_pre_pto_bnn.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void Check(int rc, const char* step) {
    if (rc != 0) throw std::runtime_error(std::string(step) + ": " + std::to_string(rc));
}
struct Tensor {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
    size_t bytes;
    Tensor(std::vector<int64_t> shape, aclDataType dtype, const std::filesystem::path& file, bool input) {
        std::vector<int64_t> strides(shape.size(), 1);
        int64_t count = 1;
        for (size_t i = shape.size(); i-- > 0;) { strides[i] = count; count *= shape[i]; }
        bytes = count * (dtype == ACL_BF16 ? 2 : 4);
        Check(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "malloc tensor");
        if (input) {
            if (std::filesystem::file_size(file) != bytes) throw std::runtime_error("Wrong input size: " + file.string());
            std::vector<char> data(bytes);
            std::ifstream f(file, std::ios::binary); f.read(data.data(), bytes);
            if (!f) throw std::runtime_error("Cannot read input");
            Check(aclrtMemcpy(device, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
        }
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), device);
        if (!tensor) throw std::runtime_error("create tensor");
    }
    ~Tensor() { if (tensor) aclDestroyTensor(tensor); if (device) aclrtFree(device); }
    void Poison() { Check(aclrtMemset(device, bytes, 0xff, bytes), "poison output"); }
    void Save(const std::filesystem::path& p) {
        std::vector<char> data(bytes);
        Check(aclrtMemcpy(data.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
        std::ofstream f(p, std::ios::binary); f.write(data.data(), bytes);
        if (!f) throw std::runtime_error("Cannot write output");
    }
};
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "Usage: run_aclnn DEVICE INPUT_DIR OUTPUT_DIR\n"; return 2; }
    try {
        const int device = std::stoi(argv[1]);
        std::filesystem::path input(argv[2]), output(argv[3]);
        std::filesystem::create_directories(output);
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device), "set device");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        {
            Tensor x({96,4,4096}, ACL_BF16, input/"x.bin", true);
            Tensor weight({24,16384}, ACL_FLOAT, input/"weight.bin", true);
            Tensor scale({3}, ACL_FLOAT, input/"scale.bin", true);
            Tensor base({24}, ACL_FLOAT, input/"base.bin", true);
            Tensor mixed({96,4096}, ACL_BF16, {}, false);
            Tensor post({96,4}, ACL_FLOAT, {}, false);
            Tensor comb({96,4,4}, ACL_FLOAT, {}, false);
            std::ofstream timings(output/"timing.json");
            timings << "{\"scope\":\"ACL event around ACLNN launch; not Simpler L1\",\"samples_us\":[";
            for (int iteration = 0; iteration < 8; ++iteration) {
                mixed.Poison(); post.Poison(); comb.Poison();
                uint64_t size = 0; aclOpExecutor* executor = nullptr;
                Check(aclnnHcPrePtoBnnGetWorkspaceSize(x.tensor, weight.tensor, scale.tensor, base.tensor,
                    mixed.tensor, post.tensor, comb.tensor, &size, &executor), "workspace query");
                void* workspace = nullptr;
                if (size) Check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_NORMAL_ONLY), "workspace malloc");
                aclrtEvent begin = nullptr, end = nullptr;
                Check(aclrtCreateEvent(&begin), "create event");
                Check(aclrtCreateEvent(&end), "create event");
                Check(aclrtRecordEvent(begin, stream), "start");
                Check(aclnnHcPrePtoBnn(workspace, size, executor, stream), "ACLNN launch");
                Check(aclrtRecordEvent(end, stream), "end");
                Check(aclrtSynchronizeStream(stream), "synchronize");
                float ms = 0;
                Check(aclrtEventElapsedTime(&ms, begin, end), "elapsed");
                if (iteration >= 5) { if (iteration > 5) timings << ','; timings << ms * 1000; }
                aclrtDestroyEvent(begin); aclrtDestroyEvent(end);
                if (workspace) aclrtFree(workspace);
            }
            timings << "]}\n";
            mixed.Save(output/"mixed.bin"); post.Save(output/"post.bin"); comb.Save(output/"comb.bin");
        }
        aclrtDestroyStream(stream); aclrtResetDevice(device); aclFinalize();
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
