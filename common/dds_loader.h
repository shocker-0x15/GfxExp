/*

   Copyright 2021 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

#pragma once

#include <cstdint>

// For DDS image read (block compressed format)
namespace dds {
    enum class Format : uint32_t {
        BC1_UNorm = 71,
        BC1_UNorm_sRGB = 72,
        BC2_UNorm = 74,
        BC2_UNorm_sRGB = 75,
        BC3_UNorm = 77,
        BC3_UNorm_sRGB = 78,
        BC4_UNorm = 80,
        BC4_SNorm = 81,
        BC5_UNorm = 83,
        BC5_SNorm = 84,
        BC6H_UF16 = 95,
        BC6H_SF16 = 96,
        BC7_UNorm = 98,
        BC7_UNorm_sRGB = 99,
    };

    [[nodiscard]]
    uint8_t** load(const char* filepath, int32_t* width, int32_t* height, int32_t* mipCount, size_t** sizes, Format* format);
    void free(uint8_t** data, int32_t mipCount, size_t* sizes);
}
