#pragma once

#include "common.h"
#include "../../ext/tiny_obj_loader.h"

namespace obj {
    struct Vertex {
        float3 position;
        float3 normal;
        float2 texCoord;
    };

    struct Triangle {
        uint32_t v[3];
    };

    struct Material {
        float diffuse[3];
        std::filesystem::path diffuseTexPath;
    };

    struct MaterialGroup {
        std::vector<Triangle> triangles;
        uint32_t materialIndex;
    };

    void load(const std::filesystem::path &filepath,
              std::vector<Vertex>* vertices, std::vector<MaterialGroup>* matGroups,
              std::vector<Material>* materials);


    // JP: マテリアルを区別せずに形状だけを読み込む。
    // EN: Load only the shape without distinguishing materials.
    void load(const std::filesystem::path &filepath,
              std::vector<Vertex>* vertices, std::vector<Triangle>* triangles);
}
