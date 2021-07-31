#include "obj_loader.h"

namespace obj {
    void load(const std::filesystem::path &filepath,
              std::vector<Vertex>* vertices, std::vector<MaterialGroup>* matGroups,
              std::vector<Material>* materials) {
        std::filesystem::path matBaseDir = filepath;
        matBaseDir.remove_filename();

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> objShapes;
        std::vector<tinyobj::material_t> objMaterials;
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &objShapes, &objMaterials, &warn, &err,
                                    filepath.string().c_str(), matBaseDir.string().c_str());
        if (!ret) {
            printf("failed to load obj %s.n\n", filepath.string().c_str());
            printf("error: %s\n", err.c_str());
            printf("warning: %s\n", warn.c_str());
            return;
        }

        if (materials) {
            materials->resize(objMaterials.size());
            for (int mIdx = 0; mIdx < objMaterials.size(); ++mIdx) {
                const tinyobj::material_t &srcMat = objMaterials[mIdx];
                Material &dstMat = (*materials)[mIdx];
                dstMat.diffuse[0] = srcMat.diffuse[0];
                dstMat.diffuse[1] = srcMat.diffuse[1];
                dstMat.diffuse[2] = srcMat.diffuse[2];
                if (!srcMat.diffuse_texname.empty())
                    dstMat.diffuseTexPath = matBaseDir / srcMat.diffuse_texname;
            }
        }

        // Record unified unique vertices.
        using VertexKey = std::tuple<uint32_t, int32_t, int32_t, int32_t>;
        std::map<VertexKey, Vertex> unifiedVertexMap;
        for (uint32_t sIdx = 0; sIdx < objShapes.size(); ++sIdx) {
            const tinyobj::shape_t &shape = objShapes[sIdx];
            size_t idxOffset = 0;
            for (uint32_t fIdx = 0; fIdx < shape.mesh.num_face_vertices.size(); ++fIdx) {
                uint32_t numFaceVertices = shape.mesh.num_face_vertices[fIdx];
                if (numFaceVertices != 3) {
                    idxOffset += numFaceVertices;
                    continue;
                }

                uint32_t smoothGroupIdx = shape.mesh.smoothing_group_ids[fIdx];

                VertexKey vKeys[3];
                Vertex vs[3];
                for (uint32_t vIdx = 0; vIdx < 3; ++vIdx) {
                    tinyobj::index_t idx = shape.mesh.indices[idxOffset + vIdx];

                    vKeys[vIdx] = std::make_tuple(smoothGroupIdx,
                                                  idx.vertex_index,
                                                  idx.normal_index >= 0 ? idx.normal_index : static_cast<int32_t>(fIdx),
                                                  idx.texcoord_index);
                    if (unifiedVertexMap.count(vKeys[vIdx])) {
                        vs[vIdx] = unifiedVertexMap.at(vKeys[vIdx]);
                        continue;
                    }

                    vs[vIdx].position = float3(attrib.vertices[static_cast<uint32_t>(3 * idx.vertex_index + 0)],
                                               attrib.vertices[static_cast<uint32_t>(3 * idx.vertex_index + 1)],
                                               attrib.vertices[static_cast<uint32_t>(3 * idx.vertex_index + 2)]);
                    if (attrib.normals.size() && idx.normal_index >= 0)
                        vs[vIdx].normal = float3(attrib.normals[static_cast<uint32_t>(3 * idx.normal_index + 0)],
                                                 attrib.normals[static_cast<uint32_t>(3 * idx.normal_index + 1)],
                                                 attrib.normals[static_cast<uint32_t>(3 * idx.normal_index + 2)]);
                    else
                        vs[vIdx].normal = float3(NAN, NAN, NAN);
                    if (attrib.texcoords.size() && idx.texcoord_index >= 0)
                        vs[vIdx].texCoord = float2(attrib.texcoords[static_cast<uint32_t>(2 * idx.texcoord_index + 0)],
                                                   1 - attrib.texcoords[static_cast<uint32_t>(2 * idx.texcoord_index + 1)]); // flip V dir
                    else
                        vs[vIdx].texCoord = float2(0.0f, 0.0f);
                }

                float3 gn = normalize(cross(vs[1].position - vs[0].position,
                                            vs[2].position - vs[0].position));

                for (int32_t vIdx = 0; vIdx < 3; ++vIdx) {
                    const VertexKey &key = vKeys[vIdx];
                    Vertex &v = vs[vIdx];
                    if (std::isnan(v.normal.x))
                        v.normal = gn;
                    unifiedVertexMap[key] = v;
                }

                idxOffset += numFaceVertices;
            }
        }

        // Assign a vertex index to each of unified unique vertices.
        std::map<VertexKey, uint32_t> vertexIndices;
        vertices->resize(unifiedVertexMap.size());
        uint32_t vertexIndex = 0;
        for (const auto &kv : unifiedVertexMap) {
            (*vertices)[vertexIndex] = kv.second;
            vertexIndices[kv.first] = vertexIndex++;
        }
        unifiedVertexMap.clear();

        // Extract material groups and accumulate vertex normals.
        for (uint32_t sIdx = 0; sIdx < objShapes.size(); ++sIdx) {
            const tinyobj::shape_t &shape = objShapes[sIdx];
            size_t idxOffset;

            // Count the number of faces of each material group.
            std::unordered_map<uint32_t, uint32_t> matGroupNumFaces;
            idxOffset = 0;
            for (uint32_t fIdx = 0; fIdx < shape.mesh.num_face_vertices.size(); ++fIdx) {
                uint32_t numFaceVertices = shape.mesh.num_face_vertices[fIdx];
                if (numFaceVertices != 3) {
                    idxOffset += numFaceVertices;
                    continue;
                }

                uint32_t matIdx = uint32_t(shape.mesh.material_ids[fIdx]);
                if (matGroupNumFaces.count(matIdx) == 0)
                    matGroupNumFaces[matIdx] = 0;
                ++matGroupNumFaces[matIdx];

                idxOffset += numFaceVertices;
            }

            // Prepare triangle list array for each material group.
            std::unordered_map<uint32_t, MaterialGroup> shapeMatGroups;
            for (auto it = matGroupNumFaces.cbegin(); it != matGroupNumFaces.cend(); ++it) {
                MaterialGroup &matGroup = shapeMatGroups[it->first];
                matGroup.triangles.reserve(it->second);
                matGroup.materialIndex = it->first;
            }

            // Write triangle list for each material group.
            idxOffset = 0;
            for (uint32_t fIdx = 0; fIdx < shape.mesh.num_face_vertices.size(); ++fIdx) {
                uint32_t numFaceVertices = shape.mesh.num_face_vertices[fIdx];
                if (numFaceVertices != 3) {
                    idxOffset += numFaceVertices;
                    continue;
                }

                uint32_t smoothGroupIdx = shape.mesh.smoothing_group_ids[fIdx];

                tinyobj::index_t idx0 = shape.mesh.indices[idxOffset + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[idxOffset + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[idxOffset + 2];
                auto key0 = std::make_tuple(smoothGroupIdx,
                                            idx0.vertex_index,
                                            idx0.normal_index >= 0 ? idx0.normal_index : static_cast<int32_t>(fIdx),
                                            idx0.texcoord_index);
                auto key1 = std::make_tuple(smoothGroupIdx,
                                            idx1.vertex_index,
                                            idx1.normal_index >= 0 ? idx1.normal_index : static_cast<int32_t>(fIdx),
                                            idx1.texcoord_index);
                auto key2 = std::make_tuple(smoothGroupIdx,
                                            idx2.vertex_index,
                                            idx2.normal_index >= 0 ? idx2.normal_index : static_cast<int32_t>(fIdx),
                                            idx2.texcoord_index);

                Triangle triangle;
                triangle.v[0] = vertexIndices.at(key0);
                triangle.v[1] = vertexIndices.at(key1);
                triangle.v[2] = vertexIndices.at(key2);

                uint32_t matIdx = uint32_t(shape.mesh.material_ids[fIdx]);
                shapeMatGroups[matIdx].triangles.push_back(triangle);

                idxOffset += numFaceVertices;
            }

            for (auto it = shapeMatGroups.cbegin(); it != shapeMatGroups.cend(); ++it)
                matGroups->push_back(std::move(it->second));
        }
        vertexIndices.clear();

        // Normalize accumulated vertex normals.
        for (uint32_t vIdx = 0; vIdx < vertices->size(); ++vIdx) {
            Vertex &v = (*vertices)[vIdx];
            v.normal = normalize(v.normal);
        }
    }

    void load(const std::filesystem::path &filepath,
              std::vector<Vertex>* vertices, std::vector<Triangle>* triangles) {
        std::vector<obj::MaterialGroup> objMatGroups;
        load(filepath, vertices, &objMatGroups, nullptr);

        triangles->clear();
        for (int mIdx = 0; mIdx < objMatGroups.size(); ++mIdx) {
            const obj::MaterialGroup &matGroup = objMatGroups[mIdx];
            uint32_t baseIndex = static_cast<uint32_t>(triangles->size());
            triangles->resize(triangles->size() + matGroup.triangles.size());
            std::copy_n(matGroup.triangles.data(),
                        matGroup.triangles.size(),
                        triangles->data() + baseIndex);
        }
    }
}
