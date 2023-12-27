# Graphics Experiments

適当にグラフィックス関連の論文などを読んで実装・検証したものを置きます。

このリポジトリを正しくcloneするには[Git LFS](https://git-lfs.github.com/)のインストールが必要です。

I'll randomly put something for implementing/validating graphics papers here.

You need to install [Git LFS](https://git-lfs.github.com/) to correctly clone this repository.



## 実装 / Implementations

### ReSTIR: Reservoir-based Spatiotemporal Importance Resampling
Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting\
https://research.nvidia.com/publication/2020-07_Spatiotemporal-reservoir-resampling

ReSTIRでは、Resampled Importance Sampling (RIS), Weighted Reservoir Sampling (WRS)、そして複数のReservoirを結合する際の特性を利用することで、プライマリーヒットにおいて大量の発光プリミティブからの効率的なサンプリングが可能となります。

ReSTIR enables efficient sampling from a massive amount of emitter primitives at primary hit by Resampled Importance Sampling (RIS), Weighted Reservoir Sampling (WRS) and utilizing the property of combining multiple reservoirs.

- [x] Basic Implementation (Biased RIS Estimator, Spatio-temporal Reuse)
- [x] Advanced Items
  - [x] Diffuse + Glossy BRDF
  - [x] Environmental Light
  - [x] Unbiased RIS Estimator with MIS weights
  - [x] Implement the improved ReSTIR algorithm:\
        "Rearchitecting Spatiotemporal Resampling for Production"\
        https://research.nvidia.com/publication/2021-07_Rearchitecting-Spatiotemporal-Resampling

![example](restir/comparison.jpg)
Amazon Lumberyard Bistro (Exterior) from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)

### ReGIR: Reservoir-based Grid Importance Resampling
Chapter 23. "Rendering Many Lights with Grid-based Reservoirs", Ray Tracing Gems II\
https://www.realtimerendering.com/raytracinggems/rtg2/index.html

ReGIRでは、ReSTIRと同様にStreaming RISを用いて大量の発光プリミティブからの効率的なサンプリングが可能となります。ReSTIRとは異なり、セカンダリー以降の光源サンプリングにも対応するため、Reservoirをワールド空間のグリッドに記録し、2段階のStreaming RISを行います。

ReGIR enables efficient sampling from a massive amount of emitter primitives by using streaming RIS similar to ReSTIR. Unlike ReSTIR, ReGIR stores reservoirs in a world space grid and performs two-stage streaming RIS to support light sampling after secondary visibility.

- [x] Basic Implementation (Uniform Grid, Temporal Reuse)
- [ ] Advanced Items
  - [x] Diffuse + Glossy BRDF
  - [x] Environmental Light
  - [ ] Scrolling Clipmap or Sparse Grid using Hash Map
  - [ ] ReGIR + Multiple Importance Sampling (Impossible?)

![example](regir/comparison.jpg)
Amazon Lumberyard Bistro (Interior) from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)

### NRC: Neural Radiance Caching
Real-time Neural Radiance Caching for Path Tracing\
https://research.nvidia.com/publication/2021-06_Real-time-Neural-Radiance

Path Tracing + Neural Radiance Cacheは、ある経路長より先から得られる寄与をニューラルネットワークによるキャッシュからの値によって置き換えることで、少しのバイアスと引き換えに低い分散の推定値(、さらにシーンによっては短いレンダリング時間)を実現します。NRCは比較的小さなネットワークであり、トレーニングはレンダリングの最中に行うオンラインラーニングとすることで、「適応による汎化」を実現、推論の実行時間もリアルタイムレンダリングに適した短いものとなります。

Path Tracing + Neural Radiance Cache replaces contributions given from beyond a certain path length by a value from the cache based on a neural network. This achieves low variance estimates at the cost of a little bias (, and additionally rendering time can even be reduced depending on the scene). NRC is a relatively small network, and training is online learning during rendering. This achieves "generalization via adaptation", and short inference time appropriate to real-time rendering.

- [x] Basic Implementation (based on simple path tracing, frequency/one-blob input encoding)
- [ ] Advanced Items
  - [ ] Combine with many-light sampling techniques like ReSTIR/ReGIR
  - [x] Add multi-resolution hash grid input encoding:\
        "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"\
        https://nvlabs.github.io/instant-ngp/

![example](neural_radiance_caching/comparison.jpg)
Zero-Day from [Open Research Content Archive (ORCA)](https://developer.nvidia.com/orca/beeple-zero-day)

### SVGF: Spatiotemporal Variance-Guided Filtering
Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination\
https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced

SVGFはパストレーシングなどによって得られたライティング結果を、物体表面のパラメターを参照しつつ画像空間でフィルタリングします。各ピクセルのライティングの分散を時間的・空間的にトラッキングし、分散が小さな箇所では小さなフィルター半径、分散が大きな箇所では大きなフィルター半径とすることで画像のぼけを抑えつつレンダリング画像における視覚的なノイズを低減します。フィルターにはà-trous Filterを用いることで大きなフィルタリング半径を比較的小さなコストで実現します。

SVGF filters the lighting result in screen-space obtained by methods like path tracing with references to surface parameters. It tracks the variance of the lighting for each pixel in spatially and temporally, then uses smaller filter radii at lower variance parts and larger filter radii at higher variance parts to get rid of perceptual noises from the rendered image while avoiding excessive blurs in the image. It uses an à-trous filter so that large filter radii can be used with relatively low costs.

- [x] Basic Implementation (temporal accumulation, SVGF, temporal AA)
- [ ] Advanced Items

![example](svgf/comparison.jpg)
Crytek Sponza from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)

### TFDM: Tessellation-Free Displacement Mapping
Tessellation-Free Displacement Mapping for Ray Tracing\
https://research.adobe.com/publication/tessellation-free-displacement-mapping-for-ray-tracing/

TFDMではハイトマップの各テクセルの最小値・最大値を階層的に記録したMinmaxミップマップを、ベースメッシュ形状と切り離した暗黙的なBVH(の一部)として用いることで、事前テッセレーションを行うことなく、省メモリに緻密なジオメトリに対するレイトレーシングを可能とします。トラバーサル中にはベース三角形ごとにIntersectionシェーダーが起動されます。三角形の各頂点における位置、法線、テクスチャー座標とMinmaxミップマップの値から、アフィン演算を用いてその場で階層的なAABBの計算とレイの交叉判定、最終的な形状との交叉判定を行います。

In TFDM, a minmax mipmap is used to store the minimum and maximum values of each texel hierarchically as (a part of) an implicit BVH, which is decoupled from the base mesh shape. This allows for ray tracing against detailed geometry without pre-tessellation, resulting in a low memory footprint. Intersection shader is invoked during traversal for each base triangle. Using the position, normal, texture coordinates of the triangle vertices and the values from the minmax mipmap, AABB computation on the fly by affine arithmetic and ray intersection test is performed hierarchically, finally ray intersection test against the final shape.

- [x] Basic Implementation (Minmax mipmap traversal, box/two-triangle local intersection, non-wrapping texture)
- [ ] Advanced Items
  - [x] Better root choice
  - [x] Flexible traversal order
  - [x] Bilinear local intersection
  - [ ] B-spline local intersection
  - [x] Texture wrapping
  - [x] Texture transform
  - [ ] Watertightness consideration
  - [ ] Continous LoD

![example](tfdm/comparison.jpg)
Height map from [textures.com](https://www.textures.com/download/3DScans0422/133306)

### Nonlinear Ray Tracing for Displacement and Shell Mapping
Nonlinear Ray Tracing for Displacement and Shell Mapping\
https://github.com/shinjiogaki/nonlinear-ray-tracing

シェル空間(ベース三角形と頂点法線からつくられるオフセット三角形に囲まれる空間)とテクスチャー空間(ディスプレイスメントマッピングにおけるハイトフィールドやシェルマッピングにおけるインスタンスのBVHが「歪みなく」存在する)のマッピングを考えると、テクスチャー空間内ではレイは曲線、具体的には二次の有理関数で表されます。同手法では曲線レイと、MinmaxミップマップやインスタンスのBVHによって与えられるAABBやテクスチャー空間中でのマイクロ三角形の交叉判定を直接解くことで省メモリかつ面倒な初期化処理が不要で効率的なディスプレイスメントマッピングやシェルマッピングを実現します。

Given the mapping between shell space (a space enclosed by the base triangle and the offset triangle formed by vertex normals) and texture space (where height fields in displacement mapping and instanced BVHs in shell mapping exist without "distortion"), rays in texture space are represented as curves, specifically degree-2 as rational functions. The proposed method directly solves the intersection test between a curved ray and an AABB given by a minmax mipmap or an instanced BVH, and the test between the curved ray and a micro triangle in texture space to achieve efficient and low-memory displacement mapping and shell mapping without troublesome initialization.

- [ ] Basic Implementation (Displacement mapping, non-wrapping texture)
- [ ] Advanced Items
  - [ ] Shell mapping
  - [x] Better root choice
  - [x] Flexible traversal order
  - [ ] Traversal order based on ray-box hit distance
  - [x] Texture wrapping
  - [x] Texture transform

![example](nrtdsm/comparison.jpg)
Height map from [aaa](bbb)



## その他 / Miscellaneous
OptiX/CUDAのラッパーとして[OptiX Utility](https://github.com/shocker-0x15/OptiX_Utility)を使用しています。

Programs here use [OptiX Utility](https://github.com/shocker-0x15/OptiX_Utility) as OptiX/CUDA wrapper.



## 動作環境 / Confirmed Environment
現状以下の環境で動作を確認しています。\
I've confirmed that the program runs correctly in the following environment.

* Windows 11 (23H2) & Visual Studio Community 2022 (17.8.2)
* Ryzen 9 7950X, 64GB, RTX 4080 16GB
* NVIDIA Driver 546.17 (Note that versions around 510-512 had several OptiX issues.)

動作させるにあたっては以下のライブラリが必要です。\
It requires the following libraries.

* CUDA 12.2
* OptiX 8.0.0 (requires Maxwell or later generation NVIDIA GPU)



## オープンソースソフトウェア / Open Source Software
- [Open Asset Import Library (assimp)](https://github.com/assimp/assimp)
- [CUBd](https://github.com/shocker-0x15/CUBd)
- [Dear ImGui](https://github.com/ocornut/imgui)
- [gl3w](https://github.com/skaslev/gl3w)
- [GLFW](https://github.com/glfw/glfw)
- [OptiX Utility](https://github.com/shocker-0x15/OptiX_Utility)
- [stb](https://github.com/nothings/stb)
- [Tiny CUDA Neural Networks (tiny-cuda-nn)](https://github.com/NVlabs/tiny-cuda-nn)
- [Tiny OpenEXR image library (tinyexr)](https://github.com/syoyo/tinyexr)

----
2023 [@Shocker_0x15](https://twitter.com/Shocker_0x15)
