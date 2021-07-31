# Graphics Experiments

適当にグラフィックス関連の論文などを読んで実装・検証したものを置きます。

I'll randomly put something for implementing/validating graphics papers here.

## 実装 / Implementations

### ReSTIR
Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting\
https://research.nvidia.com/publication/2020-07_Spatiotemporal-reservoir-resampling

![example](restir/comparison.png)

## その他 / Miscellaneous
OptiX/CUDAのラッパーとして[OptiX Utility](https://github.com/shocker-0x15/OptiX_Utility)を使用しています。

## 動作環境 / Confirmed Environment
現状以下の環境で動作を確認しています。\
I've confirmed that the program runs correctly in the following environment.

* Windows 10 (21H1) & Visual Studio Community 2019 (16.10.4)
* Core i9-9900K, 32GB, RTX 3080 10GB
* NVIDIA Driver 471.41

動作させるにあたっては以下のライブラリが必要です。\
It requires the following libraries.

* CUDA 11.3 Update 1 \
  OptiX Utilityは少し古いバージョンでも動作するとは思います。単にサンプルコードがこのバージョンに依存しているだけです。\
  ※CUDA 11.3.0にはバグがあり、OptiX Utilityと一緒に使用することができません。Update 1以降が必要です。\
  OptiX Utility may work with a bit older versions. The sample code just assumes this version.\
  \* CUDA 11.3.0 has a bug which prevents to use it with OptiX Utility. You need to use Update 1 or later.
* OptiX 7.3.0 (requires Maxwell or later generation NVIDIA GPU)

----
2021 [@Shocker_0x15](https://twitter.com/Shocker_0x15)
