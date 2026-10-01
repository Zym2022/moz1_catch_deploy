# Moz1 回放网格

本目录保存 `../moz1_boxer.urdf` 引用的全部 20 个 STL，合计约 33.5 MiB，
用于离线实物回放。文件保持原始字节，不简化网格。

来源：MozBoxer 仓库 `source/MozBoxer/assets/moz1/meshes/`，
提交 `f603888ad7677c5c86f47809dcc105575c9152a0`。

手掌由部署代码中的 12 球＋刚芯常数生成，未使用旧手掌 STL。
当前 URDF 不引用的头、车轮与底盘网格不在本目录中。

网格、URDF 与回放代码随 Git 同步；每台机器的实验日志和生成网页存放在
被忽略的 `output/`。导出回放不需要旁边存在 MozBoxer 仿真仓库。
