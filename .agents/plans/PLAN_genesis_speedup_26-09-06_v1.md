# Genesis FEM/SAP 三项新加速的实施与一次性验收计划

## 一、目标、当前基线与不可变边界

在当前保留旧优化 1/2/5 的实现上，一次性加入三项新优化：

1. 融合 implicit FEM 六刚体模态 PCG 的数据遍历。
2. 在同一物理子步内复用 FEM 四面体裁剪平面。
3. 用按 rigid geom 建立的局部坐标静态 BVH forest，替代 rigid--FEM 路径每子步重建的世界坐标 LBVH。

当前 Genesis 分支为 `guanxiong/main`，基准提交为 `884f475dbbe6038eb4639ca590d595904c77ceca`。工作树已有的 `sap_coupler.py` 修改是要保留的旧优化 1/2/5（active-prefix contact export、compact whitelist、ancestor-sparse rigid contact algebra）；不得整体还原该文件。当前 `fem_solver.py` 与 `bvh.py` 无工作树修改。HAG4R-MPC 的现有修改也保持原样；本轮不需要修改 HAG4R 源码或配置。

保持以下运行与物理合同不变：

- USB hub 网格、材料、质量、FP64 精度。
- `B=2`、`H=565`、60 Hz 控制频率、每控制步 10 个物理子步。
- FEM/SAP 迭代上限、收敛阈值、接触和摩擦模型。
- HAG4R 控制轨迹、相机、渲染频率和现有遥测逻辑。
- query result、candidate、pair 以及求解缓冲区的现有绝对容量和 overflow 语义。
- 非 rigid-mode FEM PCG、true-residual probe、其他接触类型及其通用 BVH。

不增加 feature flag、旧路径 fallback、动态性能 gate、运行时算法选择、新诊断、新 guard/checker 或防御性分支；不加入 CUDA Graph、warm start、材料 coarse modes、BSR operator 或其他顺手优化。

历史参照固定使用已有 `usb_speed_without34`，不重跑 baseline：

- rollout loop：`2319.699810778955 s`
- end-to-end：`2365.73 s`

参考资料：

- [三项新优化的圆桌报告](/home/eric/research/genesis-world/.agents/reports/2026-09-06_genesis_fem_sap_new_speedup_methods_roundtable.md)
- [当前 implicit FEM PCG](/home/eric/research/genesis-world/genesis/engine/solvers/fem_solver.py:1345)
- [rigid--FEM tet/triangle contact](/home/eric/research/genesis-world/genesis/engine/couplers/sap_coupler.py:7374)
- [当前通用 LBVH](/home/eric/research/genesis-world/genesis/engine/bvh.py:68)
- [本分支最新复用知识](/home/eric/research/genesis-world/.agents/knowledge/guanxiong__main.md:466)
- [USB hub 配置](/home/eric/research/HAG4R-MPC/configs/control_usb_hub.yaml)
- [历史 receipt](/home/eric/research/HAG4R-MPC/outputs/usb_speed_without34/run_receipt.json)

## 二、Step-by-step implementation instructions

### Step 1：融合 rigid-mode PCG 数据流

只修改 `genesis/engine/solvers/fem_solver.py` 的 `FEMSolver.pcg_solve()` rigid-mode 分支及其私有 helper；矩阵无组装 `compute_Ap()`、六模态 coarse operator、零初值、归约量、breakdown/active 更新和停止公式不变。

初始化严格保留三个有序 Quadrants 入口：

1. 重写 `_init_pcg_solve_rigid_mode()`：初始化 batch PCG 状态并清零 `rigid_mode_coarse_rhs`；在同一次顶点遍历中完成 `x=0`、`r=force`、`z=prec @ r`，并按 `rigid_mode_component_by_vertex` 累加六维 coarse RHS。
2. 保留独立的 `_solve_rigid_mode_coarse_rhs()`。coarse RHS 的全局累加必须在该入口前完成，coarse coefficient 必须在下一入口前完成；不得跨越该屏障融合。
3. 重写 `_finish_init_pcg_solve_rigid_mode()`：在一次顶点遍历中将 coarse correction 加到 `z` 并累计 `rTr/rTz`，随后保持原公式设置 `rTr_initial`、`termination_threshold`、breakdown/active，最后令仍 active 的 `p=z`。

每轮迭代严格使用四个有序入口：

1. `_rigid_mode_compute_Ap_and_pTAp()` 同时为进入本轮时仍 active 的 batch 增加 `batch_pcg_iterations`、执行 `compute_Ap(False)` 并归约 `pTAp`。
2. 新的 `_rigid_mode_update_x_r_and_coarse_rhs()` 先按原条件计算/校验 `alpha` 并清零 coarse RHS，再在一次顶点遍历中更新 `x/r`、计算 block `z=prec @ r`、累加六维 coarse RHS。
3. 独立调用 `_solve_rigid_mode_coarse_rhs()`。
4. 新的 `_finish_rigid_mode_pcg_iter_and_update_p()` 在一次顶点遍历中加回 coarse correction 并累计 `rTr_new/rTz_new`，再按原顺序计算/校验 `beta`、发布 `rTr/rTz`、更新 breakdown/active，最后在同一入口的末次顶点遍历更新 active batch 的 `p`。

`pcg_solve()` 的 rigid-mode 分支直接调用上述三入口初始化与四入口迭代；true-residual probe 仍在原 schedule 位置调用。非 rigid-mode 分支继续使用 `_count_active_pcg_iterations()` 和 `one_pcg_iter()`，不得随本次融合改写。

删除只为旧碎片化 rigid-mode 路径服务的 `_apply_rigid_mode_block_preconditioner()`、`_accumulate_rigid_mode_rhs()`、`_add_rigid_mode_correction()`、`_apply_rigid_mode_preconditioner()`、`_rigid_mode_update_x_r()`、`_finish_rigid_mode_pcg_iter()`、`_update_rigid_mode_pcg_direction()` 和 `_one_rigid_mode_pcg_iter()`；不保留新旧两套 rigid-mode 流程。保留 `_count_active_pcg_iterations()`，因为普通 PCG 仍使用它。

### Step 2：在同一 `update_contact` kernel 内复用 tet 裁剪平面

修改 `genesis/engine/couplers/sap_coupler.py` 的 `RigidFemTriTetContactHandler`。在 `__init__()` 分配三个内部 field：

- `active_tet: bool[B, n_elements]`
- `tet_clip_plane_points: vec3[B, n_elements, 4]`
- `tet_clip_plane_normals: vec3[B, n_elements, 4]`

不得把 plane population 做成从 `detection()` 内嵌套启动的 `@qd.kernel`。`detection()` 当前是 `SAPCoupler.update_contact()` 这个外层 `@qd.kernel` 中调用的 `@qd.func`；因此 candidate creation、plane population、pair clipping 都必须是该外层入口内按顺序执行的 `@qd.func` 阶段：

1. `compute_candidates()` 在清零现有 candidate counters 的同时清空 `active_tet`。
2. `_append_candidate()` 只有在 candidate slot 成功保留并写完后，才把 `(batch_idx, fem_element_idx)` 的 `active_tet` 写为 `True`；重复写 `True`，不引入原子计数、unique-tet list 或 scan。
3. 新增 `_populate_active_tet_clip_planes(f)`（`@qd.func`），遍历 `(B, n_elements)`，仅处理 active tet。每个 face 继续使用 `x = tet_vertices[:, (face + 1) % 4]`、`(v[(face + 2) % 4] - x).cross(v[(face + 3) % 4] - x) * [1,-1,1,-1][face]`、`normal /= normal.norm()`；顶点顺序、符号和归一化公式逐字保持，不添加退化面检查。
4. `compute_pairs()` 的四次裁剪只读取缓存的 point/normal。三角形先照旧加载；只有裁剪后 `polygon_n_vertices >= 3` 才加载完整 `tet_vertices` 和 `tet_pressures`，用于面积阈值、重心坐标、压力和刚度的原公式。
5. `detection()` 顺序固定为 local-forest query → `compute_candidates()` → `_populate_active_tet_clip_planes()` → `compute_pairs()`。

不缓存重心坐标、边长、压力或 contact result。保留 7-vertex polygon 容量、candidate/pair reservation 次序、attempted/dropped/overflow counters 和绝对容量公式。

### Step 3：加入每个 rigid geom 的局部静态 BVH forest

#### 3.1 `genesis/engine/bvh.py`

新增内部类 `RigidLocalTriBVHForest`，不修改 `LBVH`、`FEMSurfaceTetLBVH` 或 `RigidTetLBVH` 的行为。

构建期在 CPU 上按 whitelist-enabled collision geom 分树，直接使用每个 `geom.init_verts` 与 `geom.init_faces`。叶子保存现有 compact face index；compact-to-global face mapping 继续由 SAP 持有。每棵树按以下确定性规则构建一次：

- 三角面局部 AABB 与 centroid 从 init mesh 计算。
- 选择 centroid extent 最大的轴；并列时取最低轴。
- 按 `(centroid[axis], compact_face_index)` 排序，从 `n // 2` 划分，直到单面叶子。
- 以前序顺序 flatten；`geom_root` 是首节点，`geom_end` 是 exclusive end；每个节点的 `escape` 指向其子树后的首节点，整棵树的 escape 为 `geom_end`。

GPU 只保留以下 forest 数据：

- `geom_indices[n_enabled_geoms]`：forest slot 到 global geom index。
- `geom_root[n_enabled_geoms]`、`geom_end[n_enabled_geoms]`。
- `node_aabb_min[n_nodes]`、`node_aabb_max[n_nodes]`。
- `node_compact_face[n_nodes]`：internal 为 `-1`，leaf 为 compact face index。
- `node_escape[n_nodes]`。
- `world_root_aabb_min/max[B, n_enabled_geoms]`。
- 与现有 rigid-triangle BVH 完全相同绝对上限的 `query_result` 和 `query_result_count`；record 仍为 `(batch, compact_face, surface_tet_index)`。

每个物理子步只更新 root world boxes：对每个 `(batch, forest geom)`，先沿用 `links_info.geom_start/geom_end` 判断该 global geom 是否属于该环境；不属于时写 empty box，属于时把局部 root AABB 的八个角按当前 `geoms_state.pos/quat` 变换到世界坐标并取 min/max。

forest query 是供 contact `detection()` 调用的 `@qd.func`，不是嵌套 kernel。对每个 FEM surface tet：

1. 先用现有 world tet AABB 与各有效 geom 的 world root AABB 相交测试。
2. 对通过的 geom，用 `gu.qd_inv_transform_by_trans_quat` 将该 tet 当前四个真实顶点变换到 geom local frame，并生成 local tet AABB。
3. 从 `[geom_root, geom_end)` stackless 遍历：命中 internal 时走下一前序节点，未命中时跳 `node_escape`，命中 leaf 时取得 compact face。
4. 在预留 query record 之前，将 compact face 映射到 global face，读取当前世界三角形顶点并重算 leaf 的 world AABB；只有它与原 world tet AABB 相交才按原 counter/capacity 语义写入 record。该最终世界 leaf recheck 不得移到 capacity reservation 之后。

#### 3.2 `genesis/engine/couplers/sap_coupler.py`

- 从 `genesis.engine.bvh` 导入 `RigidLocalTriBVHForest`，并在 rigid--FEM `_init_bvh()` 分支以现有 `old_max_query_results` 绝对值构造 `self.rigid_local_tri_bvh_forest`。
- `update_bvh()` 保留 FEM surface-tet BVH 与 rigid-tet BVH 的原更新，仅把 rigid--FEM 的世界 root-box 更新交给 forest；不再构建 rigid-triangle LBVH。
- `RigidFemTriTetContactHandler.detection()` 调用 forest query。candidate 阶段仍立即用 `rigid_fem_compact_to_global_face` 恢复 global face，再执行现有世界三角形 normal、pressure-gradient direction 和 tet 跨平面判断。
- forest 只产生 whitelist leaf，因此删除 candidate/pair 阶段重复的 device-side `rigid_fem_face_enabled` 检查；保留该 field 及 public export 中已有的 whitelist identity 检查，因为 receipt/public contract 仍使用它。
- 将 `_compute_candidates_legacy_view()` 改名并收敛为唯一的 local-forest candidate consumer；不保留 legacy world-LBVH fallback。

删除 rigid--FEM 专用的 `rigid_tri_aabb`、`rigid_tri_bvh`、`update_rigid_tri_bvh()`、`compute_rigid_tri_aabb()` 及随之未使用的 `LBVH` import。candidate、pair、Jacobian/Delassus、public export 的现有容量和字段不缩减；通用 BVH 类与其他 contact handler 不动。运行时只支持 scene build 后 mesh/scale 固定、pose 变化这一现有目标合同，不增加动态重建分支。

### Step 4：合并与死代码收口

最终生产修改只涉及：

- `genesis/engine/solvers/fem_solver.py`
- `genesis/engine/couplers/sap_coupler.py`
- `genesis/engine/bvh.py`

在当前 dirty `sap_coupler.py` 上增量修改并保留旧优化 1/2/5。清除上述明确被替代的 helper、field、import 和 legacy consumer 名称，不触碰 HAG4R、USB YAML、控制轨迹、相机或渲染频率。

## 三、仅保留的静态与编译准备

三项全部实现后只做一次 plan-consistency/static audit：审阅最终 diff，运行 `git diff --check`，并确认以下事实：四入口 rigid-mode 顺序及 coarse barrier 存在；普通 PCG/true-residual probe 未改；contact 三阶段在同一外层 kernel 内有序；forest 的 compact/global mapping、heterogeneous geom ownership、world leaf recheck 和旧绝对容量保留；generic BVH 与旧优化 1/2/5 未被删除；旧 rigid-triangle rebuild 和碎片化 rigid-PCG symbols 已消失。该审阅不增加运行时 checker 或测试代码。

随后只运行达到正式启动所需的 Python syntax compile：

```bash
/home/eric/research/HAG4R-MPC/.conda/hag4rmpc-genesis-cu126/bin/python -m py_compile \
  /home/eric/research/genesis-world/genesis/engine/solvers/fem_solver.py \
  /home/eric/research/genesis-world/genesis/engine/couplers/sap_coupler.py \
  /home/eric/research/genesis-world/genesis/engine/bvh.py
```

不运行 import/scene smoke、单元测试、短 horizon、单项 rollout、baseline、profiler 或逐项计时。Quadrants 的目标路径编译随唯一正式 full run 发生。

HAG 环境中的 Genesis 是 non-editable wheel。syntax compile 后仅刷新本地 Genesis wheel，不同步/安装其他依赖：

```bash
/snap/bin/uv pip install /home/eric/research/genesis-world \
  --python /home/eric/research/HAG4R-MPC/.conda/hag4rmpc-genesis-cu126/bin/python \
  --no-deps --reinstall \
  --build-constraints /home/eric/research/HAG4R-MPC/environment/genesis-cu126-build-constraints.txt \
  --no-python-downloads --no-config
```

正式运行前用一次只读 inline Python provenance 检查确认：`genesis.__file__` 位于该 conda 环境的 `site-packages`、distribution 不是 editable，且 installed/source 的 `engine/bvh.py`、`engine/couplers/sap_coupler.py`、`engine/solvers/fem_solver.py` SHA-256 分别相等。记录路径和三组 hash；不新增 checker 文件。

## 四、唯一一次 USB-hub integrated run

三项合并、静态审阅、syntax compile 和 wheel provenance 完成后，只启动一次完整 `B=2、H=565` run；不传 `--control-steps`，不先运行 baseline：

```bash
cd /home/eric/research/HAG4R-MPC
mkdir -p outputs/usb_speed_pcg_planes_local_bvh
/usr/bin/time -f '%e' \
  -o outputs/usb_speed_pcg_planes_local_bvh/end_to_end_seconds.txt \
  ./.conda/hag4rmpc-genesis-cu126/bin/python \
  scripts/run_waypoint_control.py \
  --config configs/control_usb_hub.yaml \
  --experiment-id usb_speed_pcg_planes_local_bvh \
  > outputs/usb_speed_pcg_planes_local_bvh/run.log 2>&1
```

本轮只接受这一份三项合并后的完整 rollout 作为运行证据；不做逐项归因或追加性能 run。

## 五、MP4-first 验收、receipt/telemetry 与历史计时

### 1. 先看 MP4

运行完成后，在读取新计时前，完整检查两个新 `rollout.mp4`，并与 `usb_speed_without34` 的同名历史视频对照。只保留六个关键画面：

| 视频帧 | 控制步 | 阶段 |
|---:|---:|---|
| 0 | 1 | 初始 |
| 74 | 149 | 稳定抓取 |
| 109 | 219 | 抬升完成 |
| 176 | 353 | 降低物体 |
| 202 | 405 | 释放 |
| 281 | 563 | 最终 |

视觉标准：网格连续且无爆炸/突然拉长/高速飞走；无明显穿过夹爪、桌面或托盘壁；抓取、抬升、搬运、降低、释放顺序正常且接触后无持续高频抖动。`0.024` 仍完成放置；`0.028` 至少保持历史结果（主体进入托盘，尾部或 connector 可像历史视频一样留在托盘外，但不得明显恶化）。视觉不合格时将结果报告为物理质量不合格，不因速度数字改变结论。

### 2. 只读取本次 full run 已生成的 receipt/telemetry

- `run_receipt.json`：`B == 2`、`H == 565`、`completed_steps == 565`、`simulation_substeps_per_control_step == 10`，并列出两个 case 的 MP4、commands 与 telemetry artifact。
- 两个 telemetry 的 `step == 219` 行：`object_part0_centroid_m.z >= 0.045 m`。
- 两个 telemetry 的最终 `step == 564` 行：part-0 centroid 满足 `x ∈ [0.127, 0.387] m`、`y ∈ [0.116, 0.296] m`、`z ∈ [0.0, 0.05] m`。

不要求与历史逐帧坐标、contact 数或归约顺序完全相等，也不从这些 artifact 派生新的诊断 gate。

### 3. 对固定历史 baseline 计算计时

从新 receipt 读取 `T_new = rollout_loop_wall_s`，从新 `end_to_end_seconds.txt` 读取 `T_e2e,new`：

\[
S_{\text{loop}}=\frac{2319.699810778955}{T_{\text{new}}},\qquad
R_{\text{loop}}=\left(1-\frac{T_{\text{new}}}{2319.699810778955}\right)\times100\%
\]

\[
S_{\text{e2e}}=\frac{2365.73}{T_{\text{e2e,new}}}
\]

计时只作结果报告，不设 `1.05x` 等动态通过门槛，也不因结果快慢重跑。最终报告仅列历史 1/2/5 与新三项合并版本的 loop/E2E 时间、加速倍数、loop 降低百分比、wheel/source provenance，以及两个 case 的 MP4 和 telemetry 结论。
