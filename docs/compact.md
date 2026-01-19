会话总结 - ComfyUI-PainterAIO 节点重构项目
项目位置
F:\AI_TOOLS\ComfyUI-aki-v2\ComfyUI\custom_nodes\ComfyUI-PainterAIO.disabled\
已完成工作
1. 项目结构创建
- 合并了 6 个 Painter 系列上游仓库到 ref/ 目录
- 创建了 modules/ 目录存放重构后的节点代码
- 统一入口 __init__.py 支持 ComfyExtension API + 传统 API
2. PainterI2V 节点重构 (已完成)
- 文件: modules/painteri2v/nodes.py
- 合并了原 PainterFLF2V 功能
- 三模式自动切换：T2V / I2V / FLF2V
- 添加了 enable_reference_latent debug 开关
- 输入: start_image, end_image, clip_vision_start, clip_vision_end
- 输出: 2 conditioning + latent
3. 深度分析文档
- docs/NODE_ANALYSIS.md: 完整的 Painter vs Wan22FMLF 机制对比分析
- docs/IMPLEMENTATION_PLAN.md: 三节点重构规划
核心分析结论
Painter vs Wan 设计差异
| 维度 | Painter 系列 | Wan22FMLF |
|------|-------------|-----------|
| 参考机制 | reference_latents (软引导) | 仅 concat_latent (硬约束) |
| 运动增强 | latent 层面 (频率分离) | mask 层面 (空间梯度) |
| 阶段分离 | latent 分离 (增强/原始) | mask 强度分离 |
| 续接 | IMAGE 空间 + 软引导 | 支持 LATENT 无损续接 |
| 输出 | 2 或 4 cond | 3 cond |
关键发现
1. SVI 模式中 start_image 作为风格锚点，prev_latent 作为续接参考，类似 PainterLongVideo 的设计
2. reference_latents 可能影响运动幅度（控制度过高），已添加 debug 开关测试
3. Wan 不使用 reference_latents 字段，全部依赖 concat_latent_image
三节点重构规划
1. PainterI2V (基础-首发)
   - 无前置视频
   - T2V / I2V / FLF2V 模式
   - 2 cond 输出
   - [已基本完成]
2. PainterI2V Extend (基础-续接)
   - 需要 previous_video 输入
   - reference_motion + reference_latents
   - 2 cond 输出
   - [待实现，基于 PainterLongVideo 重构]
3. PainterI2V Advanced (高级-全功能)
   - 支持所有场景
   - 支持无损 latent 续接 (借鉴 Wan SVI)
   - 4 cond 输出
   - 色彩保护算法
   - [待实现]
关键文件状态
| 文件 | 状态 |
|------|------|
| modules/painteri2v/nodes.py | ✅ 已重构，含 debug 开关 |
| modules/painteri2v_advanced/nodes.py | 原版，待重构 |
| modules/painterlongvideo/nodes.py | 原版，待重构为 Extend |
| modules/painterflf2v/ | 已废弃，功能合并到 painteri2v |
| modules/paintersampler/ | 保持原样 |
| modules/paintersampler_advanced/ | 保持原样 |
| docs/NODE_ANALYSIS.md | ✅ 完整分析文档 |
| docs/IMPLEMENTATION_PLAN.md | ✅ 实现规划 |
| docs/Just_notes.md | 用户的设计笔记 |
待确认问题
1. Extend 是否需要 motion_amplitude？
2. Advanced 的 output_latent 是否需要单独输出？
3. 是否加入 Wan 的空间梯度增强作为可选项？
4. 命名：PainterI2V Extend vs PainterI2V Continue？
下一步工作
1. 确认规划 - 回答待确认问题
2. 实现 PainterI2V Extend - 基于 PainterLongVideo 重构
3. 实现 PainterI2V Advanced - 整合所有高级功能
4. 测试 - 验证三节点工作流
技术要点备忘
- reference_motion: 从 previous_video 最后 73 帧提取，编码后取 19 个 latent timesteps
- 频率分离增强: high_freq = diff - low_freq; result = original + high_freq * boost
- 色彩保护: 检测通道漂移 > 18%，微调修正 + 亮度保护
- 无损续接: prev_latent 直接注入，跳过 VAE encode/decode
- 双 CLIP concat: 首尾帧 CLIP hidden states 在 sequence 维度拼接
用户偏好
- 不考虑中间帧控制功能
- 倾向简单分层设计（基础 + 高级）
- 保留 Painter 的软引导优势
- 借鉴 Wan 的无损续接能力