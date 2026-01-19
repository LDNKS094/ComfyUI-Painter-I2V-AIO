我有初步构思了，最终应该保留三个初始化节点

1.PainterI2V

使用场景：
- 无前置motion_frames的情况下生成视频
- 支持无首帧、仅首帧、首尾帧模式生成
- 双conditioning输出
- 基础节点

2.PainterI2V Extend(原PainterLongVideo)

使用场景：
- 有前置视频，可以是外部load的视频，也可以是模型生成前置视频
- 支持仅首帧，首尾帧模式
- 使用reference_latent类似机制控制一致性
- 基础节点

3.PainterI2V Advanced

使用场景：
- 完全支持所有使用场景，通过输入参数是否存在自动切换合适的模式
- 支持无首帧、仅首帧、首尾帧模式生成
- 支持视频接续
- 支持4 conditioning输出
- 支持无损接续（输入输出latent）
- 高级节点，要求使用者掌握原理

把wan22fmlf里合适的技术思路加入以上三个节点。
使用场景：
- 仅使用painterI2V做简单生成
- 仅使用painterI2V Extend做视频延续
- 使用painterI2V 做启动视频 + PainterI2V Extend 循环生成长视频，注重简单且低理解成本
- 仅使用painterI2V advanced 结合loop实现自由控制，包括初始启动，要求用户自己处理latent以实现无损
