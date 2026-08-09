"""多相机渲染器 — mujoco_camrender RenderPool 并行（RGB + 深度线性化）。

背景：
  - camrender 的 RGB 与 mujoco.Renderer 逐像素一致（已验证）。
  - camrender 的深度读取原始 GL 深度缓冲（NDC 值），可通过透视投影公式
    线性化恢复真实距离：z = 2·near·far / (far + near - (2·raw - 1)·(far - near))。
    已对单臂/双臂所有相机拟合验证（near=0.01·extent, far=50·extent，误差 ~0）。
  - 深度单位为米（float32），交给 LeRobot 编码器按 depth_min/depth_max 量化。
  - camrender 不可用时回退 mujoco.Renderer（串行，同样正确）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# 无头（无 DISPLAY，如 SSH 服务器）：camrender 改用 EGL 后端离屏渲染
# （无需 X 服务，需 GPU + EGL 驱动，如 NVIDIA）。
_HEADLESS = not bool(os.environ.get("DISPLAY"))

try:  # pragma: no cover - 取决于 C++ 绑定是否构建
    from mujoco_camrender import (
        MultiCameraRenderer,
        CameraParams as CamRenderParams,
        RenderMode as CamRenderMode,
    )
    _HAS_CAMRENDER = True
except Exception:  # pragma: no cover
    _HAS_CAMRENDER = False

# MuJoCo 默认渲染近/远平面（与模型统计 extent 相关）
_NEAR_SCALE = 0.01
_FAR_SCALE = 50.0


@dataclass
class RenderedFrame:
    name: str
    rgb: np.ndarray    # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 米制


class CameraRenderer:
    """多相机渲染器（camrender 并行，深度线性化）。"""

    def __init__(
        self,
        cameras: list,
        model,
        data,
        num_threads: int | None = None,
        backend: str = "opengl",
    ) -> None:
        import mujoco

        # 无头时确保 mujoco.Renderer 走 EGL 离屏渲染（需 GPU + EGL 驱动）
        if _HEADLESS:
            os.environ.setdefault("MUJOCO_GL", "egl")

        self.cameras = cameras
        self._num_threads = num_threads or max(2, len(cameras))
        self._use_camrender = False
        self._camrender = None
        self._mj_renderers: dict[str, object] = {}
        # eval 视频自由视角渲染器（render_view 专用，惰性创建）
        self._view_renderer = None
        self._model = model
        self._data = data

        # 线性化参数（MuJoCo 默认渲染近/远平面）
        ext = float(model.stat.extent)
        self._near = _NEAR_SCALE * ext
        self._far = _FAR_SCALE * ext

        # 注意：不要在此处创建 mujoco.Renderer（会污染 camrender 的 GL 状态），
        # 仅在 camrender 不可用/无头时惰性创建。
        self._cam_ids: dict[str, int] = {
            c.name: int(model.camera(c.name).id) for c in cameras
        }

        # 无头（无 DISPLAY）时 camrender 使用 EGL 后端离屏渲染（无需显示，
        # 需 GPU + EGL 驱动），保持并行渲染性能。
        effective_backend = "egl" if _HEADLESS else backend
        if _HAS_CAMRENDER:
            try:
                self._camrender = MultiCameraRenderer(
                    model, data, num_threads=self._num_threads, backend=effective_backend
                )
                # 应用 yaml 图像尺寸到各相机（覆盖 mjcf 默认分辨率）。
                # 显式指定为固定相机（mjCAMERA_FIXED=2）+ 模型相机索引，
                # 避免部分覆盖时被默认 FREE 相机覆盖。
                for c in cameras:
                    params = CamRenderParams()
                    params.name = c.name
                    params.width = c.width
                    params.height = c.height
                    params.cam_type = 2  # mjCAMERA_FIXED
                    params.fixedcamid = self._cam_ids[c.name]
                    self._camrender.set_camera_params(self._cam_ids[c.name], params)
                self._use_camrender = True
                log.info(
                    "CameraRenderer: 使用 mujoco_camrender 并行 "
                    "(%d 相机, %d 线程, backend=%s, near=%.3f far=%.1f)",
                    len(cameras), self._num_threads, effective_backend,
                    self._near, self._far,
                )
            except Exception as exc:  # pragma: no cover
                log.warning("mujoco_camrender 初始化失败，回退 mujoco.Renderer: %s", exc)
                self._use_camrender = False

        if not self._use_camrender:
            import mujoco
            for c in cameras:
                self._mj_renderers[c.name] = mujoco.Renderer(
                    model, height=c.height, width=c.width
                )
            log.info("CameraRenderer: 使用 mujoco.Renderer（串行）")

    @property
    def using_camrender(self) -> bool:
        return self._use_camrender

    def render_view(self, view) -> np.ndarray:
        """用 view 配置（MjvCamera 自由视角）渲染一帧 RGB，供 eval 视频使用。

        走 MuJoCo 原生 Renderer + MjvCamera（lookat/distance/elevation/azimuth），
        与数据采集的固定相机（render_all）相互独立，互不影响。
        返回 (H, W, 3) uint8。
        """
        import mujoco

        if self._view_renderer is None:
            self._view_renderer = mujoco.Renderer(
                self._model, height=view.height, width=view.width
            )
        # 尺寸变化时重建
        r = self._view_renderer
        if r.height != view.height or r.width != view.width:
            r.close()
            self._view_renderer = mujoco.Renderer(
                self._model, height=view.height, width=view.width
            )
            r = self._view_renderer

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = view.lookat
        cam.distance = view.distance
        cam.elevation = view.elevation
        cam.azimuth = view.azimuth

        r.disable_depth_rendering()
        r.update_scene(self._data, camera=cam)
        return r.render()

    def render_all(self, data) -> list[RenderedFrame]:
        """渲染全部相机（RGB + 深度），返回按配置顺序排列的帧。"""
        names = [c.name for c in self.cameras]

        if self._use_camrender:
            self._camrender.update_scene(data)
            outs = self._camrender.render_all(
                [self._cam_ids[n] for n in names], CamRenderMode.All
            )
            frames: list[RenderedFrame] = []
            for c, out in zip(self.cameras, outs):
                depth = self._linearize_depth(out["depth"])
                frames.append(
                    RenderedFrame(name=c.name, rgb=out["rgb"], depth=depth)
                )
            return frames

        # 回退：mujoco.Renderer 串行
        import mujoco
        frames = []
        for c in self.cameras:
            r = self._mj_renderers[c.name]
            cam_id = self._cam_ids[c.name]
            r.disable_depth_rendering()
            r.update_scene(data, camera=cam_id)
            rgb = r.render()
            r.enable_depth_rendering()
            r.update_scene(data, camera=cam_id)
            depth = r.render()
            frames.append(RenderedFrame(name=c.name, rgb=rgb, depth=depth))
        return frames

    def _linearize_depth(self, raw: np.ndarray) -> np.ndarray:
        """将 camrender 原始 NDC 深度线性化为真实距离（米）。"""
        near, far = self._near, self._far
        r = np.clip(raw, 0.0, 1.0)
        z = (2.0 * near * far) / (far + near - (2.0 * r - 1.0) * (far - near))
        # 远平面外（背景）置为 far
        z[~(np.isfinite(z))] = far
        return z.astype(np.float32)

    def close(self) -> None:
        self._camrender = None
        for r in self._mj_renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._mj_renderers.clear()
        if self._view_renderer is not None:
            try:
                self._view_renderer.close()
            except Exception:
                pass
            self._view_renderer = None
