"""PushT 鼠标遥操作专属模块 — pygame 2D 俯视窗口。

本模块全部为 **push_t teacher 专属**（不进入通用 simulate 组件）：
- ``TeleopParams`` / ``TeleopSnapshot`` / ``TeleopState``：线程安全纯逻辑
  （不依赖 pygame，可无头测试）。窗口线程写入（鼠标增量 / 事件 / 灵敏度），
  仿真线程读取（期望 TCP）并回写物理状态（真实 TCP、T 物块位姿、录制状态…）。
- ``TeleopWindow``：pygame 渲染线程。**不渲染 3D 场景**，而是把世界坐标
  (x, y) 直接映射到屏幕像素做 2D 俯视投影，只绘制关心的元素
  （桌面范围 / T 物块 / 目标 T / 当前 TCP / 期望 TCP / 状态栏），
  因此不存在机械臂遮挡问题。

录制生命周期由 ``PushTTeacher.recording_decision`` 控制（见 run_episode 文档）：
本模块只提供共享状态与窗口，不涉及 episode 驱动循环。

鼠标交互：
- 左键      切换"鼠标控制期望 TCP"（开启时鼠标被锚定，FPS 式增量移动）
- 右键      开始 / 结束并保存当前 episode
- 中键      丢弃当前 episode（仅录制中）
- 滚轮      TCP 上下移动（裁剪到 [z_min, z_max]）
- [ / ]     调节鼠标灵敏度
- - / =     调节滚轮灵敏度系数
- Esc       退出
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np


# ═══════════════════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════════════════

@dataclass
class TeleopParams:
    """遥操作窗口参数（对应 teacher 配置中的 window / workspace / sens）。"""

    window_width: int = 800
    window_height: int = 600
    workspace_x: tuple[float, float] = (-0.6, 0.6)   # 世界 x 范围（全部展示）
    workspace_y: tuple[float, float] = (-0.45, 0.45)  # 世界 y 范围（全部展示）
    mouse_sens: float = 1.5      # 鼠标灵敏度（像素 → 世界比例系数）
    wheel_sens: float = 0.008    # 滚轮灵敏度（每格 → 世界 z 增量，米）
    sens_step: float = 0.1       # 灵敏度调节步长
    tcp_z_min: float = 0.62      # TCP 高度下限
    tcp_z_max: float = 0.85      # TCP 高度上限
    push_z_min: float = 0.66     # 可推动高度带下限（TCP z）
    push_z_max: float = 0.72     # 可推动高度带上限（TCP z）
    # 鼠标增量方向反转（环境修正）：某些环境（如 grab 模式下 XInput2
    # raw motion 轴映射异常）SDL 报告的鼠标增量与物理移动方向相反，
    # 表现为"鼠标右移但期望 TCP 左移"。默认 False 为标准方向。
    invert_mouse_x: bool = False
    invert_mouse_y: bool = False

    _margin_px: int = 30         # 绘图留边（像素）
    _status_h: int = 56          # 底部状态栏高度（像素）


# ═══════════════════════════════════════════════════════
# 共享状态
# ═══════════════════════════════════════════════════════

@dataclass
class TeleopSnapshot:
    """窗口渲染用的只读快照（仿真线程 publish 的最新物理状态）。"""

    desired: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tcp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    t_obj_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    t_obj_yaw: float = 0.0
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sim_time: float = 0.0
    recording: bool = False
    control: bool = False
    pushable: bool = False
    mouse_sens: float = 1.0
    wheel_sens: float = 0.01
    message: str = ""


class TeleopState:
    """PushT 遥操作共享状态（线程安全）。

    窗口线程（写入）：``toggle_control`` / ``apply_mouse_delta`` /
    ``wheel_scroll`` / ``request_*`` / ``adjust_*_sens``
    仿真线程（读取/写入）：``desired`` / ``consume_*`` / ``publish``
    """

    def __init__(self, params: TeleopParams | None = None) -> None:
        self.params = params or TeleopParams()
        p = self.params
        self._lock = threading.Lock()
        self._desired = np.zeros(3, dtype=np.float64)
        self._desired[2] = p.tcp_z_min + 0.05
        self._tcp = np.zeros(3, dtype=np.float64)
        self._t_obj_pos = np.zeros(3, dtype=np.float64)
        self._t_obj_yaw = 0.0
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._sim_time = 0.0
        self._recording = False
        self._control = False
        self._pushable = False
        self._mouse_sens = p.mouse_sens
        self._wheel_sens = p.wheel_sens
        self._message = ""
        self._start = False
        self._stop = False
        self._discard = False
        self._quit = False
        self._published = False  # 仿真线程是否已 publish 过物理状态

    # ── 坐标变换（窗口渲染与鼠标增量共用，保证一致） ──

    def screen_transform(self) -> tuple[float, float, float]:
        """返回 (scale, u0, v0)：世界 x/y → 屏幕 u/v 的线性映射参数。

        u = u0 + (x - ws.x0) * scale
        v = v0 + (ws.y1 - y) * scale   （y 轴翻转）
        """
        p = self.params
        span_x = p.workspace_x[1] - p.workspace_x[0]
        span_y = p.workspace_y[1] - p.workspace_y[0]
        avail_w = p.window_width - 2 * p._margin_px
        avail_h = p.window_height - 2 * p._margin_px - p._status_h
        scale = min(avail_w / span_x, avail_h / span_y)
        return scale, float(p._margin_px), float(p._margin_px)

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        scale, u0, v0 = self.screen_transform()
        p = self.params
        u = u0 + (x - p.workspace_x[0]) * scale
        v = v0 + (p.workspace_y[1] - y) * scale
        return u, v

    # ── 窗口线程写入 ──────────────────────────────────

    def toggle_control(self, anchor: np.ndarray | None = None) -> None:
        """切换鼠标控制。开启时将期望 TCP 锚定到 anchor（当前 TCP 位置）。

        防御：仿真线程尚未 publish 时（``_published`` 为 False，tcp 为全零），
        不锚定 x/y/z —— 否则黄十字会被直接赋值为原点 (0,0)，表现为
        "跳变到固定位置"。anchor 的 z 也只在合理高度范围才覆盖期望 z。
        """
        with self._lock:
            self._control = not self._control
            if self._control:
                if anchor is not None and self._published:
                    a = np.asarray(anchor, dtype=np.float64)
                    if a.size >= 2:
                        self._desired[:2] = a[:2]
                        p = self.params
                        if (np.isfinite(a[2])
                                and p.tcp_z_min - 0.05 <= a[2] <= p.tcp_z_max + 0.05):
                            self._desired[2] = np.clip(
                                a[2], p.tcp_z_min, p.tcp_z_max
                            )
                self._message = "Mouse control: ON (move mouse to steer TCP)"
            else:
                self._message = "Mouse control: OFF"

    def anchor_desired(self, pos: np.ndarray) -> None:
        """将期望 TCP 锚定到给定位置（仿真线程在 episode 开始时调用）。"""
        a = np.asarray(pos, dtype=np.float64)
        with self._lock:
            if a.size >= 2:
                self._desired[:2] = a[:2]
            if a.size >= 3 and np.isfinite(a[2]):
                self._desired[2] = np.clip(
                    a[2], self.params.tcp_z_min, self.params.tcp_z_max
                )

    def apply_mouse_delta(self, dx_px: float, dy_px: float) -> None:
        """鼠标增量（像素）→ 期望 TCP 水平面增量（世界）。仅控制开启时生效。

        方向修正：某些环境 grab 模式下 SDL 报告的鼠标增量与物理移动相反
        （XInput2 raw motion 轴映射异常），按 ``invert_mouse_x/y`` 反转。
        """
        if not dx_px and not dy_px:
            return
        with self._lock:
            if not self._control:
                return
            if self.params.invert_mouse_x:
                dx_px = -dx_px
            if self.params.invert_mouse_y:
                dy_px = -dy_px
            scale, _, _ = self.screen_transform()
            k = self._mouse_sens / scale
            self._desired[0] += dx_px * k
            self._desired[1] -= dy_px * k
            p = self.params
            self._desired[0] = float(np.clip(
                self._desired[0], p.workspace_x[0], p.workspace_x[1]))
            self._desired[1] = float(np.clip(
                self._desired[1], p.workspace_y[0], p.workspace_y[1]))

    def wheel_scroll(self, amount: int) -> None:
        """滚轮：控制期望 TCP 的 z（裁剪到 [z_min, z_max]）。"""
        if amount == 0:
            return
        with self._lock:
            self._desired[2] = float(np.clip(
                self._desired[2] + amount * self._wheel_sens,
                self.params.tcp_z_min, self.params.tcp_z_max,
            ))

    def request_start(self) -> None:
        with self._lock:
            self._start = True
            self._message = "> Start recording..."

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True
            self._message = "# Stop and save..."

    def request_discard(self) -> None:
        with self._lock:
            self._discard = True
            self._message = "x Discard episode..."

    def request_quit(self) -> None:
        with self._lock:
            self._quit = True
            self._message = "Quit..."

    def adjust_mouse_sens(self, delta: float) -> None:
        with self._lock:
            self._mouse_sens = max(0.1, self._mouse_sens + delta)
            self._message = f"Mouse sens: {self._mouse_sens:.2f}"

    def adjust_wheel_sens(self, delta: float) -> None:
        """滚轮灵敏度：**乘法**调节 ×(1+delta)，限制在 [0.001, 0.1]。

        不用加法：滚轮灵敏度的量级是 0.008（米/格），而灵敏度键的步长
        sens_step=0.1 相对它过大——加法一次按键会放大 13 倍，滚轮一格就
        clip 到 z 上下限（表现为"期望 TCP 跳变到固定位置"）。
        """
        with self._lock:
            self._wheel_sens = min(0.1, max(0.001, self._wheel_sens * (1.0 + delta)))
            self._message = f"Wheel sens: {self._wheel_sens:.4f}"

    def set_message(self, msg: str) -> None:
        with self._lock:
            self._message = msg

    # ── 仿真线程读取 / 写入 ────────────────────────────

    def desired(self) -> np.ndarray:
        with self._lock:
            return self._desired.copy()

    def consume_start(self) -> bool:
        with self._lock:
            v, self._start = self._start, False
            return v

    def consume_stop(self) -> bool:
        with self._lock:
            v, self._stop = self._stop, False
            return v

    def consume_discard(self) -> bool:
        with self._lock:
            v, self._discard = self._discard, False
            return v

    def consume_quit(self) -> bool:
        with self._lock:
            v, self._quit = self._quit, False
            return v

    def publish(
        self,
        *,
        tcp: np.ndarray,
        t_obj_pos: np.ndarray,
        t_obj_yaw: float,
        target_pos: np.ndarray,
        sim_time: float,
        recording: bool,
        pushable: bool,
    ) -> None:
        """仿真线程发布最新物理状态（窗口渲染用）。"""
        with self._lock:
            self._tcp = np.asarray(tcp, dtype=np.float64).copy()
            self._t_obj_pos = np.asarray(t_obj_pos, dtype=np.float64).copy()
            self._t_obj_yaw = float(t_obj_yaw)
            self._target_pos = np.asarray(target_pos, dtype=np.float64).copy()
            self._sim_time = float(sim_time)
            self._recording = bool(recording)
            self._pushable = bool(pushable)
            self._published = True

    def snapshot(self) -> TeleopSnapshot:
        with self._lock:
            return TeleopSnapshot(
                desired=self._desired.copy(),
                tcp=self._tcp.copy(),
                t_obj_pos=self._t_obj_pos.copy(),
                t_obj_yaw=self._t_obj_yaw,
                target_pos=self._target_pos.copy(),
                sim_time=self._sim_time,
                recording=self._recording,
                control=self._control,
                pushable=self._pushable,
                mouse_sens=self._mouse_sens,
                wheel_sens=self._wheel_sens,
                message=self._message,
            )


# ═══════════════════════════════════════════════════════
# T 物块几何（与 assets/mujoco/objects/t_obj.xml 一致）
# ═══════════════════════════════════════════════════════

# 相对 T 原点的两段矩形：(半宽 hw, 半长 hh, 段中心 y 偏移 cy0)
#   stem   中心 (0, 0)     半宽 0.01 × 半长 0.035
#   cross  中心 (0, 0.045) 半宽 0.04 × 半长 0.01
_T_PARTS: tuple[tuple[float, float, float], ...] = (
    (0.01, 0.035, 0.0),
    (0.04, 0.01, 0.045),
)


def _t_local_points(yaw: float, part_index: int) -> list[tuple[float, float]]:
    """返回 T 第 part_index 段（0=stem, 1=cross）的 4 个局部角点。

    角点相对 T 原点 (0, 0) 定义，整体绕原点旋转 yaw（与物理 freejoint 一致）。
    """
    hw, hh, cy0 = _T_PARTS[part_index]
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for wx, wy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        px, py = wx, wy + cy0          # 先加段中心偏移（相对 T 原点）
        pts.append((px * c - py * s, px * s + py * c))
    return pts


# ═══════════════════════════════════════════════════════
# 渲染窗口
# ═══════════════════════════════════════════════════════

class TeleopWindow:
    """pygame 2D 俯视遥操作窗口（独立线程）。"""

    def __init__(self, state: TeleopState, title: str = "PushT Teleop") -> None:
        self._state = state
        self._title = title
        self._thread: threading.Thread | None = None
        self._running = False

    # ── 生命周期 ───────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── 线程体 ─────────────────────────────────────────

    def _run(self) -> None:
        try:
            import pygame
        except ImportError:
            self._state.set_message("pygame not installed; teleop window unavailable")
            self._running = False
            return
        pygame.init()
        p = self._state.params
        screen = pygame.display.set_mode((p.window_width, p.window_height))
        pygame.display.set_caption(self._title)
        font = pygame.font.SysFont(None, 22)
        clock = pygame.time.Clock()
        grabbed = False

        while self._running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._state.request_quit()
                elif ev.type == pygame.KEYDOWN:
                    self._handle_key(ev.key)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_button(ev.button)
                elif ev.type == pygame.MOUSEWHEEL:
                    self._state.wheel_scroll(ev.y)

            snap = self._state.snapshot()
            # 鼠标锚定（FPS 式）：控制开启时抓取并隐藏光标
            if snap.control and not grabbed:
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                grabbed = True
            elif not snap.control and grabbed:
                pygame.event.set_grab(False)
                pygame.mouse.set_visible(True)
                grabbed = False

            if snap.control:
                dx, dy = pygame.mouse.get_rel()
                if dx or dy:
                    self._state.apply_mouse_delta(dx, dy)

            self._draw(screen, font, snap)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        self._running = False

    def _handle_key(self, key: int) -> None:
        step = self._state.params.sens_step
        if key in (91, 93):  # [ ]
            self._state.adjust_mouse_sens(-step if key == 91 else step)
        elif key in (45, 61, 95, 43):  # - = _ +
            self._state.adjust_wheel_sens(-step if key in (45, 95) else step)
        elif key == 27:  # Esc
            self._state.request_quit()

    def _handle_mouse_button(self, button: int) -> None:
        snap = self._state.snapshot()
        if button == 1:  # 左键：切换鼠标控制（锚定到当前 TCP）
            self._state.toggle_control(snap.tcp)
        elif button == 2:  # 中键：丢弃当前 episode
            if snap.recording:
                self._state.request_discard()
            else:
                self._state.set_message("Not recording; middle-click discard ignored")
        elif button == 3:  # 右键：开始 / 结束并保存
            if snap.recording:
                self._state.request_stop()
            else:
                self._state.request_start()

    # ── 绘制 ───────────────────────────────────────────

    def _draw(self, screen, font, snap: TeleopSnapshot) -> None:
        import pygame

        screen.fill((24, 24, 28))
        p = self._state.params

        # 桌面范围
        x0, y0 = self._state.world_to_screen(p.workspace_x[0], p.workspace_y[0])
        x1, y1 = self._state.world_to_screen(p.workspace_x[1], p.workspace_y[1])
        # y0 > y1（屏幕 y 向下，世界 y 向上翻转）→ 用左上角 + 正宽高，
        # 否则 pygame.Rect 负高度画不出来（桌面边框会消失）
        rect = pygame.Rect(int(x0), int(y1), int(x1 - x0), int(y0 - y1))
        pygame.draw.rect(screen, (46, 44, 40), rect)
        pygame.draw.rect(screen, (90, 86, 78), rect, 2)

        # 目标 T（蓝，半透明）
        self._draw_t(screen, snap.target_pos[0], snap.target_pos[1], 0.0, (80, 130, 230, 140))
        # 当前 T（绿，按 yaw 旋转）
        self._draw_t(screen, snap.t_obj_pos[0], snap.t_obj_pos[1], snap.t_obj_yaw, (60, 200, 110))

        # 当前 TCP（白点）
        cu, cv = self._state.world_to_screen(snap.tcp[0], snap.tcp[1])
        pygame.draw.circle(screen, (255, 255, 255), (int(cu), int(cv)), 8)
        pygame.draw.circle(screen, (120, 120, 120), (int(cu), int(cv)), 8, 2)
        # 期望 TCP（黄十字）
        du, dv = self._state.world_to_screen(snap.desired[0], snap.desired[1])
        r = 12
        pygame.draw.line(screen, (255, 220, 80), (du - r, dv), (du + r, dv), 2)
        pygame.draw.line(screen, (255, 220, 80), (du, dv - r), (du, dv + r), 2)

        self._draw_status(screen, font, snap)

    def _draw_t(self, screen, cx: float, cy: float, yaw: float, color) -> None:
        """绘制 T 形物块（stem + cross 两段矩形，**整体**绕 T 原点 (cx, cy) 旋转 yaw）。

        与 ``assets/mujoco/objects/t_obj.xml`` 的几何一致：stem 中心 (0, 0)、
        cross 中心 (0, 0.045)。两段都先相对 T 原点定义局部角点，再统一旋转，
        保证 yaw≠0 时形状不变。
        """
        import pygame

        surf = pygame.Surface(
            (self._state.params.window_width, self._state.params.window_height),
            pygame.SRCALPHA,
        )
        for i in range(len(_T_PARTS)):
            pts = [
                self._state.world_to_screen(cx + rx, cy + ry)
                for rx, ry in _t_local_points(yaw, i)
            ]
            pygame.draw.polygon(surf, color, pts)
        screen.blit(surf, (0, 0))

    def _draw_status(self, screen, font, snap: TeleopSnapshot) -> None:
        import pygame

        p = self._state.params
        status_y = p.window_height - p._status_h
        pygame.draw.rect(screen, (40, 40, 46), (0, status_y, p.window_width, p._status_h))

        if snap.recording:
            rec = "*REC"
            rec_color = (240, 90, 90)
        else:
            rec = "WAIT"
            rec_color = (200, 200, 200)
        control = "MOUSE ON" if snap.control else "MOUSE OFF"
        push = "PUSH OK" if snap.pushable else "PUSH NO"
        push_color = (110, 220, 110) if snap.pushable else (150, 150, 150)
        line1 = (f"{rec}  t={snap.sim_time:6.1f}s   {control}   "
                 f"tcp z={snap.tcp[2]:.3f}   {push}")
        line2 = (f"msens {snap.mouse_sens:.2f}  wsens {snap.wheel_sens:.4f}"
                 f"   |  {snap.message}")
        s1 = font.render(line1, True, rec_color if snap.recording else (230, 230, 230))
        s2 = font.render(line2, True, (180, 180, 200))
        screen.blit(s1, (10, status_y + 6))
        screen.blit(s2, (10, status_y + 30))
