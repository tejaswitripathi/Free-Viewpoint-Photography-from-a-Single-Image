from utils import *

import pygame


class InteractiveSplatViewer:
    """
    Pygame-based interactive viewer for the CPU Gaussian splat renderer.

    Controls:
      - trackpad / mouse wheel: physically move camera forward/back
      - left click + drag: rotate yaw/pitch
      - W/S: move forward/back
      - A/D: move left/right
      - R/F: move up/down
      - arrow keys: rotate yaw/pitch
      - ESC/Q: quit

    Camera translation is clamped to a restricted local box.
    """

    def __init__(
        self,
        points,
        colors,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        box_limits=(-0.4, 0.4, -0.25, 0.25, -0.4, 0.4),
        base_world_scale=0.004,
        opacity=0.55,
        min_radius_px=1.5,
        max_radius_px=8.0,
        max_points=None,
        window_scale=2,
    ):
        self.points0 = points.astype(np.float32)
        self.colors = colors.astype(np.float32)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = int(width)
        self.height = int(height)

        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = box_limits

        self.base_world_scale = base_world_scale
        self.opacity = opacity
        self.min_radius_px = min_radius_px
        self.max_radius_px = max_radius_px
        self.max_points = max_points
        self.window_scale = max(1, int(window_scale))

        self.cam_t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.dragging = False
        self.last_mouse = None
        self.needs_render = True
        self.current_image = None

    def clamp_camera(self):
        self.cam_t[0] = np.clip(self.cam_t[0], self.xmin, self.xmax)
        self.cam_t[1] = np.clip(self.cam_t[1], self.ymin, self.ymax)
        self.cam_t[2] = np.clip(self.cam_t[2], self.zmin, self.zmax)
        self.pitch = float(np.clip(self.pitch, -25.0, 25.0))

    def camera_transform_points(self):
        return transform_points_camera(
            self.points0,
            tx=float(self.cam_t[0]),
            ty=float(self.cam_t[1]),
            tz=float(self.cam_t[2]),
            yaw_deg=float(self.yaw),
            pitch_deg=float(self.pitch),
            roll_deg=float(self.roll),
        )

    def render(self):
        pts = self.camera_transform_points()

        radii_px = compute_depth_based_radii(
            pts,
            self.fx,
            base_world_scale=self.base_world_scale,
            min_px=self.min_radius_px,
            max_px=self.max_radius_px,
        )

        opacities = np.full((len(pts),), self.opacity, dtype=np.float32)

        self.current_image = render_gaussian_splats(
            pts,
            self.colors,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            width=self.width,
            height=self.height,
            radii_px=radii_px,
            opacities=opacities,
            max_points=self.max_points,
        )
        self.needs_render = False

    def handle_keyboard(self, dt):
        keys = pygame.key.get_pressed()
        moved = False

        # dt-normalized motion so holding keys feels smooth.
        move_speed = 0.65  # meters / second in the local bounded box
        rot_speed = 55.0   # degrees / second

        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            move_speed *= 2.0
            rot_speed *= 1.5

        step = move_speed * dt
        rot = rot_speed * dt

        if keys[pygame.K_w]:
            self.cam_t[2] += step
            moved = True
        if keys[pygame.K_s]:
            self.cam_t[2] -= step
            moved = True
        if keys[pygame.K_a]:
            self.cam_t[0] -= step
            moved = True
        if keys[pygame.K_d]:
            self.cam_t[0] += step
            moved = True
        if keys[pygame.K_r]:
            self.cam_t[1] += step
            moved = True
        if keys[pygame.K_f]:
            self.cam_t[1] -= step
            moved = True

        if keys[pygame.K_LEFT]:
            self.yaw -= rot
            moved = True
        if keys[pygame.K_RIGHT]:
            self.yaw += rot
            moved = True
        if keys[pygame.K_UP]:
            self.pitch -= rot
            moved = True
        if keys[pygame.K_DOWN]:
            self.pitch += rot
            moved = True

        if moved:
            self.clamp_camera()
            self.needs_render = True

    def draw_overlay(self, screen, font):
        line1 = (
            f"x={self.cam_t[0]:+.2f}  y={self.cam_t[1]:+.2f}  z={self.cam_t[2]:+.2f}  "
            f"yaw={self.yaw:+.1f}  pitch={self.pitch:+.1f}"
        )
        line2 = "wheel: forward/back | drag: look | WASD/RF: move | arrows: look | Q/ESC: quit"

        for i, text in enumerate([line1, line2]):
            surf = font.render(text, True, (0, 0, 0))
            bg = pygame.Surface((surf.get_width() + 12, surf.get_height() + 8), pygame.SRCALPHA)
            bg.fill((255, 255, 255, 175))
            y = 10 + i * (surf.get_height() + 8)
            screen.blit(bg, (10, y))
            screen.blit(surf, (16, y + 4))

    def image_to_surface(self, image):
        arr = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        # pygame.surfarray expects shape (width, height, 3)
        arr = np.transpose(arr, (1, 0, 2))
        return pygame.surfarray.make_surface(arr)

    def run(self):
        pygame.init()
        pygame.display.set_caption("Interactive Gaussian Splat Viewer")

        screen_w = self.width * self.window_scale
        screen_h = self.height * self.window_scale
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Menlo", 16) or pygame.font.SysFont(None, 16)

        print()
        print("Interactive controls:")
        print("  trackpad/mouse wheel : physically move camera forward/back")
        print("  left click + drag    : rotate camera yaw/pitch")
        print("  W/S                  : move forward/back")
        print("  A/D                  : move left/right")
        print("  R/F                  : move up/down")
        print("  arrow keys           : rotate yaw/pitch")
        print("  Shift                : faster movement")
        print("  ESC or Q             : quit")
        print()

        running = True
        while running:
            dt = clock.tick(30) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.dragging = True
                        self.last_mouse = event.pos
                    # Pygame 1/older compatibility: wheel can be button 4/5.
                    elif event.button == 4:
                        self.cam_t[2] += 0.08
                        self.clamp_camera()
                        self.needs_render = True
                    elif event.button == 5:
                        self.cam_t[2] -= 0.08
                        self.clamp_camera()
                        self.needs_render = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                        self.last_mouse = None

                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    x, y = event.pos
                    lx, ly = self.last_mouse
                    dx = x - lx
                    dy = y - ly

                    self.yaw += dx * 0.12
                    self.pitch += dy * 0.12

                    self.last_mouse = event.pos
                    self.clamp_camera()
                    self.needs_render = True

                elif event.type == pygame.MOUSEWHEEL:
                    # On macOS trackpads, event.y may be small but consistent.
                    # Positive y generally means scroll up/forward.
                    self.cam_t[2] += float(event.y) * 0.08
                    self.cam_t[0] += float(event.x) * 0.04
                    self.clamp_camera()
                    self.needs_render = True

            self.handle_keyboard(dt)

            if self.needs_render or self.current_image is None:
                self.render()

            surf = self.image_to_surface(self.current_image)
            if self.window_scale != 1 or screen.get_size() != (self.width, self.height):
                surf = pygame.transform.smoothscale(surf, screen.get_size())

            screen.blit(surf, (0, 0))
            self.draw_overlay(screen, font)
            pygame.display.flip()

        pygame.quit()
