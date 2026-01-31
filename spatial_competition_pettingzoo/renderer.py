"""Pygame-based renderer for the Spatial Competition environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pygame

if TYPE_CHECKING:
    from spatial_competition_pettingzoo.buyer import Buyer
    from spatial_competition_pettingzoo.competition import Competition
    from spatial_competition_pettingzoo.seller import Seller


@dataclass
class EntityInfo:
    """Information about a hovered or selected entity."""

    entity_type: str  # "seller" or "buyer"
    entity: Seller | Buyer
    screen_pos: tuple[int, int]
    color: tuple[int, int, int]


class PygameRenderer:
    """Pygame-based renderer for visualizing spatial competition environments."""

    def __init__(
        self,
        competition: Competition,
        max_env_steps: int = 100,
    ) -> None:
        """
        Initialize the renderer.

        Args:
            competition: The competition instance to render.
            max_env_steps: Maximum environment steps (for display purposes).

        """
        self._competition = competition
        self._max_env_steps = max_env_steps

        # Pygame rendering attributes (lazy initialization)
        self._pygame_initialized = False
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._window_size = (900, 650)
        self._seller_colors: list[tuple[int, int, int]] = self._generate_seller_colors()
        self._cumulative_rewards: dict[str, float] = {}

        # Interactive state
        self._paused = False
        self._speed_multiplier = 1.0  # 0.25x to 4x
        self._selected_entity: EntityInfo | None = None
        self._hovered_entity: EntityInfo | None = None

        # Entity screen positions for hit detection (updated each frame)
        self._seller_positions: list[tuple[int, int, int, Seller]] = []  # (x, y, radius, seller)
        self._buyer_positions: list[tuple[int, int, int, Buyer]] = []  # (x, y, radius, buyer)

        # UI element positions
        self._pause_button_rect: pygame.Rect | None = None
        self._slider_rect: pygame.Rect | None = None
        self._slider_dragging = False

        # Leaderboard click areas (updated each frame)
        self._leaderboard_items: list[tuple[pygame.Rect, Seller, tuple[int, int, int]]] = []

    @property
    def paused(self) -> bool:
        """Return whether the simulation is paused."""
        return self._paused

    @property
    def speed_multiplier(self) -> float:
        """Return the current speed multiplier."""
        return self._speed_multiplier

    def _get_buyer_color(self, buyer: Buyer) -> tuple[int, int, int]:
        """Get the color for a buyer based on purchase or preference."""
        sellers = self._competition.space.sellers

        # If buyer has already purchased, use the color of the seller they bought from
        target_seller_id = buyer.purchased_from_id

        if target_seller_id is None:
            return (80, 80, 90)  # Neutral gray for buyers who won't buy

        # Find the seller's index to get their color
        for idx, seller in enumerate(sellers):
            if seller.agent_id == target_seller_id:
                return self._seller_colors[idx % len(self._seller_colors)]

        return (80, 80, 90)  # Fallback

    def _generate_seller_colors(self) -> list[tuple[int, int, int]]:
        """Generate distinct colors for each seller using HSV color space."""
        num_sellers = len(self._competition.space.sellers)
        colors = []
        for i in range(num_sellers):
            hue = (i * 360 / num_sellers) % 360 if num_sellers > 0 else 0
            # Convert HSV to RGB (saturation=0.8, value=0.65 for darker colors)
            h = hue / 60
            saturation = 0.8
            value = 0.65
            c = value * saturation
            x = c * (1 - abs(h % 2 - 1))
            m = value - c

            r: float
            g: float
            b: float
            if h < 1:
                r, g, b = c, x, 0.0
            elif h < 2:
                r, g, b = x, c, 0.0
            elif h < 3:
                r, g, b = 0.0, c, x
            elif h < 4:
                r, g, b = 0.0, x, c
            elif h < 5:
                r, g, b = x, 0.0, c
            else:
                r, g, b = c, 0.0, x

            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
        return colors

    def _init_pygame(self) -> None:
        """Initialize Pygame for rendering."""
        if self._pygame_initialized:
            return

        pygame.init()
        pygame.display.set_caption("Spatial Competition Environment")
        self._screen = pygame.display.set_mode(self._window_size)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._small_font = pygame.font.SysFont("monospace", 12)
        self._pygame_initialized = True

    def _handle_events(self) -> bool:
        """Handle Pygame events. Returns False if window was closed."""
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            self._process_event(event, mouse_pos)

        return True

    def _process_event(self, event: pygame.event.Event, mouse_pos: tuple[int, int]) -> None:
        """Process a single Pygame event."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._paused = not self._paused
            elif event.key == pygame.K_ESCAPE:
                self._selected_entity = None
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(mouse_pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._slider_dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self._slider_dragging:
                self._update_slider_from_mouse(mouse_pos)
            else:
                self._update_hover(mouse_pos)

    def _handle_click(self, mouse_pos: tuple[int, int]) -> None:
        """Handle mouse click at the given position."""
        # Check pause button
        if self._pause_button_rect and self._pause_button_rect.collidepoint(mouse_pos):
            self._paused = not self._paused
            return

        # Check slider
        if self._slider_rect and self._slider_rect.collidepoint(mouse_pos):
            self._slider_dragging = True
            self._update_slider_from_mouse(mouse_pos)
            return

        # Check sellers
        for x, y, radius, seller in self._seller_positions:
            if (mouse_pos[0] - x) ** 2 + (mouse_pos[1] - y) ** 2 <= radius**2:
                idx = self._competition.space.sellers.index(seller)
                color = self._seller_colors[idx % len(self._seller_colors)]
                self._selected_entity = EntityInfo("seller", seller, (x, y), color)
                return

        # Check buyers
        for x, y, radius, buyer in self._buyer_positions:
            if (mouse_pos[0] - x) ** 2 + (mouse_pos[1] - y) ** 2 <= (radius + 2) ** 2:
                color = self._get_buyer_color(buyer)
                self._selected_entity = EntityInfo("buyer", buyer, (x, y), color)
                return

        # Check leaderboard items
        for rect, seller, color in self._leaderboard_items:
            if rect.collidepoint(mouse_pos):
                # Find the seller's screen position for the highlight
                screen_pos = (0, 0)
                for x, y, _, s in self._seller_positions:
                    if s is seller:
                        screen_pos = (x, y)
                        break
                self._selected_entity = EntityInfo("seller", seller, screen_pos, color)
                return

        # Clicked empty space - deselect
        self._selected_entity = None

    def _update_slider_from_mouse(self, mouse_pos: tuple[int, int]) -> None:
        """Update speed multiplier based on mouse position on slider."""
        if self._slider_rect is None:
            return

        # Calculate position within slider (0 to 1)
        rel_x = (mouse_pos[0] - self._slider_rect.x) / self._slider_rect.width
        rel_x = max(0.0, min(1.0, rel_x))

        # Map to speed range: 0.25x to 4x (logarithmic scale), with MAX at far right
        # rel_x=0 -> 0.25, rel_x=0.5 -> 1.0, rel_x=0.9 -> 4.0, rel_x=1.0 -> MAX (infinity)
        if rel_x >= 0.95:
            self._speed_multiplier = float("inf")
        else:
            # Scale rel_x from [0, 0.95) to [0, 1] for the logarithmic range
            scaled_x = rel_x / 0.95
            self._speed_multiplier = 0.25 * (16**scaled_x)

    def _update_hover(self, mouse_pos: tuple[int, int]) -> None:
        """Update hovered entity based on mouse position."""
        # Check sellers first (they're on top)
        for x, y, radius, seller in self._seller_positions:
            if (mouse_pos[0] - x) ** 2 + (mouse_pos[1] - y) ** 2 <= radius**2:
                idx = self._competition.space.sellers.index(seller)
                color = self._seller_colors[idx % len(self._seller_colors)]
                self._hovered_entity = EntityInfo("seller", seller, (x, y), color)
                return

        # Check buyers
        for x, y, radius, buyer in self._buyer_positions:
            if (mouse_pos[0] - x) ** 2 + (mouse_pos[1] - y) ** 2 <= (radius + 2) ** 2:
                color = self._get_buyer_color(buyer)
                self._hovered_entity = EntityInfo("buyer", buyer, (x, y), color)
                return

        self._hovered_entity = None

    def render(
        self,
        current_step: int = 0,
        cumulative_rewards: dict[str, float] | None = None,
    ) -> bool:
        """
        Render the current state of the competition.

        Args:
            current_step: The current environment step number.
            cumulative_rewards: Optional dict mapping agent_id to cumulative reward.

        Returns:
            True if rendering succeeded, False if the window was closed.

        """
        self._init_pygame()
        assert self._screen is not None
        assert self._font is not None
        assert self._clock is not None

        self._cumulative_rewards = cumulative_rewards or {}

        # Handle Pygame events (uses entity positions from previous frame for hit detection)
        if not self._handle_events():
            return False

        # Clear entity positions for this frame (after event handling so clicks work)
        self._seller_positions = []
        self._buyer_positions = []

        # Clear screen with dark background
        bg_color = (30, 30, 40)
        self._screen.fill(bg_color)

        # Get dimensions for rendering
        width, height = self._window_size
        margin = 60

        dimensions = self._competition.dimensions
        if dimensions == 1:
            self._render_1d(width, height, margin)
        elif dimensions == 2:
            self._render_2d(width, height, margin)
        else:
            # For dimensions > 2, show stats in the main area
            self._render_high_dim(width, height, margin, dimensions)

        # Draw UI controls
        self._draw_controls(width, current_step)

        # Draw leaderboard (top 10 sellers)
        self._draw_leaderboard(width, height, margin)

        # Clear selection if entity no longer exists (but keep hover until mouse moves)
        if self._selected_entity and not self._entity_exists(self._selected_entity):
            self._selected_entity = None

        # Draw highlight around hovered entity (only if entity still exists)
        if self._hovered_entity:
            self._draw_hover_highlight(self._hovered_entity)
            self._draw_tooltip(self._hovered_entity)

        # Draw highlight around selected entity (only if entity still exists)
        if self._selected_entity:
            self._draw_hover_highlight(self._selected_entity)
            self._draw_detail_panel(self._selected_entity)

        pygame.display.flip()
        self._clock.tick(60)  # 60 FPS for smooth interaction
        return True

    def render_and_wait(
        self,
        base_delay: float,
        current_step: int = 0,
        cumulative_rewards: dict[str, float] | None = None,
    ) -> bool:
        """
        Render and wait for delay while keeping UI responsive.

        This method renders the current state and waits for the specified delay,
        continuously processing events and re-rendering to keep the UI interactive.
        The delay is adjusted by the speed multiplier, and pausing freezes the timer.

        Args:
            base_delay: Base delay in seconds before speed adjustment.
            current_step: The current environment step number.
            cumulative_rewards: Optional dict mapping agent_id to cumulative reward.

        Returns:
            True if ready to continue, False if the window was closed.

        """
        import time

        start_time = time.time()
        elapsed = 0.0

        while True:
            # Render frame
            if not self.render(current_step, cumulative_rewards):
                return False  # Window closed

            # If paused, reset timer and keep looping
            if self._paused:
                start_time = time.time()
                elapsed = 0.0
                time.sleep(0.016)  # ~60fps while paused
                continue

            # MAX speed: skip delay entirely (just render one frame)
            if self._speed_multiplier == float("inf"):
                break

            # Calculate elapsed time and check if delay is complete
            elapsed = time.time() - start_time
            adjusted_delay = base_delay / self._speed_multiplier

            if elapsed >= adjusted_delay:
                break

            # Small sleep to prevent busy-waiting
            remaining = adjusted_delay - elapsed
            time.sleep(min(0.016, remaining))  # ~60fps or remaining time

        return True

    def _draw_controls(self, width: int, current_step: int) -> None:
        """Draw UI controls (step counter, pause button, speed slider)."""
        assert self._screen is not None
        assert self._font is not None

        # Step counter
        step_text = self._font.render(f"Step: {current_step}/{self._max_env_steps}", True, (200, 200, 200))
        self._screen.blit(step_text, (10, 10))

        # Sellers and buyers count
        num_sellers = len(self._competition.space.sellers)
        num_buyers = len(self._competition.space.buyers)
        counts_text = self._font.render(f"Sellers: {num_sellers}  Buyers: {num_buyers}", True, (150, 150, 150))
        self._screen.blit(counts_text, (10, 28))

        # Speed label and slider (from left to right: label, slider, pause button)
        speed_label = self._font.render("Speed:", True, (150, 150, 150))
        if self._speed_multiplier == float("inf"):
            speed_value = self._font.render("MAX", True, (255, 200, 100))
        else:
            speed_value = self._font.render(f"{self._speed_multiplier:.1f}x", True, (200, 200, 200))

        # Position elements from right edge
        button_width = 70
        button_height = 25
        slider_width = 80
        slider_height = 10

        # Pause button (rightmost)
        button_x = width - button_width - 10
        button_y = 10

        self._pause_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        button_color = (80, 120, 80) if self._paused else (60, 60, 80)
        pygame.draw.rect(self._screen, button_color, self._pause_button_rect, border_radius=4)
        pygame.draw.rect(self._screen, (100, 100, 120), self._pause_button_rect, 1, border_radius=4)

        pause_text = "Resume" if self._paused else "Pause"
        text = self._font.render(pause_text, True, (200, 200, 200))
        text_x = button_x + (button_width - text.get_width()) // 2
        text_y = button_y + (button_height - text.get_height()) // 2
        self._screen.blit(text, (text_x, text_y))

        # Speed value (left of button)
        value_x = button_x - speed_value.get_width() - 10
        self._screen.blit(speed_value, (value_x, 13))

        # Slider (left of value)
        slider_x = value_x - slider_width - 10
        slider_y = 17

        self._slider_rect = pygame.Rect(slider_x, slider_y, slider_width, slider_height)
        pygame.draw.rect(self._screen, (50, 50, 65), self._slider_rect, border_radius=3)

        # Slider handle position (logarithmic: 0.25 -> 0, 1.0 -> ~0.47, 4.0 -> 0.95, MAX -> 1.0)
        if self._speed_multiplier == float("inf"):
            handle_pos = 1.0
        else:
            handle_pos = np.log(self._speed_multiplier / 0.25) / np.log(16) * 0.95
        handle_x = int(slider_x + handle_pos * slider_width)
        handle_rect = pygame.Rect(handle_x - 4, slider_y - 2, 8, slider_height + 4)
        pygame.draw.rect(self._screen, (120, 120, 150), handle_rect, border_radius=2)

        # Speed label (left of slider)
        label_x = slider_x - speed_label.get_width() - 8
        self._screen.blit(speed_label, (label_x, 13))

        # Paused indicator
        if self._paused:
            paused_text = self._font.render("PAUSED (Space to resume)", True, (255, 200, 100))
            self._screen.blit(paused_text, (width // 2 - paused_text.get_width() // 2, 10))

    def _render_1d(self, width: int, height: int, margin: int) -> None:
        """Render 1D environment as a horizontal line with entities."""
        assert self._screen is not None
        assert self._font is not None

        # Draw the market line (add extra left margin for leaderboard, same as 2D)
        legend_width = 220
        line_y = height // 2
        line_start = legend_width + margin
        line_end = width - margin
        line_length = line_end - line_start

        # Draw axis line
        pygame.draw.line(self._screen, (100, 100, 120), (line_start, line_y), (line_end, line_y), 2)

        # Draw axis markers
        for i in range(11):
            x = line_start + (line_length * i) // 10
            pygame.draw.line(self._screen, (100, 100, 120), (x, line_y - 5), (x, line_y + 5), 1)
            label = self._font.render(f"{i / 10:.1f}", True, (150, 150, 150))
            self._screen.blit(label, (x - label.get_width() // 2, line_y + 15))

        # Draw buyers as small dots below the line, colored by preferred seller
        buyer_radius = 3
        for buyer in self._competition.space.buyers:
            x_pos = buyer.position.space_coordinates[0]
            screen_x = int(line_start + x_pos * line_length)
            screen_y = line_y + 40
            buyer_color = self._get_buyer_color(buyer)
            pygame.draw.circle(self._screen, buyer_color, (screen_x, screen_y), buyer_radius)
            self._buyer_positions.append((screen_x, screen_y, buyer_radius, buyer))

        # Draw sellers as larger circles above the line with labels
        seller_radius = 12
        for idx, seller in enumerate(self._competition.space.sellers):
            x_pos = seller.position.space_coordinates[0]
            screen_x = int(line_start + x_pos * line_length)
            screen_y = line_y - 30
            color = self._seller_colors[idx % len(self._seller_colors)]

            # Draw seller circle
            pygame.draw.circle(self._screen, color, (screen_x, screen_y), seller_radius)
            self._seller_positions.append((screen_x, screen_y, seller_radius, seller))

            # Draw seller label (ID, price, and optionally quality)
            label = self._font.render(f"{seller.agent_id}", True, color)
            self._screen.blit(label, (screen_x - label.get_width() // 2, line_y - 85))

            info_text = f"p={seller.price:.1f}"
            if self._competition.include_quality:
                info_text += f" q={seller.quality:.1f}"
            price_label = self._font.render(info_text, True, (200, 200, 200))
            self._screen.blit(price_label, (screen_x - price_label.get_width() // 2, line_y - 70))

    def _render_2d(self, width: int, height: int, margin: int) -> None:
        """Render 2D environment as a grid with entities."""
        assert self._screen is not None
        assert self._font is not None

        # Calculate drawing area (square to maintain aspect ratio)
        # Add extra left margin for legend
        legend_width = 220
        available_width = width - legend_width
        draw_size = min(available_width, height) - 2 * margin
        offset_x = legend_width + (available_width - draw_size) // 2
        offset_y = (height - draw_size) // 2

        # Draw grid background
        grid_rect = pygame.Rect(offset_x, offset_y, draw_size, draw_size)
        pygame.draw.rect(self._screen, (40, 40, 55), grid_rect)
        pygame.draw.rect(self._screen, (80, 80, 100), grid_rect, 2)

        # Draw grid lines
        num_grid_lines = 10
        for i in range(num_grid_lines + 1):
            # Vertical lines
            x = offset_x + (draw_size * i) // num_grid_lines
            pygame.draw.line(self._screen, (60, 60, 75), (x, offset_y), (x, offset_y + draw_size), 1)
            # Horizontal lines
            y = offset_y + (draw_size * i) // num_grid_lines
            pygame.draw.line(self._screen, (60, 60, 75), (offset_x, y), (offset_x + draw_size, y), 1)

        # Draw buyers as small dots, colored by preferred seller
        buyer_radius = 3
        for buyer in self._competition.space.buyers:
            coords = buyer.position.space_coordinates
            screen_x = int(offset_x + coords[0] * draw_size)
            screen_y = int(offset_y + (1 - coords[1]) * draw_size)  # Flip Y for screen coords
            buyer_color = self._get_buyer_color(buyer)
            pygame.draw.circle(self._screen, buyer_color, (screen_x, screen_y), buyer_radius)
            self._buyer_positions.append((screen_x, screen_y, buyer_radius, buyer))

        # Draw sellers as larger circles with labels
        seller_radius = 10
        for idx, seller in enumerate(self._competition.space.sellers):
            coords = seller.position.space_coordinates
            screen_x = int(offset_x + coords[0] * draw_size)
            screen_y = int(offset_y + (1 - coords[1]) * draw_size)  # Flip Y for screen coords
            color = self._seller_colors[idx % len(self._seller_colors)]

            # Draw seller circle
            pygame.draw.circle(self._screen, color, (screen_x, screen_y), seller_radius)
            self._seller_positions.append((screen_x, screen_y, seller_radius, seller))

            # Draw seller label (ID, price, and optionally quality)
            label = self._font.render(f"{seller.agent_id}", True, color)
            self._screen.blit(label, (screen_x + 15, screen_y - 25))

            info_text = f"p={seller.price:.1f}"
            if self._competition.include_quality:
                info_text += f" q={seller.quality:.1f}"
            price_label = self._font.render(info_text, True, (200, 200, 200))
            self._screen.blit(price_label, (screen_x + 15, screen_y - 10))

    def _render_high_dim(self, width: int, height: int, margin: int, dimensions: int) -> None:
        """Render high-dimensional environment (>2D) with stats display."""
        assert self._screen is not None
        assert self._font is not None
        assert self._small_font is not None

        # Left margin for leaderboard
        legend_width = 220
        content_x = legend_width + margin
        content_width = width - legend_width - 2 * margin

        # Title
        title = self._font.render(f"🌀 {dimensions}D Spatial Competition", True, (200, 200, 200))
        self._screen.blit(title, (content_x + (content_width - title.get_width()) // 2, 60))

        subtitle = self._small_font.render(
            "(Space cannot be visualized - see leaderboard for rankings)", True, (120, 120, 140)
        )
        self._screen.blit(subtitle, (content_x + (content_width - subtitle.get_width()) // 2, 85))

        # Draw stats boxes
        box_y = 120
        box_width = min(300, content_width - 40)
        box_x = content_x + (content_width - box_width) // 2

        # Competition stats
        num_sellers = len(self._competition.space.sellers)
        num_buyers = len(self._competition.space.buyers)
        total_sales = sum(s.total_sales for s in self._competition.space.sellers)
        total_reward = sum(self._cumulative_rewards.values()) if self._cumulative_rewards else 0

        stats = [
            ("Dimensions", str(dimensions)),
            ("Active Sellers", str(num_sellers)),
            ("Active Buyers", str(num_buyers)),
            ("Total Sales", str(total_sales)),
            ("Total Reward", f"{total_reward:.1f}"),
        ]

        # Draw stats box background
        box_height = len(stats) * 28 + 20
        pygame.draw.rect(self._screen, (35, 35, 50), (box_x, box_y, box_width, box_height), border_radius=8)
        pygame.draw.rect(self._screen, (60, 60, 80), (box_x, box_y, box_width, box_height), 2, border_radius=8)

        # Draw stats
        for i, (label, value) in enumerate(stats):
            y = box_y + 12 + i * 28
            label_text = self._font.render(label, True, (150, 150, 170))
            value_text = self._font.render(value, True, (200, 200, 220))
            self._screen.blit(label_text, (box_x + 15, y))
            self._screen.blit(value_text, (box_x + box_width - value_text.get_width() - 15, y))

        # Top 5 sellers summary (quick view)
        top_sellers_y = box_y + box_height + 30
        top_label = self._font.render("Quick Stats - Top 5 Sellers:", True, (180, 180, 200))
        self._screen.blit(top_label, (box_x, top_sellers_y))

        # Sort sellers by cumulative reward
        if self._cumulative_rewards:
            sorted_sellers = sorted(
                self._competition.space.sellers,
                key=lambda s: self._cumulative_rewards.get(s.agent_id, 0),
                reverse=True,
            )[:5]

            for i, seller in enumerate(sorted_sellers):
                y = top_sellers_y + 25 + i * 22
                idx = self._competition.space.sellers.index(seller)
                color = self._seller_colors[idx % len(self._seller_colors)]
                reward = self._cumulative_rewards.get(seller.agent_id, 0)

                # Color dot
                pygame.draw.circle(self._screen, color, (box_x + 10, y + 6), 5)

                # Seller info
                info = f"{seller.agent_id}: {reward:.0f} pts, {seller.total_sales} sales"
                if self._competition.include_quality:
                    info += f", q={seller.quality:.1f}"
                text = self._small_font.render(info, True, (160, 160, 180))
                self._screen.blit(text, (box_x + 25, y))

        # Hint at bottom
        hint = self._small_font.render("Click leaderboard items for details", True, (100, 100, 120))
        self._screen.blit(hint, (content_x + (content_width - hint.get_width()) // 2, height - 40))

    def _format_position(self, coords: np.ndarray, precision: int = 2) -> str | None:
        """Format position coordinates. Returns None if dimensions > 3."""
        dims = len(coords)
        if dims > 3:
            return None
        formatted = ", ".join(f"{c:.{precision}f}" for c in coords)
        return f"({formatted})"

    def _draw_tooltip(self, entity_info: EntityInfo) -> None:
        """Draw a tooltip near the hovered entity."""
        assert self._screen is not None
        assert self._small_font is not None

        lines: list[str] = []

        if entity_info.entity_type == "seller":
            seller: Seller = entity_info.entity  # type: ignore[assignment]
            lines.append(f"{seller.agent_id}")
            pos_str = self._format_position(seller.position.space_coordinates)
            if pos_str:
                lines.append(f"Position: {pos_str}")
            lines.append(f"Price: {seller.price:.2f}")
            if self._competition.include_quality:
                lines.append(f"Quality: {seller.quality:.2f}")
            lines.append(f"Sales: {seller.running_sales} (total: {seller.total_sales})")
        else:
            buyer: Buyer = entity_info.entity  # type: ignore[assignment]
            lines.append("Buyer")
            pos_str = self._format_position(buyer.position.space_coordinates)
            if pos_str:
                lines.append(f"Position: {pos_str}")
            if buyer.value is not None:
                lines.append(f"Value: {buyer.value:.2f}")
            if self._competition.include_quality:
                lines.append(f"Quality taste: {buyer.quality_taste:.2f}")
            lines.append(f"Distance factor: {buyer.distance_factor:.2f}")
            if buyer.purchased_from_id:
                lines.append(f"Bought from: {buyer.purchased_from_id}")

        # Use original screen position (box stays in place while highlight follows entity)
        self._draw_info_box(entity_info.screen_pos, lines, entity_info.color)

    def _draw_hover_highlight(self, entity_info: EntityInfo) -> None:
        """Draw a white border highlight around the hovered entity."""
        assert self._screen is not None

        # Find current screen position of the entity (it may have moved)
        current_pos = self._find_entity_screen_pos(entity_info)
        if current_pos is None:
            return  # Entity no longer exists

        x, y = current_pos

        if entity_info.entity_type == "buyer":
            # Draw a white circle border around the buyer
            pygame.draw.circle(self._screen, (255, 255, 255), (x, y), 6, 2)
        else:
            # For sellers, draw a slightly larger white circle
            pygame.draw.circle(self._screen, (255, 255, 255), (x, y), 14, 2)

    def _find_entity_screen_pos(self, entity_info: EntityInfo) -> tuple[int, int] | None:
        """Find the current screen position of an entity, or None if not found."""
        if entity_info.entity_type == "seller":
            for x, y, _, seller in self._seller_positions:
                if seller is entity_info.entity:
                    return (x, y)
        else:
            for x, y, _, buyer in self._buyer_positions:
                if buyer is entity_info.entity:
                    return (x, y)
        return None

    def _entity_exists(self, entity_info: EntityInfo) -> bool:
        """Check if an entity still exists in the competition."""
        if entity_info.entity_type == "seller":
            return entity_info.entity in self._competition.space.sellers
        return entity_info.entity in self._competition.space.buyers

    def _draw_detail_panel(self, entity_info: EntityInfo) -> None:
        """Draw a detailed info panel for the selected entity."""
        assert self._screen is not None
        assert self._font is not None

        lines: list[str] = []

        if entity_info.entity_type == "seller":
            seller: Seller = entity_info.entity  # type: ignore[assignment]
            lines.append(f"=== {seller.agent_id} ===")
            lines.append("")
            pos_str = self._format_position(seller.position.space_coordinates, precision=3)
            if pos_str:
                lines.append(f"Position: {pos_str}")
            lines.append(f"Price: {seller.price:.3f}")
            if self._competition.include_quality:
                lines.append(f"Quality: {seller.quality:.3f}")
            lines.append("")
            lines.append(f"Running sales: {seller.running_sales}")
            lines.append(f"Total sales: {seller.total_sales}")
            reward = self._cumulative_rewards.get(seller.agent_id, 0.0)
            lines.append(f"Cumulative reward: {reward:.2f}")
        else:
            buyer: Buyer = entity_info.entity  # type: ignore[assignment]
            lines.append("=== Buyer ===")
            lines.append("")
            pos_str = self._format_position(buyer.position.space_coordinates, precision=3)
            if pos_str:
                lines.append(f"Position: {pos_str}")
            if buyer.value is not None:
                lines.append(f"Value: {buyer.value:.3f}")
            if self._competition.include_quality:
                lines.append(f"Quality taste: {buyer.quality_taste:.3f}")
            lines.append(f"Distance factor: {buyer.distance_factor:.3f}")
            lines.append("")
            if buyer.purchased_from_id:
                lines.append(f"Purchased from: {buyer.purchased_from_id}")
            else:
                lines.append("Has not purchased yet")

        # Draw panel on the bottom left
        _, height = self._window_size
        line_height = 18
        padding = 15

        # Calculate panel dimensions based on content
        max_text_width = max(self._font.size(line)[0] for line in lines) if lines else 100
        panel_width = max_text_width + padding * 2
        panel_height = len(lines) * line_height + padding * 2

        panel_x = 10
        panel_y = height - panel_height - 10

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self._screen, (45, 45, 60), panel_rect, border_radius=5)
        pygame.draw.rect(self._screen, entity_info.color, panel_rect, 2, border_radius=5)

        # Draw lines
        for i, line in enumerate(lines):
            color = entity_info.color if i == 0 else (200, 200, 200)
            text = self._font.render(line, True, color)
            self._screen.blit(text, (panel_x + padding, panel_y + padding + i * line_height))

    def _draw_info_box(
        self,
        anchor_pos: tuple[int, int],
        lines: list[str],
        border_color: tuple[int, int, int],
    ) -> None:
        """Draw an info box near the given anchor position."""
        assert self._screen is not None
        assert self._small_font is not None

        if not lines:
            return

        # Calculate box dimensions
        line_height = 16
        padding = 8
        max_width = max(self._small_font.size(line)[0] for line in lines)
        box_width = max_width + padding * 2
        box_height = len(lines) * line_height + padding * 2

        # Position box (offset from anchor, keep on screen)
        width, height = self._window_size
        box_x = anchor_pos[0] + 20
        box_y = anchor_pos[1] - box_height // 2

        # Keep on screen
        if box_x + box_width > width - 10:
            box_x = anchor_pos[0] - box_width - 20
        box_y = max(10, min(box_y, height - box_height - 10))

        # Draw box
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self._screen, (50, 50, 65), box_rect, border_radius=4)
        pygame.draw.rect(self._screen, border_color, box_rect, 1, border_radius=4)

        # Draw text
        for i, line in enumerate(lines):
            text = self._small_font.render(line, True, (200, 200, 200))
            self._screen.blit(text, (box_x + padding, box_y + padding + i * line_height))

    def _draw_leaderboard(self, width: int, height: int, margin: int) -> None:  # noqa: ARG002
        """Draw a leaderboard chart showing top 10 sellers by cumulative reward."""
        assert self._screen is not None
        assert self._font is not None

        # Get sellers sorted by cumulative reward
        sellers = self._competition.space.sellers
        seller_data = []
        for idx, seller in enumerate(sellers):
            reward = self._cumulative_rewards.get(seller.agent_id, 0.0)
            color = self._seller_colors[idx % len(self._seller_colors)]
            seller_data.append((seller, reward, color, idx))

        # Sort by reward descending and take top 10
        seller_data.sort(key=lambda x: x[1], reverse=True)
        top_sellers = seller_data[:10]

        if not top_sellers:
            self._leaderboard_items = []
            return

        # Chart dimensions
        chart_x = 10
        chart_y = 80  # Below the step/counts info with margin
        chart_width = 200
        bar_height = 26
        bar_spacing = 4

        # Draw chart title
        title = self._font.render("Top Sellers (by reward)", True, (180, 180, 180))
        self._screen.blit(title, (chart_x, chart_y - 20))

        # Find max reward for scaling
        max_reward = max(s[1] for s in top_sellers) if top_sellers else 1
        if max_reward <= 0:
            max_reward = 1

        # Clear and rebuild leaderboard click areas
        self._leaderboard_items = []

        # Draw bars
        for i, (seller, reward, color, _orig_idx) in enumerate(top_sellers):
            y = chart_y + i * (bar_height + bar_spacing)

            # Bar background
            bg_rect = pygame.Rect(chart_x, y, chart_width, bar_height)
            pygame.draw.rect(self._screen, (40, 40, 55), bg_rect, border_radius=3)

            # Store clickable area
            self._leaderboard_items.append((bg_rect, seller, color))

            # Highlight if this seller is selected
            is_selected = (
                self._selected_entity is not None
                and self._selected_entity.entity_type == "seller"
                and self._selected_entity.entity is seller
            )
            if is_selected:
                pygame.draw.rect(self._screen, (255, 255, 255), bg_rect, 2, border_radius=3)

            # Bar fill (proportional to reward)
            bar_width = int((reward / max_reward) * (chart_width - 4)) if max_reward > 0 else 0
            if bar_width > 0:
                bar_rect = pygame.Rect(chart_x + 2, y + 2, bar_width, bar_height - 4)
                pygame.draw.rect(self._screen, color, bar_rect, border_radius=2)

            # Seller label and reward (vertically centered)
            label = self._font.render(f"{seller.agent_id}", True, (255, 255, 255))
            label_y = y + (bar_height - label.get_height()) // 2
            self._screen.blit(label, (chart_x + 5, label_y))

            reward_text = self._font.render(f"{reward:.0f}", True, (200, 200, 200))
            reward_y = y + (bar_height - reward_text.get_height()) // 2
            self._screen.blit(reward_text, (chart_x + chart_width - reward_text.get_width() - 5, reward_y))

    def close(self) -> None:
        """Close the renderer and cleanup Pygame resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self._screen = None
            self._clock = None
            self._font = None
