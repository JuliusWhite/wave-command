import unittest
from math import hypot

from src.translation.pointer_controller import PointerController


class PointerControllerTests(unittest.TestCase):
    def test_maps_normalized_coordinates_to_display(self) -> None:
        positions: list[tuple[int, int]] = []
        cursor = [0, 0]
        now = [0.0]

        def move_cursor(x: int, y: int) -> None:
            positions.append((x, y))
            cursor[:] = [x, y]

        controller = PointerController(
            move_cursor=move_cursor,
            screen_size=lambda: (101, 51),
            active_margin=0,
            smoothing=1,
            deadzone_px=0,
            screen_margin_px=0,
            max_speed_px_per_sec=10000,
            cursor_position=lambda: (cursor[0], cursor[1]),
            clock=lambda: now[0],
        )

        controller.move_to(0, 0)
        now[0] = 1.0
        controller.move_to(1, 1)

        self.assertEqual(positions, [(100, 50)])

    def test_smoothing_and_reset(self) -> None:
        positions: list[tuple[int, int]] = []
        cursor = [0, 0]
        now = [0.0]

        def move_cursor(x: int, y: int) -> None:
            positions.append((x, y))
            cursor[:] = [x, y]

        controller = PointerController(
            move_cursor=move_cursor,
            screen_size=lambda: (101, 101),
            active_margin=0,
            smoothing=0.5,
            deadzone_px=0,
            screen_margin_px=0,
            max_speed_px_per_sec=10000,
            cursor_position=lambda: (cursor[0], cursor[1]),
            clock=lambda: now[0],
        )

        controller.move_to(0, 0)
        now[0] = 0.1
        controller.move_to(1, 1)
        controller.reset()
        now[0] = 0.2
        controller.move_to(1, 0)
        now[0] = 0.3
        controller.move_to(1, 0)

        self.assertEqual(positions, [(50, 50), (75, 25)])

    def test_active_area_and_mirroring_keep_cursor_off_screen_edges(self) -> None:
        positions: list[tuple[int, int]] = []
        cursor = [0, 0]
        now = [0.0]

        def move_cursor(x: int, y: int) -> None:
            positions.append((x, y))
            cursor[:] = [x, y]

        controller = PointerController(
            move_cursor=move_cursor,
            screen_size=lambda: (101, 101),
            smoothing=1,
            deadzone_px=0,
            screen_margin_px=3,
            mirror_x=True,
            max_speed_px_per_sec=10000,
            cursor_position=lambda: (cursor[0], cursor[1]),
            clock=lambda: now[0],
        )

        controller.move_to(0, 1)
        now[0] = 1.0
        controller.move_to(0, 1)

        self.assertEqual(positions, [(97, 97)])

    def test_limits_cursor_speed_after_a_large_landmark_jump(self) -> None:
        positions: list[tuple[int, int]] = []
        cursor = [1000, 500]
        now = [0.0]

        def move_cursor(x: int, y: int) -> None:
            positions.append((x, y))
            cursor[:] = [x, y]

        controller = PointerController(
            move_cursor=move_cursor,
            screen_size=lambda: (2001, 1001),
            active_margin=0,
            smoothing=1,
            deadzone_px=0,
            screen_margin_px=0,
            max_speed_px_per_sec=100,
            cursor_position=lambda: (cursor[0], cursor[1]),
            clock=lambda: now[0],
        )

        controller.move_to(1, 1)
        now[0] = 10.0
        controller.move_to(1, 1)

        self.assertEqual(len(positions), 1)
        self.assertLessEqual(
            hypot(positions[0][0] - 1000, positions[0][1] - 500),
            8,
        )

    def test_rejects_non_finite_coordinates(self) -> None:
        controller = PointerController(
            screen_size=lambda: (101, 101),
            active_margin=0,
            screen_margin_px=0,
        )

        with self.assertRaisesRegex(ValueError, "must be finite"):
            controller.move_to(float("nan"), 0.5)


if __name__ == "__main__":
    unittest.main()
