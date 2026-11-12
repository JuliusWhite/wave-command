import unittest

from src.translation.pointer_controller import PointerController


class PointerControllerTests(unittest.TestCase):
    def test_maps_normalized_coordinates_to_display(self) -> None:
        positions: list[tuple[int, int]] = []
        controller = PointerController(
            move_cursor=lambda x, y: positions.append((x, y)),
            screen_size=lambda: (101, 51),
            active_margin=0,
            smoothing=1,
            deadzone_px=0,
            screen_margin_px=0,
        )

        controller.move_to(0, 0)
        controller.move_to(1, 1)

        self.assertEqual(positions, [(0, 0), (100, 50)])

    def test_smoothing_and_reset(self) -> None:
        positions: list[tuple[int, int]] = []
        controller = PointerController(
            move_cursor=lambda x, y: positions.append((x, y)),
            screen_size=lambda: (101, 101),
            active_margin=0,
            smoothing=0.5,
            deadzone_px=0,
            screen_margin_px=0,
        )

        controller.move_to(0, 0)
        controller.move_to(1, 1)
        controller.reset()
        controller.move_to(1, 0)

        self.assertEqual(positions, [(0, 0), (50, 50), (100, 0)])

    def test_active_area_and_mirroring_keep_cursor_off_screen_edges(self) -> None:
        positions: list[tuple[int, int]] = []
        controller = PointerController(
            move_cursor=lambda x, y: positions.append((x, y)),
            screen_size=lambda: (101, 101),
            smoothing=1,
            deadzone_px=0,
            screen_margin_px=3,
            mirror_x=True,
        )

        controller.move_to(0, 1)

        self.assertEqual(positions, [(97, 97)])

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
