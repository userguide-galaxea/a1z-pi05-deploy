# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Data-only adapter checks; no model, hardware or network access is needed."""

import unittest

import numpy as np

from pi05_a1z.robot_adapter import A1ZInputs, A1ZOutputs, RepackA1ZDataset
from pi05_a1z.robot_adapter import action_sequence_keys, delta_action_mask

try:
    from openpi import transforms as openpi_transforms
except ImportError:
    openpi_transforms = None


def sample(mode="single", scalar_gripper=True):
    data = {
        "observation.state.left_arm": np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6]),
        "observation.state.left_gripper": np.array(0.7) if scalar_gripper else np.array([0.7]),
        "action.left_arm": np.tile(np.array([0.2, 0.3, -0.4, 0.5, 0.6, -0.7]), (4, 1)),
        "action.left_gripper": np.array([0.8, 0.7, 0.5, 0.2]),
        "observation.images.head_rgb": np.full((3, 8, 12), 0.5, dtype=np.float32),
        "observation.images.left_wrist_rgb": np.full((6, 10, 3), 30, dtype=np.uint8),
        "prompt": "pick up the cube and place it in the box",
    }
    if mode == "dual":
        data.update({
            "observation.state.right_arm": np.array([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]),
            "observation.state.right_gripper": np.array([0.4]),
            "action.right_arm": np.tile(np.array([-0.2, 0.3, -0.4, 0.5, -0.6, 0.7]), (4, 1)),
            "action.right_gripper": np.array([[0.1], [0.2], [0.3], [0.4]]),
            "observation.images.right_wrist_rgb": np.full((3, 7, 11), 80, dtype=np.uint8),
        })
    if not scalar_gripper:
        data["action.left_gripper"] = data["action.left_gripper"][:, None]
    return data


class AdapterTests(unittest.TestCase):
    def test_existing_scalar_and_vector_gripper_columns(self):
        for mode, width in (("single", 7), ("dual", 14)):
            for scalar in (True, False):
                with self.subTest(mode=mode, scalar=scalar):
                    data = sample(mode, scalar)
                    packed = RepackA1ZDataset(mode)(data)
                    self.assertEqual(packed["state"].shape, (width,))
                    self.assertEqual(packed["actions"].shape, (4, width))
                    np.testing.assert_allclose(packed["actions"][:, 6], [0.8, 0.7, 0.5, 0.2])
                    np.testing.assert_allclose(packed["state"][:6], data["observation.state.left_arm"])
                    if mode == "dual":
                        np.testing.assert_allclose(packed["state"][7:13], data["observation.state.right_arm"])
                        np.testing.assert_allclose(packed["actions"][:, 13], [0.1, 0.2, 0.3, 0.4])
                    self.assertEqual(packed["prompt"], data["prompt"])

    def test_camera_layout_dtype_and_missing_mask(self):
        transformed = A1ZInputs("single")(RepackA1ZDataset("single")(sample()))
        self.assertEqual(transformed["image"]["base_0_rgb"].shape, (8, 12, 3))
        self.assertEqual(transformed["image"]["base_0_rgb"].dtype, np.uint8)
        self.assertTrue(np.all(transformed["image"]["base_0_rgb"] == 128))
        self.assertTrue(transformed["image_mask"]["left_wrist_0_rgb"])
        self.assertFalse(transformed["image_mask"]["right_wrist_0_rgb"])
        self.assertFalse(np.any(transformed["image"]["right_wrist_0_rgb"]))
        dual = A1ZInputs("dual")(RepackA1ZDataset("dual")(sample("dual")))
        self.assertTrue(all(dual["image_mask"].values()))
        self.assertEqual(dual["image"]["right_wrist_0_rgb"].shape, (7, 11, 3))

    def test_dataset_fields_are_not_modified(self):
        raw = sample()
        packed = RepackA1ZDataset("single")(raw)
        transformed = A1ZInputs("single")(packed)
        transformed["actions"][0, 0] = 10
        transformed["state"][0] = 10
        self.assertAlmostEqual(float(raw["action.left_arm"][0, 0]), 0.2)
        self.assertAlmostEqual(float(packed["actions"][0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(packed["state"][0]), 0.1, places=6)

    def test_action_timestamp_keys_and_delta_mask_keep_grippers_absolute(self):
        self.assertEqual(action_sequence_keys("single"), ("action.left_arm", "action.left_gripper"))
        self.assertEqual(action_sequence_keys("dual"), (
            "action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper",
        ))
        self.assertEqual(delta_action_mask("single"), (True,) * 6 + (False,))
        self.assertEqual(delta_action_mask("dual"), ((True,) * 6 + (False,)) * 2)

    def test_no_aloha_sign_or_gripper_calibration(self):
        packed = RepackA1ZDataset("dual")(sample("dual"))
        transformed = A1ZInputs("dual")(packed)
        np.testing.assert_array_equal(transformed["state"], packed["state"])
        np.testing.assert_array_equal(transformed["actions"], packed["actions"])

    def test_inference_has_no_action_targets_and_output_removes_padding(self):
        data = RepackA1ZDataset("single")(sample())
        del data["actions"]
        transformed = A1ZInputs("single")(data)
        self.assertNotIn("actions", transformed)
        prediction = np.arange(4 * 32).reshape(4, 32).astype(np.float32)
        for mode, width in (("single", 7), ("dual", 14)):
            output = A1ZOutputs(mode)({"actions": prediction})
            np.testing.assert_array_equal(output["actions"], prediction[:, :width])

    def test_partial_action_chunks_nonfinite_and_wrong_units_fail(self):
        data = sample()
        del data["action.left_gripper"]
        with self.assertRaisesRegex(ValueError, "missing"):
            RepackA1ZDataset("single")(data)
        data = sample()
        data["action.left_gripper"][0] = 100
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            RepackA1ZDataset("single")(data)
        data = sample()
        data["action.left_arm"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            RepackA1ZDataset("single")(data)
        data = sample()
        data["action.left_gripper"] = np.ones(3)
        with self.assertRaisesRegex(ValueError, "horizon"):
            RepackA1ZDataset("single")(data)

    def test_invalid_images_and_missing_head_fail(self):
        data = RepackA1ZDataset("single")(sample())
        data["images"]["cam_high"][0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            A1ZInputs("single")(data)
        del data["images"]["cam_high"]
        with self.assertRaisesRegex(ValueError, "head camera"):
            A1ZInputs("single")(data)

    def test_required_wrist_streams_cannot_be_silently_masked(self):
        for mode, camera in (("single", "cam_left_wrist"), ("dual", "cam_right_wrist")):
            data = RepackA1ZDataset(mode)(sample(mode))
            del data["images"][camera]
            with self.assertRaisesRegex(ValueError, "Missing required A1Z camera"):
                A1ZInputs(mode)(data)

    @unittest.skipIf(openpi_transforms is None, "Requires the separate openpi model environment")
    def test_official_delta_padding_absolute_roundtrip(self):
        for mode in ("single", "dual"):
            packed = RepackA1ZDataset(mode)(sample(mode))
            original = packed["actions"].copy()
            inputs = A1ZInputs(mode)(packed)
            inputs = openpi_transforms.DeltaActions(delta_action_mask(mode))(inputs)
            np.testing.assert_array_equal(inputs["actions"][:, 6], original[:, 6])
            inputs = openpi_transforms.PadStatesAndActions(32)(inputs)
            self.assertEqual(inputs["state"].shape, (32,))
            self.assertEqual(inputs["actions"].shape, (4, 32))
            outputs = openpi_transforms.AbsoluteActions(delta_action_mask(mode))(inputs)
            outputs = A1ZOutputs(mode)(outputs)
            np.testing.assert_allclose(outputs["actions"], original, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
