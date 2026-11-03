# This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Protocol checks with synthetic observations; no robot or GPU is opened."""

import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import msgpack
import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from pi05_a1z.policy_bridge import (
    BridgeConfig, BridgeSession, PARTS, PolicyBridge,
    make_upstream_factory, observation_for_openpi, pack, unpack, validate_chunk,
)


def observation(mode="single"):
    state = {key: np.full(size, 0.5 if "gripper" in key else 0.0, np.float32)
             for key, size in PARTS[mode].items()}
    images = {
        "head_rgb": np.full((3, 8, 12), 41, np.uint8),
        "left_wrist_rgb": np.full((3, 5, 7), 82, np.uint8),
    }
    if mode == "dual":
        images["right_wrist_rgb"] = np.full((3, 6, 9), 123, np.uint8)
    return {"state": state, "images": images, "task": "pick up the cube",
            "frequency": 15, "embodiment_type": "galaxea_a1z"}


def prediction(mode="single", *, offset=0.0, horizon=50):
    result = np.full((horizon, sum(PARTS[mode].values())), offset, np.float32)
    result[:, 0] = np.arange(horizon, dtype=np.float32) / 100 + offset
    result[:, 6] = 0.5
    if mode == "dual":
        result[:, 13] = 0.75
    return {"actions": result}


class FakePolicy:
    def __init__(self, mode="single", *, offset=0.0, result=None, delay=0.0):
        self.mode, self.offset, self.delay = mode, offset, delay
        self.result = result
        self.observations = []
        self.closed = False

    def get_server_metadata(self):
        return {"a1z_mode": self.mode, "a1z_action_dim": sum(PARTS[self.mode].values()),
                "action_representation": "absolute", "action_horizon": 50}

    def infer(self, obs):
        time.sleep(self.delay)
        self.observations.append(obs)
        return copy.deepcopy(self.result if self.result is not None
                             else prediction(self.mode, offset=self.offset))

    def close(self):
        self.closed = True


class CodecAndMappingTests(unittest.TestCase):
    def test_roundtrip_legacy_string_key_numpy_envelopes(self):
        original = observation("dual")
        restored = unpack(pack(original))
        np.testing.assert_array_equal(restored["state"]["left_arm"], original["state"]["left_arm"])
        np.testing.assert_array_equal(restored["images"]["head_rgb"], original["images"]["head_rgb"])
        wire = msgpack.unpackb(pack({"array": np.zeros(2, np.float32)}), raw=False)
        self.assertIs(wire["array"]["__ndarray__"], True)
        with self.assertRaisesRegex(ValueError, "binary"):
            unpack("not binary")
        with self.assertRaises(ValueError):
            pack({"array": np.zeros(2, dtype=object)})

    def test_single_and_dual_mapping_order_rgb_and_distinct_image_sizes(self):
        for mode, dim in (("single", 7), ("dual", 14)):
            with self.subTest(mode=mode):
                source = observation(mode)
                for i, (key, size) in enumerate(PARTS[mode].items()):
                    source["state"][key] = np.full(size, 0.5 if "gripper" in key else i, np.float32)
                source["images"]["head_rgb"][:, 0, 0] = [11, 22, 33]
                mapped = observation_for_openpi(source, BridgeConfig(mode))
                self.assertEqual(mapped["state"].shape, (dim,))
                self.assertEqual(mapped["state"].dtype, np.float32)
                np.testing.assert_array_equal(mapped["state"][:6], np.zeros(6))
                self.assertEqual(mapped["state"][6], 0.5)
                if mode == "dual":
                    np.testing.assert_array_equal(mapped["state"][7:13], np.full(6, 2))
                    self.assertEqual(mapped["state"][13], 0.5)
                self.assertEqual(mapped["images"]["cam_high"].shape, (3, 8, 12))
                self.assertEqual(mapped["images"]["cam_left_wrist"].shape, (3, 5, 7))
                np.testing.assert_array_equal(mapped["images"]["cam_high"][:, 0, 0], [11, 22, 33])
                self.assertEqual(mapped["prompt"], "pick up the cube")

    def test_bad_observation_is_rejected(self):
        bad = observation()
        bad["state"]["left_gripper"][0] = 100
        with self.assertRaisesRegex(ValueError, "normalized"):
            observation_for_openpi(bad, BridgeConfig())
        bad = observation()
        bad["images"]["head_rgb"] = np.zeros((8, 12, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "CHW"):
            observation_for_openpi(bad, BridgeConfig())
        with self.assertRaisesRegex(ValueError, "State layout"):
            observation_for_openpi(observation("dual"), BridgeConfig())

    def test_entire_horizon_is_checked_before_any_action(self):
        for failure in ("nonfinite", "gripper", "shape", "horizon"):
            with self.subTest(failure=failure):
                response = prediction()
                if failure == "nonfinite":
                    response["actions"][49, 0] = np.nan
                elif failure == "gripper":
                    response["actions"][49, 6] = 1.021
                elif failure == "shape":
                    response["actions"] = np.zeros((50, 32))
                else:
                    response["actions"] = response["actions"][:3]
                with self.assertRaises(ValueError):
                    validate_chunk(response, BridgeConfig())
        response = prediction("dual")
        response["actions"][0, 6] = 1.01
        self.assertEqual(validate_chunk(response, BridgeConfig("dual")).shape, (16, 14))


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunk_need_obs_and_no_extra_inference(self):
        policy = FakePolicy()
        session = BridgeSession(BridgeConfig(action_steps=3), lambda: policy)
        for index, request in enumerate((observation(), {}, {})):
            response = await session.respond(request)
            self.assertAlmostEqual(response["action"]["left_arm"][0], index / 100)
            self.assertEqual(response["need_obs"], index == 2)
        self.assertEqual(len(policy.observations), 1)
        with self.assertRaisesRegex(ValueError, "fresh observation"):
            await session.respond({})
        self.assertTrue(policy.closed)

    async def test_reset_clears_cache_and_opens_a_fresh_upstream(self):
        created = []

        def factory():
            policy = FakePolicy(offset=len(created) / 10)
            created.append(policy)
            return policy

        session = BridgeSession(BridgeConfig(), factory)
        await session.respond(observation())
        self.assertTrue(session.actions)
        self.assertEqual(await session.respond({"reset": True}), {"reset": True, "need_obs": True})
        self.assertFalse(session.actions)
        self.assertTrue(created[0].closed)
        response = await session.respond(observation())
        self.assertAlmostEqual(response["action"]["left_arm"][0], 0.1)
        self.assertEqual(len(created), 2)
        await session.close()

    async def test_prediction_error_exposes_no_partial_chunk(self):
        bad = prediction()
        bad["actions"][25, 6] = 2
        policy = FakePolicy(result=bad)
        session = BridgeSession(BridgeConfig(), lambda: policy)
        with self.assertRaisesRegex(ValueError, "entire chunk refused"):
            await session.respond(observation())
        self.assertFalse(session.actions)
        self.assertTrue(policy.closed)

    async def test_returned_horizon_must_match_upstream_metadata(self):
        policy = FakePolicy(result=prediction(horizon=40))
        session = BridgeSession(BridgeConfig(), lambda: policy)
        with self.assertRaisesRegex(ValueError, "different action horizon"):
            await session.respond(observation())
        self.assertFalse(session.actions)
        self.assertTrue(policy.closed)

    async def test_session_caches_are_independent(self):
        one = BridgeSession(BridgeConfig(), lambda: FakePolicy(offset=0.1))
        two = BridgeSession(BridgeConfig(), lambda: FakePolicy(offset=0.6))
        first, second = await asyncio.gather(one.respond(observation()), two.respond(observation()))
        self.assertAlmostEqual(first["action"]["left_arm"][0], 0.1)
        self.assertAlmostEqual(second["action"]["left_arm"][0], 0.6)
        self.assertEqual(len(one.actions), 15)
        self.assertEqual(len(two.actions), 15)
        await one.close()
        self.assertEqual(len(two.actions), 15)
        await two.close()

    async def test_sync_inference_does_not_block_event_loop(self):
        policy = FakePolicy(delay=0.08)
        session = BridgeSession(BridgeConfig(), lambda: policy)
        pending = asyncio.create_task(session.respond(observation()))
        ticks = 0
        while not pending.done():
            await asyncio.sleep(0.005)
            ticks += 1
        await pending
        self.assertGreaterEqual(ticks, 3)
        await session.close()

    async def test_metadata_mismatch_is_rejected_before_handshake(self):
        policy = FakePolicy("dual")
        session = BridgeSession(BridgeConfig("single"), lambda: policy)
        with self.assertRaisesRegex(ValueError, "a1z_mode"):
            await session.open()
        self.assertTrue(policy.closed)

    async def test_missing_a1z_metadata_does_not_accept_an_aloha_server(self):
        policy = FakePolicy("dual")
        policy.get_server_metadata = lambda: {}
        session = BridgeSession(BridgeConfig("dual"), lambda: policy)
        with self.assertRaisesRegex(ValueError, "use an A1Z policy server"):
            await session.open()
        self.assertTrue(policy.closed)

    async def test_inference_timeout_clears_and_closes_session(self):
        policy = FakePolicy(delay=0.06)
        session = BridgeSession(BridgeConfig(upstream_timeout=0.01), lambda: policy)
        with self.assertRaises(TimeoutError):
            await session.respond(observation())
        self.assertTrue(policy.closed)
        self.assertFalse(session.actions)

    async def test_cancelled_connection_is_closed_when_factory_finishes(self):
        policy = FakePolicy()

        def factory():
            time.sleep(0.04)
            return policy

        session = BridgeSession(BridgeConfig(), factory)
        pending = asyncio.create_task(session.open())
        await asyncio.sleep(0.005)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await asyncio.sleep(0.06)
        self.assertTrue(policy.closed)


class WebsocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_single_and_dual_clients_complete_synthetic_cycles(self):
        root = Path(__file__).resolve().parents[1]
        for mode, filename, config_name in (
            ("single", "single_client.py", "pc-single.example.json"),
            ("dual", "dual_arm_client.py", "pc-direct.example.json"),
        ):
            with self.subTest(mode=mode):
                bridge = PolicyBridge(BridgeConfig(mode, action_steps=2), lambda: FakePolicy(mode))
                async with serve(bridge.handler, "127.0.0.1", 0, compression=None) as server:
                    config = json.loads((root / "configs" / "deployment" / config_name).read_text())
                    config.update(server_port=server.sockets[0].getsockname()[1],
                                  max_cycles=3, image_size=[12, 8], control_frequency=100)
                    with tempfile.TemporaryDirectory(prefix="pi05-bridge-test-") as directory:
                        path = Path(directory) / "synthetic.json"
                        path.write_text(json.dumps(config))
                        process = await asyncio.create_subprocess_exec(
                            sys.executable, str(root / "a1z_runtime" / "client" / filename),
                            "--config", str(path), stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        try:
                            output, _ = await asyncio.wait_for(process.communicate(), timeout=10)
                        except BaseException:
                            process.kill()
                            await process.wait()
                            raise
                        self.assertEqual(process.returncode, 0, output.decode())
                        self.assertIn(b"dry-run step 2:", output)

    async def test_real_downstream_socket_and_legacy_synthetic_clients(self):
        runtime = Path(__file__).resolve().parents[1] / "a1z_runtime" / "client"
        self.assertTrue(runtime.is_dir(), "The original A1Z client runtime must be present")
        sys.path.insert(0, str(runtime))
        try:
            from dual_arm_client import validate_handshake as dual_handshake
            from single_left_client import validate_handshake as single_handshake
            for mode, handshake in (("single", single_handshake), ("dual", dual_handshake)):
                with self.subTest(mode=mode):
                    created = []

                    def factory():
                        policy = FakePolicy(mode)
                        created.append(policy)
                        return policy

                    bridge = PolicyBridge(BridgeConfig(mode, action_steps=2), factory)
                    async with serve(bridge.handler, "127.0.0.1", 0, compression=None) as server:
                        port = server.sockets[0].getsockname()[1]
                        async with connect(f"ws://127.0.0.1:{port}", compression=None) as ws:
                            handshake(unpack(await ws.recv()))
                            await ws.send(pack(observation(mode)))
                            first = unpack(await ws.recv())
                            self.assertEqual(set(first["action"]), set(PARTS[mode]))
                            self.assertFalse(first["need_obs"])
                            await ws.send(pack({}))
                            self.assertTrue(unpack(await ws.recv())["need_obs"])
                            await ws.send(pack({"reset": True}))
                            self.assertTrue(unpack(await ws.recv())["reset"])
                        await asyncio.sleep(0)
                    self.assertTrue(all(policy.closed for policy in created))
        finally:
            sys.path.remove(str(runtime))

    async def test_official_upstream_client_wire_with_mock_model_server(self):
        # This imports only openpi-client, never the JAX/Torch model package.
        from openpi_client import msgpack_numpy

        seen = []

        async def upstream(ws):
            await ws.send(msgpack_numpy.packb(FakePolicy("dual").get_server_metadata()))
            async for raw in ws:
                seen.append(msgpack_numpy.unpackb(raw))
                await ws.send(msgpack_numpy.packb(prediction("dual")))

        async with serve(upstream, "127.0.0.1", 0, compression=None) as server:
            port = server.sockets[0].getsockname()[1]
            factory = make_upstream_factory("127.0.0.1", port, 2)
            session = BridgeSession(BridgeConfig("dual", upstream_timeout=2), factory)
            first = await session.respond(observation("dual"))
            self.assertEqual(first["action"]["right_gripper"].shape, (1,))
            self.assertEqual(seen[0]["state"].shape, (14,))
            self.assertIsInstance(seen[0]["state"], np.ndarray)
            self.assertEqual(seen[0]["images"]["cam_right_wrist"].dtype, np.uint8)
            await session.close()


if __name__ == "__main__":
    unittest.main()
