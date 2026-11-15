import copy
import struct
import tempfile
import unittest
from pathlib import Path

from tools.opponent_study import bootstrap, paired_clusters, parameter_payload_digest


def report(scores):
    games = [
        {"config": {"game_seed": index // 2 + 10, "agent_seeds": [17, 18], "max_actions": 1000},
         "learner_seat": index % 2, "score": score}
        for index, score in enumerate(scores)
    ]
    return {"device": "cuda:0", "config": {"games": len(games), "seed": 81, "inference_batch_games": 128},
            "baselines": [{"baseline": {"kind": "greedy", "weights": [0, 0, 1, 5, 10, 100]},
                           "summary": {"opponent": "greedy"}, "games": games}]}


class PairedStudyTests(unittest.TestCase):
    def test_parameter_fingerprints_ignore_metadata_ids_but_detect_weight_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, (metadata, payload) in enumerate([(b"id1", b"weights"), (b"id222", b"weights"), (b"id1", b"changed")]):
                path = Path(directory) / f"{index}.bpk"
                header = struct.pack("<IHI", 0x4255524E, 1, len(metadata))
                path.write_bytes((header + metadata).ljust(256, b"\0") + payload)
                paths.append(path)
            self.assertEqual(parameter_payload_digest(paths[0]), parameter_payload_digest(paths[1]))
            self.assertNotEqual(parameter_payload_digest(paths[0]), parameter_payload_digest(paths[2]))

    def test_seat_pairs_form_one_bootstrap_cluster(self):
        left = report([1, 1, 0, 1])
        right = report([0, 0, 1, 0])
        self.assertEqual(paired_clusters(left, right, ["greedy"]), [1, 0])
        self.assertEqual(bootstrap([0, 0, 0]), [0, 0])

    def test_mismatched_seeds_device_and_opponent_are_rejected(self):
        reference = report([1, 0, .5, .5])
        for mutation in ["device", "seed", "opponent"]:
            changed = copy.deepcopy(reference)
            if mutation == "device":
                changed["device"] = "cpu"
            elif mutation == "seed":
                changed["baselines"][0]["games"][0]["config"]["game_seed"] += 1
            else:
                changed["baselines"][0]["baseline"]["weights"][0] = 1
            with self.assertRaises(AssertionError):
                paired_clusters(reference, changed, ["greedy"])


if __name__ == "__main__":
    unittest.main()
