"""Read and verify the versioned forecasting protocol without loading test arrays."""

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments/forecasting-v1.json"


def protocol_digest(protocol):
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("protocol_sha256") != protocol_digest(protocol):
        raise ValueError("Protocol checksum mismatch; create a reviewed protocol revision")
    return protocol


def file_digest(path):
    with Path(path).open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    digest = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(root=ROOT, protocol_path=DEFAULT_PROTOCOL):
    """Check opaque file bytes only, including the sealed test archive."""
    root = Path(root).resolve()
    protocol = load_protocol(protocol_path)
    checked = []
    for relative, expected in protocol["dataset"]["files_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Dataset path escapes the project: {relative}")
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Frozen dataset file missing or changed: {relative}")
        checked.append(relative)
    return checked


if __name__ == "__main__":
    checked = verify_dataset()
    print(f"Verified {len(checked)} frozen files. No arrays loaded; provenance approval remains separate.")
