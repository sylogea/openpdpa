import os
import pathlib
import zipfile

from dotenv import load_dotenv
from store import QDRANT_LOCATION, init_store


ROOT = pathlib.Path(__file__).resolve().parent
SNAPSHOT_NAME = "store.zip"


def _build_store():
	init_store()


def _zip_store():
	target_dir = pathlib.Path(QDRANT_LOCATION).resolve()
	if not target_dir.exists():
		raise RuntimeError(f"QDRANT_LOCATION does not exist: {target_dir}")
	output_path = ROOT / SNAPSHOT_NAME
	with zipfile.ZipFile(str(output_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
		for directory_path, _, filenames in os.walk(target_dir):
			for filename in filenames:
				full_path = pathlib.Path(directory_path) / filename
				archive.write(
					str(full_path),
					arcname=str(full_path.relative_to(target_dir))
				)
	return output_path


def main():
	load_dotenv(ROOT / ".env")
	_build_store()
	output_path = _zip_store()
	print(f"Snapshot written to: {output_path}")


if __name__ == "__main__":
	main()
