import os
import pathlib
import zipfile

from dotenv import load_dotenv
from store import QDRANT_LOCATION, init_store


ROOT = pathlib.Path(__file__).resolve().parent
SNAPSHOT_NAME = "store.zip"


def _build_store():
	print("Building Qdrant store from PDFs...")
	init_store()
	print("Store build complete.")


def _zip_store():
	target_dir = pathlib.Path(QDRANT_LOCATION).resolve()
	if not target_dir.exists():
		raise RuntimeError(f"QDRANT_LOCATION does not exist: {target_dir}")

	all_files = []
	for directory_path, _, filenames in os.walk(target_dir):
		for filename in filenames:
			all_files.append(pathlib.Path(directory_path) / filename)

	if not all_files:
		raise RuntimeError(f"No files found under QDRANT_LOCATION: {target_dir}")

	output_path = ROOT / SNAPSHOT_NAME
	total_files = len(all_files)

	print("Creating snapshot...")
	with zipfile.ZipFile(str(output_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
		for index, full_path in enumerate(all_files, start=1):
			archive.write(
				str(full_path),
				arcname=str(full_path.relative_to(target_dir))
			)
			progress = int(index * 100 / total_files)
			print(
				f"\rZipping store: {progress:3d}% ({index}/{total_files} files)",
				end="",
				flush=True
			)
	print()  # move to next line after progress bar
	print("Snapshot creation complete.")
	return output_path


def main():
	load_dotenv(ROOT / ".env")
	_build_store()
	output_path = _zip_store()
	print(f"Snapshot written to: {output_path}")


if __name__ == "__main__":
	main()
