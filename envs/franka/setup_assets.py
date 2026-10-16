"""Download only the pinned Menagerie Panda, including its upstream license."""
import argparse
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

REVISION = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
DEFAULT_MODEL = "data/menagerie/franka_emika_panda/panda.xml"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/franka_emika_panda"


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        path = directory / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(f"{BASE}/{name}", timeout=60) as response:
                content = response.read()
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
        return path

    xml = fetch("panda.xml")
    for name in ("LICENSE", "README.md"):
        fetch(name)
    for mesh in ET.parse(xml).findall("asset/mesh"):
        fetch("assets/" + mesh.attrib["file"])
    (directory / "REVISION").write_text(REVISION + "\n")
    print(f"Panda assets: {xml} (Menagerie {REVISION})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", default=str(Path(DEFAULT_MODEL).parent))
    download(ap.parse_args().directory)
