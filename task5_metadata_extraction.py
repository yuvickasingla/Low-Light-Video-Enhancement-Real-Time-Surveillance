import argparse
import os
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_basic_info(path: str):
    img = Image.open(path)
    print("== Basic file info ==")
    print(f"  Path: {path}")
    print(f"  Format: {img.format}")
    print(f"  Dimensions: {img.size[0]} x {img.size[1]}")
    print(f"  Mode (colour profile): {img.mode}")
    print(f"  File size: {os.path.getsize(path)/1024:.1f} KB")
    return img


def get_exif(img: Image.Image):
    print("\n== EXIF metadata ==")
    exif_data = img.getexif()
    if not exif_data:
        print("  No EXIF block found (common for PNG, screenshots, or "
              "re-saved/edited images which strip EXIF).")
        return

    found_any = False
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            gps_info = {GPSTAGS.get(t, t): v for t, v in value.items()}
            print(f"  GPSInfo: {gps_info}")
            found_any = True
            continue
        print(f"  {tag}: {value}")
        found_any = True

    if not found_any:
        print("  EXIF block present but empty.")


def get_tiff_tags(path: str):
    try:
        import tifffile
    except ImportError:
        print("  (install `tifffile` to inspect TIFF/GeoTIFF tags: "
              "pip install tifffile)")
        return

    print("\n== TIFF tag dump ==")
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        for tag in page.tags.values():
            print(f"  {tag.name}: {tag.value}")
        if tif.is_geotiff:
            print("  -> GeoTIFF detected: coordinate reference system / "
                  "geotransform tags present (see GeoKeyDirectoryTag, "
                  "ModelPixelScaleTag, ModelTiepointTag above).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    ext = Path(args.image).suffix.lower()
    img = get_basic_info(args.image)
    get_exif(img)

    if ext in (".tif", ".tiff"):
        get_tiff_tags(args.image)

    print("\n== Discussion prompts to answer in your report ==")
    print("- Which of the fields above (dimensions, date/time, camera model,")
    print("  colour profile, GPS, exposure, CRS, sensor name...) are present")
    print("  here, and which are missing?")
    print("- Privacy: does GPSInfo appear? If so, this file encodes the exact")
    print("  location the photo was taken — a real privacy concern for shared")
    print("  low-light/surveillance datasets like ExDark.")
    print("- Reproducibility: exposure/ISO/camera model let you judge whether")
    print("  a 'low-light' label is about scene lighting or camera settings.")


if __name__ == "__main__":
    main()
