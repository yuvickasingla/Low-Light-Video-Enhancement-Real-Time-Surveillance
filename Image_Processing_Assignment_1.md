# Image Processing Assignment 1

**Name:** Yuvicka\
**Roll No.:** 1024240016\
**Group:** 3X11

------------------------------------------------------------------------

## Task 1 --- Problem Statement and Dataset

### Low-Light Video Enhancement & Real-Time Surveillance

**Problem:** Standard object detectors and image-enhancement pipelines
are trained and validated almost entirely on well-lit imagery, yet a big
share of real-world applications --- night-time surveillance, autonomous
driving after dusk, wildlife monitoring --- must operate in low light,
where noise, low contrast, and colour distortion degrade both human
interpretation and downstream computer-vision algorithms. This project
evaluates how basic image-processing operations (format conversion,
thresholding, filtering) behave on genuinely dark, real-world images, as
a first step toward a low-light enhancement pipeline.

### Why image processing helps

Simple representation changes (RGB → grayscale, adaptive thresholding
for binarisation, YCbCr separation of luminance from colour) are the
building blocks of every low-light enhancement method ---
brightness/contrast correction is normally done on the luminance (Y)
channel alone so that colour information isn't distorted. Understanding
how these conversions behave on dark, noisy images --- where the usual
value ranges and thresholds break down --- is a prerequisite for any
later enhancement or detection work.

### Dataset chosen

The **Exclusively Dark (ExDark) dataset (Loh & Chan, 2019)** is a
collection of **7,363 real-world low-light JPEG images** spanning **10
lighting conditions**, with both image-level class labels and object
bounding boxes across 12 categories:

-   Bicycle
-   Boat
-   Bottle
-   Bus
-   Car
-   Cat
-   Chair
-   Cup
-   Dog
-   Motorbike
-   People
-   Table

Each image shows one or more everyday objects photographed in dim,
naturally or artificially lit environments such as streets, rooms, and
vehicles.

### Origin

Collected and curated by Loh & Chan for the paper **"Getting to Know
Low-light Images with the Exclusively Dark Dataset"** (*Computer Vision
and Image Understanding*, 2019); hosted at:

https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

### Limitations

-   Images vary widely in lighting, introducing heavy sensor noise and
    motion blur.
-   Resolutions and aspect ratios are inconsistent across images.
-   Class distribution is imbalanced.
-   No depth or spectral information is present; only standard 3-channel
    RGB is available.
-   Because the images were sourced from the wider internet,
    provenance/privacy of any people or license plates visible in the
    "People"/"Car" classes isn't individually verifiable.

------------------------------------------------------------------------

## Task 2 --- Format Conversion

### Results

1.  **RGB**
    -   output: C:\Users\HP\Desktop\Image_Processing\outputs\task2\rgb.png
    -   Shape: `(334, 500, 3)`
    -   Range: `[0, 255]`
2.  **Grayscale**
    -   output: C:\Users\HP\Desktop\Image_Processing\outputs\task2\grayscale.png
    -   Shape: `(334, 500)`
    -   Range: `[0, 255]`
3.  **Binary**
    -   output: C:\Users\HP\Desktop\Image_Processing\outputs\task2\binary_otsu.png
    -   Threshold chosen automatically: `86.0`
    -   Pixel values present: `[0, 255]`

### Thresholding Method

An automatic thresholding method, **Otsu's thresholding**, was used to
convert the grayscale ExDark image into a binary image. A fixed
threshold was not manually selected. Instead, Otsu's method
automatically determines the optimal threshold value from the grayscale
image by analyzing the distribution of pixel intensities.

### Features that become clearer or disappear after grayscale and binary conversion

After converting the ExDark image from RGB to grayscale, the **colour
information is removed**, but brightness and intensity differences are
preserved. This helps in identifying structure, as the **main car/object
becomes more visible against the dark background due to its brighter
regions appearing in gray shades**. Bright areas like **lights or
reflections** are still distinguishable.

After converting the grayscale image to a binary image using **Otsu's
thresholding**, pixels become only black or white. This makes **strong
edges and bright regions more prominent**, with illuminated parts of the
car appearing white against a black background.

However, many details are lost. **Colour information is completely
removed**, and subtle brightness variations also disappear, causing some
parts of the car and background to merge or vanish if they fall below
the threshold.

Overall, grayscale keeps structural intensity information, while binary
simplifies the image to only strong foreground and background
separation.

------------------------------------------------------------------------

## Task 3 --- Hyperspectral

### Dataset / Sensor

-   **Sensor:** AVIRIS airborne imaging spectrometer
-   **Scene:** Indian Pines test site, NW Indiana, USA
-   **Spatial dimensions:** `145 × 145`
-   **Number of bands:** `200` (20 water-absorption bands already
    removed from the original 224)
-   **Approximate wavelength range:** `~400 nm – 2500 nm` (visible to
    SWIR)

-   output: C:\Users\HP\Desktop\Image_Processing\outputs\task3\false_color_composite.png
            C:\Users\HP\Desktop\Image_Processing\outputs\task3\pixel_spectrum.png

    

### Interpretation

Two objects that look identical in an RGB image (same colour to the
naked eye/camera) can have very different reflectance curves across the
200 bands here --- e.g. healthy vs. stressed vegetation reflect
similarly in visible light but diverge sharply in the near-infrared
bands. This is the basis for applications like crop health monitoring,
mineral identification, and food-quality inspection, where hyperspectral
imaging distinguishes materials that RGB cameras cannot.

------------------------------------------------------------------------

## Task 4 --- Image-File Extension Analysis

  --------------------------------------------------------------------------------------------------------
  Extension        Normally represents   Compression    Lossy /       Why is it          Limitation
                                                        Lossless /    suitable?          
                                                        None                             
  ---------------- --------------------- -------------- ------------- ------------------ -----------------
  `.jpg / .jpeg`   Standard              Compressed     Lossy         Very small file    Repeated saving
                   photographic/raster                                size; suitable for can introduce
                   images                                             photographs and    compression
                                                                      ExDark low-light   artifacts; no
                                                                      images; easy to    transparency;
                                                                      store and process  generally not
                                                                                         ideal for
                                                                                         scientific data
                                                                                         requiring exact
                                                                                         pixel values

  `.png`           Raster/graphic images Compressed     Lossless      Preserves pixel    Usually larger
                                                                      information;       than JPEG for
                                                                      supports           photographs; file
                                                                      transparency; good size can become
                                                                      for storing        large for
                                                                      processed images   high-resolution
                                                                      such as            images
                                                                      grayscale/binary   
                                                                      results            

  `.bmp`           Bitmap/raster images  Usually        None          Simple format with Very large file
                                         uncompressed                 direct pixel       size; inefficient
                                                                      storage; useful    for storage and
                                                                      for basic          transmission;
                                                                      image-processing   limited features
                                                                      experiments        compared with
                                                                                         modern formats

  `.tif / .tiff`   High-quality          Can be         Lossless or   Suitable for       Files can be
                   raster/scientific     compressed or  none,         scientific imaging considerably
                   images                uncompressed   depending on  because it can     larger than JPEG;
                                                        compression   preserve           more complex than
                                                                      high-quality pixel simple image
                                                                      data and support   formats
                                                                      high bit depths;   
                                                                      commonly used for  
                                                                      professional image 
                                                                      processing         

  `.dcm`           DICOM medical images  Can be         Lossless or   Designed for       More specialized
                                         compressed or  lossy,        medical imaging;   and complex; not
                                         uncompressed   depending on  can store image    normally used for
                                                        the DICOM     data together with ordinary
                                                        transfer      important patient, photographs
                                                        syntax        scanner, and       
                                                                      examination        
                                                                      metadata           
  --------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## Task 5 --- Metadata Extraction and Analysis

### Selected Image

**Path:**
`C:\Users\HP\Desktop\Image_Processing\data\exdark\Car\2015_02406.jpg`

  Field                   Result
  ----------------------- -----------
  Format                  JPEG
  Dimensions              500 × 334
  Mode (colour profile)   RGB
  File size               82.9 KB

### EXIF Metadata

No EXIF metadata block was found in the image. Therefore, information
such as the **date and time of capture, camera model, exposure settings,
ISO value, and GPS location** is not available in this particular image
file.

Since no GPS information is present, the image does not contain embedded
GPS coordinates that could reveal the exact capture location.

### Reproducibility

The absence of metadata also limits reproducibility. For example,
without camera model, ISO, exposure time, or other acquisition
information, it is difficult to determine whether the low-light
appearance resulted mainly from the actual scene illumination or from
camera settings.

### Conclusion

Overall, the image contains basic file information such as its JPEG
format, dimensions, RGB colour mode, and file size, but it does not
contain EXIF acquisition metadata.
