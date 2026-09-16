# Automatic PDF / Image / Text Editor

A command-line Python tool for automatically editing PDF files.

The tool supports:

* Deleting individual PDF pages
* Deleting page ranges
* Finding and deleting multiple text strings
* Finding all occurrences of text
* PDF text redaction
* Replacing text
* Detecting images or image regions from screenshots/crops
* Detecting images across multiple PDF pages
* Detecting multiple occurrences of the same image
* SIFT-based image detection
* ORB-based image detection
* Template Matching
* Edge Matching
* Combining results from multiple detection methods
* Removing detected image regions
* Reconstructing the removed background using OpenCV Inpainting
* Replacing detected images with another image
* Generating debug images showing detected regions
* Keeping the original PDF unchanged and creating a new edited PDF

---

## 1. Features

### PDF Page Operations

The application can delete:

* A single page
* Multiple pages
* A page range
* Multiple page ranges combined together

Example:

```text
2 5 8-12 20
```

This removes:

```text
2
5
8
9
10
11
12
20
```

---

### Text Operations

The application can:

* Search for multiple text strings
* Find every occurrence of each text
* Delete text using PDF Redaction
* Replace deleted text with new text

Example:

```text
Confidential
Internal Use Only
Copyright 2025
```

Each occurrence is searched throughout the entire PDF.

---

### Image Operations

The application can detect an image even when the reference image is only:

* A screenshot
* A crop
* A small part of the original image
* A section of a larger image

The reference image does **not** have to be an independent PDF Image Object.

The application uses several computer-vision methods:

```text
SIFT
ORB
Template Matching
Edge Matching
```

Their results are combined and verified before a detection is accepted.

---

## 2. Requirements

### Python

Python 3.10 or newer is recommended.

Check your Python version:

```bash
python3 --version
```

or:

```bash
python --version
```

Example:

```text
Python 3.11.9
```

---

## 3. Supported Operating Systems

The application can run on systems that support Python and the required packages.

Supported platforms include:

* Linux
* macOS
* Windows

---

# 4. Installation

## Linux / macOS

Clone or copy the project to your computer and enter the project directory:

```bash
cd pdf-editor
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
pip install --upgrade pip
```

Install the dependencies:

```bash
pip install pymupdf opencv-python numpy
```

---

## Windows

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install pymupdf opencv-python numpy
```

---

# 5. Dependencies

The project requires the following Python packages:

```text
PyMuPDF
OpenCV
NumPy
```

Install them with:

```bash
pip install pymupdf opencv-python numpy
```

### PyMuPDF

Used for:

* Opening PDF files
* Reading PDF pages
* Searching text
* Applying redactions
* Deleting pages
* Inserting images
* Saving modified PDFs

### OpenCV

Used for:

* SIFT
* ORB
* Template Matching
* Edge detection
* Image processing
* Image inpainting

### NumPy

Used for:

* Image arrays
* Pixel processing
* OpenCV data conversion

---

# 6. Verify Installation

Check OpenCV:

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

Check NumPy:

```bash
python3 -c "import numpy; print(numpy.__version__)"
```

Check PyMuPDF:

```bash
python3 -c "import fitz; print(fitz.__doc__)"
```

If these commands run without errors, the dependencies are installed correctly.

---

# 7. Project Structure

A recommended project structure is:

```text
pdf-editor/
│
├── pdf_editor.py
├── README.md
│
├── input/
│   └── document.pdf
│
├── references/
│   ├── reference.png
│   └── replacement.png
│
└── .venv/

PDF
 ├── Delete Pages
 ├── Delete / Replace Text
 │    └── PDF Redaction
 │
 └── Delete / Replace Image / Logo
      ├── SIFT
      ├── ORB
      ├── Multi-scale Template Matching
      ├── Edge Matching
      ├── RANSAC + Homography
      ├── Candidate Verification
      ├── IoU / Overlap Merge
      ├── PDF Image Object Check
      ├── Vector Drawing Check
      ├── TELEA / NS Inpainting
      ├── Seamless Clone
      └── Optional Replacement Image
```

The Python filename can be different.

This README assumes the main file is:

```text
pdf_editor.py
```

---

# 8. Run the Application

Activate the virtual environment first.

Linux/macOS:

```bash
source .venv/bin/activate
```

Then run:

```bash
python3 pdf_editor.py
```

Windows:

```powershell
.venv\Scripts\activate
python pdf_editor.py
```

The application is interactive and will ask questions in the terminal.

---

# 9. Main Workflow

When the application starts, it will ask for the PDF:

```text
======================================================================
AUTOMATIC PDF / IMAGE / TEXT EDITOR
======================================================================

Main PDF path:
```

Enter the path to your PDF.

Example:

```text
/home/user/Documents/document.pdf
```

or:

```text
~/Documents/document.pdf
```

Windows example:

```text
C:\Users\User\Documents\document.pdf
```

---

# 10. Output File

The original PDF is not overwritten.

If the input file is:

```text
document.pdf
```

the output will be:

```text
document_edited.pdf
```

For example:

```text
document.pdf
document_edited.pdf
```

This allows you to keep the original file as a backup.

---

# 11. Delete Pages

The application asks:

```text
Are there pages to delete? [y/n]:
```

Enter:

```text
y
```

Then:

```text
Page numbers/ranges (example: 2 5 8-12):
```

---

## Delete One Page

Input:

```text
5
```

Result:

```text
Page 5
```

will be deleted.

---

## Delete Multiple Pages

Input:

```text
2 5 8
```

Deletes:

```text
Page 2
Page 5
Page 8
```

---

## Delete a Page Range

Input:

```text
8-12
```

Deletes:

```text
8
9
10
11
12
```

---

## Combine Pages and Ranges

Input:

```text
2 5 8-12 20
```

Deletes:

```text
2
5
8
9
10
11
12
20
```

---

# 12. Page Numbering

Page numbers entered by the user are **1-based**.

For example:

```text
1 = first page
2 = second page
3 = third page
```

The application internally converts them to zero-based indexes.

---

# 13. Delete Text

To delete text:

```text
Is there text to delete? [y/n]:
```

Enter:

```text
y
```

The application will show:

```text
============================================================
TEXTS TO DELETE
============================================================

Text to delete:
```

Enter the text you want to remove.

---

# 14. Delete Multiple Texts

You can enter multiple strings.

Example:

```text
Text to delete: Confidential
Text to delete: Internal Use Only
Text to delete: Copyright 2025
Text to delete:
```

Pressing Enter on an empty line finishes text input.

The application will search for every specified string.

---

# 15. Find All Text Occurrences

If the same text appears multiple times, all occurrences are searched.

For example:

```text
Confidential
```

might appear:

```text
Page 1: 2 occurrences
Page 4: 3 occurrences
Page 8: 5 occurrences
```

The application processes all detected occurrences.

---

# 16. Text Redaction

Text is removed using PDF Redaction rather than simply drawing a white rectangle over the text.

The application uses:

```python
page.add_redact_annot(...)
page.apply_redactions()
```

This is important because the goal is to remove the text from the PDF content rather than merely visually hide it.

---

# 17. Replace Text

After entering the texts, the application asks:

```text
Is there text to replace? [y/n]:
```

Enter:

```text
y
```

For each text:

```text
Original: "Confidential"
Replacement:
```

Example:

```text
Original: "Confidential"
Replacement: Public
```

The original text is redacted first, and the replacement text is then inserted into the original text rectangle.

---

# 18. Delete Text Without Replacement

If you only want to remove text:

```text
Is there text to replace? [y/n]:
```

Enter:

```text
n
```

The original text will be deleted without inserting replacement text.

---

# 19. Image Detection

To enable image processing:

```text
Is there an image / image area to delete? [y/n]:
```

Enter:

```text
y
```

The application will ask:

```text
Reference image path:
```

Enter the path to the reference image.

Example:

```text
references/logo.png
```

---

# 20. What Is a Reference Image?

The reference image is the image or image region that the application should find inside the PDF.

It does **not** have to be the original PDF image object.

For example, if the PDF contains:

```text
+--------------------------------+
|                                |
|            LOGO                |
|                                |
+--------------------------------+
```

you can provide:

```text
logo.png
```

as a screenshot or crop.

---

# 21. Reference Image Can Be a Crop

The reference image can be only part of the target.

For example:

```text
PDF:

+--------------------------------------+
|                                      |
|       +----------------------+       |
|       |                      |       |
|       |       TARGET         |       |
|       |                      |       |
|       +----------------------+       |
|                                      |
+--------------------------------------+
```

You may provide only a section of the target as the reference.

The application attempts to locate that region in the rendered PDF page.

---

# 22. Image Detection Methods

The application uses four detection methods.

```text
1. SIFT
2. ORB
3. Template Matching
4. Edge Matching
```

The results are combined to improve detection reliability.

---

# 23. SIFT

SIFT is used for feature-based matching.

It can be useful when the target:

* Has recognizable visual features
* Has been resized
* Is a crop of a larger image
* Has moderate transformations

Configuration:

```python
SIFT_RATIO = 0.76
SIFT_MIN_MATCHES = 6
SIFT_MIN_INLIERS = 5
```

---

# 24. ORB

ORB is another feature-based matching algorithm.

Configuration:

```python
ORB_RATIO = 0.80
ORB_MIN_MATCHES = 8
ORB_MIN_INLIERS = 5
```

ORB is generally faster than SIFT.

---

# 25. Template Matching

Template Matching compares the reference image against different areas of the rendered PDF page.

The application tests multiple scales.

Current configuration:

```python
TEMPLATE_SCALES = np.linspace(0.30, 2.50, 23)
```

This means the reference is tested at approximately:

```text
30%
...
250%
```

of its original size.

---

# 26. Edge Matching

Edge Matching extracts image edges using Canny Edge Detection and compares them against the PDF page.

This can be useful when structural shapes are more reliable than colors.

Configuration:

```python
EDGE_MIN_SCORE = 0.58
EDGE_STRONG_SCORE = 0.72
```

---

# 27. Multiple Detection Methods

The application does not rely on a single algorithm.

For every page, it can run:

```text
SIFT
    ↓
ORB
    ↓
Template Matching
    ↓
Edge Matching
    ↓
Merge Candidates
    ↓
Verify Candidates
    ↓
Accept Valid Matches
```

This reduces dependence on any one detection technique.

---

# 28. Multiple Occurrences

The same reference image can occur multiple times in a PDF.

For example:

```text
Page 1  → 1 match
Page 3  → 2 matches
Page 8  → 1 match
Page 15 → 3 matches
```

The application attempts to detect each occurrence.

---

# 29. Detection Candidate Merging

Different detection methods may find the same target.

For example:

```text
SIFT       → Bounding Box A
Template   → Bounding Box A
ORB        → Bounding Box A
```

These should not become three separate deletions.

The application compares bounding boxes using:

```text
IoU
Overlap
```

and merges overlapping candidates.

Configuration:

```python
IOU_MERGE_THRESHOLD = 0.25
OVERLAP_SMALLER_THRESHOLD = 0.55
```

---

# 30. Candidate Verification

Not every detected candidate is automatically accepted.

The application uses several acceptance conditions.

For example, SIFT requires a minimum number of inliers:

```python
SIFT_MIN_INLIERS = 5
```

ORB requires:

```python
ORB_MIN_INLIERS = 5
```

Strong Template Matching candidates can be accepted using:

```python
TEMPLATE_STRONG_SCORE = 0.80
```

Strong Edge Matching candidates can be accepted using:

```python
EDGE_STRONG_SCORE = 0.72
```

---

# 31. Image Object Removal

When a target region is detected, the application first attempts to determine whether it corresponds to an actual PDF Image Object.

It uses:

```python
page.get_image_info(xrefs=True)
```

If an underlying image object overlaps sufficiently with the detected region, the application attempts to delete it.

The current overlap requirement is:

```text
60%
```

---

# 32. Image Areas That Are Not Independent PDF Objects

A target image does not always exist as an independent Image Object.

For example, the target could be:

* Part of a screenshot
* Part of a larger raster image
* Embedded inside another image
* Rasterized together with other content

In these cases, simply calling:

```python
page.delete_image()
```

is not sufficient.

The application therefore uses another approach.

---

# 33. Background Reconstruction

When an image region is detected, the application creates a mask over that area.

The processing pipeline is:

```text
Detected Image
       ↓
Create Mask
       ↓
Expand Mask
       ↓
OpenCV Inpainting
       ↓
Reconstructed Background
       ↓
Insert Reconstructed Region
```

The application uses OpenCV Telea Inpainting:

```python
cv2.inpaint(
    image,
    mask,
    INPAINT_RADIUS,
    cv2.INPAINT_TELEA
)
```

---

# 34. Inpainting Radius

The current configuration is:

```python
INPAINT_RADIUS = 5
```

Increasing the radius can sometimes improve reconstruction for larger removed areas, but it can also increase processing time and produce less accurate results.

---

# 35. Removal Padding

The application slightly expands the area around the detected target before reconstructing the background.

Current configuration:

```python
REMOVE_PADDING_PX = 3
```

This can help make the boundary around the removed region less noticeable.

---

# 36. Replace an Image

After detecting the original image, the application can optionally insert another image.

It asks:

```text
Is there a replacement image? [y/n]:
```

Enter:

```text
y
```

Then:

```text
Replacement image path:
```

Example:

```text
references/new-logo.png
```

The processing pipeline becomes:

```text
Detect Original
       ↓
Remove Original
       ↓
Reconstruct Background
       ↓
Insert Replacement
```

---

# 37. Image Replacement Position

The replacement image is inserted into the same PDF rectangle as the detected original target.

This means the detected bounding box determines the replacement position and size.

---

# 38. Remove Image Without Replacement

If you do not want a replacement image:

```text
Is there a replacement image? [y/n]:
```

Enter:

```text
n
```

The target region is removed and the background is reconstructed.

---

# 39. PDF Rendering DPI

For image detection, each PDF page is rendered to a raster image.

Current configuration:

```python
RENDER_DPI = 220
```

Higher DPI generally provides:

* More pixels
* More visual detail
* Potentially better detection

But it also requires:

* More RAM
* More CPU
* More processing time

---

# 40. Changing Render DPI

For faster processing:

```python
RENDER_DPI = 150
```

For higher resolution:

```python
RENDER_DPI = 300
```

For very large PDFs, lower DPI may be necessary to reduce memory usage.

---

# 41. Debug Images

Debug images are enabled by default:

```python
CREATE_DEBUG_IMAGES = True
```

When matches are detected, the application generates images showing the detected regions.

Example:

```text
document_debug/
│
├── page_0001_matches.png
├── page_0003_matches.png
└── page_0012_matches.png
```

---

# 42. What Do Debug Images Show?

Detected regions are displayed using bounding boxes.

The detection methods are also displayed.

For example:

```text
1: SIFT,TEMPLATE
```

or:

```text
2: ORB,SIFT,TEMPLATE
```

These images are useful for verifying whether the computer-vision algorithms detected the correct region.

---

# 43. Disable Debug Images

If debug images are not required:

```python
CREATE_DEBUG_IMAGES = False
```

---

# 44. Performance

Image detection can be computationally expensive.

For every PDF page, several algorithms may run:

```text
SIFT
ORB
Template Matching
Edge Matching
```

Template and Edge Matching are also performed at multiple scales.

Therefore, large PDFs may take significant time to process.

---

# 45. False Positives

Computer-vision detection cannot guarantee perfect detection in every situation.

False positives can occur when:

* The reference image is very small
* The reference has very few visual features
* The image is heavily blurred
* The image has strong compression artifacts
* The target is extremely simple
* The background is visually similar
* The target is heavily transformed
* The crop is too small

Always inspect the Debug Images and output PDF before using the result for important documents.

---

# 46. Improving Detection

If the application cannot detect the target, consider:

1. Use a larger reference image.
2. Include more visual features in the reference.
3. Increase `RENDER_DPI`.
4. Try a less aggressive crop.
5. Adjust Template Matching thresholds.
6. Adjust Edge Matching thresholds.

---

# 47. Lower Detection Thresholds

If the application is too strict, thresholds can be lowered.

For example:

```python
TEMPLATE_MIN_SCORE = 0.62
```

and:

```python
EDGE_MIN_SCORE = 0.52
```

However, lowering thresholds increases the probability of false positives.

---

# 48. Increase Detection Thresholds

If too many incorrect regions are detected, increase thresholds.

Example:

```python
TEMPLATE_MIN_SCORE = 0.72
```

and:

```python
EDGE_MIN_SCORE = 0.65
```

You can also increase the minimum number of SIFT/ORB inliers.

---

# 49. Detection Limits

The application is designed for robust image-area detection, but image detection is not guaranteed to work for every PDF.

Particularly difficult cases include:

* Very small images
* Extremely low-resolution PDFs
* Strong perspective distortion
* Heavy rotation
* Very blurry targets
* Nearly featureless images
* Highly compressed images
* Very small reference crops

---

# 50. Text Detection Limitations

Text processing uses:

```python
page.search_for()
```

Therefore, normal text search works with actual PDF text content.

It will not automatically detect text that is only part of an image.

For example:

```text
PDF Image
    ↓
"CONFIDENTIAL"
```

If `CONFIDENTIAL` is inside the image, normal PDF text search will not find it.

OCR would be required for that use case.

This version does not include OCR.

---

# 51. Complete Example — Delete Pages

Run:

```bash
python3 pdf_editor.py
```

Input:

```text
Main PDF path: book.pdf

Are there pages to delete? [y/n]: y

Page numbers/ranges (example: 2 5 8-12): 2 5 10-12

Is there text to delete? [y/n]: n

Is there an image / image area to delete? [y/n]: n
```

Output:

```text
book_edited.pdf
```

---

# 52. Complete Example — Delete Text

```text
Main PDF path: book.pdf

Are there pages to delete? [y/n]: n

Is there text to delete? [y/n]: y

Text to delete: Confidential
Text to delete: Internal Use Only
Text to delete:
```

Then:

```text
Is there text to replace? [y/n]: n
```

The selected text is removed from the PDF.

---

# 53. Complete Example — Replace Text

Input:

```text
Main PDF path: book.pdf

Are there pages to delete? [y/n]: n

Is there text to delete? [y/n]: y

Text to delete: Company ABC
Text to delete:
```

Then:

```text
Is there text to replace? [y/n]: y

Original: "Company ABC"
Replacement: Company XYZ
```

The application replaces:

```text
Company ABC
```

with:

```text
Company XYZ
```

---

# 54. Complete Example — Remove an Image

Suppose you have:

```text
book.pdf
logo.png
```

Run:

```bash
python3 pdf_editor.py
```

Then:

```text
Is there an image / image area to delete? [y/n]: y

Reference image path: logo.png

Is there a replacement image? [y/n]: n
```

The application scans the PDF pages and removes accepted matches.

---

# 55. Complete Example — Replace an Image

Suppose you have:

```text
book.pdf
old-logo.png
new-logo.png
```

Enter:

```text
Reference image path: old-logo.png

Is there a replacement image? [y/n]: y

Replacement image path: new-logo.png
```

The application will:

```text
Find old logo
     ↓
Remove old logo
     ↓
Reconstruct background
     ↓
Insert new logo
```

---

# 56. Complete Example — Everything

A complete workflow can look like:

```text
======================================================================
AUTOMATIC PDF / IMAGE / TEXT EDITOR
======================================================================

Main PDF path: /home/user/Documents/book.pdf

Opening PDF...
Pages: 120

Are there pages to delete? [y/n]: y

Page numbers/ranges (example: 2 5 8-12): 2 10-12

Pages selected: [2, 10, 11, 12]

Is there text to delete? [y/n]: y

Text to delete: Confidential
Text to delete: Internal Use Only
Text to delete:

Is there text to replace? [y/n]: y

Original: "Confidential"
Replacement: Public

Original: "Internal Use Only"
Replacement: Public Document

Is there an image / image area to delete? [y/n]: y

Reference image path: /home/user/logo.png

Is there a replacement image? [y/n]: y

Replacement image path: /home/user/new-logo.png
```

The application then processes the selected operations and saves:

```text
book_edited.pdf
```

---

# 57. Final Report

At the end, the application prints a report similar to:

```text
======================================================================
FINAL REPORT
======================================================================

Original pages: 120
Final pages: 116
Deleted pages: 4

TEXT
  Found: 3
  Deleted: 3
  Replaced: 3

IMAGE
  Pages scanned: 116
  Pages with matches: 8
  Matches: 8
  Removed/reconstructed: 8
  Replaced: 8
  PDF image objects deleted: 5

Output file:

/home/user/Documents/book_edited.pdf

Debug images:

/home/user/Documents/book_debug

======================================================================
DONE
======================================================================
```

---

# 58. Generated Files

If the input is:

```text
document.pdf
```

the main output is:

```text
document_edited.pdf
```

If Debug Images are enabled:

```text
document_debug/
```

may also be created.

Example:

```text
document.pdf
document_edited.pdf

document_debug/
├── page_0001_matches.png
├── page_0004_matches.png
└── page_0010_matches.png
```

---

# 59. Recommended Workflow

For important PDF files, use the following workflow:

```text
Backup Original PDF
        ↓
Prepare Reference Image
        ↓
Run Application
        ↓
Process a Copy
        ↓
Check Debug Images
        ↓
Open Edited PDF
        ↓
Verify Text Changes
        ↓
Verify Image Changes
        ↓
Use Final PDF
```

---

# 60. Common Errors

## File Not Found

If you see:

```text
File not found
```

check the file path.

Example:

```text
/home/user/Documents/document.pdf
```

---

## Cannot Open PDF

Possible causes:

* Corrupted PDF
* Invalid PDF file
* Insufficient permissions
* File is locked
* Unsupported/corrupted PDF structure

---

## Cannot Read Reference Image

If you see:

```text
Cannot read reference image
```

check:

* File path
* Image format
* File permissions
* Image integrity

Common formats include:

```text
PNG
JPG
JPEG
WEBP
```

---

## Save Error

If the application reports:

```text
SAVE ERROR
```

check:

1. Output directory permissions
2. Available disk space
3. Whether the output file is open in another application
4. Whether the source PDF is valid

---

# 61. Performance Tuning

For large PDFs, the most important settings are:

```python
RENDER_DPI
TEMPLATE_SCALES
MAX_DETECTIONS_PER_METHOD
MAX_FINAL_DETECTIONS
```

Higher detection quality generally requires more processing.

For faster processing, consider:

```python
RENDER_DPI = 150
```

For higher-resolution detection:

```python
RENDER_DPI = 300
```

---

# 62. Main Configuration

The main configuration is located near the beginning of the Python file.

Important settings include:

```python
RENDER_DPI = 220
```

```python
SIFT_RATIO = 0.76
SIFT_MIN_MATCHES = 6
SIFT_MIN_INLIERS = 5
```

```python
ORB_RATIO = 0.80
ORB_MIN_MATCHES = 8
ORB_MIN_INLIERS = 5
```

```python
TEMPLATE_MIN_SCORE = 0.68
TEMPLATE_STRONG_SCORE = 0.80
```

```python
EDGE_MIN_SCORE = 0.58
EDGE_STRONG_SCORE = 0.72
```

```python
REMOVE_PADDING_PX = 3
```

```python
INPAINT_RADIUS = 5
```

```python
CREATE_DEBUG_IMAGES = True
```

---

# 63. Disable Image Detection

If you only need page and text operations, answer:

```text
Is there an image / image area to delete? [y/n]: n
```

The image-processing algorithms will not run.

---

# 64. Local Processing

The application is designed to process files locally.

The PDF and reference images are processed by the Python application using:

```text
PyMuPDF
OpenCV
NumPy
```

No online PDF editing service is required.

---

# 65. Important Notes

### Original PDF

The original PDF is not intentionally overwritten.

The edited version is saved with:

```text
_edited.pdf
```

### Reference Images

Reference images can be screenshots or crops.

They do not need to correspond to independent PDF Image Objects.

### Image Detection

Multiple computer-vision techniques are combined:

```text
SIFT
ORB
Template Matching
Edge Matching
```

### Image Removal

If an underlying PDF Image Object cannot be removed directly, the application reconstructs the affected area using OpenCV Inpainting.

### Debugging

Debug images should be reviewed when image detection is important.

### Accuracy

Computer vision is probabilistic. Detection is not guaranteed to be 100% accurate for every document.

---

# 66. Feature Summary

| Feature                          | Supported |
| -------------------------------- | --------- |
| Delete single page               | Yes       |
| Delete multiple pages            | Yes       |
| Delete page ranges               | Yes       |
| Delete multiple text strings     | Yes       |
| Find all text occurrences        | Yes       |
| PDF text redaction               | Yes       |
| Replace text                     | Yes       |
| Screenshot as image reference    | Yes       |
| Crop as image reference          | Yes       |
| SIFT                             | Yes       |
| ORB                              | Yes       |
| Template Matching                | Yes       |
| Edge Matching                    | Yes       |
| Multiple image occurrences       | Yes       |
| Multiple pages                   | Yes       |
| PDF Image Object deletion        | Yes       |
| OpenCV Inpainting                | Yes       |
| Remove image without replacement | Yes       |
| Replace image                    | Yes       |
| Debug images                     | Yes       |
| Local processing                 | Yes       |
| OCR                              | No        |
| Guaranteed 100% image detection  | No        |

---

# 67. Quick Start

For a quick installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pymupdf opencv-python numpy
python3 pdf_editor.py
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pymupdf opencv-python numpy
python pdf_editor.py
```

Then follow the interactive prompts.

---

# 68. Deactivate the Virtual Environment

When you are finished:

```bash
deactivate
```

---

# 69. Run the Project Again

Linux/macOS:

```bash
cd pdf-editor
source .venv/bin/activate
python3 pdf_editor.py
```

Windows:

```powershell
cd pdf-editor
.venv\Scripts\activate
python pdf_editor.py
```

---

# 70. Technology Stack

This project uses:

```text
Python 3
PyMuPDF
OpenCV
NumPy
```

Main areas of functionality:

```text
PDF Processing
Computer Vision
Image Matching
Image Inpainting
Text Redaction
PDF Image Manipulation
```

---

# 71. License

Add the project's license here.

For example:

```text
MIT License
```

Replace this section with the actual license used by the project.
