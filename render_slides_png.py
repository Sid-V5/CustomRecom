import os
import win32com.client

def export_slides(pptx_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    try:
        pres = ppt.Presentations.Open(os.path.abspath(pptx_path), WithWindow=False)
        print(f"Opened presentation with {pres.Slides.Count} slides.")
        for i, slide in enumerate(pres.Slides, 1):
            out_file = os.path.join(out_dir, f"slide_{i:02d}.png")
            slide.Export(os.path.abspath(out_file), "PNG", 1920, 1080)
            print(f"Exported slide {i} -> {out_file}")
        pres.Close()
    finally:
        ppt.Quit()

if __name__ == "__main__":
    export_slides("Varsity_RecSys_Presentation.pptx", "rendered_slides")
