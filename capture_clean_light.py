import os
from PIL import Image
from playwright.sync_api import sync_playwright

def run():
    out_dir = os.path.abspath("presentation_assets")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto("http://127.0.0.1:8000/")
        page.wait_for_timeout(2500)
        
        # 1. Ensure Light Theme
        page.evaluate("setTheme('light')")
        page.wait_for_timeout(800)
        
        # Capture Varsity Light Layout (Slide 2)
        page.screenshot(path=os.path.join(out_dir, "varsity_light_layout.png"))
        print("Captured varsity_light_layout.png")
        
        # 2. Expand 'Why this post?' on the first card
        page.evaluate("""() => {
            const buttons = Array.from(document.querySelectorAll("button"));
            const btn = buttons.find(b => b.textContent.includes("Why this post?"));
            if (btn) btn.click();
        }""")
        page.wait_for_timeout(1000)
        
        # Find the expanded post card and capture it
        first_post = page.query_selector("article, .timeline-card, [id*='post-']")
        if first_post:
            first_post.screenshot(path=os.path.join(out_dir, "why_this_post_light.png"))
            print("Captured why_this_post_light.png")
            
        # 3. Open Persona Modal
        page.evaluate("openProfileModal()")
        page.wait_for_timeout(800)
        modal_content = page.query_selector("#profile-modal .modal-box, #profile-modal > div:last-child")
        if modal_content:
            modal_content.screenshot(path=os.path.join(out_dir, "persona_modal_light.png"))
            print("Captured persona_modal_light.png")
        page.evaluate("closeProfileModal()")
        page.wait_for_timeout(500)
        
        # 4. Open Settings Drawer - Sliders Tab
        page.evaluate("openSettingsDrawer(); switchDrawerTab('sliders');")
        page.wait_for_timeout(800)
        drawer_panel = page.query_selector("#settings-drawer-panel")
        if drawer_panel:
            drawer_panel.screenshot(path=os.path.join(out_dir, "crop_drawer_sliders_light.png"))
            print("Captured crop_drawer_sliders_light.png")
            
        # 5. Settings Drawer - Sandbox Tab
        page.evaluate("switchDrawerTab('sandbox')")
        page.wait_for_timeout(800)
        if drawer_panel:
            drawer_panel.screenshot(path=os.path.join(out_dir, "crop_sandbox_controls_light.png"))
            print("Captured crop_sandbox_controls_light.png")
        page.evaluate("closeSettingsDrawer()")
        page.wait_for_timeout(500)
        
        # 6. Open Benchmark Modal
        page.evaluate("openBenchmarkModal()")
        page.wait_for_timeout(1000)
        bench_box = page.query_selector("#benchmark-modal .modal-box, #benchmark-modal > div:last-child")
        if bench_box:
            bench_box.screenshot(path=os.path.join(out_dir, "benchmark_table_light.png"))
            print("Captured benchmark_table_light.png")
        page.evaluate("closeBenchmarkModal()")
        page.wait_for_timeout(500)

        browser.close()

    # Now make targeted crops from why_this_post_light.png
    wtp_path = os.path.join(out_dir, "why_this_post_light.png")
    if os.path.exists(wtp_path):
        wtp_im = Image.open(wtp_path)
        w, h = wtp_im.size
        print(f"why_this_post_light dimensions: {w}x{h}")
        
        # Action bar crop (Like, Reply, Bookmark, Skip)
        # Usually located around 30% to 50% height
        # Let's crop the top section containing the post actions
        action_crop = wtp_im.crop((0, int(h * 0.22), int(w * 0.55), int(h * 0.44)))
        action_crop.save(os.path.join(out_dir, "crop_post_actions_light.png"))
        print("Cropped crop_post_actions_light.png")
        
        # Probability boxes (P(Like), P(Reply), P(Skip))
        prob_crop = wtp_im.crop((0, int(h * 0.45), w, int(h * 0.68)))
        prob_crop.save(os.path.join(out_dir, "crop_heavy_ranker_probs_light.png"))
        print("Cropped crop_heavy_ranker_probs_light.png")
        
        # Similarity bars
        bars_crop = wtp_im.crop((0, int(h * 0.67), w, h))
        bars_crop.save(os.path.join(out_dir, "crop_similarity_bars_light.png"))
        print("Cropped crop_similarity_bars_light.png")

if __name__ == "__main__":
    run()
