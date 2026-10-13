from playwright.sync_api import sync_playwright
import time

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto("http://127.0.0.1:8000/")
        page.wait_for_timeout(2500)
        
        # Ensure light mode
        page.evaluate("setTheme('light')")
        page.wait_for_timeout(1000)
        
        # Capture main layout
        page.screenshot(path="presentation_assets/varsity_light_layout.png")
        print("Captured varsity_light_layout.png")
        
        # Open Why This Post on first card
        why_btn = page.query_selector("button:has-text('Why this post?')")
        if why_btn:
            why_btn.click()
            page.wait_for_timeout(800)
            
            # Screenshot of the first post with why this post expanded
            card = page.query_selector(".timeline-card, article")
            if card:
                card.screenshot(path="presentation_assets/why_this_post_light.png")
                print("Captured why_this_post_light.png")
                
        # Open Persona modal
        profile_btn = page.query_selector("button:has-text('Profile & Persona'), #nav-profile, [onclick*='openProfileModal']")
        if profile_btn:
            profile_btn.click()
            page.wait_for_timeout(1000)
            modal = page.query_selector("#profile-modal, .modal-content, [id*='modal']")
            if modal:
                modal.screenshot(path="presentation_assets/persona_modal_light.png")
                print("Captured persona_modal_light.png")
            # Close modal
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)
            
        # Open Sliders drawer
        sliders_btn = page.query_selector("button:has-text('Sliders'), button:has-text('Settings'), [onclick*='openSettingsDrawer']")
        if sliders_btn:
            sliders_btn.click()
            page.wait_for_timeout(1000)
            drawer = page.query_selector("#settings-drawer, #drawer, [id*='settings']")
            if drawer:
                drawer.screenshot(path="presentation_assets/crop_drawer_sliders_light.png")
                print("Captured crop_drawer_sliders_light.png")
                
            # Click attack sandbox tab
            sandbox_tab = page.query_selector("button:has-text('Attack Sandbox'), [onclick*='switchDrawerTab']")
            if sandbox_tab:
                sandbox_tab.click()
                page.wait_for_timeout(800)
                if drawer:
                    drawer.screenshot(path="presentation_assets/crop_sandbox_controls_light.png")
                    print("Captured crop_sandbox_controls_light.png")
                    
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)

        # Open Benchmarks tab
        bench_nav = page.query_selector("a:has-text('Benchmarks'), [onclick*='benchmarks']")
        if bench_nav:
            bench_nav.click()
            page.wait_for_timeout(1200)
            bench_area = page.query_selector("#timeline-container, main, #benchmarks-tab")
            if bench_area:
                bench_area.screenshot(path="presentation_assets/benchmark_table_light.png")
                print("Captured benchmark_table_light.png")

        browser.close()

if __name__ == "__main__":
    capture()
