"""
=============================================================
  src/auto_apply.py — Deep Auto-Applier Engine
=============================================================
  A robust Playwright-style automation script using
  undetected-chromedriver for LinkedIn "Easy Apply" flows.
"""

import time
import random
import os
import threading
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from src.logger import get_logger

log = get_logger("auto_applier")

def _random_sleep(min_s=1.5, max_s=3.0):
    time.sleep(random.uniform(min_s, max_s))


def init_stealth_driver():
    """Boot a highly stealthy Chrome profile."""
    options = uc.ChromeOptions()
    options.add_argument('--headless') # Run in background as requested
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--window-size=1280,1024")

    # Use a temp profile to avoid corrupting main browser
    driver = uc.Chrome(options=options, use_subprocess=True)
    return driver


def process_linkedin_easy_apply(driver, job_url: str, cv_path: str, profile: dict) -> bool:
    """
    The 'Holy Grail' Feature: Automates the entire LinkedIn Easy Apply flow.
    Note: Requires the user to already be logged into LinkedIn in the browser profile,
    or you must prompt for login. For this MVP, we assume the user logs in manually once.
    """
    log.info("Attempting Easy Apply for: %s", job_url)

    try:
        driver.get(job_url)
        _random_sleep(3, 5)

        # 1. Find the "Easy Apply" button
        # LinkedIn frequently changes class names, so we look for text or specific aria-labels.
        easy_apply_btn = None
        try:
            # Look for button containing exactly "Easy Apply"
            buttons = driver.find_elements(By.XPATH, "//button[contains(@class, 'jobs-apply-button')]")
            for btn in buttons:
                if "Easy Apply" in btn.text:
                    easy_apply_btn = btn
                    break
        except:
            pass

        if not easy_apply_btn:
            log.warning("No 'Easy Apply' button found (might be external 'Apply' or already applied). Skipping.")
            return False

        log.info("Found Easy Apply button. Initiating sequence...")
        easy_apply_btn.click()
        _random_sleep(2, 3)

        # 2. Cycle through the Modal Steps
        max_steps = 10
        step = 0
        while step < max_steps:
            step += 1

            # --- Check for specific fields to fill based on profile ---

            # Resume Upload Step
            try:
                upload_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file'][name='file']")
                if upload_inputs and os.path.exists(cv_path):
                    # LinkedIn usually hides the input, we send keys directly to it
                    upload_inputs[0].send_keys(os.path.abspath(cv_path))
                    log.info("Resume uploaded successfully.")
                    _random_sleep(2, 3)
            except Exception as e:
                pass

            # Phone Number
            try:
                phone_inputs = driver.find_elements(By.XPATH, "//input[contains(@id, 'phoneNumber') or contains(@class, 'fb-single-line-text__input')]")
                if phone_inputs:
                    phone = profile.get("personal_details", {}).get("phone", "9876543210")
                    if not phone_inputs[0].get_attribute('value'):
                        phone_inputs[0].clear()
                        phone_inputs[0].send_keys(phone)
                        _random_sleep(1, 2)
            except:
                pass

            # Generic Text/Number Questions (Years of exp, etc.)
            try:
                text_inputs = driver.find_elements(By.XPATH, "//input[@type='text' or @type='number']")
                for inp in text_inputs:
                    if not inp.get_attribute('value'):
                        # Very generic fallback: answer '3' for years of exp questions
                        inp.send_keys("3")
                        _random_sleep(0.5, 1)
            except:
                pass

            # Dropdowns (Yes/No questions)
            try:
                selects = driver.find_elements(By.TAG_NAME, "select")
                for sel in selects:
                    if not sel.get_attribute('value'):
                        # Default to "Yes" or the first valid option
                        options = sel.find_elements(By.TAG_NAME, "option")
                        for opt in options:
                            if "Yes" in opt.text or "English" in opt.text:
                                opt.click()
                                break
                        _random_sleep(0.5, 1)
            except:
                pass

            # --- Navigation Buttons (Next / Review / Submit) ---
            try:
                footer = driver.find_element(By.CLASS_NAME, "ph5") # LinkedIn modal footer
                buttons = footer.find_elements(By.TAG_NAME, "button")

                action_taken = False
                for btn in buttons:
                    text = btn.text.strip().lower()
                    if text == "submit application":
                        btn.click()
                        log.info("✅ Successfully Submitted Application!")
                        _random_sleep(2, 3)
                        return True
                    elif text == "review":
                        btn.click()
                        log.info("Clicked Review...")
                        action_taken = True
                        _random_sleep(2, 3)
                        break
                    elif text == "next":
                        btn.click()
                        log.info("Clicked Next...")
                        action_taken = True
                        _random_sleep(2, 3)
                        break

                if not action_taken:
                    # Stuck or no recognized button
                    log.warning("Could not find Next/Review/Submit. Modal might require manual intervention.")
                    break
            except Exception as e:
                log.warning("Modal footer not found or interaction failed: %s", str(e))
                break

        log.warning("Failed to complete application flow after %d steps.", step)
        return False

    except Exception as e:
        log.error("Automation Error on %s: %s", job_url, str(e))
        return False


def run_background_applier(jobs: list, cv_path: str, profile: dict, max_applications: int = 3):
    """
    Background worker that iterates through the top safe jobs and attempts to apply.
    """
    safe_jobs = [j for j in jobs if j.get("probability_fake", 1.0) < 0.3 and "linkedin.com" in j.get("url", "")]

    if not safe_jobs:
        log.info("No safe LinkedIn jobs available for Auto-Apply.")
        return

    log.info("Auto-Applier started in background. Targeting %d safe jobs...", min(len(safe_jobs), max_applications))

    try:
        driver = init_stealth_driver()

        # In a real SaaS, we would pause here and let the user login to LinkedIn once.
        # For this PoC, we will give the browser 15 seconds so the user can see it open
        # and manually login if they aren't already.
        log.info("Giving browser 15 seconds to establish session/login...")
        _random_sleep(10, 15)

        applied_count = 0
        for job in safe_jobs:
            if applied_count >= max_applications:
                break

            url = job.get("url", "")
            success = process_linkedin_easy_apply(driver, url, cv_path, profile)
            if success:
                applied_count += 1

        log.info("Auto-Applier finished. Total successful applications: %d", applied_count)

    except Exception as e:
        log.error("Fatal error in Auto-Applier background thread: %s", str(e))
    finally:
        try:
            driver.quit()
        except:
            pass

def trigger_auto_apply_async(jobs: list, cv_path: str, profile: dict):
    """Spawns the auto-applier in a non-blocking thread so the API can return immediately."""
    thread = threading.Thread(
        target=run_background_applier,
        args=(jobs, cv_path, profile, 3) # Max 3 applications per run to avoid bans
    )
    thread.daemon = True
    thread.start()
    log.info("Triggered Auto-Applier background thread.")
