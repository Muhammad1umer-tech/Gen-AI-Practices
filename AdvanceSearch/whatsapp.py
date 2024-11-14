import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# Path to a custom user profile directory for Chrome
chrome_options = webdriver.ChromeOptions()
user_data_dir = os.path.expanduser("~") + "/.whatsapp-session"
chrome_options.add_argument(f"--user-data-dir={user_data_dir}")  # Use persistent session data

# Initialize Chrome WebDriver with the custom user profile
driver = webdriver.Chrome(options=chrome_options)

# Open WhatsApp Web
driver.get('https://web.whatsapp.com')

# Wait for the user to scan the QR code
time.sleep(10)

# Define the recipient's phone number or name
contact_name = "Discipline"  # Name as it appears on WhatsApp
document_path = "/home/arsal/Desktop/learning/Gen-AI/langGraph/AdvanceSearch/Document.docx"  # Path to the document on your local system

try:
    # Search for the contact
    search_box = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]'))
    )
    search_box.click()
    search_box.send_keys(contact_name)
    search_box.send_keys(Keys.RETURN)
    
    # Wait for chat to open and then click the attachment icon
    plus_icon = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, '//span[@data-icon="plus"]'))
    )
    plus_icon.click()
    
    # Upload the document
    document_icon = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, '//input[@accept="*"]'))
    )
    document_icon.send_keys(document_path)
    
    # Wait for the document to upload and then click the send button
    send_button = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, '//span[@data-icon="send"]'))
    )
    send_button.click()

    print("Document sent successfully!")

except Exception as e:
    print("An error occurred:", e)
