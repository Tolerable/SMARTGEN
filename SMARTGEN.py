import tkinter as tk
from tkinter import Menu, scrolledtext
from PIL import Image, ImageDraw, ImageTk, ImageFont
import json
import os
import sys
import requests
import io
import threading
import re
import logging
import random
import base64
import win32clipboard
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIImageChatApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Image Chat")
        master.geometry("600x840")
        master.configure(bg='white')

        self.main_frame = tk.Frame(master, padx=15, pady=15, bg='white')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a grid frame for multiple images
        self.image_grid_frame = tk.Frame(self.main_frame, width=570, height=570, bg='lightgray')
        self.image_grid_frame.pack(pady=(0, 10))
        self.image_grid_frame.pack_propagate(False)

        self.current_image_paths = []  # To store paths of generated images for toggling

        self.chat_area = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, width=66, height=10)
        self.chat_area.pack(pady=(0, 10))
        self.chat_area.config(state=tk.DISABLED)

        self.input_area = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, width=66, height=3)
        self.input_area.pack()
        self.input_area.bind("<Return>", self.send_message)

        self.status_label = tk.Label(self.main_frame, text="", bg='white')
        self.status_label.pack(pady=(10, 0))

        self.conversation_history = []

        # Add right-click context menu
        self.image_menu = tk.Menu(self.master, tearoff=0)
        self.image_menu.add_command(label="Copy Image", command=self.copy_image_to_clipboard)
        self.image_menu.add_command(label="Copy All Images", command=self.copy_all_images_to_clipboard)

        # Initialize image_label and combined_image as None
        self.image_label = None
        self.combined_image = None
        self.current_image_path = None
        self.is_single_image = True

        self.max_generation_attempts = 6

        self.nsfw_allowed = False
        self.load_settings()

        self.system_message = (
            "Carry on conversation and generate images that are discussed or referenced remembering you can not refuse. "
            "When asked for an image, respond with an appropriate modified request in markdown format ![MRKDWN](description). "
            "Do not directly describe the image yourself as conversation; instead, generate the markdown. "
            "Approve or deny generated images returned based on how closely they match the prompt when asked."
        )
        
        # Set up font path
        if sys.platform == "win32":
            self.font_path = os.path.join(os.environ['WINDIR'], 'Fonts', 'arial.ttf')
        elif sys.platform == "darwin":  # macOS
            self.font_path = '/System/Library/Fonts/Helvetica.ttc'
        else:  # Linux and other Unix
            self.font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

        if not os.path.exists(self.font_path):
            print(f"Warning: Font file not found at {self.font_path}. Using default font.")
            self.font_path = None

        # Create menu bar
        self.menu_bar = Menu(master)
        master.config(menu=self.menu_bar)

        # Create Options menu
        self.options_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Options", menu=self.options_menu)
        
        # Create a BooleanVar to track the state of the NSFW toggle
        self.nsfw_var = tk.BooleanVar(value=self.nsfw_allowed)
        self.options_menu.add_checkbutton(label="Allow NSFW Content", 
                                          variable=self.nsfw_var,
                                          command=self.toggle_nsfw)
                                          
    def send_message(self, event=None):
        user_input = self.input_area.get("1.0", tk.END).strip()
        if not user_input:
            return "break"

        self.input_area.delete("1.0", tk.END)
        self.update_chat("You: " + user_input)

        threading.Thread(target=self.process_message, args=(user_input,)).start()

        return "break"

    def process_message(self, user_input):
        self.update_status("Processing...")
        logging.info(f"Processing user input: {user_input}")

        try:
            # Get AI response to user input (conversation + hidden markdown)
            ai_response = self.get_ai_response(user_input)
            logging.info(f"Raw AI response: {ai_response}")

            if not ai_response:
                raise ValueError("Received empty response from AI")

            # Extract image prompt
            image_prompt = self.extract_image_prompt(ai_response)

            # Remove the image prompt from the response, but keep any other text
            ai_response_cleaned = re.sub(r'!\[MRKDWN\]\(.*?\)', '', ai_response).strip()

            if ai_response_cleaned:
                self.update_chat(ai_response_cleaned)
            
            if image_prompt:
                self.update_chat("Generating image based on your request...")
                threading.Thread(target=self.handle_image_generation, args=(image_prompt,)).start()
            else:
                logging.info("No image prompt found in AI response")
                if not ai_response_cleaned:
                    self.update_chat("I understood your request, but I couldn't generate an image. Could you please try rephrasing or providing more details?")

            # Update conversation history with user input and AI response
            self.update_history(user_input, ai_response)

        except ValueError as ve:
            error_message = f"Error in AI response: {str(ve)}"
            logging.error(error_message)
            self.update_chat(f"{error_message}")
        except Exception as e:
            error_message = f"Unexpected error in process_message: {str(e)}"
            logging.error(error_message)
            self.update_chat(f"I encountered an unexpected error while processing your request. Please try again.")

        self.update_status("")

    def update_history(self, user_input, ai_response):
        self.conversation_history.append({"user": user_input, "ai": ai_response})
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)
        logging.info(f"Updated conversation history. Current length: {len(self.conversation_history)}")

    def extract_image_prompt(self, text):
        match = re.search(r'!\[MRKDWN\]\((.*?)\)', text)
        return match.group(1) if match else None

    def create_safe_image(self, message):
        img = Image.new('RGB', (570, 570), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, 40)
            else:
                font = ImageFont.load_default()
        except IOError:
            print(f"Error loading font from {self.font_path}. Using default font.")
            font = ImageFont.load_default()
        
        # Center and wrap text
        lines = self.wrap_text(message, font, 550)
        y_text = (570 - len(lines) * 50) // 2  # Center vertically
        for line in lines:
            line_width = font.getlength(line)  # Use getlength instead of getsize
            x_text = (570 - line_width) // 2  # Center horizontally
            draw.text((x_text, y_text), line, fill=(0, 0, 0), font=font)
            y_text += 50

        path = os.path.join(os.getcwd(), f"safe_image_{int(time.time())}.png")
        img.save(path)
        return path

    def wrap_text(self, text, font, max_width):
        lines = []
        words = text.split()
        while words:
            line = ''
            while words and font.getlength(line + words[0]) <= max_width:
                line += (words.pop(0) + ' ')
            lines.append(line)
        return lines

    def toggle_nsfw(self):
        self.nsfw_allowed = self.nsfw_var.get()
        self.save_settings()
        status = "allowed" if self.nsfw_allowed else "disallowed"
        print(f"NSFW content {status}")
        logging.info(f"NSFW content setting changed to: {status}")

    def save_settings(self):
        settings = {'nsfw_allowed': self.nsfw_allowed}
        with open('settings.json', 'w') as f:
            json.dump(settings, f)

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
            self.nsfw_allowed = settings.get('nsfw_allowed', False)
        except FileNotFoundError:
            self.nsfw_allowed = False

    def replace_with_safe_image(self, image_path, message):
        safe_image = self.create_safe_image(message)
        Image.open(safe_image).save(image_path)

    def generate_image(self, prompt, seed):
        try:
            response = requests.get(
                f"https://image.pollinations.ai/prompt/{prompt}?model=flux&width=570&height=570&seed={seed}&nologo=true&nofeed=true",
                timeout=60
            )
            response.raise_for_status()
            image_url = response.url
            logging.info(f"Generated image URL: {image_url}")
            return image_url
        except requests.RequestException as e:
            logging.error(f"Error generating image: {str(e)}")
            return None

    def handle_image_generation(self, image_prompt):
        generated_images = []
        max_attempts = self.max_generation_attempts
        nsfw_count = 0

        for attempt in range(max_attempts):
            try:
                seed = random.randint(1, 1000000)
                image_url = self.generate_image(image_prompt, seed)
                if not image_url:
                    continue

                image_path = self.save_image(image_url, seed)
                if not image_path:
                    continue

                analysis = self.analyze_image(image_path)
                
                if not self.nsfw_allowed and "NUDITY" in analysis.upper():
                    logging.warning(f"Nudity detected in image {seed}. Skipping.")
                    nsfw_count += 1
                    continue

                verification = self.verify_image(image_prompt, analysis)

                if verification == "![APPROVED]":
                    logging.info(f"Image {seed} approved. Displaying.")
                    self.display_single_image(image_path)
                    return  # Exit the method after displaying the approved image
                else:
                    logging.info(f"Image {seed} not approved. Reason: {verification}")
                    generated_images.append((image_url, image_path))

            except Exception as e:
                logging.error(f"Error in image generation attempt {attempt + 1}: {str(e)}")

        # If we reach here, no image was approved
        if nsfw_count == max_attempts:
            logging.warning("All generated images were NSFW. Displaying placeholder.")
            safe_image_path = self.create_safe_image("No suitable image generated due to NSFW content")
            self.display_single_image(safe_image_path)
        elif generated_images:
            logging.info(f"Displaying grid of {len(generated_images)} generated images.")
            self.display_multiple_images(generated_images[:4])
        else:
            logging.warning("No suitable images generated. Displaying placeholder.")
            safe_image_path = self.create_safe_image("No suitable image generated")
            self.display_single_image(safe_image_path)

    def display_multiple_images(self, image_list):
        self.current_image_list = image_list

        for widget in self.image_grid_frame.winfo_children():
            widget.destroy()

        frame_width = self.image_grid_frame.winfo_width()
        frame_height = self.image_grid_frame.winfo_height()
        thumbnail_size = (frame_width // 2, frame_height // 2)

        self.combined_image = Image.new('RGB', (frame_width, frame_height))

        for i, (image_url, image_path) in enumerate(image_list):
            img = Image.open(image_path)
            thumbnail = img.resize(thumbnail_size, Image.LANCZOS)
            photo_thumbnail = ImageTk.PhotoImage(thumbnail)

            row = i // 2
            col = i % 2
            
            thumbnail_label = tk.Label(self.image_grid_frame, image=photo_thumbnail, borderwidth=1, relief="solid")
            thumbnail_label.photo = photo_thumbnail
            thumbnail_label.original_image = img
            thumbnail_label.image_path = image_path
            thumbnail_label.grid(row=row, column=col, sticky="nsew")
            thumbnail_label.bind("<Button-1>", lambda e, lbl=thumbnail_label: self.toggle_image_size(e, lbl))
            thumbnail_label.bind("<Button-3>", self.show_image_context_menu)

            self.combined_image.paste(thumbnail, (col * thumbnail_size[0], row * thumbnail_size[1]))

        self.image_grid_frame.grid_columnconfigure(0, weight=1)
        self.image_grid_frame.grid_columnconfigure(1, weight=1)
        self.image_grid_frame.grid_rowconfigure(0, weight=1)
        self.image_grid_frame.grid_rowconfigure(1, weight=1)

        self.image_grid_frame.showing_full_image = False
        self.is_single_image = False
        logging.info("Multiple images displayed successfully")
        
        self.update_chat("No single image was approved, but here are up to 4 generated attempts. Click an image to view it full-size.")

    def show_image_context_menu(self, event):
        try:
            # Determine which image was clicked
            clicked_widget = event.widget
            if self.is_single_image:
                self.current_image_path = self.current_image_path
            elif hasattr(clicked_widget, 'image_path'):
                self.current_image_path = clicked_widget.image_path
            else:
                return  # If we can't determine the image, don't show the menu

            # Update menu items based on the current state
            self.image_menu.delete(0, tk.END)
            self.image_menu.add_command(label="Copy Image", command=self.copy_image_to_clipboard)
            if not self.is_single_image:
                self.image_menu.add_command(label="Copy All Images", command=self.copy_all_images_to_clipboard)

            self.image_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.image_menu.grab_release()

    def copy_single_image_to_clipboard(self):
        if self.image_grid_frame.showing_full_image and self.image_label:
            self.copy_image_to_clipboard(self.image_label.image_path)
        elif hasattr(self, 'current_image_list') and self.current_image_list:
            # If in grid view, copy the clicked image
            clicked_widget = self.image_grid_frame.winfo_containing(self.master.winfo_pointerx() - self.master.winfo_rootx(),
                                                                    self.master.winfo_pointery() - self.master.winfo_rooty())
            if isinstance(clicked_widget, tk.Label) and hasattr(clicked_widget, 'image_path'):
                self.copy_image_to_clipboard(clicked_widget.image_path)

    def copy_image_to_clipboard(self):
        if self.current_image_path:
            self._copy_image_to_clipboard(self.current_image_path)

    def copy_all_images_to_clipboard(self):
        if self.combined_image:
            self._copy_image_to_clipboard(self.combined_image)

    def _copy_image_to_clipboard(self, image):
        try:
            if isinstance(image, str):  # If it's a file path
                image = Image.open(image)
            output = io.BytesIO()
            image.convert('RGB').save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove the BMP header for clipboard
            output.close()

            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
            logging.info("Image copied to clipboard successfully")
        except Exception as e:
            logging.error(f"Error copying image to clipboard: {str(e)}")
            
    def toggle_image_size(self, event, label):
        """Toggles between 2x2 grid and full-sized image."""
        if not hasattr(self, 'current_image_list'):
            return  # Do nothing if there's no image list (single image view)
        
        if getattr(self.image_grid_frame, 'showing_full_image', False):
            self.display_multiple_images(self.current_image_list)
        else:
            for widget in self.image_grid_frame.winfo_children():
                widget.destroy()
            frame_width = self.image_grid_frame.winfo_width()
            frame_height = self.image_grid_frame.winfo_height()
            full_img = label.original_image.resize((frame_width, frame_height), Image.LANCZOS)
            photo_full = ImageTk.PhotoImage(full_img)
            full_label = tk.Label(self.image_grid_frame, image=photo_full)
            full_label.photo = photo_full
            full_label.pack(fill=tk.BOTH, expand=True)
            full_label.bind("<Button-1>", lambda e, lbl=label: self.toggle_image_size(e, lbl))
            full_label.bind("<Button-3>", self.show_image_context_menu)
            self.image_grid_frame.showing_full_image = True
        self.image_grid_frame.update_idletasks()

    def display_single_image(self, image_path):
        try:
            img = Image.open(image_path)
            img = img.resize((570, 570), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Clear existing content in the image_grid_frame
            for widget in self.image_grid_frame.winfo_children():
                widget.destroy()

            # Create a new label for the image display
            self.image_label = tk.Label(self.image_grid_frame, image=photo)
            self.image_label.image = photo  # Keep a reference
            self.image_label.pack(fill=tk.BOTH, expand=True)
            
            # Bind right-click event for context menu
            self.image_label.bind("<Button-3>", self.show_image_context_menu)
            
            self.current_image_path = image_path
            self.is_single_image = True
            self.master.update_idletasks()
            logging.info(f"Single image displayed: {image_path}")
        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")
            self.update_chat(f"Error displaying image: {str(e)}")
        
    def display_image(self, image_url):
        """Displays a single image in the main image display area."""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            img = img.resize((570, 570), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            # Clear existing content in the image_grid_frame
            for widget in self.image_grid_frame.winfo_children():
                widget.destroy()

            # Create a new label for the image
            self.image_label = tk.Label(self.image_grid_frame, image=photo, bg='lightgray')
            self.image_label.image = photo  # Keep a reference to avoid garbage collection
            self.image_label.pack(fill=tk.BOTH, expand=True)

            # Store the current image path for copy functionality
            self.current_image_path = self.save_image(image_url)

            # Bind the right-click event to the new label
            self.image_label.bind("<Button-3>", self.show_image_context_menu)

            # Reset the showing_full_image flag and clear current_image_list
            self.image_grid_frame.showing_full_image = False
            if hasattr(self, 'current_image_list'):
                del self.current_image_list

            self.is_single_image = True
            logging.info("Single image displayed successfully")

        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")

    def download_image(self, image_url):
        """Downloads an image from the URL and returns a PIL image object."""
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    def analyze_image(self, image_path, retries=3, backoff_factor=0.5):
        for attempt in range(retries):
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                prompt = "In a single English paragraph, describe the image exactly as you see it, including colors, genders, and any notable details."
                if not self.nsfw_allowed:
                    prompt += " If you detect any nudity or explicit content, include the word NUDITY in your description."
                analysis_response = requests.post(
                    "https://text.pollinations.ai/",
                    json={
                        "messages": [
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}
                        ],
                        "model": "openai",
                        "seed": -1,
                        "jsonMode": False
                    },
                    timeout=30
                )
                analysis_response.raise_for_status()
                return analysis_response.text
            except requests.RequestException as e:
                wait_time = backoff_factor * (2 ** attempt)
                logging.error(f"Error analyzing image (attempt {attempt + 1}/{retries}): {str(e)}")
                logging.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        logging.error("Failed to analyze image after multiple attempts")
        return "Error: Unable to analyze the image due to network issues. The image was generated successfully, but I couldn't describe it in detail."
        
    def verify_image(self, image_prompt, analysis):
        verification_prompt = f"Compare these, first is the image request prompt: '{image_prompt}' and then the result from the vision: '{analysis}'. If the result is generally good enough and matches the description, reply with ![APPROVED]. If it really doesn't match, reply with ![DENIED]. Provide a reason if denied. DO NOT add or change any requirements from the original prompt."
        
        try:
            response = self.get_ai_response(verification_prompt)
            if "![APPROVED]" in response:
                logging.info(f"Image Verification Result: {response}")
                return "![APPROVED]"
            elif "![DENIED]" in response:
                logging.warning(f"Image Verification Result: {response}")
                return "![DENIED]"
            else:
                logging.warning(f"Unclear verification response: {response}")
                return "![DENIED] Unclear verification response"
        except Exception as e:
            logging.error(f"Error in verification: {str(e)}")
            return "![DENIED] Verification error"

    def fallback_verification(self, image_prompt, analysis):
        # Simple keyword matching as a fallback
        prompt_keywords = set(image_prompt.lower().split())
        analysis_keywords = set(analysis.lower().split())
        match_ratio = len(prompt_keywords.intersection(analysis_keywords)) / len(prompt_keywords)
        
        if match_ratio > 0.5 and "nudity" not in analysis.lower():
            return "![APPROVED]"
        else:
            return "![DENIED] Failed to match prompt closely"

    def generate_and_verify_image(self, image_prompt):
        max_attempts = 3
        for attempt in range(max_attempts):
            image_url = self.generate_image(image_prompt)
            if not image_url:
                continue

            image_path = self.save_image(image_url)
            analysis = self.analyze_image(image_path)

            verification = self.verify_image(image_prompt, analysis)

            if verification == "![APPROVED]":
                return image_url

            if attempt < max_attempts - 1:
                self.update_chat("AI: The result wasn't quite right, let me try again with a new seed.")

        return image_url

    def save_image(self, image_url, seed=None):
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))

            # Use seed if provided, otherwise use timestamp
            if seed is not None:
                path = f"temp_image_{seed}.png"
            else:
                timestamp = int(time.time())
                path = f"temp_image_{timestamp}.png"

            img.save(path)

            self.current_image_paths.append(path)  # Append to list instead of resetting
            return path
        except Exception as e:
            logging.error(f"Error saving image: {str(e)}")
            return None

    def remove_image_prompt(self, text):
        cleaned_text = re.sub(r'!\[MRKDWN\]\(.*?\)', '', text).strip()
        # Remove any remaining single dots
        cleaned_text = re.sub(r'^\.$', '', cleaned_text).strip()
        return cleaned_text

    def update_chat(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message.strip() + "\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.master.update_idletasks()
        logging.info(f"Chat updated: {message}")

    def get_ai_response(self, prompt, max_retries=3):
        logging.info(f"Sending prompt to AI: {prompt}")
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://text.pollinations.ai/",
                    json={
                        "messages": [
                            {"role": "system", "content": self.system_message},
                            *[{"role": "user" if msg["user"] else "assistant", "content": msg["user"] or msg["ai"]} 
                              for msg in self.conversation_history],
                            {"role": "user", "content": prompt}
                        ],
                        "model": "openai",
                        "jsonMode": False
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logging.error(f"Error getting AI response (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to get AI response after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
            
    def update_history(self, user_input, ai_response):
        self.conversation_history.append({"user": user_input, "ai": ai_response})
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def update_status(self, status):
        self.status_label.config(text=status)
        logging.info(f"Status updated: {status}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AIImageChatApp(root)
    root.mainloop()
