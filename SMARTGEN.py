import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
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

        # Create a grid frame for multiple images (must come before image_label)
        self.image_grid_frame = tk.Frame(self.main_frame, width=570, height=570, bg='lightgray')
        self.image_grid_frame.pack(pady=(0, 10))
        self.image_grid_frame.pack_propagate(False)

        # Now define image_label after image_grid_frame is created
        self.image_label = tk.Label(self.image_grid_frame, bg='lightgray')
        self.image_label.pack(fill=tk.BOTH, expand=True)

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
        self.image_menu.add_command(label="COPY", command=self.copy_image_to_clipboard)
        self.image_label.bind("<Button-3>", self.show_image_context_menu)

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
            logging.info(f"AI response: {ai_response}")

            # Remove the "AI:" prefix from here, since update_chat adds it
            ai_response_cleaned = self.remove_image_prompt(ai_response).strip()

            if ai_response_cleaned:
                self.update_chat(ai_response_cleaned)  # Send the clean AI response to chat

            # Check for image prompt hidden in markdown
            image_prompt = self.extract_image_prompt(ai_response)

            if image_prompt:
                self.update_chat("AI: Generating image based on your request...")
                threading.Thread(target=self.handle_image_generation, args=(image_prompt,)).start()

            # Update conversation history with user input and AI response
            self.update_history(user_input, ai_response)

        except Exception as e:
            logging.error(f"Error in process_message: {str(e)}")
            self.update_chat("AI: I'm sorry, but I encountered an error while processing your request.")

        self.update_status("")

    def update_history(self, user_input, ai_response):
        self.conversation_history.append({"user": user_input, "ai": ai_response})
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def extract_image_prompt(self, text):
        match = re.search(r'!\[MRKDWN\]\((.*?)\)', text)
        return match.group(1) if match else None

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
        """Handles the full flow of generating up to 4 images and deciding whether to display one or all as thumbnails"""
        generated_images = []
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            seed = random.randint(1, 1000000)  # Random seed for each attempt
            image_url = self.generate_image(image_prompt, seed)  # Pass seed to image generation
            if not image_url:
                continue  # Skip to next attempt if image generation failed

            image_path = self.save_image(image_url, seed)  # Pass seed to save_image
            if not image_path:
                continue  # Skip if saving failed

            analysis = self.analyze_image(image_path)
            verification = self.verify_image(image_prompt, analysis)

            # Store each generated image and its path
            generated_images.append((image_url, image_path))

            if verification == "![APPROVED]":
                # If approved, display only the approved image and stop
                self.display_image(image_url)
                self.update_chat(f"AI: Here's the image I created for you on attempt {attempt}. I hope you like it!")
                return

        # If no approved image after 4 attempts, display all generated images as clickable thumbnails
        self.display_multiple_images(generated_images)

    def display_multiple_images(self, image_list):
        """Displays multiple images in a 2x2 grid within the existing frame."""
        self.current_image_list = image_list

        for widget in self.image_grid_frame.winfo_children():
            widget.destroy()

        frame_width = self.image_grid_frame.winfo_width()
        frame_height = self.image_grid_frame.winfo_height()
        thumbnail_size = (frame_width // 2, frame_height // 2)

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
            thumbnail_label.bind("<Button-3>", lambda e, path=image_path: self.show_image_context_menu(e, path))

        self.image_grid_frame.grid_columnconfigure(0, weight=1)
        self.image_grid_frame.grid_columnconfigure(1, weight=1)
        self.image_grid_frame.grid_rowconfigure(0, weight=1)
        self.image_grid_frame.grid_rowconfigure(1, weight=1)

        self.image_grid_frame.showing_full_image = False

    def show_image_context_menu(self, event, img_path):
        """Shows the right-click context menu on the image frame and stores the image path to be copied."""
        self.current_image_to_copy = img_path
        try:
            self.image_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.image_menu.grab_release()

    def copy_image_to_clipboard(self):
        """Copies the right-clicked image to the clipboard."""
        if hasattr(self, 'current_image_to_copy'):
            try:
                image = Image.open(self.current_image_to_copy)
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
        else:
            logging.info("No image available to copy")

    def toggle_image_size(self, event, label):
        """Toggles between 2x2 grid and full-sized image."""
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
            full_label.bind("<Button-3>", lambda e, path=label.image_path: self.show_image_context_menu(e, path))

            self.image_grid_frame.showing_full_image = True

        self.image_grid_frame.update_idletasks()
    
    def display_image(self, image_url):
        """Displays a single image in the main image display area."""
        try:
            response = requests.get(image_url, timeout=30)  # Download the image
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))  # Open the image
            img = img.resize((570, 570), Image.LANCZOS)  # Resize to fit the frame
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
            self.image_label.bind("<Button-3>", lambda e: self.show_image_context_menu(e, self.current_image_path))

            logging.info("Image displayed successfully")

        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")

    def download_image(self, image_url):
        """Downloads an image from the URL and returns a PIL image object."""
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    def analyze_image(self, image_path, retries=3):
        for attempt in range(retries):
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                # Send image to AI for analysis
                prompt = "In a single English paragraph, describe the image exactly as you see it, including colors, genders, and any notable details."
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
                logging.error(f"Error analyzing image: {str(e)}")
                if attempt == retries - 1:  # If last attempt
                    return "Error analyzing image."

    def verify_image(self, image_prompt, analysis):
        verification_prompt = f"Compare these, first is the image request prompt: '{image_prompt}' and then the result from the vision: '{analysis}'. If the result is generally good enough and matches the description, reply with ![APPROVED]. If it really doesn't match, reply with ![DENIED]. Provide a reason if denied."
        response = self.get_ai_response(verification_prompt)
        logging.info(f"Verification response: {response}")
        
        # Log the reason for denial
        if "![DENIED]" in response:
            logging.error(f"Image Denied: Reason - {response}")
        
        return "![APPROVED]" if "![APPROVED]" in response else "![DENIED]"

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
        return cleaned_text + "\n"

    def update_chat(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message.strip() + "\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.master.update_idletasks()
        logging.info(f"Chat updated: {message}")

    def get_ai_response(self, prompt):
        logging.info(f"Sending prompt to AI: {prompt}")
        try:
            conversation_history_text = "\n".join([f"User: {item['user']}\nAI: {item['ai']}" for item in self.conversation_history])

            system_message = (
                "You are an AI that engages in conversation and generates image prompts. "
                "When asked for an image, respond with the appropriate markdown format ![MRKDWN](description). "
                "Do not directly describe the image yourself; instead, generate the markdown. "
                "Approve or deny generated images based on how close they match the prompt."
            )

            response = requests.post(
                "https://text.pollinations.ai/",
                json={
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": conversation_history_text + f"\nUser: {prompt}"}
                    ],
                    "model": "openai",
                    "jsonMode": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Error getting AI response: {str(e)}")
            return "Error getting AI response."

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
