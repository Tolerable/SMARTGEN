import tkinter as tk
from tkinter import scrolledtext, Menu
from PIL import Image, ImageTk
import requests
import io
import re
import os
import threading
import logging
import random
import base64
import time
import math
import win32clipboard  # For clipboard copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIImageChatApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Image Chat")
        master.geometry("600x800")
        master.configure(bg='white')

        self.main_frame = tk.Frame(master, padx=15, pady=15, bg='white')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Image display area - fixed 2x2 grid with placeholders
        self.image_grid_frame = tk.Frame(self.main_frame, width=500, height=500, bg='lightgray')
        self.image_grid_frame.pack(pady=(0, 10))
        self.image_grid_frame.pack_propagate(False)

        # Placeholder for the images in 2x2 grid (4 slots)
        self.image_placeholders = [None] * 4
        self.create_placeholders()

        self.chat_area = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, width=66, height=10)
        self.chat_area.pack(pady=(0, 10))
        self.chat_area.config(state=tk.DISABLED)

        self.input_area = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, width=66, height=3)
        self.input_area.pack()
        self.input_area.bind("<Return>", self.send_message)

        # Adjust the status bar to stay at the bottom
        self.status_label = tk.Label(self.master, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg='white')
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.conversation_history = []
        self.current_image_paths = []  # Store paths of generated images
        self.is_zoomed = False  # Track if an image is zoomed

        # System message for the AI's initial persona
        self.system_message = (
            "You are an AI responding in conversation and also generating image prompts on the fly without hesitation. "
            "Respond conversationally only when chatting, and when asked to or inferred instantly create image prompts. "
            "Use the format: ![MRKDWN](Description User Requested) without commenting about the process."
        )

        # Right-click context menu for copying images
        self.image_menu = Menu(self.master, tearoff=0)
        self.image_menu.add_command(label="Copy Image", command=self.copy_image_to_clipboard)
        self.image_menu.add_command(label="Copy All", command=self.copy_all_images_to_clipboard)
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.cleanup_temp_files()  # Initial cleanup
        self.master.after(3600000, self.periodic_cleanup)
        
    def periodic_cleanup(self):
        self.cleanup_temp_files()
        self.master.after(3600000, self.periodic_cleanup)  # Schedule next cleanup
        
    def refine_prompt(self, original_prompt, previous_attempt, analysis):
        keywords = self.extract_keywords(original_prompt)
        refined_prompt = (
            f"Original concept: '{original_prompt}'. "
            f"Key elements: {', '.join(keywords)}. "
            f"Previous attempt: '{previous_attempt}'. "
            f"Current result: '{analysis}'. "
            "Enhance the image while staying true to the original concept. "
            "Feel free to creatively interpret and expand on the key elements, "
            "but ensure the core idea remains intact. "
            "GENERATE NEW PROMPT: ![MRKDWN](ENHANCED PROMPT FAITHFUL TO ORIGINAL CONCEPT)"
        )
        return refined_prompt

    def get_enhancement_strategy(self, original_prompt):
        strategy_prompt = (
            f"Given this image concept: '{original_prompt}', "
            "explain in a few sentences how you would enhance it "
            "while staying true to the original idea. Focus on visual improvements "
            "and creative interpretations that amplify the concept's core elements."
        )
        return self.get_ai_response(strategy_prompt)

    def clean_prompt(self, prompt):
        """
        Cleans redundant metadata like 'USER REQUESTED' and nested prompt markers from the text.
        """
        clean_prompt = re.sub(r"USER REQUESTED THIS IMAGE:|Your last attempt.*?resulted in.*?image:|RESUBMIT WITH.*", "", prompt)
        clean_prompt = clean_prompt.replace(';;', ';').strip()  # Clean up extra semi-colons if present
        return clean_prompt

    def clean_analysis(self, analysis):
        """
        Clean any unnecessary text from the vision analysis to ensure only the descriptive part remains.
        """
        return analysis.strip()

    def create_placeholders(self):
        """Create placeholders for the 2x2 image grid."""
        placeholder_image = Image.new('RGB', (250, 250), color='lightgray')
        self.placeholders = [ImageTk.PhotoImage(placeholder_image) for _ in range(4)]

        for i in range(4):
            row = i // 2
            col = i % 2
            label = tk.Label(self.image_grid_frame, image=self.placeholders[i], borderwidth=1, relief="solid")
            label.grid(row=row, column=col, sticky="nsew")
            label.bind("<Button-1>", lambda e, label=label: self.toggle_image_size(e, label))
            label.bind("<Button-3>", lambda e, index=i: self.show_image_context_menu(index))
            self.image_placeholders[i] = label

    def send_message(self, event=None):
        user_input = self.input_area.get("1.0", tk.END).strip()
        if not user_input:
            return "break"

        self.input_area.delete("1.0", tk.END)
        self.update_chat(f"You: {user_input}")
        self.conversation_history.append({"role": "user", "content": user_input})

        threading.Thread(target=self.process_message, args=(user_input,)).start()

        return "break"

    def process_message(self, user_input):
        self.update_status("Processing...")
        logging.info(f"Processing user input: {user_input}")

        try:
            ai_response = self.get_ai_response(user_input)
            ai_response_cleaned, image_prompt = self.extract_image_prompt(ai_response)

            if ai_response_cleaned:
                self.update_chat(ai_response_cleaned)

            if image_prompt:
                self.update_chat("Generating image based on your request...")
                threading.Thread(target=self.image_enhancement_pipeline, args=(image_prompt,)).start()

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            self.update_chat(f"Error processing your request. Please try again.")
        
        self.update_status("")

    def update_history(self, user_input, ai_response):
        """Updates the conversation history."""
        self.conversation_history.append({"user": user_input, "ai": ai_response})
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)
        logging.info(f"Updated conversation history. Current length: {len(self.conversation_history)}")

    def extract_image_prompt(self, text):
        """
        Extract the image prompt from the AI response and clean the response text.
        This will remove the markdown-style image prompt and return the cleaned text and the image prompt.
        """
        match = re.search(r'!\[MRKDWN\]\((.*?)\)', text)
        image_prompt = match.group(1) if match else None
        cleaned_text = re.sub(r'!\[MRKDWN\]\(.*?\)', '', text).strip()
        return cleaned_text, image_prompt

    def image_enhancement_pipeline(self, base_prompt):
        keywords = self.extract_keywords(base_prompt)
        enhancement_strategy = self.get_enhancement_strategy(base_prompt)
        
        previous_enhanced_prompt = base_prompt
        previous_analysis = "No previous analysis"

        for i in range(4):
            if i == 0:
                enhanced_prompt = (
                    f"Original concept: '{base_prompt}'. "
                    f"Key elements: {', '.join(keywords)}. "
                    f"Enhancement strategy: {enhancement_strategy}. "
                    "GENERATE NEW PROMPT: ![MRKDWN](ENHANCED PROMPT)"
                )
            else:
                enhanced_prompt = self.refine_prompt(base_prompt, previous_enhanced_prompt, previous_analysis)
            
            # Generate image based on enhanced_prompt
            seed = random.randint(1, 1000000)
            image_url = self.generate_image(enhanced_prompt, seed=seed)
            if not image_url:
                self.update_chat(f"Failed to generate image for iteration {i+1}.")
                continue

            # Save and display the generated image
            image_path = self.save_image(image_url)
            self.display_image(image_path, i)

            # Analyze the generated image
            analysis = self.analyze_image(image_path)

            # Update previous_enhanced_prompt and previous_analysis for the next iteration
            previous_enhanced_prompt = enhanced_prompt
            previous_analysis = analysis

        self.update_chat("All images generated.")
        self.cleanup_temp_files()
        
    def enhance_prompt(self, base_prompt):
        """Enhance the base prompt for better image generation."""
        enhanced_prompt = f"Enhanced version of: {base_prompt} with more vivid details."
        return enhanced_prompt

    def extract_keywords(self, prompt):
        common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        return [word.lower() for word in prompt.split() if word.lower() not in common_words]

    def generate_image(self, prompt, seed):
        """Call the image generation API with the enhanced prompt."""
        try:
            url = f"https://image.pollinations.ai/prompt/{prompt}?model=flux&width=570&height=570&seed={seed}&nologo=true"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.url
        except requests.RequestException as e:
            logging.error(f"Error generating image: {str(e)}")
            return None

    def save_image(self, image_url):
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            path = f"temp_image_{int(time.time())}.png"
            img.save(path)
            self.current_image_paths.append(path)
            # Keep only the last 4 image paths
            self.current_image_paths = self.current_image_paths[-4:]
            return path
        except Exception as e:
            logging.error(f"Error saving image: {str(e)}")
            return None

    def cleanup_temp_files(self):
        current_time = time.time()
        for filename in os.listdir('.'):
            if filename.startswith('temp_image_') and filename.endswith('.png'):
                file_time = int(filename.split('_')[2].split('.')[0])
                if current_time - file_time > 3600:  # Delete files older than 1 hour
                    try:
                        os.remove(filename)
                        logging.info(f"Deleted old temporary file: {filename}")
                    except Exception as e:
                        logging.error(f"Error deleting temporary file {filename}: {str(e)}")
    
    def analyze_image(self, image_path):
        """Send the image to the vision API to extract a text description."""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            prompt = "In a single English paragraph, describe the image exactly as you see it, including colors, genders, and any notable details."
            analysis_response = requests.post(
                "https://text.pollinations.ai/",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
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
            return "Error analyzing image."

    def display_image(self, image_path, index):
        img = Image.open(image_path)
        thumbnail = img.copy().resize((250, 250), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumbnail)
        label = self.image_placeholders[index]
        label.config(image=photo)
        label.photo = photo
        label.original_image = img
        label.image_path = image_path
        label.bind("<Button-3>", self.show_image_context_menu)

    def toggle_image_size(self, event, label):
        frame_width = 500
        frame_height = 500

        if getattr(self.image_grid_frame, 'showing_full_image', False):
            # Return to grid view
            for widget in self.image_grid_frame.winfo_children():
                if isinstance(widget, tk.Label):
                    widget.grid()
            self.image_grid_frame.showing_full_image = False
            # Remove the full-size image label
            for widget in self.image_grid_frame.winfo_children():
                if widget.grid_info().get('columnspan') == 2:
                    widget.destroy()
        else:
            # Show full-size image
            if hasattr(label, 'original_image'):
                full_img = label.original_image.copy().resize((frame_width, frame_height), Image.LANCZOS)
                photo_full = ImageTk.PhotoImage(full_img)
                
                full_label = tk.Label(self.image_grid_frame, image=photo_full)
                full_label.photo = photo_full
                full_label.original_image = label.original_image
                full_label.image_path = label.image_path
                full_label.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")
                full_label.bind("<Button-1>", lambda e, lbl=full_label: self.toggle_image_size(e, lbl))
                full_label.bind("<Button-3>", self.show_image_context_menu)  # Corrected this line
                
                # Hide other images but keep them in the grid
                for widget in self.image_grid_frame.winfo_children():
                    if widget != full_label:
                        widget.grid_remove()
                
                self.image_grid_frame.showing_full_image = True

        self.image_grid_frame.update_idletasks()

    def copy_image_to_clipboard(self):
        """Copy the selected image to the clipboard."""
        try:
            if hasattr(self, 'current_copied_image') and os.path.exists(self.current_copied_image):
                img = Image.open(self.current_copied_image)
                output = io.BytesIO()
                img.convert('RGB').save(output, 'BMP')
                data = output.getvalue()[14:]  # Remove BMP header
                output.close()

                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()

                logging.info(f"Image copied to clipboard: {self.current_copied_image}")
            else:
                logging.error("No image selected or image file not found.")
        except Exception as e:
            logging.error(f"Error copying image to clipboard: {str(e)}")

    def copy_all_images_to_clipboard(self):
        """Combine all 4 images into one and copy it to the clipboard."""
        try:
            if len(self.current_image_paths) != 4:
                logging.error(f"Expected 4 images, but found {len(self.current_image_paths)}.")
                return

            combined_image = Image.new('RGB', (500, 500))
            for i, image_path in enumerate(self.current_image_paths):
                img = Image.open(image_path)
                img = img.resize((250, 250), Image.LANCZOS)
                x = (i % 2) * 250
                y = (i // 2) * 250
                combined_image.paste(img, (x, y))

            output = io.BytesIO()
            combined_image.save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove BMP header
            output.close()

            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()

            logging.info("All 4 images copied to clipboard as a single image.")
        except Exception as e:
            logging.error(f"Error copying all images to clipboard: {str(e)}")

    def show_image_context_menu(self, event):
        label = event.widget
        if hasattr(label, 'image_path'):
            self.current_copied_image = label.image_path
            self.image_menu.tk_popup(event.x_root, event.y_root)

    def get_ai_response(self, prompt, retry_count=3):
        """Get a response from the AI using the text generation API."""
        logging.info(f"Sending prompt to AI: {prompt}")

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    "https://text.pollinations.ai/",
                    json={
                        "messages": [
                            {"role": "system", "content": self.system_message},
                            {"role": "user", "content": prompt}
                        ]
                    },
                    timeout=60
                )
                response.raise_for_status()

                # Process the response as raw text
                response_text = response.text.strip()

                if response_text:
                    return response_text  # Directly return the raw text
                else:
                    raise ValueError("Received empty response from AI")

            except requests.RequestException as e:
                logging.error(f"Error getting AI response (attempt {attempt + 1}): {str(e)}")

                if attempt == retry_count - 1:
                    # If all retry attempts fail, raise an error or return a default response
                    raise ValueError(f"Failed to get AI response after {retry_count} attempts: {str(e)}")

    def update_chat(self, message):
        """Update the chat area with a new message."""
        # Remove colon at the end of the message if present
        if message.strip().endswith(":"):
            message = message.strip()[:-1] + "."

        # Clean up the message, remove extra newlines or spaces
        cleaned_message = message.strip()
        
        # Replace ellipses and colons with cleaner output
        cleaned_message = cleaned_message.replace("...", "")
        
        # Reformat the image generation status
        cleaned_message = cleaned_message.replace("Generating image based on your request", "[GENERATING IMAGES]")
        cleaned_message = cleaned_message.replace("All images generated.", "[IMAGES GENERATED]")
        
        # Remove any double newlines or excessive whitespace
        cleaned_message = re.sub(r'\n\s*\n', '\n', cleaned_message).strip()

        # Update chat area
        self.chat_area.config(state=tk.NORMAL)
        if cleaned_message:
            self.chat_area.insert(tk.END, cleaned_message + "\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.master.update_idletasks()

    def update_status(self, status):
        """Update the status label to display the current status."""
        self.status_label.config(text=status)
        logging.info(f"Status updated: {status}")


    def on_closing(self):
        self.cleanup_temp_files()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AIImageChatApp(root)
    root.mainloop()
