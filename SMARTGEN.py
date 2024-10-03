import tkinter as tk
from tkinter import scrolledtext, Menu, ttk
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
import win32clipboard 
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StyleEditor:
    def __init__(self, master, callback, initial_styles=None, initial_checkboxes=None, initial_models=None):
        self.master = master
        self.callback = callback
        self.styles = initial_styles if initial_styles else [""] * 4
        self.checkboxes = initial_checkboxes if initial_checkboxes else [False] * 4
        self.models = initial_models if initial_models else ["flux"] * 4
        
        self.window = tk.Toplevel(master)
        self.window.title("Style Editor")
        
        self.entries = []
        self.checkbox_vars = []
        self.model_vars = []
        
        # Fetch available models
        self.available_models = self.fetch_models()
        
        for i in range(4):
            var = tk.BooleanVar(value=self.checkboxes[i])
            cb = ttk.Checkbutton(self.window, variable=var)
            cb.grid(row=i, column=0)
            self.checkbox_vars.append(var)
            
            entry = ttk.Entry(self.window, width=40)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, self.styles[i])
            self.entries.append(entry)
            
            model_var = tk.StringVar(value=self.models[i])
            model_dropdown = ttk.Combobox(self.window, textvariable=model_var, values=self.available_models, width=10)
            model_dropdown.grid(row=i, column=2, padx=5)
            self.model_vars.append(model_var)
        
        ttk.Button(self.window, text="Save", command=self.save).grid(row=4, column=1)
        ttk.Button(self.window, text="Revert", command=self.revert).grid(row=4, column=2)
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def fetch_models(self):
        try:
            response = requests.get("https://image.pollinations.ai/models")
            return response.json()
        except:
            return ["flux", "flux-realism", "flux-anime", "flux-3d", "any-dark", "turbo"]
    
    def save(self):
        styles = [entry.get() for entry in self.entries]
        checkboxes = [var.get() for var in self.checkbox_vars]
        models = [var.get() for var in self.model_vars]
        self.callback(styles, checkboxes, models)
        logging.info(f"StyleEditor save: styles={styles}, checkboxes={checkboxes}, models={models}")

    def on_closing(self):
        self.save()
        self.window.destroy()
    
    def revert(self):
        for entry, style in zip(self.entries, self.styles):
            entry.delete(0, tk.END)
            entry.insert(0, style)
        for var, checked in zip(self.checkbox_vars, self.checkboxes):
            var.set(checked)
        for var, model in zip(self.model_vars, self.models):
            var.set(model)
        logging.info("Styles and models reverted to last saved state")
    
class AIImageChatApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Image Chat")
        master.geometry("600x800")
        master.configure(bg='white')
        
        # Create menu bar
        self.menu_bar = Menu(master)
        self.master.config(menu=self.menu_bar)

        # Create Tools menu
        self.tools_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=self.tools_menu)
        self.tools_menu.add_command(label="Style Editor", command=self.open_style_editor)

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
        
        self.styles = [""] * 4
        self.style_checkboxes = [False] * 4
        self.models = ["flux"] * 4
        
        self.load_settings()
        # print(f"Loaded settings: styles={self.styles}, checkboxes={self.style_checkboxes}, models={self.models}")
        
    def open_style_editor(self):
        StyleEditor(self.master, self.update_styles, self.styles, self.style_checkboxes, self.models)

    def update_styles(self, styles, checkboxes, models):
        self.styles = styles
        self.style_checkboxes = checkboxes
        self.models = models
        self.save_settings()
        logging.info(f"Styles and models updated and saved: styles={styles}, checkboxes={checkboxes}, models={models}")
        
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

    def get_enhancement_strategy(self, original_prompt, style_index):
        user_style = self.styles[style_index] if self.style_checkboxes[style_index] else ""
        strategy_prompt = (
            f"Given this image concept: '{original_prompt}', and considering the style '{user_style}', "
            "provide a brief strategy on how to enhance it visually in a single paragraph format. "
            "Focus on specific visual elements, styles, or techniques that could be added or modified, "
            f"with emphasis on the '{user_style}' if provided. "
            "Keep your response concise, around 2-3 sentences confined to one paragraph."
        )
        response = self.get_ai_response(strategy_prompt)
        return self.clean_paragraph(response)

    def clean_paragraph(self, text):
        """Ensure the AI response is formatted as a single paragraph."""
        # Replace multiple newlines with a single space
        cleaned = re.sub(r'\s*\n\s*', ' ', text).strip()
        # Remove any remaining newlines
        cleaned = cleaned.replace('\n', ' ')
        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned


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
            # Get AI response for chatting
            ai_response = self.get_ai_response(user_input)
            
            # Extract image prompt and clean the response
            cleaned_response, image_prompt = self.extract_image_prompt(ai_response)
            cleaned_response = self.clean_paragraph(cleaned_response)

            # Update chat with AI's cleaned response, preserving the "AI:" prefix
            if cleaned_response.strip():
                self.update_chat(f"AI: {cleaned_response}")

            if image_prompt:
                self.update_chat("[GENERATING IMAGES]")
                threading.Thread(target=self.image_enhancement_pipeline, args=(image_prompt,)).start()

            # Update conversation history with cleaned response
            self.update_history(user_input, cleaned_response)

        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            self.update_chat("Error processing your request. Please try again.")
        
        self.update_status("")

    def update_history(self, user_input, ai_response):
        """Updates the conversation history."""
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Trim conversation history to the last 5 exchanges
        if len(self.conversation_history) > 10:  # 5 exchanges = 10 messages (user + AI)
            self.conversation_history = self.conversation_history[-10:]

        logging.info(f"Updated conversation history. Current length: {len(self.conversation_history)}")

    def extract_image_prompt(self, text):
        match = re.search(r'!\[MRKDWN\]\((.*?)\)', text)
        image_prompt = match.group(1) if match else None
        cleaned_text = re.sub(r'!\[MRKDWN\]\(.*?\)', '', text).strip()
        return cleaned_text, image_prompt

    def image_enhancement_pipeline(self, base_prompt):
        keywords = self.extract_keywords(base_prompt)
        enhancement_strategy = self.get_enhancement_strategy(base_prompt, 0)  # Use 0 as default
        
        for i in range(4):
            if i == 0:
                enhanced_prompt = self.get_ai_response(
                    f"Original concept: '{base_prompt}'. "
                    f"Key elements: {', '.join(keywords)}. "
                    f"Enhancement strategy: {enhancement_strategy}. "
                    "GENERATE NEW PROMPT: ![MRKDWN](ENHANCED PROMPT)"
                )
            else:
                previous_image_path = self.current_image_paths[i-1]
                vision_analysis = self.clean_paragraph(self.analyze_image(previous_image_path))
                # Get a new enhancement strategy for each iteration
                enhancement_strategy = self.get_enhancement_strategy(base_prompt, i)
                enhanced_prompt = self.get_ai_response(
                    f"Original concept: '{base_prompt}'. "
                    f"Key elements: {', '.join(keywords)}. "
                    f"Enhancement strategy: {enhancement_strategy}. "
                    f"Previous attempt: {vision_analysis}. "
                    "GENERATE NEW PROMPT: ![MRKDWN](REFINED PROMPT)"
                )

            # Extract the actual prompt from the AI response
            _, image_prompt = self.extract_image_prompt(enhanced_prompt)
            
            if not image_prompt:
                image_prompt = base_prompt  # Fallback to original if extraction fails

            # Append user style if checkbox is checked
            if self.style_checkboxes[i] and self.styles[i]:
                image_prompt += f". Apply a strong {self.styles[i]} to the entire image"

            # Log the final prompt for debugging
            logging.info(f"Final prompt for image {i+1}: {image_prompt}")

            # Generate image based on enhanced_prompt with user style
            seed = random.randint(1, 1000000)
            selected_model = self.models[i] if self.style_checkboxes[i] else "flux"
            image_url = self.generate_image(image_prompt, seed=seed, model=selected_model)
            if not image_url:
                self.update_chat(f"Failed to generate image for iteration {i+1}.")
                continue
            # Save and display the generated image
            image_path = self.save_image(image_url)
            self.display_image(image_path, i)

            # Analyze the generated image
            vision_analysis = self.analyze_image(image_path)
            self.update_chat(f"Image {i+1} generated and analyzed.")

        self.update_chat("All images generated and analyzed.")
        self.cleanup_temp_files()
    
    def enhance_prompt(self, base_prompt):
        """Enhance the base prompt for better image generation."""
        enhanced_prompt = f"Enhanced version of: {base_prompt} with more vivid details."
        return enhanced_prompt

    def extract_keywords(self, prompt):
        common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        return [word.lower() for word in prompt.split() if word.lower() not in common_words]

    def generate_image(self, prompt, seed=None, model="flux"):
        try:
            url = f"https://image.pollinations.ai/prompt/{prompt}?model={model}&width=570&height=570&seed={seed}&nologo=true"
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
        logging.info(f"Sending prompt to AI: {prompt}")
        
        # Include conversation history to maintain context
        conversation = [{"role": "system", "content": self.system_message}]
        conversation.extend(self.conversation_history)  # Add previous conversation history
        conversation.append({"role": "user", "content": prompt})  # Add the new prompt

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    "https://text.pollinations.ai/",
                    json={
                        "messages": conversation,  # Pass the conversation to the API
                        "model": "openai",
                        "seed": -1,
                        "jsonMode": False
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                if response.status_code == 200:
                    return response.text.strip()
                else:
                    logging.error(f"Unexpected status code: {response.status_code}")
                    return None

            except requests.RequestException as e:
                logging.error(f"Error getting AI response (attempt {attempt + 1}): {str(e)}")
                if attempt == retry_count - 1:
                    return f"I'm sorry, but I'm having trouble connecting to my language model right now. Let's proceed with image generation using your original prompt: {prompt}"

        return f"I apologize, but I'm unable to enhance the prompt at the moment. Let's use your original idea: {prompt}"

    def update_chat(self, message):
        """Update the chat area with a new message."""
        # Remove leading/trailing whitespace
        message = message.strip()
        
        # Remove colon at the end of the message if present
        if message.endswith(":"):
            message = message[:-1] + "."
        
        # Replace ellipses with cleaner output
        message = message.replace("...", "")
        
        # Reformat the image generation status
        message = message.replace("Generating image based on your request", "[GENERATING IMAGES]")
        message = message.replace("All images generated.", "[IMAGES GENERATED]")
        
        # Remove any double newlines or excessive whitespace
        message = re.sub(r'\n\s*\n', '\n', message)
        
        # Remove any remaining leading/trailing whitespace
        message = message.strip()
        
        # Only update chat if there's actual content
        if message:
            self.chat_area.config(state=tk.NORMAL)
            self.chat_area.insert(tk.END, message + "\n\n")
            self.chat_area.see(tk.END)
            self.chat_area.config(state=tk.DISABLED)
            self.master.update_idletasks()

    def update_status(self, status):
        """Update the status label to display the current status."""
        self.status_label.config(text=status)
        logging.info(f"Status updated: {status}")

    def save_settings(self):
        settings = {
            "styles": self.styles,
            "style_checkboxes": self.style_checkboxes,
            "models": self.models
        }
        with open("smartgen_settings.json", "w") as f:
            json.dump(settings, f)
        logging.info(f"Settings saved: {settings}")

    def load_settings(self):
        try:
            with open("smartgen_settings.json", "r") as f:  # Changed from "settings.json" to "smartgen_settings.json"
                settings = json.load(f)
            self.styles = settings.get("styles", [""] * 4)
            self.style_checkboxes = settings.get("style_checkboxes", [False] * 4)
            self.models = settings.get("models", ["flux"] * 4)
        except FileNotFoundError:
            self.styles = [""] * 4
            self.style_checkboxes = [False] * 4
            self.models = ["flux"] * 4

    def update_styles(self, styles, checkboxes, models):
        self.styles = styles
        self.style_checkboxes = checkboxes
        self.models = models
        self.save_settings()
        logging.info("Styles and models updated and saved")

    def on_closing(self):
        self.save_settings()
        self.cleanup_temp_files()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AIImageChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()