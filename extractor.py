import numpy as np
import cv2
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
from PIL import Image, ImageTk
import pytesseract
import os
from pdf2image import convert_from_path
import re
from enum import Enum


class Language(Enum):
    ENGLISH = 'eng'
    HUNGARIAN = 'hun'
    GERMAN = 'deu'
    FRENCH = 'fra'


def replace_multiple_spaces_with_tabs(input_string):
    # Replace three or more spaces with a tab character
    return re.sub(r'\s{3,}', ', \t ', input_string)


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


MAX_WIDTH = 1600
MAX_HEIGHT = 1000

def analyze(image, lang):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to the image
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('thresholded', thresholded_image)
    # config = r'--oem 3 --psm 6'
    config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    try:
        extracted_text = pytesseract.image_to_string(thresholded_image, lang=lang.value, config=config)
    except Exception as e:
        print(f"An error occurred: {e}")
        extracted_text = ""
    return extracted_text


def crop_masked_image(image, mask):
    # Check if the image and the mask have the same dimensions
    assert image.shape[:2] == mask.shape[:2], "Image and mask must have the same dimensions."
    # Apply the mask to the image
    result = cv2.bitwise_and(image, mask)
    # Convert the result to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Find contours in the grayscale image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Crop the image using the bounding box
    cropped_im = result[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # most common color
    color = np.median(cropped_im, axis=(0, 1))

    inverted_mask = cv2.bitwise_not(cropped_mask)
    # fill mask with most common color
    mcc_mask = np.full_like(cropped_im, color)
    final_mask = cv2.bitwise_and(mcc_mask, inverted_mask)
    cropped_im = cv2.bitwise_or(cropped_im, final_mask)

    print(cropped_im.shape)
    return cropped_im


class MaskDrawerGUI:
    def __init__(self, master, images):
        self.master = master
        self.images = images
        self.current_page_index = 0
        self.current_image = self.images[self.current_page_index].copy()
        self.languages = [language for language in Language]
        self.drawing_mode = 'rectangle'
        self.drawing_modes = ['free', 'rectangle']
        self.initialize_gui()

    def initialize_gui(self):
        self.initialize_frames()
        self.initialize_buttons()
        self.initialize_menus()
        self.initialize_text_field()
        self.initialize_canvas()

    def initialize_frames(self):
        self.load_and_settings_frame = tk.Frame(self.master)
        self.load_and_settings_frame.grid(row=0, column=6)
        self.page_turn_frame = tk.Frame(self.master)
        self.page_turn_frame.grid(row=3, column=6)
        self.text_field_frame = tk.Frame(self.master)
        self.text_field_frame.grid(row=4, column=6)
        self.frame3 = tk.Frame(self.master)
        self.frame3.grid(row=6, column=6)

    def initialize_buttons(self):
        self.button_read_file = ttk.Button(self.load_and_settings_frame, text="Load pdf", command=self.master.load_pdf)
        self.button_read_file.config(width=40)
        self.button_read_file.pack()
        self.button_open_folder = ttk.Button(self.load_and_settings_frame, text="Open Folder",
                                             command=self.master.open_folder_dialog)
        self.button_open_folder.config(width=40)
        self.button_open_folder.pack()
        self.button_rotate = ttk.Button(self.load_and_settings_frame, text="Rotate 90 degrees",
                                        command=self.master.rotate_current_image_by_90_degrees)
        self.button_rotate.config(width=40)
        self.button_rotate.pack()

        self.next_button = ttk.Button(self.page_turn_frame, text="Next Image", command=self.master.next_image)
        self.next_button.config(width=17)
        self.next_button.pack(side="right")
        self.prev_button = ttk.Button(self.page_turn_frame, text="Previous Image", command=self.master.previous_image)
        self.prev_button.config(width=17)
        self.prev_button.pack(side="left")
        self.page_label = tk.Label(self.page_turn_frame)
        self.page_label.config(text=f"{self.current_page_index + 1} / {len(self.images)}")
        self.page_label.pack()
        self.save_button = ttk.Button(self.frame3, text="Save text", command=self.master.save_text_to_file)
        self.save_button.config(width=20)
        self.save_button.pack(side="left")
        self.load_text_button = ttk.Button(self.frame3, text="Load text", command=self.master.load_text_file_dialog)
        self.load_text_button.config(width=20)
        self.load_text_button.pack(side="right")

    def initialize_menus(self):
        self.lang_var = tk.StringVar(self.load_and_settings_frame)
        self.lang_var.set(Language.ENGLISH.value)
        self.lang_menu = ttk.OptionMenu(self.load_and_settings_frame, self.lang_var, Language.ENGLISH.value, *[lang.value for lang in self.languages])
        self.lang_menu.config(width=10)
        self.lang_menu.pack()
        self.drawing_mode_var = tk.StringVar(self.load_and_settings_frame)
        self.drawing_mode_var.set('rectangle')
        self.drawing_mode_var.trace('w', self.update_drawing_mode)
        self.drawing_mode_menu = ttk.OptionMenu(self.load_and_settings_frame, self.drawing_mode_var, 'rectangle', *self.drawing_modes)
        self.drawing_mode_menu.config(width=10)
        self.drawing_mode_menu.pack()
        self.preserve_text_structure_var = tk.IntVar()
        self.preserve_text_structure_var.set(1)
        self.preserve_text_structure_checkbox = ttk.Checkbutton(self.load_and_settings_frame, text="Preserve text structure (csv if not checked)",
                                                                variable=self.preserve_text_structure_var)
        self.preserve_text_structure_checkbox.pack()

    def update_drawing_mode(self, *args):
        # Update the drawing mode based on the value of drawing_mode_var
        self.drawing_mode = self.drawing_mode_var.get()

    def initialize_text_field(self):
        self.text_field = tk.Text(self.text_field_frame, width=30, height=10, wrap="none")
        scrollbar_horizontal = ttk.Scrollbar(self.text_field_frame, orient="horizontal", command=self.text_field.xview)
        scrollbar_horizontal.pack(side="bottom", fill="x")
        scrollbar_vertical = ttk.Scrollbar(self.text_field_frame, orient="vertical", command=self.text_field.yview)
        scrollbar_vertical.pack(side="right", fill="y")
        self.text_field.configure(xscrollcommand=scrollbar_horizontal.set, font=("Helvetica", 9))
        self.text_field.configure(yscrollcommand=scrollbar_vertical.set)
        self.text_field.pack(side="left", fill="both", expand=True)

    def initialize_canvas(self):
        tk_image = ImageTk.PhotoImage(Image.fromarray(self.current_image))
        self.canvas = tk.Canvas(self.master, width=tk_image.width(), height=tk_image.height())
        self.canvas.grid(row=0, column=1, rowspan=8, columnspan=5)
        self.image_id = self.canvas.create_image(0, 0, image=tk_image, anchor="nw")
        self.canvas.bind('<Button-1>', self.master.start_draw)
        self.canvas.bind('<B1-Motion>', self.master.draw)
        self.canvas.bind('<ButtonRelease-1>', self.master.stop_draw)

    def update_image(self, new_image):
        if new_image is None:
            print("Warning: Attempted to update the image with a None image.")
            return
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(new_image))
        self.canvas.delete(self.image_id)
        self.image_id = self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())


def save_to_text_file(text):
    filename = tkinter.filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if filename:
        with open(filename, 'w') as f:
            f.write(text)


class MaskDrawer(tk.Tk):
    def __init__(self, images=[]):
        super().__init__()
        self.current_page_index = 0
        if images is None or len(images) == 0:
            self.images = [np.full((250, 250, 3), 255, np.uint8)]
            self.original_shapes = [image.shape for image in self.images]
            self.resized_images = [self.resize_image(image, 2, 2) for image in self.images]
        else:
            self.images = images
            self.original_shapes = [image.shape for image in self.images]
            self.resized_images = [self.resize_image(image) for image in self.images]
        self.current_image = self.resized_images[self.current_page_index].copy()
        mask = np.zeros((self.current_image.shape[0:2]), np.uint8)
        self.mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        self.drawing = False
        self.ix, self.iy = -1, -1
        self.points = []
        self.rect_id = None
        self.gui = MaskDrawerGUI(self, self.resized_images)
        self.mainloop()

    def resize_image(self, img, height=MAX_HEIGHT, width=MAX_WIDTH):
        # Check if the image is None
        if img is None:
            print("Warning: Attempted to resize a None image.")
            return None

        # Calculate the ratios of the new width and height to the old width and height
        width_ratio = width / float(self.original_shapes[self.current_page_index][1])
        height_ratio = height / float(self.original_shapes[self.current_page_index][0])
        # Choose the smallest ratio
        ratio = min(width_ratio, height_ratio)
        # Calculate the new dimensions
        new_width = int(self.original_shapes[self.current_page_index][1] * ratio)
        new_height = int(self.original_shapes[self.current_page_index][0] * ratio)
        # Resize the image
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def load_pdf(self):
        file_path = tkinter.filedialog.askopenfilename()
        if file_path:
            # The user selected a file, you can now use the file path
            print(f"Selected file: {file_path}")

            # Check if the selected file is a PDF file
            if file_path.endswith('.pdf'):
                # Open the PDF file
                images = convert_from_path(file_path, dpi=400)
                self.images = [np.array(image) for image in images]
                self.original_shapes = [image.shape for image in self.images]
                self.resized_images = [self.resize_image(image) for image in self.images]
                self.current_page_index = 0
                self.current_image = self.resized_images[self.current_page_index].copy()
                self.update_image(self.current_image)
                self.update_page_label()
                self.gui.text_field.config(width=50, height=55)
            else:
                print("Invalid file format. Please select a PDF file.")

    def next_image(self):
        if self.current_page_index < len(self.images) - 1:
            self.current_page_index += 1
            self.current_image = self.resized_images[self.current_page_index].copy()
            self.gui.update_image(self.current_image)
            self.update_page_label()

    def previous_image(self):
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self.current_image = self.resized_images[self.current_page_index].copy()
            self.gui.update_image(self.current_image)
            self.update_page_label()

    def update_page_label(self):
        self.gui.page_label.config(text=f"{self.current_page_index + 1} / {len(self.images)}")

    def save_text_to_file(self):
        text = self.gui.text_field.get("1.0", "end")
        save_to_text_file(text)

    def start_draw(self, event):
        self.drawing = True
        self.ix, self.iy = event.x, event.y
        if self.gui.drawing_mode == 'free':
            self.points.append((event.x, event.y))

    def draw(self, event):
        if self.drawing:
            if self.gui.drawing_mode == 'free':
                self.gui.canvas.create_line(self.ix, self.iy, event.x, event.y, fill="black")
                self.ix, self.iy = event.x, event.y
                self.points.append((event.x, event.y))
            elif self.gui.drawing_mode == 'rectangle':
                if self.rect_id:
                    self.gui.canvas.delete(self.rect_id)
                self.rect_id = self.gui.canvas.create_rectangle(self.ix, self.iy, event.x, event.y, outline='black')

    def stop_draw(self, event):
        width, height = self.current_image.shape[1], self.current_image.shape[0]
        mask = np.zeros((height, width), np.uint8)
        self.drawing = False
        if self.gui.drawing_mode == 'free':
            self.points.append((event.x, event.y))
            pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))
            self.points.clear()
        elif self.gui.drawing_mode == 'rectangle':
            if self.rect_id:
                self.gui.canvas.delete(self.rect_id)
            self.rect_id = self.gui.canvas.create_rectangle(self.ix, self.iy, event.x, event.y, outline='black',
                                                        fill='gray')
            cv2.rectangle(mask, (self.ix, self.iy), (event.x, event.y), (255), thickness=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        print('Mask shape: ',mask.shape)
        print('Current image shape: ', self.current_image.shape)

        weighted_image = cv2.addWeighted(self.current_image, 0.7, mask, 0.3, 0)
        self.update_image(weighted_image)
        mask = self.resize_image(mask, *self.original_shapes[self.current_page_index][0:2])
        cropped_image = crop_masked_image(self.images[self.current_page_index], mask)
        self.extracted_text = analyze(cropped_image, Language(self.gui.lang_var.get()))
        self.copy_text()

    def load_text_file_dialog(self):
        # Open a file dialog for selecting a file
        file_path = tkinter.filedialog.askopenfilename()
        if file_path:
            print(f"Selected file: {file_path}")
            if file_path.endswith('.csv') or file_path.endswith('.txt'):
                # Open the CSV file with a specific encoding
                with open(file_path, 'r', encoding='utf-8') as file:  # Replace 'utf-8' with your desired encoding
                    self.gui.text_field.delete("1.0", tk.END)
                    for line in file:
                        # Insert each line into the text field
                        self.gui.text_field.insert(tk.END, line)

    def open_folder_dialog(self):
        folder_path = tkinter.filedialog.askdirectory()
        if folder_path:
            images = []
            for filename in os.listdir(folder_path):
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        # change color space to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
            if images:
                self.images = images
                self.original_shapes = [image.shape for image in self.images]
                self.resized_images = [self.resize_image(image) for image in self.images]
                self.current_page_index = 0
                self.current_image = self.resized_images[self.current_page_index].copy()
                self.update_image(self.current_image)
                self.update_page_label()
                self.gui.text_field.config(width=50, height=55)
            else:
                print("No valid images found in the selected folder.")

    def update_drawing_mode(self, *args):
        # Update the drawing mode based on the value of drawing_mode_var
        self.drawing_mode = self.gui.drawing_mode_var.get()

    def add_new_line(self, event=None):
        # Delete the last character from text_field2
        self.gui.text_field.delete('end-2c', 'end-1c')
        # Insert a new line into text_field2
        self.gui.text_field.insert(tk.END, '\n')

    def rotate_current_image_by_90_degrees(self):
        # Rotate the current image by 90 degrees
        self.current_image = np.rot90(self.current_image)
        # print(self.current_image.shape)
        self.resized_images[self.current_page_index] = self.current_image.copy()
        # print(self.resized_images[self.current_page_index].shape)
        self.images[self.current_page_index] = np.rot90(self.images[self.current_page_index])
        self.original_shapes[self.current_page_index] = self.images[self.current_page_index].shape
        # print(self.original_shapes[self.current_page_index])
        # self.resized_images[self.current_page_index] = self.resize_image(self.images[self.current_page_index])
        print('resized images current page shape: ' , self.resized_images[self.current_page_index].shape)
        self.update_image(self.current_image)

        # self.gui.update_image(self.current_image)


    def copy_text(self):
        text = self.extracted_text
        text = text.replace('\n\n', '\n')
        if self.gui.preserve_text_structure_var.get() == 0:
            text = replace_multiple_spaces_with_tabs(text)
        self.gui.text_field.insert(tk.END, text)

    def update_image(self, new_image):
        if new_image is None:
            print("Warning: Attempted to update the image with a None image.")
            return
        # Convert the new image to a format that Tkinter can display
        self.gui.tk_image = ImageTk.PhotoImage(Image.fromarray(new_image))
        # Delete the current image from the canvas
        self.gui.canvas.delete(self.gui.image_id)
        # Display the new image on the canvas and update the ID of the image
        self.gui.image_id = self.gui.canvas.create_image(0, 0, image=self.gui.tk_image, anchor="nw")
        self.gui.canvas.config(width=self.gui.tk_image.width(), height=self.gui.tk_image.height())


def main():
    drawer = MaskDrawer()
    drawer.mainloop()

if __name__ == "__main__":
    main()

