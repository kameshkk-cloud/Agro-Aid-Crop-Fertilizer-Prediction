import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os, sys, subprocess
import difflib

# ================= LOAD MODELS =================

def ensure_models():
    required = {
        'ensemble_crop_model.pkl': 'Crop_training_model.py',
        'crop_label_encoders.pkl': 'Crop_training_model.py',
        'fertilizer_model.pkl': 'Fertilizer_training_model.py',
        'fertilizer_label_encoders.pkl': 'Fertilizer_training_model.py'
    }

    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        root = tk.Tk()
        root.withdraw()
        msg = (
            "Missing model files:\n" + "\n".join(missing) +
            "\n\nWould you like to try to create them by running the training scripts?"
        )
        if messagebox.askyesno("Missing model files", msg):
            train_scripts = set(required[file] for file in missing)
            for script in train_scripts:
                try:
                    subprocess.check_call([sys.executable, script])
                except Exception as e:
                    messagebox.showerror("Training error", f"Failed to run {script}: {e}")
                    sys.exit(1)

            missing = [f for f in required if not os.path.exists(f)]
            if missing:
                messagebox.showerror("Error", "Still missing: " + ", ".join(missing))
                sys.exit(1)
        else:
            messagebox.showerror("Missing files", "Please run the training scripts to create missing files.")
            sys.exit(1)


ensure_models()

crop_model = joblib.load('ensemble_crop_model.pkl')
crop_label_encoders = joblib.load('crop_label_encoders.pkl')
fertilizer_model = joblib.load('fertilizer_model.pkl')
fertilizer_label_encoders = joblib.load('fertilizer_label_encoders.pkl')


# ================= GUI CLASS =================
class AgroAidGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Agro Aid - Crop & Fertilizer Recommendation")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')

        self.bot = AgroAidBot()
        self.bot.display_message = self.display_bot_message
        self.bot.gui = self            # ⭐ IMPORTANT LINK

        self.setup_gui()

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#1a1a1a')
        style.configure('Dark.TButton', background='#444444', foreground='white', padding=8)
        style.map('Dark.TButton', background=[('active', '#666666')])
        style.configure('Input.TEntry', fieldbackground='#333333',
                        foreground='white', insertcolor='white')

        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.chat_display = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=70, height=20,
            font=('Arial', 11), bg='#1a1a1a',
            fg='white', insertbackground='white'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.tag_configure('bot', foreground='#008cff', font=('Arial', 11, 'bold'))
        self.chat_display.tag_configure('user', foreground='#ffffff')

        input_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.input_field = ttk.Entry(input_frame, font=('Arial', 11), style='Input.TEntry')
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_field.bind('<Return>', lambda e: self.send_message())

        send_button = ttk.Button(input_frame, text="Send",
                                 style='Dark.TButton', command=self.send_message)
        send_button.pack(side=tk.RIGHT)

        # Quick-select area shown below the input for categorical choices (Soil_Type, Crop)
        self.quick_select_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        self.quick_select_frame.pack(fill=tk.X, pady=(8, 0))
        self.quick_options_visible = False

        options_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        options_frame.pack(fill=tk.X)

        options = [("Crop Prediction", "1"), ("Fertilizer Recommendation", "2")]
        for text, value in options:
            ttk.Button(
                options_frame, text=text, style='Dark.TButton',
                command=lambda v=value: self.quick_option(v)
            ).pack(side=tk.LEFT, padx=5)

        self.display_bot_message("Hello! I am Agro Aid. How can I help you today?\n")
        self.display_bot_message(
            "Please choose an option:\n1) Crop prediction\n2) Fertilizer recommendation\nType 'quit' to exit"
        )

    # ========== GRAPH FUNCTION (STEP 2) ==========
    def show_graph(self, data, title, position='center'):
        # Position the new graph relative to the main window
        self.root.update_idletasks()
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_w = self.root.winfo_width()
        win_w = 600
        win_h = 400

        if position == 'left':
            x = root_x + 50
            y = root_y + 50
        elif position == 'right':
            x = root_x + 50 + win_w + 50
            y = root_y + 50
        else:
            x = root_x + max(0, (root_w - win_w) // 2)
            y = root_y + 50

        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry(f"{win_w}x{win_h}+{x}+{y}")

        labels = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color='#008cff')
        ax.set_title(title, color='white')

        # Friendly parameter names with units when known
        param_label_map = {
            'Temperature': 'Temperature (°C)',
            'Humidity': 'Humidity (%)',
            'Moisture': 'Moisture (%)',
            'Nitrogen': 'Nitrogen Level',
            'Phosphorus': 'Phosphorus Level',
            'Potassium': 'Potassium Level',
            'pH_Value': 'pH Value',
            'Rainfall': 'Rainfall (mm)'
        }

        # Customize axis labels per-chart according to user request
        if 'Prediction Probabilities' in title or '%' in title or 'Probabilities' in title:
            # Probabilities charts: X = class names, Y = percent
            if 'Crop' in title:
                xlabel = 'Crop Types'
            elif 'Fertilizer' in title:
                xlabel = 'Fertilizer Types'
            else:
                xlabel = 'Classes'
            ylabel = 'Prediction Probability (%)'
            display_labels = labels
        elif title == 'Crop Input Analysis':
            xlabel = 'Soil & Environmental Parameters'
            ylabel = 'Parameter Value (Units)'
            display_labels = [param_label_map.get(lbl, lbl) for lbl in labels]
        elif title == 'Fertilizer Decision Factors':
            xlabel = 'Nutrient & Soil Factors'
            ylabel = 'Nutrient Value (kg/ha or %)'
            display_labels = [param_label_map.get(lbl, lbl) for lbl in labels]
        else:
            xlabel = 'Parameters'
            ylabel = 'Measured Value'
            display_labels = [param_label_map.get(lbl, lbl) for lbl in labels]

        ax.set_xlabel(xlabel, color='white', labelpad=25)
        ax.set_ylabel(ylabel, color='white')

        # Ensure tick positions match labels to avoid matplotlib warnings
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(display_labels, rotation=30, ha='right', color='white')
        ax.tick_params(colors='white')
        # Match the GUI dark background for better contrast
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')

        # Annotate bars with their exact values for clarity
        try:
            max_val = max(values) if len(values) else 0
        except Exception:
            max_val = 0
        y_offset = max_val * 0.02 if max_val else 0.5
        for bar, val in zip(bars, values):
            # Format value: integer if close to int, else 2 decimal places
            try:
                fval = float(val)
                if abs(fval - round(fval)) < 1e-6:
                    label_text = f"{int(round(fval))}"
                else:
                    label_text = f"{fval:.2f}"
            except Exception:
                label_text = str(val)

            # Append percent sign for percentage charts
            if '%' in title or 'Percent' in title:
                label_text = f"{label_text}%"

            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset, label_text,
                    ha='center', color='white', fontsize=10, weight='bold')

        canvas = FigureCanvasTkAgg(fig, master=window)
        # Use tight_layout to avoid x-label clipping where possible
        try:
            fig.tight_layout()
        except Exception:
            pass
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True) 

    def display_bot_message(self, message):
        self.chat_display.insert(tk.END, f"🤖 Bot: {message}\n", 'bot')
        self.chat_display.see(tk.END)

    def display_user_message(self, message):
        self.chat_display.insert(tk.END, f"👤 You: {message}\n", 'user')
        self.chat_display.see(tk.END)

    def send_message(self):
        message = self.input_field.get().strip()
        if message:
            self.input_field.delete(0, tk.END)
            self.display_user_message(message)
            if message.lower() == 'quit':
                self.display_bot_message("Goodbye! 👋")
                self.root.after(1000, self.root.destroy)
                return
            self.bot.process_input(message)

    def quick_option(self, option):
        self.input_field.delete(0, tk.END)
        self.input_field.insert(0, option)
        self.send_message()

    def run(self):
        self.root.mainloop()

    # ---------- Inline quick-select helpers (GUI methods) ----------
    def show_quick_options(self, key):
        # Clear existing
        for child in self.quick_select_frame.winfo_children():
            child.destroy()

        options = []
        label_text = key
        if key == 'Soil_Type':
            options = list(fertilizer_label_encoders['Soil_Type'].classes_)
            label_text = 'Soil Types:'
        elif key == 'Crop':
            options = list(fertilizer_label_encoders['Crop'].classes_)
            label_text = 'Crops:'

        # Title label
        lbl = tk.Label(self.quick_select_frame, text=label_text, bg='#1a1a1a', fg='white')
        lbl.pack(anchor='w', padx=5)

        btn_frame = ttk.Frame(self.quick_select_frame, style='Dark.TFrame')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create buttons for options
        for opt in options:
            b = tk.Button(btn_frame, text=opt, bg='#333333', fg='white', activebackground='#555555',
                          relief=tk.RAISED, bd=1, command=lambda o=opt: self.select_quick_option(o))
            b.pack(side=tk.LEFT, padx=4, pady=2)

        # Also print options in the chat for visibility (in case buttons aren't visible)
        if options:
            try:
                self.display_bot_message('Options: ' + ', '.join(options))
            except Exception:
                pass

        self.quick_options_visible = True

    def hide_quick_options(self):
        for child in self.quick_select_frame.winfo_children():
            child.destroy()
        self.quick_options_visible = False

    def select_quick_option(self, option):
        # Insert into input and submit
        self.input_field.delete(0, tk.END)
        self.input_field.insert(0, option)
        # Auto-send selection
        self.send_message()


# ================= BOT CLASS =================
class AgroAidBot:
    def __init__(self):
        self.state = 'main_menu'
        self.inputs = {}
        self.step_index = 0
        self.display_message = print
        self.gui = None

        self.crop_steps = [
            'Nitrogen', 'Phosphorus', 'Potassium',
            'Temperature', 'Humidity', 'pH_Value', 'Rainfall'
        ]

        self.ferti_steps = [
            'Temperature', 'Humidity', 'Moisture',
            'Soil_Type', 'Crop', 'Nitrogen', 'Potassium', 'Phosphorus'
        ]

    def process_input(self, message):
        if self.state == 'main_menu':
            if message == '1':
                self.state = 'crop_prediction'
                self.step_index = 0
                self.inputs.clear()
                self.display_message("You've selected Crop Prediction.")
                self.display_message(f"Please enter {self.crop_steps[0]}:")
            elif message == '2':
                self.state = 'fertilizer_start'
                self.step_index = 0
                self.inputs.clear()
                self.display_message("You've selected Fertilizer Recommendation.")
                self.display_message(f"Please enter {self.ferti_steps[0]}:")
            else:
                self.display_message("❌ Invalid option. Choose 1 or 2.")

        elif self.state == 'crop_prediction':
            key = self.crop_steps[self.step_index]
            self.inputs[key] = message
            self.step_index += 1
            if self.step_index < len(self.crop_steps):
                self.display_message(f"Please enter {self.crop_steps[self.step_index]}:")
            else:
                self.run_crop_prediction()
                self.state = 'main_menu'

        elif self.state == 'fertilizer_start':
            key = self.ferti_steps[self.step_index]
            # For categorical steps, show inline quick-select options first; accept typed input too
            if key in ('Soil_Type', 'Crop'):
                if not self.gui.quick_options_visible:
                    # Show options and ask user to pick (or type and press Send)
                    self.gui.show_quick_options(key)
                    self.display_message(f"Please select {key} from the list below or type and press Send:")
                    return
                else:
                    # The message contains the user's choice (from button or typed)
                    self.inputs[key] = message
                    self.gui.hide_quick_options()
            else:
                self.inputs[key] = message

            self.step_index += 1
            if self.step_index < len(self.ferti_steps):
                next_key = self.ferti_steps[self.step_index]
                if next_key in ('Soil_Type', 'Crop'):
                    # Show quick options immediately when prompting for categorical input
                    self.gui.show_quick_options(next_key)
                    self.display_message(f"Please select {next_key} from the list below or type and press Send:")
                else:
                    self.display_message(f"Please enter {next_key}:")
            else:
                self.run_fertilizer_prediction()
                self.state = 'main_menu'

    def prompt_selection(self, key):
        # Keep modal selection for backward compatibility (not used by inline flow)
        options = []
        title = key
        if key == 'Soil_Type':
            options = list(fertilizer_label_encoders['Soil_Type'].classes_)
            title = 'Select Soil Type'
        elif key == 'Crop':
            options = list(fertilizer_label_encoders['Crop'].classes_)
            title = 'Select Crop'
        else:
            return None

        win = tk.Toplevel(self.gui.root)
        win.title(title)
        win.transient(self.gui.root)
        win.grab_set()

        ttk.Label(win, text=f"Please select {key}:").pack(padx=10, pady=10)
        combo = ttk.Combobox(win, values=options, state='readonly')
        combo.pack(padx=10, pady=5)
        if options:
            combo.current(0)

        selected = {'value': None}
        def on_ok():
            selected['value'] = combo.get()
            win.destroy()
        def on_cancel():
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text='OK', command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='Cancel', command=on_cancel).pack(side=tk.LEFT, padx=5)

        self.gui.root.wait_window(win)
        return selected['value']


    def select_quick_option(self, option):
        # Insert into input and submit
        self.input_field.delete(0, tk.END)
        self.input_field.insert(0, option)
        # Auto-send selection
        self.send_message()

    def run_crop_prediction(self):
        try:
            for key in self.crop_steps:
                self.inputs[key] = float(self.inputs[key])

            df = pd.DataFrame([self.inputs])
            pred = crop_model.predict(df)[0]

            # Correctly map outputs: [Crop, Soil_Type, Variety]
            crop_pred = crop_label_encoders['Crop'].inverse_transform([pred[0]])[0]
            soil = crop_label_encoders['Soil_Type'].inverse_transform([pred[1]])[0]
            variety = crop_label_encoders['Variety'].inverse_transform([pred[2]])[0]

            self.display_message(f"🌾 Recommended Crop: {crop_pred}")
            self.display_message(f"✅ Suitable Soil Type: {soil}")
            self.display_message(f"🌾 Recommended Crop Variety: {variety}")

            # Show input factors graph
            graph_data = {
                "Nitrogen": self.inputs["Nitrogen"],
                "Phosphorus": self.inputs["Phosphorus"],
                "Potassium": self.inputs["Potassium"],
                "Temperature": self.inputs["Temperature"],
                "Humidity": self.inputs["Humidity"],
                "Rainfall": self.inputs["Rainfall"]
            }
            self.gui.show_graph(graph_data, "Crop Input Analysis", position='left')

            # Show crop probability percentages (model confidence)
            try:
                prob_list = crop_model.predict_proba(df)
                # prob_list[0] -> array shape (n_samples, n_classes) for 'Crop' output
                crop_probs = prob_list[0][0]
                crop_encoder = crop_label_encoders['Crop']
                classes = list(range(len(crop_encoder.classes_)))
                class_names = crop_encoder.inverse_transform(classes)
                crop_percentages = {name: round(float(prob) * 100, 2) for name, prob in zip(class_names, crop_probs)}
                self.gui.show_graph(crop_percentages, "Crop Prediction Probabilities (%)", position='right')
            except Exception:
                # If predict_proba not supported for some reason, skip gracefully
                pass

        except Exception as e:
            self.display_message(f"❌ Error: {e}")

    def run_fertilizer_prediction(self):
        try:
            for key in ['Temperature', 'Humidity', 'Moisture',
                        'Nitrogen', 'Potassium', 'Phosphorus']:
                self.inputs[key] = float(self.inputs[key])

            # Validate Soil_Type input against known encoder classes
            soil = self.inputs['Soil_Type']
            soil_encoder = fertilizer_label_encoders['Soil_Type']
            if soil not in soil_encoder.classes_:
                suggestion = difflib.get_close_matches(soil, soil_encoder.classes_, n=1, cutoff=0.6)
                if suggestion:
                    self.display_message(f"❌ Unknown Soil_Type '{soil}'. Did you mean '{suggestion[0]}'? Allowed: {', '.join(soil_encoder.classes_)}")
                else:
                    self.display_message(f"❌ Unknown Soil_Type '{soil}'. Allowed values: {', '.join(soil_encoder.classes_)}")
                return
            soil_encoded = soil_encoder.transform([soil])[0]

            # Validate Crop input against known encoder classes
            crop = self.inputs['Crop']
            crop_encoder = fertilizer_label_encoders['Crop']
            if crop not in crop_encoder.classes_:
                suggestion = difflib.get_close_matches(crop, crop_encoder.classes_, n=1, cutoff=0.6)
                if suggestion:
                    self.display_message(f"❌ Unknown Crop '{crop}'. Did you mean '{suggestion[0]}'? Allowed: {', '.join(crop_encoder.classes_)}")
                else:
                    self.display_message(f"❌ Unknown Crop '{crop}'. Allowed values: {', '.join(crop_encoder.classes_)}")
                return
            crop_encoded = crop_encoder.transform([crop])[0]

            fert_input = [
                int(self.inputs['Temperature']),
                int(self.inputs['Humidity']),
                int(self.inputs['Moisture']),
                soil_encoded,
                crop_encoded,
                int(self.inputs['Nitrogen']),
                int(self.inputs['Potassium']),
                int(self.inputs['Phosphorus'])
            ]

            # Use DataFrame with column names to avoid sklearn warning about feature names
            cols = ['Temperature','Humidity','Moisture','Soil_Type','Crop','Nitrogen','Potassium','Phosphorus']
            fert_df = pd.DataFrame([fert_input], columns=cols)

            result = fertilizer_model.predict(fert_df)[0]
            fert_name = fertilizer_label_encoders['FertilizerName'].inverse_transform([result])[0]

            self.display_message(f"💡 Recommended Fertilizer: {fert_name}")

            # Echo the numeric inputs used so users can verify values match the graph
            try:
                nit = int(round(self.inputs['Nitrogen']))
                phos = int(round(self.inputs['Phosphorus']))
                pot = int(round(self.inputs['Potassium']))
                moist = int(round(self.inputs['Moisture']))
                self.display_message(f"🔢 Inputs used — Nitrogen: {nit}, Phosphorus: {phos}, Potassium: {pot}, Moisture: {moist}")
            except Exception:
                # Fallback: print raw dictionary
                self.display_message(f"🔢 Inputs used — { {k: self.inputs[k] for k in ['Nitrogen','Phosphorus','Potassium','Moisture'] if k in self.inputs} }")

            graph_data = {
                "Nitrogen": self.inputs["Nitrogen"],
                "Phosphorus": self.inputs["Phosphorus"],
                "Potassium": self.inputs["Potassium"],
                "Moisture": self.inputs["Moisture"]
            }
            # Show decision factors on the left
            self.gui.show_graph(graph_data, "Fertilizer Decision Factors", position='left')

            # Show fertilizer prediction probabilities on the right (if model supports predict_proba)
            try:
                probs = fertilizer_model.predict_proba(fert_df)[0]
                fert_encoder = fertilizer_label_encoders['FertilizerName']
                class_names = list(fert_encoder.classes_)
                fert_percentages = {name: round(float(p) * 100, 2) for name, p in zip(class_names, probs)}
                # Place probability chart to the right
                self.gui.show_graph(fert_percentages, "Fertilizer Prediction Probabilities (%)", position='right')
            except Exception:
                # If predict_proba not available, skip
                pass

        except Exception as e:
            self.display_message(f"❌ Error: {e}")


# ================= RUN APP =================
if __name__ == '__main__':
    try:
        AgroAidGUI().run()
    except KeyboardInterrupt:
        # Gracefully handle user interrupt from the terminal (Ctrl+C)
        print("\nInterrupted by user. Exiting.")
        try:
            import sys
            sys.exit(0)
        except SystemExit:
            pass
