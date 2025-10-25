\"\"\"
DataPrep Pro - Data Cleaning and Preprocessing Tool
Copyright (c) 2024 Hexatech Solutions Limited
Email: ahmedkansulum@hexatech.ng
GitHub: https://github.com/yourusername/DataPrep-Pro
Licensed under MIT License
\"\"\"

import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.simpledialog import Dialog
from datetime import datetime
import chardet
import warnings
import zipfile
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import webbrowser

warnings.filterwarnings('ignore')


class DownloadDialog(Dialog):
    \"\"\"Custom dialog for download options\"\"\"

    def __init__(self, parent, title, is_merged=False):
        self.is_merged = is_merged
        self.format_var = tk.StringVar(value='csv')
        super().__init__(parent, title=title)

    def body(self, master):
        ttk.Label(master, text=\"Select file format:\").grid(row=0, column=0, sticky=tk.W)

        formats = [('CSV', 'csv'), ('Excel', 'xlsx'), ('JSON', 'json')]
        for i, (text, value) in enumerate(formats, start=1):
            ttk.Radiobutton(master, text=text, variable=self.format_var,
                            value=value).grid(row=i, column=0, sticky=tk.W)

        if not self.is_merged:
            ttk.Label(master, text=\"\nInclude all files in one zip archive?\").grid(row=4, column=0, sticky=tk.W)
            self.zip_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(master, variable=self.zip_var).grid(row=4, column=1, sticky=tk.W)

        return master

    def apply(self):
        self.result = {
            'format': self.format_var.get(),
            'zip': getattr(self, 'zip_var', False)
        }


class SelectableText(tk.Frame):
    \"\"\"A frame containing selectable and copyable text\"\"\"
    
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create text widget with scrollbar
        self.text_widget = tk.Text(self, height=4, wrap=tk.WORD, font=('TkDefaultFont', 9),
                                  relief='solid', borderwidth=1, bg='white')
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Insert text and make it read-only but selectable
        self.text_widget.insert(1.0, text)
        self.text_widget.configure(state='disabled')
        
        # Pack widgets
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add right-click context menu for copy
        self.create_context_menu()

    def create_context_menu(self):
        \"\"\"Create right-click context menu for copy functionality\"\"\"
        self.context_menu = tk.Menu(self.text_widget, tearoff=0)
        self.context_menu.add_command(label=\"Copy\", command=self.copy_text)
        
        # Bind right-click to show context menu
        self.text_widget.bind(\"<Button-3>\", self.show_context_menu)

    def show_context_menu(self, event):
        \"\"\"Show context menu on right-click\"\"\"
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def copy_text(self):
        \"\"\"Copy selected text to clipboard\"\"\"
        try:
            selected_text = self.text_widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.clipboard_clear()
            self.clipboard_append(selected_text)
            messagebox.showinfo(\"Copied\", \"Text copied to clipboard!\")
        except tk.TclError:
            messagebox.showwarning(\"No Selection\", \"Please select some text first.\")


class AboutDialog(Dialog):
    \"\"\"About dialog showing version and license information\"\"\"

    def body(self, master):
        info_text = \"\"\"DataPrep Pro - Data Cleaning and Preprocessing Tool

Version: 2.0.0
Release Date: December 2024
Developed by: Hexatech Solutions Limited

Features:
• Data Cleaning and Preprocessing
• Feature Engineering
• Encoding Techniques
• File Merging Capabilities
• Multiple Export Formats

Licensed to: Hexatech Solutions Limited
All rights reserved.
\"\"\"
        ttk.Label(master, text=info_text, justify=tk.LEFT).pack(padx=10, pady=5)
        
        # Email section with selectable text
        ttk.Label(master, text=\"Contact Email:\", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 2))
        email_frame = SelectableText(master, \"ahmedkansulum@hexatech.ng\", height=2)
        email_frame.pack(fill=tk.X, padx=10, pady=2)
        
        return master

    def buttonbox(self):
        box = ttk.Frame(self)
        ttk.Button(box, text=\"OK\", width=10, command=self.ok, default=tk.ACTIVE).pack(pady=5)
        self.bind(\"<Return>\", self.ok)
        self.bind(\"<Escape>\", self.ok)
        box.pack()


class DonateDialog(Dialog):
    \"\"\"Donate dialog showing PayPal information\"\"\"
    
    def body(self, master):
        donate_text = \"\"\"Support DataPrep Pro Development

If you find this tool helpful and would like to support 
its continued development, please consider making a donation.

Your contribution helps maintain and improve this software.

Thank you for your support!
\"\"\"
        ttk.Label(master, text=donate_text, justify=tk.CENTER).pack(padx=10, pady=5)
        
        # PayPal section with selectable text
        ttk.Label(master, text=\"PayPal Email:\", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 2))
        paypal_frame = SelectableText(master, \"kansulumahmed@paypal.com\", height=2)
        paypal_frame.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(master, text=\"Right-click on the email above to copy it.\", 
                 font=('TkDefaultFont', 8), foreground='gray').pack(pady=2)
        
        return master

    def buttonbox(self):
        box = ttk.Frame(self)
        ttk.Button(box, text=\"Open PayPal\", width=12, command=self.open_paypal).pack(side=tk.LEFT, padx=5)
        ttk.Button(box, text=\"OK\", width=10, command=self.ok, default=tk.ACTIVE).pack(side=tk.LEFT, padx=5)
        self.bind(\"<Return>\", self.ok)
        self.bind(\"<Escape>\", self.ok)
        box.pack()
    
    def open_paypal(self):
        \"\"\"Open PayPal donation link\"\"\"
        webbrowser.open(\"https://www.paypal.com/paypalme/kansulumahmed\")
        self.ok()


class DataPrepProApp:
    def __init__(self, root):
        self.root = root
        self.root.title(\"DataPrep Pro - Data Cleaning and Preprocessing Tool v2.0.0\")
        self.root.geometry(\"950x750\")

        # Create menu bar
        self.create_menu()

        # Variables
        self.files = []
        self.cleaned_files = {}
        self.merged_file = None
        self.file_info = {}
        self.encoders = {}  # Store encoders for each file

        # Create GUI
        self.create_widgets()

    def create_menu(self):
        \"\"\"Create the menu bar with About and Donate menus\"\"\"
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=\"Help\", menu=help_menu)
        help_menu.add_command(label=\"About\", command=self.show_about)
        help_menu.add_separator()
        help_menu.add_command(label=\"Donate\", command=self.show_donate)

    def show_about(self):
        \"\"\"Show about dialog\"\"\"
        AboutDialog(self.root, \"About DataPrep Pro\")

    def show_donate(self):
        \"\"\"Show donate dialog\"\"\"
        DonateDialog(self.root, \"Support Development\")

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=\"10\")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Upload section
        upload_frame = ttk.LabelFrame(main_frame, text=\"File Upload\", padding=\"10\")
        upload_frame.pack(fill=tk.X, pady=5)

        self.file_listbox = tk.Listbox(upload_frame, height=5, selectmode=tk.EXTENDED)
        self.file_listbox.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(upload_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text=\"Add Files\", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=\"Remove Selected\", command=self.remove_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=\"Clear All\", command=self.clear_files).pack(side=tk.LEFT, padx=5)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Cleaning Options Tab
        cleaning_frame = ttk.Frame(notebook, padding=\"10\")
        notebook.add(cleaning_frame, text=\"Data Cleaning\")

        # Feature Engineering Tab
        feature_frame = ttk.Frame(notebook, padding=\"10\")
        notebook.add(feature_frame, text=\"Feature Engineering & Encoding\")

        # Cleaning Options
        self.remove_special_var = tk.BooleanVar(value=True)
        self.remove_na_var = tk.BooleanVar(value=True)
        self.convert_dates_var = tk.BooleanVar(value=True)
        self.merge_files_var = tk.BooleanVar(value=False)
        self.lowercase_vars_var = tk.BooleanVar(value=True)
        self.strip_whitespace_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(cleaning_frame, text=\"Remove Special Characters\", variable=self.remove_special_var).pack(
            anchor=tk.W)
        ttk.Checkbutton(cleaning_frame, text=\"Remove NaN/Null Values\", variable=self.remove_na_var).pack(anchor=tk.W)
        ttk.Checkbutton(cleaning_frame, text=\"Convert Date Formats\", variable=self.convert_dates_var).pack(anchor=tk.W)
        ttk.Checkbutton(cleaning_frame, text=\"Merge Files (on common columns)\", variable=self.merge_files_var).pack(
            anchor=tk.W)
        ttk.Checkbutton(cleaning_frame, text=\"Convert to Lowercase\", variable=self.lowercase_vars_var).pack(anchor=tk.W)
        ttk.Checkbutton(cleaning_frame, text=\"Strip Whitespace\", variable=self.strip_whitespace_var).pack(anchor=tk.W)

        # Feature Engineering Options
        feature_engineering_frame = ttk.LabelFrame(feature_frame, text=\"Feature Engineering\", padding=\"5\")
        feature_engineering_frame.pack(fill=tk.X, pady=5)

        self.create_interaction_var = tk.BooleanVar(value=False)
        self.create_polynomial_var = tk.BooleanVar(value=False)
        self.extract_datetime_var = tk.BooleanVar(value=True)
        self.bin_numerical_var = tk.BooleanVar(value=False)
        self.bins_count = tk.StringVar(value=\"5\")

        ttk.Checkbutton(feature_engineering_frame, text=\"Create Interaction Features\",
                       variable=self.create_interaction_var).pack(anchor=tk.W)
        ttk.Checkbutton(feature_engineering_frame, text=\"Create Polynomial Features (degree 2)\",
                       variable=self.create_polynomial_var).pack(anchor=tk.W)
        ttk.Checkbutton(feature_engineering_frame, text=\"Extract DateTime Features\",
                       variable=self.extract_datetime_var).pack(anchor=tk.W)

        bin_frame = ttk.Frame(feature_engineering_frame)
        bin_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(bin_frame, text=\"Bin Numerical Variables\",
                       variable=self.bin_numerical_var).pack(side=tk.LEFT)
        ttk.Entry(bin_frame, textvariable=self.bins_count, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(bin_frame, text=\"bins\").pack(side=tk.LEFT)

        # Encoding Options
        encoding_frame = ttk.LabelFrame(feature_frame, text=\"Encoding Methods\", padding=\"5\")
        encoding_frame.pack(fill=tk.X, pady=5)

        self.encode_categorical_var = tk.BooleanVar(value=True)
        self.encoding_method = tk.StringVar(value=\"label\")
        self.onehot_threshold = tk.StringVar(value=\"10\")

        ttk.Checkbutton(encoding_frame, text=\"Encode Categorical Variables\",
                       variable=self.encode_categorical_var).pack(anchor=tk.W)

        encoding_method_frame = ttk.Frame(encoding_frame)
        encoding_method_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(encoding_method_frame, text=\"Label Encoding\",
                       variable=self.encoding_method, value=\"label\").pack(side=tk.LEFT)
        ttk.Radiobutton(encoding_method_frame, text=\"One-Hot Encoding\",
                       variable=self.encoding_method, value=\"onehot\").pack(side=tk.LEFT)

        threshold_frame = ttk.Frame(encoding_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text=\"Max categories for One-Hot:\").pack(side=tk.LEFT)
        ttk.Entry(threshold_frame, textvariable=self.onehot_threshold, width=5).pack(side=tk.LEFT, padx=5)

        # Process button
        ttk.Button(main_frame, text=\"Process Files\", command=self.process_files).pack(pady=10)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text=\"Results\", padding=\"10\")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Create text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(text_frame, height=12)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Download section - Raised and more prominent
        download_frame = ttk.LabelFrame(main_frame, text=\"Download Options\", padding=\"10\")
        download_frame.pack(fill=tk.X, pady=(15, 5))

        btn_frame = ttk.Frame(download_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        # Modified buttons with black text and raised appearance
        ttk.Button(btn_frame, text=\"Download Cleaned Files\", command=self.download_cleaned,
                   style='Bold.TButton').pack(side=tk.LEFT, padx=5, expand=True, pady=3)
        ttk.Button(btn_frame, text=\"Download Merged File\", command=self.download_merged,
                   style='Bold.TButton').pack(side=tk.LEFT, padx=5, expand=True, pady=3)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(\"Ready - Licensed to Hexatech Solutions Limited\")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X)
        status_label.configure(background='#f0f0f0')

        # Style configuration for buttons
        style = ttk.Style()
        style.configure('Bold.TButton', foreground='black', font=('Helvetica', 10, 'bold'))
        style.map('Bold.TButton',
                  foreground=[('active', 'black'), ('!active', 'black')],
                  background=[('active', '#f0f0f0'), ('!active', '#f0f0f0')])

    def add_files(self):
        filetypes = (
            ('CSV files', '*.csv'),
            ('Excel files', '*.xls *.xlsx'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        )

        new_files = filedialog.askopenfilenames(
            title=\"Select files\",
            initialdir=os.getcwd(),
            filetypes=filetypes
        )

        for file in new_files:
            if file not in self.files:
                self.files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))

        self.status_var.set(f\"{len(self.files)} file(s) selected - Licensed to Hexatech Solutions Limited\")

    def remove_files(self):
        selected = self.file_listbox.curselection()
        for i in reversed(selected):
            del self.files[i]
            self.file_listbox.delete(i)
        self.status_var.set(f\"{len(self.files)} file(s) remaining - Licensed to Hexatech Solutions Limited\")

    def clear_files(self):
        self.files = []
        self.file_listbox.delete(0, tk.END)
        self.status_var.set(\"Ready - Licensed to Hexatech Solutions Limited\")

    def process_files(self):
        if not self.files:
            messagebox.showerror(\"Error\", \"No files selected for processing\")
            return

        self.cleaned_files = {}
        self.merged_file = None
        self.file_info = {}
        self.encoders = {}
        self.results_text.delete(1.0, tk.END)

        try:
            for file_path in self.files:
                self.status_var.set(f\"Processing {os.path.basename(file_path)}...\")
                self.root.update()

                # Read file
                df, file_info = self.read_file(file_path)

                # Clean data
                cleaned_df, cleaning_report = self.clean_data(df)

                # Apply feature engineering
                if any([self.create_interaction_var.get(), self.create_polynomial_var.get(),
                       self.extract_datetime_var.get(), self.bin_numerical_var.get()]):
                    cleaned_df, feature_report = self.apply_feature_engineering(cleaned_df, file_info)
                    cleaning_report.extend(feature_report)

                # Apply encoding
                if self.encode_categorical_var.get():
                    cleaned_df, encoding_report = self.apply_encoding(cleaned_df, file_info, file_path)
                    cleaning_report.extend(encoding_report)

                # Store cleaned data and info
                self.cleaned_files[file_path] = cleaned_df
                self.file_info[file_path] = file_info

                # Display results
                self.display_file_info(file_path, file_info, cleaning_report)

            # Merge files if requested
            if self.merge_files_var.get() and len(self.cleaned_files) > 1:
                self.merge_files()

            self.status_var.set(\"Processing complete - Licensed to Hexatech Solutions Limited\")
            messagebox.showinfo(\"Success\", \"Files processed successfully\")

        except Exception as e:
            self.status_var.set(\"Error occurred - Licensed to Hexatech Solutions Limited\")
            messagebox.showerror(\"Error\", f\"An error occurred: {str(e)}\")

    def read_file(self, file_path):
        file_info = {
            'filename': os.path.basename(file_path),
            'columns': [],
            'categorical': {'nominal': [], 'ordinal': []},
            'numerical': {'discrete': [], 'continuous': []},
            'date_columns': [],
            'dtypes': {},
            'shape': None
        }

        # Detect file encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']

        # Read file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t', encoding=encoding)
        else:
            # Try reading as CSV by default
            df = pd.read_csv(file_path, encoding=encoding)

        # Store basic info
        file_info['shape'] = df.shape
        file_info['columns'] = list(df.columns)
        file_info['dtypes'] = df.dtypes.to_dict()

        # Analyze columns
        for col in df.columns:
            dtype = str(df[col].dtype)

            # Check for date columns
            if self.convert_dates_var.get():
                if dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                        file_info['date_columns'].append(col)
                        continue
                    except:
                        pass

            # Categorical data (low cardinality)
            if df[col].nunique() < 20 or dtype == 'object':
                # Check if it's ordinal (has inherent order)
                if self.is_ordinal(df[col]):
                    file_info['categorical']['ordinal'].append(col)
                else:
                    file_info['categorical']['nominal'].append(col)
            else:
                # Numerical data
                if np.issubdtype(df[col].dtype, np.integer):
                    file_info['numerical']['discrete'].append(col)
                else:
                    file_info['numerical']['continuous'].append(col)

        return df, file_info

    def is_ordinal(self, series):
        # Simple heuristic to detect ordinal data
        if series.dtype == 'object':
            sample = series.dropna().sample(min(10, len(series)), replace=True)
            sample = sample.str.lower().str.strip()

            # Common ordinal patterns
            ordinal_terms = [
                ['low', 'medium', 'high'],
                ['small', 'medium', 'large'],
                ['poor', 'fair', 'good', 'excellent'],
                ['never', 'rarely', 'sometimes', 'often', 'always'],
                ['first', 'second', 'third'],
                ['beginner', 'intermediate', 'advanced']
            ]

            for terms in ordinal_terms:
                if all(any(term in val for term in terms) for val in sample if pd.notnull(val)):
                    return True

        return False

    def clean_data(self, df):
        original_shape = df.shape
        cleaning_report = []

        # Remove special characters
        if self.remove_special_var.get():
            for col in df.select_dtypes(include=['object']):
                df[col] = df[col].apply(lambda x: ''.join(e for e in str(x) if e.isalnum() or e.isspace())
                if pd.notnull(x) else x)
            cleaning_report.append(\"Removed special characters from text columns\")

        # Convert to lowercase
        if self.lowercase_vars_var.get():
            for col in df.select_dtypes(include=['object']):
                df[col] = df[col].str.lower()
            cleaning_report.append(\"Converted text to lowercase\")

        # Strip whitespace
        if self.strip_whitespace_var.get():
            for col in df.select_dtypes(include=['object']):
                df[col] = df[col].str.strip()
            cleaning_report.append(\"Stripped whitespace from text columns\")

        # Handle missing values
        if self.remove_na_var.get():
            initial_na = df.isna().sum().sum()
            df.dropna(inplace=True)
            final_na = df.isna().sum().sum()
            rows_removed = original_shape[0] - df.shape[0]
            cleaning_report.append(
                f\"Removed {initial_na} NA values ({rows_removed} rows removed)\"
            )

        # Convert date formats to consistent format
        if self.convert_dates_var.get():
            for col in df.select_dtypes(include=['datetime64']):
                df[col] = df[col].dt.strftime('%Y-%m-%d')
                cleaning_report.append(f\"Standardized date format in column '{col}'\")

        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)

        cleaning_report.append(
            f\"Final shape: {df.shape[0]} rows, {df.shape[1]} columns \"
            f\"(originally {original_shape[0]} rows, {original_shape[1]} columns)\"
        )

        return df, cleaning_report

    def apply_feature_engineering(self, df, file_info):
        \"\"\"Apply feature engineering techniques\"\"\"
        feature_report = [\"\nFeature Engineering:\"]
        original_columns = set(df.columns)

        try:
            # Create interaction features
            if self.create_interaction_var.get():
                numerical_cols = file_info['numerical']['discrete'] + file_info['numerical']['continuous']
                if len(numerical_cols) >= 2:
                    for i in range(len(numerical_cols)):
                        for j in range(i+1, len(numerical_cols)):
                            col1, col2 = numerical_cols[i], numerical_cols[j]
                            if col1 in df.columns and col2 in df.columns:
                                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    feature_report.append(\"- Created interaction features for numerical variables\")

            # Create polynomial features
            if self.create_polynomial_var.get():
                numerical_cols = file_info['numerical']['continuous']
                for col in numerical_cols:
                    if col in df.columns:
                        df[f'{col}_squared'] = df[col] ** 2
                feature_report.append(\"- Created polynomial features (squared) for continuous variables\")

            # Extract datetime features
            if self.extract_datetime_var.get():
                for col in file_info['date_columns']:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                            df[f'{col}_year'] = df[col].dt.year
                            df[f'{col}_month'] = df[col].dt.month
                            df[f'{col}_day'] = df[col].dt.day
                            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                            feature_report.append(f\"- Extracted datetime features from {col}\")
                        except:
                            pass

            # Bin numerical variables
            if self.bin_numerical_var.get():
                try:
                    bins = int(self.bins_count.get())
                    numerical_cols = file_info['numerical']['continuous']
                    for col in numerical_cols:
                        if col in df.columns and df[col].nunique() > bins:
                            df[f'{col}_binned'] = pd.cut(df[col], bins=bins, labels=False)
                            feature_report.append(f\"- Binned {col} into {bins} categories\")
                except ValueError:
                    feature_report.append(\"- Invalid bin count, skipping binning\")

            new_columns = set(df.columns) - original_columns
            if new_columns:
                feature_report.append(f\"Added {len(new_columns)} new features: {', '.join(new_columns)}\")
            else:
                feature_report.append(\"No new features added\")

        except Exception as e:
            feature_report.append(f\"- Error in feature engineering: {str(e)}\")

        return df, feature_report

    def apply_encoding(self, df, file_info, file_path):
        \"\"\"Apply encoding to categorical variables\"\"\"
        encoding_report = [\"\nEncoding:\"]
        self.encoders[file_path] = {}

        try:
            categorical_cols = file_info['categorical']['nominal'] + file_info['categorical']['ordinal']

            for col in categorical_cols:
                if col in df.columns:
                    if self.encoding_method.get() == \"label\":
                        # Label Encoding
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        self.encoders[file_path][col] = le
                        encoding_report.append(f\"- Applied Label Encoding to {col}\")

                    elif self.encoding_method.get() == \"onehot\":
                        # One-Hot Encoding (only if categories <= threshold)
                        try:
                            threshold = int(self.onehot_threshold.get())
                            if df[col].nunique() <= threshold:
                                dummies = pd.get_dummies(df[col], prefix=col)
                                df = pd.concat([df, dummies], axis=1)
                                df.drop(col, axis=1, inplace=True)
                                encoding_report.append(f\"- Applied One-Hot Encoding to {col} ({df[col].nunique()} categories)\")
                            else:
                                # Fall back to label encoding if too many categories
                                le = LabelEncoder()
                                df[col] = le.fit_transform(df[col].astype(str))
                                self.encoders[file_path][col] = le
                                encoding_report.append(f\"- Applied Label Encoding to {col} (too many categories for One-Hot)\")
                        except ValueError:
                            encoding_report.append(f\"- Invalid threshold for {col}, using Label Encoding\")
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            self.encoders[file_path][col] = le

            encoding_report.append(f\"Encoded {len(categorical_cols)} categorical variables\")

        except Exception as e:
            encoding_report.append(f\"- Error in encoding: {str(e)}\")

        return df, encoding_report

    def merge_files(self):
        try:
            # Find common columns
            common_cols = None
            for file_info in self.file_info.values():
                if common_cols is None:
                    common_cols = set(file_info['columns'])
                else:
                    common_cols = common_cols.intersection(set(file_info['columns']))

            if not common_cols:
                self.results_text.insert(tk.END, \"\n\nNo common columns found for merging\n\")
                return

            # Merge all cleaned files on common columns
            merged_df = None
            for i, (file_path, df) in enumerate(self.cleaned_files.items()):
                if i == 0:
                    merged_df = df.copy()
                else:
                    merged_df = pd.merge(merged_df, df, on=list(common_cols), how='outer')

            self.merged_file = merged_df
            self.results_text.insert(
                tk.END,
                f\"\n\nMerged file created with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns \"
                f\"using common columns: {', '.join(common_cols)}\n\"
            )
        except Exception as e:
            self.results_text.insert(tk.END, f\"\n\nError merging files: {str(e)}\n\")

    def display_file_info(self, file_path, file_info, cleaning_report):
        filename = file_info['filename']
        self.results_text.insert(tk.END, f\"\n=== File: {filename} ===\n\")
        self.results_text.insert(tk.END,
                                 f\"Original shape: {file_info['shape'][0]} rows, {file_info['shape'][1]} columns\n\n\")

        # Display column types
        self.results_text.insert(tk.END, \"Column Types:\n\")
        self.results_text.insert(tk.END, f\"- Nominal Categorical: {', '.join(file_info['categorical']['nominal'])}\n\")
        self.results_text.insert(tk.END, f\"- Ordinal Categorical: {', '.join(file_info['categorical']['ordinal'])}\n\")
        self.results_text.insert(tk.END, f\"- Discrete Numerical: {', '.join(file_info['numerical']['discrete'])}\n\")
        self.results_text.insert(tk.END, f\"- Continuous Numerical: {', '.join(file_info['numerical']['continuous'])}\n\")
        if file_info['date_columns']:
            self.results_text.insert(tk.END, f\"- Date Columns: {', '.join(file_info['date_columns'])}\n\")

        # Display cleaning report
        self.results_text.insert(tk.END, \"\nProcessing Report:\n\")
        for line in cleaning_report:
            self.results_text.insert(tk.END, f\"- {line}\n\")

        self.results_text.insert(tk.END, \"\n\")

    def download_cleaned(self):
        if not self.cleaned_files:
            messagebox.showerror(\"Error\", \"No cleaned files available to download\")
            return

        # Show format selection dialog
        dialog = DownloadDialog(self.root, \"Download Cleaned Files\")
        if not dialog.result:
            return

        selected_format = dialog.result['format']
        zip_files = dialog.result.get('zip', False)

        if zip_files and len(self.cleaned_files) > 1:
            self._download_as_zip(selected_format)
        else:
            self._download_individual_files(selected_format)

    def _download_as_zip(self, file_format):
        dir_path = filedialog.askdirectory(title=\"Select download directory\")
        if not dir_path:
            return

        zip_filename = os.path.join(dir_path, \"cleaned_files.zip\")

        try:
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for file_path, df in self.cleaned_files.items():
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]
                    ext = self._get_extension(file_format)
                    output_filename = f\"cleaned_{base_name}{ext}\"

                    # Create temporary file
                    temp_path = os.path.join(dir_path, output_filename)
                    self._save_dataframe(df, temp_path, file_format)

                    # Add to zip and remove temp file
                    zipf.write(temp_path, arcname=output_filename)
                    os.remove(temp_path)

            messagebox.showinfo(\"Success\", f\"All cleaned files saved as {zip_filename}\")
        except Exception as e:
            messagebox.showerror(\"Error\", f\"Failed to create zip file: {str(e)}\")

    def _download_individual_files(self, file_format):
        dir_path = filedialog.askdirectory(title=\"Select download directory\")
        if not dir_path:
            return

        try:
            ext = self._get_extension(file_format)
            for file_path, df in self.cleaned_files.items():
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(dir_path, f\"cleaned_{base_name}{ext}\")

                self._save_dataframe(df, save_path, file_format)

            messagebox.showinfo(\"Success\", f\"Saved {len(self.cleaned_files)} cleaned file(s) to {dir_path}\")
        except Exception as e:
            messagebox.showerror(\"Error\", f\"Failed to save files: {str(e)}\")

    def download_merged(self):
        if self.merged_file is None:
            messagebox.showerror(\"Error\", \"No merged file available to download\")
            return

        # Show format selection dialog
        dialog = DownloadDialog(self.root, \"Download Merged File\", is_merged=True)
        if not dialog.result:
            return

        selected_format = dialog.result['format']

        default_name = f\"merged_cleaned.{selected_format}\"
        file_types = self._get_file_types(selected_format)

        file_path = filedialog.asksaveasfilename(
            title=\"Save merged file\",
            initialfile=default_name,
            filetypes=file_types
        )

        if not file_path:
            return

        # Ensure proper extension
        ext = self._get_extension(selected_format)
        if not file_path.lower().endswith(ext):
            file_path += ext

        try:
            self._save_dataframe(self.merged_file, file_path, selected_format)
            messagebox.showinfo(\"Success\", f\"Merged file saved to {file_path}\")
        except Exception as e:
            messagebox.showerror(\"Error\", f\"Failed to save merged file: {str(e)}\")

    def _save_dataframe(self, df, file_path, file_format):
        \"\"\"Save dataframe to specified path and format\"\"\"
        if file_format == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format == 'xlsx':
            df.to_excel(file_path, index=False)
        elif file_format == 'json':
            df.to_json(file_path, orient='records', indent=2)

    def _get_extension(self, file_format):
        \"\"\"Get proper file extension for the selected format\"\"\"
        return {
            'csv': '.csv',
            'xlsx': '.xlsx',
            'json': '.json'
        }[file_format]

    def _get_file_types(self, file_format):
        \"\"\"Get file types for save dialog based on format\"\"\"
        return {
            'csv': [(\"CSV files\", \"*.csv\")],
            'xlsx': [(\"Excel files\", \"*.xlsx\")],
            'json': [(\"JSON files\", \"*.json\")]
        }[file_format]


def main():
    \"\"\"Main application entry point\"\"\"
    root = tk.Tk()
    app = DataPrepProApp(root)
    root.mainloop()


if __name__ == \"__main__\":
    main()
