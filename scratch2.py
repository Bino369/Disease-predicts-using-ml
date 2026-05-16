import tkinter as tk
from tkinter import ttk

def check(event):
    value = event.widget.get()
    if value == '':
        event.widget['values'] = l
    else:
        data = []
        for item in l:
            if value.lower() in item.lower():
                data.append(item)
        event.widget['values'] = data

root = tk.Tk()
l = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape']
cb = ttk.Combobox(root)
cb['values'] = l
cb.bind('<KeyRelease>', check)
cb.pack()
# root.mainloop() # don't run mainloop headless
print("Combobox code is valid")
