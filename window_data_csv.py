from tkinter import *
import tkinter.ttk as ttk
import csv

columns_data = ("No", "Company", "Product", "TypeName", "Inches", "ScreenResolution", 
                "Cpu", "Ram", "Memory", "Gpu", "OpSys", "Weight", "Price_euros")

csv_file_path = 'datasets/laptops_kaggle_com_ionaskel_2.csv'

def initialize():
    root = Tk()
    root.title("Dữ liệu mẫu")
    width = 900
    height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry("%dx%d+%d+%d" % (width, height, x, y))
    root.resizable(0, 0)

    TableMargin = Frame(root, width=500)
    TableMargin.pack(side=TOP)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin, 
                        columns=columns_data, 
                        height=400, 
                        selectmode="extended", 
                        yscrollcommand=scrollbary.set, 
                        xscrollcommand=scrollbarx.set)
    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill=X)

    # set heading text
    for column in columns_data:
        tree.heading(column, text=column, anchor=W)

    #set column size
    for x in range(len(columns_data)):
        tree.column('#' + str(x), stretch=NO, minwidth=10, width=100)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.pack()

    with open(csv_file_path) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            No = row[reader.fieldnames[0]]
            Company = row['Company']
            Product = row['Product'] 
            TypeName = row['TypeName']
            Inches = row['Inches']
            ScreenResolution = row['ScreenResolution'] 
            Cpu = row['Cpu'] 
            Ram = row['Ram'] 
            Memory = row['Memory'] 
            Gpu = row['Gpu'] 
            OpSys = row['OpSys']
            Weight = row['Weight']
            Price_euros = row['Price_euros']
            tree.insert("", END, values=(No,Company, Product, TypeName, Inches, ScreenResolution, 
                                        Cpu, Ram, Memory, Gpu, OpSys, Weight, Price_euros))
    
    root.mainloop()