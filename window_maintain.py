#Write code use Tkinter
# Library: tkinter
from tkinter import *
from tkinter import PhotoImage, messagebox
import webbrowser
import window_data_csv
import random

#region Properties
default_height_form = 600
default_width_form = 900
geometry = str(default_width_form) + "x" + str(default_height_form)
foreground_color = 'darkblue'
font_main = 'Consolas 14 bold'
columns_data = ("No", "Company", "Product", "Type Name", "Inches", "Screen Resolution", 
                "Cpu", "Ram", "Memory", "Gpu", "OpSys", "Weight", "Price_euros")
#endregion

#region Menu_Controllers
def show_data_sample():
    window_data_csv.initialize()

def show_infor_group():
    information =  '19110363 - Đàm Lưu Trung Hiếu\n'
    information += '19110365 - Phạm Đinh Quốc Hòa\n'
    information += '19110386 - Nguyễn Tấn Kiệt\n'
    information += '19110396 - Nguyễn Đức Mạnh\n'
    information += '19110410 - Nguyễn Trần Lê Nguyên\n'
    information += '19110426 - Võ Nhật Phi\n'
    information += '19110447 - Nguyễn Quang Sang\n'
    information += '19110041 - Nguyễn Văn Thành\n'
    messagebox.showinfo('Danh sách thành viên', information)

def show_infor_data():
    webbrowser.open('https://www.kaggle.com/ionaskel/laptop-prices')
#endregion

def initialize():
    root = Tk()
    root.title('Ứng dụng dự đoán giá laptop')
    root.iconphoto(False, PhotoImage(file='images/machine-learning.png'))
    root.geometry(geometry)
    root.resizable(0, 0)   
    
    #region Menu
    menuBar = Menu(root)
    
    fileMenu = Menu(menuBar, tearoff=0)
    fileMenu.add_command(label="Mở file dữ liệu", command=show_data_sample)
    fileMenu.add_separator()
    fileMenu.add_command(label="Thoát", command=root.quit)
    menuBar.add_cascade(label="File", menu=fileMenu)

    helpMenu = Menu(menuBar, tearoff=0)
    helpMenu.add_command(label="Thông tin nhóm", command=show_infor_group)
    helpMenu.add_command(label="Thông tin dữ liệu", command=show_infor_data)
    menuBar.add_cascade(label="Help", menu=helpMenu)
    #endregion

    #region Components
    index = 0
    for column in columns_data:
        lbl = Label(root, text=column,
                    foreground=foreground_color,
                    font=font_main)
        lbl.grid(row=index, column=0, sticky="w") # west east south north
        index += 1
    
    txtNo = Entry(root, font=font_main)
    txtCompany = Entry(root, font=font_main)
    txtProduct = Entry(root, font=font_main)
    txtTypeName = Entry(root, font=font_main)
    txtInches = Entry(root, font=font_main)
    txtScreen = Entry(root, font=font_main)
    txtCPU = Entry(root, font=font_main)
    txtRam = Entry(root, font=font_main)
    txtMemory = Entry(root, font=font_main)
    txtGPU = Entry(root, font=font_main)
    txtOpSys = Entry(root, font=font_main)
    txtWeight = Entry(root, font=font_main)   
    txtPrice = Entry(root, font=font_main)

    list_Entry = [txtNo, txtCompany, txtProduct, txtTypeName, txtInches,
                txtScreen, txtCPU, txtRam, txtMemory, txtGPU, txtOpSys,
                txtWeight, txtPrice]

    #region Events
    def ClearContent():
        for item in list_Entry:
            item.delete(0, END)

    def Predict_value():
        try:
            a = random.randint(100, 5000)
            txtPrice.delete(0, END)
            txtPrice.insert(0, str(a) + " Euros")
        except Exception:
            messagebox.showerror("Error", Exception)
    #endregion

    btnClear = Button(root ,text="Clear All", bg='darkblue', fg='white', 
                    command=ClearContent, font=font_main)
    btnClear.grid(row=13, column=0)                    
    btnPredict = Button(root ,text="Predict", bg='darkblue', fg='white', font=font_main,
                        command=Predict_value)
    btnPredict.grid(row=13,column=1)

    txtNo.grid(row=0, column=1)
    txtCompany.grid(row=1, column=1)
    txtProduct.grid(row=2, column=1)
    txtTypeName.grid(row=3, column=1)
    txtInches.grid(row=4, column=1)
    txtScreen.grid(row=5, column=1)
    txtCPU.grid(row=6, column=1)
    txtRam.grid(row=7, column=1)
    txtMemory.grid(row=8, column=1)
    txtGPU.grid(row=9, column=1)
    txtOpSys.grid(row=10, column=1)
    txtWeight.grid(row=11, column=1)
    txtPrice.grid(row=12, column=1)
    #endregion

    root.config(menu=menuBar)
    root.mainloop()

if __name__ == '__main__':
   initialize()