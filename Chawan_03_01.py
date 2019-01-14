# Chawan, Varsha Rani
# 1001-553-524
# 2018-10-08
# Assignment-03-01

import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import Chawan_03_02



class MainWindow(tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    Chawan Varsha Rani
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # set the properties of the row and columns in the master frame
        self.rowconfigure(0, weight=1, minsize=500)
        self.columnconfigure(0, weight=1, minsize=500)
        self.master_frame.rowconfigure(2, weight=10, minsize=100, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
        # Create an object for plotting graphs in the left frame
        self.left_frame = tk.Frame(self.master_frame)
        self.left_frame.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_error = HebbianLR(self, self.left_frame, debug_print_flag=self.debug_print_flag)


class HebbianLR:
    """
    This class creates and controls the sliders , buttons , drop down in the frame which
    are used to display decision bounrdy and generate samples and train .
    """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = 0
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 100
        self.alpha = 0.1
        self.activation_function = "Symmetrical Hard limit"
        self.learning_method = "Filtered Learning"
        self.model = Chawan_03_02.HebbModel("Data", self.activation_function, self.learning_method, self.alpha)
        self.display = None

        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Percentage Error_Rate')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the control widgets such as sliders ,buttons and dropdowns
        #########################################################################
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                     activebackground="#FF0000", highlightcolor="#00FFFF", label="Learning Rate",
                                     command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.adjust_weight_button = tk.Button(self.controls_frame, text="Learn", width=16,
                                              command=self.adjust_weight_button_callback)
        self.adjust_weight_button.grid(row=0, column=1)
        self.randomize_weight_button = tk.Button(self.controls_frame, text="Randomize Weights", width=16,
                                                 command=self.randomize_weight_button_callback)
        self.randomize_weight_button.grid(row=0, column=2)
        self.confusion_matrix_button = tk.Button(self.controls_frame, text="Display Confusion Matrix", width=20,
                                                 command=self.confusion_matrix_button_callback)
        self.confusion_matrix_button.grid(row=0, column=3)
        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################
        self.label_for_learning_method = tk.Label(self.controls_frame, text="Select Learning Method:",
                                                  justify="center")
        self.label_for_learning_method.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
        self.learning_method_variable = tk.StringVar()
        self.learning_method_drop_down = tk.OptionMenu(self.controls_frame, self.learning_method_variable,
                                                       "Filtered Learning", "Delta Rule",
                                                       "Unsupervised Heb", command=lambda
                event: self.learning_method_callback())
        self.learning_method_variable.set("Filtered Learning")
        self.learning_method_drop_down.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)
        self.label_for_activation_function = tk.Label(self.controls_frame, text="Transfer Functions:",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=6, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                          "Symmetrical Hard limit", "Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetrical Hard limit")
        self.activation_function_dropdown.grid(row=0, column=7, sticky=tk.N + tk.E + tk.S + tk.W)

    def plot_errorRate(self, epoch, error):
        #########################################################################
        #  plots the percentage error rate
        #########################################################################
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Percentage Error_Rate')
        self.axes.plot(epoch, error)
        self.xmax = len(epoch)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title("Activation = " + self.activation_function + "\nLearning Method = " + self.learning_method)
        self.canvas.draw()
        #########################################################################
        #  Re initialising the variables after 1000 epocs
        #########################################################################
        if len(epoch) == 1000:
            self.randomize_weights_callback()

    def display_confusion_mat(self):
        #######################################################################################
        #  plots the confusion matrix for 200 data and displays count of corresponding values
        #######################################################################################
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        self.axes.set_xlabel("Predicted")
        self.axes.set_ylabel("True")
        plt.title("Confusion Matrix ")
        cfg = self.model.conf_matrix()
        df_cm = pd.DataFrame(cfg,range(10),range(10))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm,annot=True,cbar=False)
        self.canvas.draw()

    def alpha_slider_callback(self):
        self.alpha = np.float(self.alpha_slider.get())
        self.model.train(self.activation_function, self.learning_method, self.alpha)
        self.plot_errorRate(self.model.epochs, self.model.error)

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        self.model.train(self.activation_function, self.learning_method, self.alpha)
        self.plot_errorRate(self.model.epochs, self.model.error)

    def learning_method_callback(self):
        self.learning_method = self.learning_method_variable.get()
        self.model.train(self.activation_function, self.learning_method, self.alpha)
        self.plot_errorRate(self.model.epochs, self.model.error)

    def randomize_weight_button_callback(self):
        self.model = Chawan_03_02.HebbModel("Data", self.activation_function, self.learning_method, self.alpha)
        self.plot_errorRate(self.model.epochs, self.model.error)

    def confusion_matrix_button_callback(self):
        self.display_confusion_mat()

    def adjust_weight_button_callback(self):
        self.model.train(self.activation_function, self.learning_method, self.alpha)
        self.plot_errorRate(self.model.epochs, self.model.error)

        #########################################################################
        #  Logic to close Main Window
        #########################################################################

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()


main_window = MainWindow(debug_print_flag=False)
main_window.wm_state('zoomed')
main_window.title('Assignment_03 --  Chawan')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()