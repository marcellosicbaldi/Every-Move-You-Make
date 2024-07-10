import sys
import numpy as np
import pandas as pd
import neurokit2 as nk
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib as mpl
mpl.rcParams["lines.linewidth"] = 0.91
mpl.rcParams["xtick.labelsize"] = 16

import neurokit2 as nk

from envelope import return_envelope_diff

class ECGViewer(FigureCanvas):
    def __init__(self, acc_data, env_data, parent=None):
        fig = Figure()
        self.axes1 = fig.add_subplot(211)
        self.axes2 = fig.add_subplot(212, sharex=self.axes1)
        self.axes1.get_xaxis().set_visible(False) # Hide x-axis of the first subplot
        self.axes1.yaxis.set_ticklabels([])
        self.axes1.yaxis.set_ticks([])
        self.axes1.set_ylabel('Acc norm', fontsize=18)
        self.axes2.yaxis.set_tick_params(labelsize=16)
        self.axes2.xaxis.set_tick_params(labelsize=16)
        self.axes2.set_ylabel('Env Diff', fontsize=18)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.mpl_connect('button_press_event', self.on_click)
        self.artifact_mode = False
        self.start_point = None
        self.start_points = []
        self.end_points = []
        
        # Plot the ECG data
        self.plot_ecg(acc_data, env_data)

    def plot_ecg(self, acc_data, env_data):
        # Assuming the data is a pd.Series with datetime index
        self.axes1.plot(acc_data.index, acc_data.values, 'r')
        self.axes2.plot(env_data.index, env_data.values)
        self.draw()

    def on_click(self, event):
        # Check if right click
        if event.button == 3:  # Right click
            if not self.artifact_mode:
                # Entering artifact mode, mark start of artifact
                self.start_point = mdates.num2date(event.xdata) # mouse x position in datetime coordinates
                self.artifact_mode = True
                self.axes1.axvline(self.start_point, color='r', linestyle='--')
                self.axes2.axvline(self.start_point, color='r', linestyle='--')
                self.draw()
            else:
                # Exiting artifact mode, mark end of artifact and draw axvspan
                end_point = mdates.num2date(event.xdata)
                self.start_points.append(self.start_point)
                self.end_points.append(end_point)
                self.axes1.axvspan(self.start_point, end_point, color='red', alpha=0.3)
                self.axes1.axvline(end_point, color='r', linestyle='--')
                self.axes2.axvspan(self.start_point, end_point, color='red', alpha=0.3)
                self.axes2.axvline(end_point, color='r', linestyle='--')
                self.draw()
                self.artifact_mode = False
                # Reset start point
                self.start_point = None

    def get_artifact_arrays(self):
        # Convert start and end points lists to numpy arrays and return them
        return pd.DataFrame(np.vstack([pd.to_datetime(self.start_points), pd.to_datetime(self.end_points)]).T, columns=['Start', 'End'])

class MainWindow(QMainWindow):
    def __init__(self, acc_data, env_data):
        super().__init__()
        self.setWindowTitle('ECG Artifact Annotation Tool')
        self.setGeometry(100, 100, 1900, 1200)  # Adjust size as needed
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.plot = ECGViewer(acc_data, env_data)
        layout.addWidget(self.plot)

        self.toolbar = NavigationToolbar(self.plot, self)
        layout.addWidget(self.toolbar)
        
        # Button to print artifact arrays for demonstration
        # self.print_button = QPushButton('Print Artifact Arrays')
        # self.print_button.clicked.connect(self.print_artifact_arrays)
        # layout.addWidget(self.print_button)

        # Button to save artifact arrays
        self.save_button = QPushButton('Save Artifact Arrays')
        self.save_button.clicked.connect(self.save_artifact_arrays)
        layout.addWidget(self.save_button)

    def print_artifact_arrays(self): 
        artifacts = self.plot.get_artifact_arrays()
        print("Artifacts array:", artifacts)
    
    def save_artifact_arrays(self):
        artifacts = self.plot.get_artifact_arrays()
        artifacts.to_csv(data_path + "/" + loc + "/bursts_ANNOT.csv", index=False)

# Select subject and location
sub = "906"
comb_location = {
    "158": [],
    "633": [],
    "906": ["rw", "ll", "t"],
    "958": [],
    "127": [],
    "098": [],
    "547": [],
    "815": [],
    "914": [],
    "971": [],
    "279": [],
    "965": [],

}

loc = "rw"

diary_SPT = {    
    "158": [pd.Timestamp('2024-02-28 23:00:00'), pd.Timestamp('2024-02-29 07:15:00')], # 158 OK
    "633": [pd.Timestamp('2024-03-07 00:05:00'), pd.Timestamp('2024-03-07 06:36:00')], # 633 OK
    "906": [pd.Timestamp('2024-03-07 00:30:00'), pd.Timestamp('2024-03-07 07:30:00')], # 906 OK
    "958": [pd.Timestamp('2024-03-13 22:00:00'), pd.Timestamp('2024-03-14 06:00:00')], # 958 OK
    "127": [pd.Timestamp('2024-03-13 23:15:00'), pd.Timestamp('2024-03-14 06:50:00')], # 127 OK
    "098": [pd.Timestamp('2024-03-16 02:01:00'), pd.Timestamp('2024-03-16 09:50:00')], # 098 OK
    "547": [pd.Timestamp('2024-03-16 01:04:00'), pd.Timestamp('2024-03-16 07:40:00')], # 547 OK
    "815": [pd.Timestamp('2024-03-20 23:00:00'), pd.Timestamp('2024-03-21 07:30:00')], # 815 OK
    "914": [pd.Timestamp('2024-03-20 21:50:00'), pd.Timestamp('2024-03-21 05:50:00')], # 914 OK
    "971": [pd.Timestamp('2024-03-20 23:50:00'), pd.Timestamp('2024-03-21 07:50:00')], # 971 OK
    "279": [pd.Timestamp('2024-03-28 00:10:00'), pd.Timestamp('2024-03-28 07:27:00')], # 279 OK
    "965": [pd.Timestamp('2024-03-28 01:25:00'), pd.Timestamp('2024-03-28 09:20:00')], # 965 OK
}

start_sleep, end_sleep = diary_SPT[sub]

# Load the data
data_path =  "/Users/marcellosicbaldi/Library/CloudStorage/OneDrive-AlmaMaterStudiorumUniversitàdiBologna/General - LG-MIAR (rehab)/SCORING_bursts/" + sub 
#### PAOLA: Change here the index of the comb_location dictionary {comb_location[sub][0], comb_location[sub][1], comb_location[sub][2]}
acc_norm_raw = pd.read_pickle(data_path + "/" + loc + "/" + loc + ".pkl")
acc_norm_raw = pd.Series(nk.signal_filter(acc_norm_raw.values, sampling_rate = 50, lowcut=0.1, highcut=5, method='butterworth', order=8), index = acc_norm_raw.index)


# Split the data according to the sleep midpoint
sleep_midPoint = start_sleep + (end_sleep - start_sleep) / 2

# RW: select the 1 hour before and after the midpoint of sleep
loc1_df_1 = acc_norm_raw.loc[sleep_midPoint - pd.Timedelta(hours = 1):sleep_midPoint]
loc1_df_2 = acc_norm_raw.loc[sleep_midPoint:sleep_midPoint + pd.Timedelta(hours = 1)]

# # LL: select the hour 2 hours before and after the midpoint of sleep
# loc2_df_1 = acc_norm_raw.loc[sleep_midPoint - pd.Timedelta(hours = 2):sleep_midPoint - pd.Timedelta(hours = 1)]
# loc2_df_2 = acc_norm_raw.loc[sleep_midPoint + pd.Timedelta(hours = 1):sleep_midPoint + pd.Timedelta(hours = 2)]

# # Trunk: select the hour 3 hours before and after the midpoint of sleep
# loc3_df_1 = acc_norm_raw.loc[sleep_midPoint - pd.Timedelta(hours = 3):sleep_midPoint - pd.Timedelta(hours = 2)]
# loc3_df_2 = acc_norm_raw.loc[sleep_midPoint + pd.Timedelta(hours = 2):sleep_midPoint + pd.Timedelta(hours = 3)]

# Extract envelope differences
loc1_env_1 = return_envelope_diff(loc1_df_1)
loc1_env_2 = return_envelope_diff(loc1_df_2)

# loc2_env_1 = return_envelope_diff(loc2_df_1)
# loc2_env_2 = return_envelope_diff(loc2_df_2)

# loc3_env_1 = return_envelope_diff(loc3_df_1)
# loc3_env_2 = return_envelope_diff(loc3_df_2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow(loc1_df_1, loc1_env_1)
    main.show()
    sys.exit(app.exec_())