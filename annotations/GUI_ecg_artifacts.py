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

class ECGViewer(FigureCanvas):
    def __init__(self, ecg_data, t_rpeaks, hr_df, parent=None):
        fig = Figure()
        self.axes1 = fig.add_subplot(211)
        self.axes2 = fig.add_subplot(212, sharex=self.axes1)
        self.axes1.get_xaxis().set_visible(False) # Hide x-axis of the first subplot
        self.axes1.yaxis.set_ticklabels([])
        self.axes1.yaxis.set_ticks([])
        self.axes1.set_ylabel('ECG', fontsize=18)
        self.axes2.yaxis.set_tick_params(labelsize=16)
        self.axes2.xaxis.set_tick_params(labelsize=16)
        self.axes2.set_ylabel('HR (bpm)', fontsize=18)
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
        self.plot_ecg(ecg_data, t_rpeaks, hr_df)

    def plot_ecg(self, ecg_data, t_rpeaks, hr_df):
        # Assuming the data is a pd.Series with datetime index
        self.axes1.plot(ecg_data.index, ecg_data.values, 'r')
        self.axes1.plot(t_rpeaks, ecg_df.loc[t_rpeaks], 'b*')
        self.axes2.plot(hr_df.index, hr_df.values)
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
    def __init__(self, ecg_data, t_rpeaks, hr_df):
        super().__init__()
        self.setWindowTitle('ECG Artifact Annotation Tool')
        self.setGeometry(100, 100, 1900, 1200)  # Adjust size as needed
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.plot = ECGViewer(ecg_data, t_rpeaks, hr_df)
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
        artifacts.to_csv(f'/Volumes/Untitled/rehab/data/{sub}/polar_processed/artifacts_ecg.csv', index=False)

sub = "971"
diary = {    
    "158": [pd.Timestamp('2024-02-28 23:05:00'), pd.Timestamp('2024-02-29 07:15:00')], # 158 OK
    "633": [pd.Timestamp('2024-03-07 00:05:00'), pd.Timestamp('2024-03-07 06:50:00')], # 633 TO CHECK
    "906": [pd.Timestamp('2024-03-07 00:45:00'), pd.Timestamp('2024-03-07 07:30:00')], # 906 OK
    "958": [pd.Timestamp('2024-03-13 22:05:00'), pd.Timestamp('2024-03-14 06:00:00')], # 958 OK
    "127": [pd.Timestamp('2024-03-13 23:25:00'), pd.Timestamp('2024-03-14 06:50:00')], # 127 OK
    "098": [pd.Timestamp('2024-03-16 02:03:00'), pd.Timestamp('2024-03-16 09:45:00')], # 098 OK
    "547": [pd.Timestamp('2024-03-16 01:15:00'), pd.Timestamp('2024-03-16 07:40:00')], # 547 OK
    "815": [pd.Timestamp('2024-03-20 23:15:00'), pd.Timestamp('2024-03-21 07:15:00')], # 815 OK
    "914": [pd.Timestamp('2024-03-20 22:25:00'), pd.Timestamp('2024-03-21 05:50:00')], # 914 OK
    "971": [pd.Timestamp('2024-03-20 23:59:00'), pd.Timestamp('2024-03-21 07:45:00')], # 971 OK
    "279": [pd.Timestamp('2024-03-28 00:10:00'), pd.Timestamp('2024-03-28 07:27:00')], # 279 OK
    "965": [pd.Timestamp('2024-03-28 01:25:00'), pd.Timestamp('2024-03-28 09:20:00')], # 965 OK
}

start_sleep, end_sleep = diary[sub]
ecg_df = pd.read_pickle(f'/Volumes/Untitled/rehab/data/{sub}/polar_processed/ecg.pkl')
ecg_df = ecg_df.loc[start_sleep:end_sleep]
ecg_filtered = nk.ecg_clean(ecg_df.values, sampling_rate=130)
# ecg_filtered = nk.signal_filter(ecg_df.values, sampling_rate=130, lowcut=0.5, highcut=15, method='butterworth', order=4)

# Extract peaks
_, results = nk.ecg_peaks(ecg_filtered, sampling_rate=130, method = 'neurokit')
rpeaks = results["ECG_R_Peaks"]
_, rpeaks_corrected = nk.signal_fixpeaks(rpeaks, sampling_rate=130, iterative=True, method="Kubios")
t_rpeaks_corrected = ecg_df.index.to_series().values[rpeaks_corrected]
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 7))
# plt.plot(ecg_df)
# plt.plot(t_rpeaks_corrected, ecg_df.loc[t_rpeaks_corrected], 'b*')
# plt.show()
rr_corrected = np.diff(t_rpeaks_corrected).astype('timedelta64[ns]').astype('float64') / 1000000000
hr_ecg_corrected = 60/rr_corrected
hr_df = pd.Series(hr_ecg_corrected, index = t_rpeaks_corrected[1:]).resample("1 s").mean()
hr_df = hr_df.interpolate(method = 'linear')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow(ecg_df, t_rpeaks_corrected, hr_df)
    main.show()
    sys.exit(app.exec_())