'''
YCWang, ChatGPT
Oct.3, 2025
This file calculate seizure onset time / pre-ictal time / inter-ictal periods
'''
from datetime import datetime, timedelta

YELLOW = "\033[33m"
BLUE = "\033[36m"
RESET = "\033[0m"

def parse_time(timestr):
    """Convert HHMMSS into datetime object."""
    return datetime.strptime(timestr, "%H%M%S")

def calculate_seizure_times(file_start, seizure_intervals):
    """
    Calculate seizure start and end times in hh:mm:ss format.

    Args:
        file_start (str): File start time in format "HHMMSS" (no colons).
        seizure_intervals (list of tuples): Each tuple contains (start_seconds, end_seconds).

    Returns:
        list of dict: Each seizure with start and end timestamps (HH:MM:SS strings).
    """
    start_dt = parse_time(file_start)

    seizure_times = []
    for idx, (s, e) in enumerate(seizure_intervals, start=1):
        seizure_start = start_dt + timedelta(seconds=s)
        seizure_end = start_dt + timedelta(seconds=e)
        seizure_times.append({
            "Seizure #": idx,
            "Start Time": seizure_start,
            "End Time": seizure_end
        })

    return seizure_times

def calculate_interictal_times(seizure_times, minute=60):
    """
    Calculate interictal periods as ±60 minutes around seizure start and end.

    Args:
        seizure_times (list of dict): Seizure start and end datetime objects.

    Returns:
        list of dict: Interictal windows for each seizure (HH:MM:SS strings).
    """
    interictals = []
    for seizure in seizure_times:
        seizure_num = seizure["Seizure #"]
        start = seizure["Start Time"]
        end = seizure["End Time"]

        interictal_start = (start - timedelta(minutes=minute)).strftime("%H:%M:%S")
        interictal_end = (end + timedelta(minutes=minute)).strftime("%H:%M:%S")

        interictals.append({
            "Seizure #": seizure_num,
            "Interictal Start": interictal_start,
            "Interictal End": interictal_end
        })

    return interictals

def timepoint_to_seconds(file_start, time_point):
    """
    Calculate which second a given time point is in the file.

    Args:
        file_start (str): File start time in format "HHMMSS".
        time_point (str): Time point in format "HHMMSS".

    Returns:
        int: Seconds from file start to time_point.
    """
    start_dt = datetime.strptime(file_start, "%H%M%S")
    point_dt = datetime.strptime(time_point, "%H%M%S")
    return int((point_dt - start_dt).total_seconds())

def cal_pre_time(start, seizure_times, interval=5):
    pre_time = start - interval*60
    for seizure in seizure_times:
        start_time = seizure["Start Time"]
        pre_start = (start_time - timedelta(minutes=interval)).strftime("%H:%M:%S")
    return pre_time, pre_start

if __name__ == "__main__":
    pre_interval = int(input("preictal interval: "))
    inter_interval = int(input("interictal interval: "))
    while True:
        # User inputs
        mode = input("Select mode: \n1. pre-ictal\n2. inter-ictal\n")
        if mode.lower() not in ["1", "2"]:
            break
        file_start_time = input("Enter file start time (HHMMSS): ").strip()
        if mode == '1':
            file_end_time = input("Enter file end time (HHMMSS): ").strip()

            seizures = []
            start_sec = int(input(f"Enter seizure start time (in seconds): "))
            end_sec = int(input(f"Enter seizure end time (in seconds): "))
            seizures.append((start_sec, end_sec))

            seizure_times = calculate_seizure_times(file_start_time, seizures)
            pre_start_time = cal_pre_time(start_sec, seizure_times, interval=pre_interval)
            interictals = calculate_interictal_times(seizure_times, minute=inter_interval)

            print(f"\n############################################\n{YELLOW}Calculated Seizure Times:")
            for seizure in seizure_times:
                print(f"Seizure {seizure['Seizure #']}: {BLUE}{seizure['Start Time'].strftime("%H:%M:%S")} - {seizure['End Time'].strftime("%H:%M:%S")}{YELLOW}")
                print(f"Pre-ictal start second: {BLUE}{pre_start_time[0]} sec{YELLOW}")
                print(f"Pre-ictal start time: {BLUE}{pre_start_time[1]}{YELLOW}")
            for interictal in interictals:
                print(f"Interictal {interictal['Seizure #']}: \nLast interictal end: {BLUE}{interictal['Interictal Start']}{YELLOW}\nNext interictal start: {BLUE}{interictal['Interictal End']}")
        if mode == '2':
            timepoint = input("Enter timepoint (HHMMSS): ").strip()
            t_timepoint = timepoint_to_seconds(file_start_time, timepoint)
            print(f"\n############################################\n{YELLOW}Timepoint (sec): {BLUE}{t_timepoint}")
        print(f"{RESET}-------------------------------------------------------------")
