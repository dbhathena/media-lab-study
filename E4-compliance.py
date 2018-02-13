import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
from dateutil import tz
import matplotlib.pyplot as plt
import os

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
SECONDS_PER_DAY = 86400
COMPLIANCE_THRESHOLD = SECONDS_PER_DAY/2
SECONDS_PER_HOUR = 3600
TEMP_DIRECTORY = '../test-temp-files'
NEW_FILES = 'new-files'
BLUE = '#2C55CA'
RED = '#D63434'
GREEN = '#5AD381'
YELLOW = '#F7D158'
PURPLE = '#7D5AD3'
GRAY = '0.2'


def get_manual_isotime_string(time_string):
    time = pd.datetime.strptime(time_string, DATE_FORMAT)
    time.isoformat(' ')
    return str(time)


def integerize(string):
    s = ''
    for char in string:
        try:
            s += str(int(char))
        except TypeError:
            pass
    return s


def get_hdf_segment(path, col=None, start=None, end=None):
    if not start and not end:
        return pd.read_hdf(path, col)
    return pd.read_hdf(path, col, where='index>' + start + '& index<' + end)


def get_temp_col(segment):
    return segment['temp'].values.tolist()


def get_eda_col(segment):
    return segment['eda'].values.tolist()


def percentage_of_one_day(segment):   # segment must be temp hdf segment for exactly one day
    temp_col = get_temp_col(segment)
    pared = 0
    for i in range(len(temp_col)):
        if temp_col[i] >= 31:
            pared += 1
    return pared*.25/SECONDS_PER_DAY


def per_day_percentages(days):
    percentages = []
    for day in days:
        percentages.append(percentage_of_one_day(day))
    return percentages


def percentage_of_one_hour(segment):    # segment must be temp hdf segment of exactly one hour
    temp_col = get_temp_col(segment)
    pared = 0
    for i in range(len(temp_col)):
        if temp_col[i] >= 31:
            pared += 1
    return min(1.0, pared*.25/3600)


def percentage_of_days_compliant(percentages_per_day):
    threshold_fraction = COMPLIANCE_THRESHOLD/SECONDS_PER_DAY
    compliant_days = 0
    for percentage in percentages_per_day:
        if percentage >= threshold_fraction:
            compliant_days += 1
    return compliant_days/len(percentages_per_day)


def show_line_plot(data_list):
    plt.plot(data_list)
    plt.show()


def show_bar_graph(data_list):
    plt.bar(data_list)
    plt.show()


def get_files(directory):
    files = os.listdir(directory)
    return [x for x in files
            if not (x.startswith('.'))]


def chart_hours_per_day_average(participant_to_percentages_per_day, title, left):
    days_to_percentages = map_days_to_percentages(participant_to_percentages_per_day)
    avg_hours_per_day = []
    std_devs = []
    labels = []
    for day in days_to_percentages:
        hours_for_day = [x*24 for x in days_to_percentages[day]]
        avg_hours_per_day.append(sum(hours_for_day)/len(hours_for_day))
        std_devs.append(np.std(hours_for_day))
        labels.append(day)
    if left:
        plt.bar(np.arange(len(labels)), avg_hours_per_day, color=[BLUE], yerr=std_devs)
    else:
        plt.bar(np.arange(len(labels)), avg_hours_per_day, color=[RED], yerr=std_devs)
    plt.title(title)
    plt.xlabel('Day in Study')
    plt.ylabel('Average # of Hours')
    plt.show()


def chart_hours_per_day_averages_together(participant_to_percentages_per_day_left,
                                          participant_to_percentages_per_day_right, title):
    # print("LEFT:", participant_to_percentages_per_day_left)
    # print()
    # print("RIGHT:", participant_to_percentages_per_day_right)
    days_to_percentages_left = map_days_to_percentages(participant_to_percentages_per_day_left)
    days_to_percentages_right = map_days_to_percentages(participant_to_percentages_per_day_right)
    avg_hours_per_day_left = []
    avg_hours_per_day_right = []
    std_devs_left = []
    std_devs_right = []
    for day in days_to_percentages_left:
        hours_for_day = [x*24 for x in days_to_percentages_left[day]]
        avg_hours_per_day_left.append(sum(hours_for_day)/len(hours_for_day))
        std_devs_left.append(np.std(hours_for_day))
    for day in days_to_percentages_right:
        hours_for_day = [x*24 for x in days_to_percentages_right[day]]
        avg_hours_per_day_right.append(sum(hours_for_day)/len(hours_for_day))
        std_devs_right.append(np.std(hours_for_day))

    ind = np.arange(max(len(avg_hours_per_day_left), len(avg_hours_per_day_right)))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    rects_left = ax.bar(np.arange(len(avg_hours_per_day_left)), avg_hours_per_day_left, width, color=BLUE,
                        label='Left', yerr=std_devs_left)
    rects_right = ax.bar(np.arange(len(avg_hours_per_day_right)) + width, avg_hours_per_day_right, width, color=RED,
                        label='Right', yerr=std_devs_right)

    rects_left.errorbar = ax.errorbar(x=np.arange(len(avg_hours_per_day_left)), y=avg_hours_per_day_left,
                                      yerr=std_devs_left, ecolor=GRAY, fmt='none')
    rects_right.errorbar = ax.errorbar(x=np.arange(len(avg_hours_per_day_right)) + width, y=avg_hours_per_day_right,
                                       yerr=std_devs_right, ecolor=GRAY, fmt='none')

    num_samples = max(len(avg_hours_per_day_left), len(avg_hours_per_day_right))
    labels = ['' for x in range(num_samples)]

    for i in range(len(labels)):
        if (i + 1) % 5 == 0:
            labels[i] = i + 1

    ax.set_xlabel('Day in Study')
    ax.set_ylabel('# hours')
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    try:
        ax.legend((rects_left[1], rects_right[1]), ('Left Hand', 'Right Hand'))
    except IndexError:
        ax.legend((rects_left[0], rects_right[0]), ('Left Hand', 'Right Hand'))

    ax.set_xticklabels(labels)

    save_hours_per_day_chart_total(range(num_samples), avg_hours_per_day_left, std_devs_left, avg_hours_per_day_right, std_devs_right)
    plt.show()


def save_hours_per_day_chart_total(days, hours_per_day_left, std_devs_left, hours_per_day_right, std_devs_right):
    file = open("average-compliance/Hours per Day/Hours_per_day_total.txt", 'w')
    file.write("Day in Study     left hand hours     left hand error     right hand hours     right hand error\n")
    file.write("------------     ---------------     ---------------     ----------------     ----------------\n")
    for day in days:
        spaces = ""
        if day < len(hours_per_day_left):
            hours_left = "{0:.2f}".format(hours_per_day_left[day])
            len_left = len(hours_left)
            for i in range(5 - len_left):
                hours_left += " "

            std_dev_left = "{0:.2f}".format(std_devs_left[day])
            len_std_left = len(std_dev_left)
            for i in range(5-len_std_left):
                std_dev_left += " "
        else:
            hours_left = "     "
            std_dev_left = "     "

        if day < len(hours_per_day_right):
            hours_right = "{0:.2f}".format(hours_per_day_right[day])
            len_right = len(hours_right)
            for i in range(5 - len_right):
                hours_right += " "
            std_dev_right = "{0:.2f}".format(std_devs_right[day])
            len_std_right = len(std_dev_right)
            for i in range(5 - len_std_right):
                std_dev_right += " "
        else:
            hours_right = "     "
            std_dev_right = "     "

        for i in range(2 - len(str(day))):
            spaces += " "
        file.write(str(day) + spaces + "                    " + hours_left + "               " + std_dev_left +
                   "               " + hours_right + "                " + std_dev_right + "\n")
    file.close()

def map_days_to_percentages(participant_to_percentages_per_day):
    days_to_percentages = {}
    for participant in participant_to_percentages_per_day:
        for day in range(len(participant_to_percentages_per_day[participant])):
            if day not in days_to_percentages:
                days_to_percentages[day] = []
            days_to_percentages[day].append(participant_to_percentages_per_day[participant][day])
    return days_to_percentages


def save_chart_hours_per_day_per_person(name, left_data, right_data, day_of_study_assessments, assessments):

    number_of_study_days = max(day_of_study_assessments) - min(day_of_study_assessments)+1

    number_samples = max(number_of_study_days, len(left_data), len(right_data))

    hours_per_day_left = [0 for x in range(number_samples)]
    hours_per_day_right = [0 for x in range(number_samples)]

    for i in range(len(left_data)):
        hours_per_day_left[i] = left_data[i]*24
    for i in range(len(right_data)):
        hours_per_day_right[i] = right_data[i]*24

    ind = np.arange(max(len(hours_per_day_left), len(hours_per_day_right)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(20, 10))
    rects_left = ax.bar(np.arange(len(hours_per_day_left)), hours_per_day_left, width, color=BLUE, label='Left')
    rects_right = ax.bar(np.arange(len(hours_per_day_right))+width, hours_per_day_right, width, color=RED, label='Right')
    labels = ['' for x in range(number_samples)]
    for day in day_of_study_assessments:
        try:
            rects_left[day].set_color(PURPLE)
            rects_right[day].set_color(YELLOW)
            labels[day] = assessments[day_of_study_assessments[day]].strftime('%x')
        except IndexError:
            print('not in range')
            pass

    ax.set_xlabel('Day in Study')
    ax.set_ylabel('# hours')
    ax.set_title(name+"'s Compliance")
    ax.set_xticks(ind+width/2)
    try:
        ax.legend((rects_left[1], rects_right[1]), ('Left Hand', 'Right Hand'))
    except IndexError:
        ax.legend((rects_left[0], rects_right[0]), ('Left Hand', 'Right Hand'))

    ax.set_xticklabels(labels)
    os.mkdir("individual-compliance-data/" + name)
    plt.savefig('individual-compliance-data/'+name+'/'+name+".png")
    print(name+'.png has been created')
    save_hours_per_day_chart_for_person(name, ind, hours_per_day_left, hours_per_day_right)
    print(name+'.txt has been created')
    print()
    plt.close()


def save_hours_per_day_chart_for_person(name, days, hours_per_day_left, hours_per_day_right):
    file = open("individual-compliance-data/"+name+"/"+name+".txt", 'w')
    file.write("Day in Study       left hand       right hand\n")
    file.write("------------       ---------       ----------\n")
    for day in days:
        spaces = ""
        hours_left = "{0:.2f}".format(hours_per_day_left[day])
        len_left = len(hours_left)
        for i in range(5 - len_left):
            hours_left += " "
        hours_right = "{0:.2f}".format(hours_per_day_right[day])
        len_right = len(hours_right)
        for i in range(5 - len_right):
            hours_right += " "
        for i in range(2-len(str(day))):
            spaces += " "
        file.write(str(day) + spaces + "                   "+hours_left+"            " + hours_right + "\n")
    file.close()

def split_into_days(hdf_temp_filepath, left, start=None):

    acc_first_row_left = pd.read_hdf(hdf_temp_filepath, 'TEMP_left')
    acc_first_row_left = acc_first_row_left.sort_index()
    acc_first_row_right = pd.read_hdf(hdf_temp_filepath, 'TEMP_right')
    acc_first_row_right = acc_first_row_right.sort_index()

    if not start:
        start_date_left = np.min(acc_first_row_left.index.values)
        start_date_right = np.min(acc_first_row_right.index.values)
        start_date = min(start_date_left, start_date_right)
        start_date = dt.utcfromtimestamp(start_date.tolist() / 1e9)
        start_date = start_date.replace(tzinfo=None)
    else:
        start_date = start
        start_date = start_date.replace(tzinfo=None)

    start_date = start_date.replace(hour=0, minute=0, second=0)

    # convert to datetime
    end_date_left = np.max(acc_first_row_left.index.values)
    end_date_right = np.max(acc_first_row_right.index.values)
    end_date = end_date_left if left else end_date_right
    # convert to datetime
    end_date = dt.utcfromtimestamp(end_date.tolist() / 1e9)
    end_date = end_date.replace(tzinfo=None)
    end_date = end_date.replace(hour=0, minute=0, second=0)
    rng = pd.date_range(start_date, end_date)

    hdf_in_days = []
    for idx, beginning in enumerate(rng):
        end = beginning + timedelta(hours=24)
        if left:
            # print('left', beginning, end)
            hdf_in_days.append(acc_first_row_left.loc[beginning:end])
        else:
            # print('right', beginning, end)
            try:
                test = acc_first_row_right.loc[beginning:end]
            except:
                print(beginning, end)
                quit()
            hdf_in_days.append(test)
    # print()
    return hdf_in_days


def split_day_into_hours(day_segment, is_print=False):
    start_date = np.min(day_segment.index.values)
    end_date = start_date + np.timedelta64(1, 'D')

    if is_print:
        print(start_date, end_date)
        print()

    # convert to datetime
    start_date = dt.utcfromtimestamp(start_date.tolist() / 1e9)

    start_date = start_date.replace(tzinfo=tz.gettz('UTC')).astimezone(
        tz.gettz('America/New_York'))
    start_date = start_date.replace(hour=0, minute=0, second=0)

    # convert to datetime
    end_date = dt.utcfromtimestamp(end_date.tolist() / 1e9)
    end_date = end_date.replace(tzinfo=tz.gettz('UTC')).astimezone(
        tz.gettz('America/New_York'))
    end_date = end_date.replace(hour=0, minute=0, second=0)
    rng = pd.date_range(start_date, end_date, freq='H', closed='left')

    hdf_in_hours = []
    for idx, beginning in enumerate(rng):
        end = beginning + timedelta(hours=1)
        hdf_in_hours.append(day_segment.loc[beginning:end])

    return hdf_in_hours


def make_minutes_per_hour_chart_from_data(directory, left):
    # maps a participant to a list of their average percentage compliance for each hour of the day
    participant_to_percentages_of_hours = {}
    date_dic = get_date_dic('HAMD_final_scores.csv')
    for filename in get_files(directory):
        participant = filename[:4]
        print("Starting " + participant)
        start_date = date_dic[participant]['Week 0']
        days = split_into_days(directory+'/'+filename, left, start=start_date)

        days_in_hours = []
        for day in days:
            if day.empty:
                percentages_per_hour_for_day = [0.0]*24
            else:
                hours_for_day = split_day_into_hours(day)
                percentages_per_hour_for_day = []
                for hour in hours_for_day:
                    percentages_per_hour_for_day.append(percentage_of_one_hour(hour))
            days_in_hours.append(percentages_per_hour_for_day)

        hour_percentages = []
        for hour in range(24):
            num_days = len(days_in_hours)
            sum_of_percentages = 0
            for day in days_in_hours:
                if hour < len(day):
                    sum_of_percentages += day[hour]
            hour_percentages.append(sum_of_percentages/num_days)
        participant_to_percentages_of_hours[participant] = hour_percentages
        print(participant + " has been finished")
        print()

    if left:
        chart_minutes_per_day_average(participant_to_percentages_of_hours, "Left Hand Compliance Data", left)
    else:
        chart_minutes_per_day_average(participant_to_percentages_of_hours, "Right Hand Compliance Data", left)


def make_minutes_per_hour_chart_both_hands(directory):
    participant_to_percentages_of_hours_left = {}
    participant_to_percentages_of_hours_right = {}
    date_dic = get_date_dic('HAMD_final_scores.csv')
    for filename in get_files(directory):
        participant = filename[:4]
        print("Starting " + participant)
        start_date = date_dic[participant]['Week 0']
        days_left = split_into_days(directory + '/' + filename, True, start=start_date)
        days_right = split_into_days(directory + '/' + filename, False, start=start_date)

        days_in_hours_left = []
        for day in days_left:
            if day.empty:
                percentages_per_hour_for_day = [0.0] * 24
            else:
                hours_for_day = split_day_into_hours(day)
                percentages_per_hour_for_day = []
                for hour in hours_for_day:
                    percentages_per_hour_for_day.append(percentage_of_one_hour(hour))
            days_in_hours_left.append(percentages_per_hour_for_day)

        days_in_hours_right = []
        for day in days_right:
            if day.empty:
                percentages_per_hour_for_day = [0.0] * 24
            else:
                hours_for_day = split_day_into_hours(day)
                percentages_per_hour_for_day = []
                for hour in hours_for_day:
                    percentages_per_hour_for_day.append(percentage_of_one_hour(hour))
            days_in_hours_right.append(percentages_per_hour_for_day)

        hour_percentages_left = []
        for hour in range(24):
            num_days = len(days_in_hours_left)
            sum_of_percentages = 0
            for day in days_in_hours_left:
                if hour < len(day):
                    sum_of_percentages += day[hour]
            hour_percentages_left.append(sum_of_percentages / num_days)

        participant_to_percentages_of_hours_left[participant] = hour_percentages_left

        hour_percentages_right = []
        for hour in range(24):
            num_days = len(days_in_hours_right)
            sum_of_percentages = 0
            for day in days_in_hours_right:
                if hour < len(day):
                    sum_of_percentages += day[hour]
            hour_percentages_right.append(sum_of_percentages / num_days)

        participant_to_percentages_of_hours_right[participant] = hour_percentages_right

        print(participant + " has been finished")
        print()

    chart_minutes_per_day_average_together(participant_to_percentages_of_hours_left,
                                           participant_to_percentages_of_hours_right,
                                           "Minutes per Hour Compliance Data")


def chart_minutes_per_day_average(participant_to_percentages_of_hours, title, left):
    avg_mins_per_hour = []
    std_devs = []
    labels = []
    for hour in range(24):
        percentages= []
        for participant in participant_to_percentages_of_hours:
            percentages.append(participant_to_percentages_of_hours[participant][hour]*60)
        avg_mins_per_hour.append(sum(percentages)/len(percentages))
        std_devs.append(np.std(percentages))
        labels.append(hour+1)
    y_pos = np.arange(len(labels))
    if left:
        plt.bar(y_pos, avg_mins_per_hour, color=[BLUE], yerr=std_devs)
    else:
        plt.bar(y_pos, avg_mins_per_hour, color=[RED], yerr=std_devs)
    plt.title(title)
    plt.xlabel('Hour of Day')
    plt.yticks([10, 20, 30, 40, 50, 60])
    plt.xticks(y_pos, labels)
    plt.ylabel('Average # of Minutes')
    plt.show()


def chart_minutes_per_day_average_together(participant_to_percentages_of_hours_left,
                                           participant_to_percentages_of_hours_right, title):
    avg_mins_per_hour_left = []
    avg_mins_per_hour_right = []
    std_devs_left = []
    std_devs_right = []
    labels = []
    for hour in range(24):
        percentages_left = []
        percentages_right = []
        for participant in participant_to_percentages_of_hours_left:
            percentages_left.append(participant_to_percentages_of_hours_left[participant][hour]*60)
            percentages_right.append(participant_to_percentages_of_hours_right[participant][hour]*60)
        avg_mins_per_hour_left.append(sum(percentages_left)/len(percentages_left))
        avg_mins_per_hour_right.append(sum(percentages_right)/len(percentages_right))
        std_devs_left.append(np.std(percentages_left))
        std_devs_right.append(np.std(percentages_right))
        labels.append(hour+1)
    y_pos = np.arange(len(labels))

    ind = np.arange(max(len(avg_mins_per_hour_left), len(avg_mins_per_hour_right)))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    rects_left = ax.bar(y_pos, avg_mins_per_hour_left, width, color=BLUE,
                        label='Left', yerr=std_devs_left)
    rects_right = ax.bar(y_pos + width, avg_mins_per_hour_right, width, color=RED,
                         label='Right', yerr=std_devs_right)

    rects_left.errorbar = ax.errorbar(x=y_pos, y=avg_mins_per_hour_left,
                                      yerr=std_devs_left, ecolor=GRAY, fmt='none')
    rects_right.errorbar = ax.errorbar(x=y_pos + width, y=avg_mins_per_hour_right,
                                       yerr=std_devs_right, ecolor=GRAY, fmt='none')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average # of minutes')
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    try:
        ax.legend((rects_left[1], rects_right[1]), ('Left Hand', 'Right Hand'))
    except IndexError:
        ax.legend((rects_left[0], rects_right[0]), ('Left Hand', 'Right Hand'))

    ax.set_xticklabels(labels)

    plt.show()


def make_average_hours_per_day_chart_both_hands(directory):
    participant_to_left_day_percentages = {}
    participant_to_right_day_percentages = {}
    date_dic = get_date_dic('HAMD_final_scores.csv')
    for filename in get_files(directory):
        participant = filename[:4]
        print("Starting " + participant)
        start_date = date_dic[participant]['Week 0']
        days_left = split_into_days(directory+'/'+filename, True, start=start_date)
        days_right = split_into_days(directory+'/'+filename, False, start=start_date)

        percentages_per_day_left = per_day_percentages(days_left)
        percentages_per_day_right = per_day_percentages(days_right)
        participant_to_left_day_percentages[filename] = percentages_per_day_left
        participant_to_right_day_percentages[filename] = percentages_per_day_right
        print(participant + " has been finished")
        print()
    chart_hours_per_day_averages_together(participant_to_left_day_percentages,
                                          participant_to_right_day_percentages,
                                          "Hours per Day Compliance Data")


def make_hours_per_day_chart_from_data(directory, left):
    percentages_per_day_by_all_participants = {}    # maps a participant to a list of their percentage compliance
    date_dic = get_date_dic('HAMD_final_scores.csv')
    for filename in get_files(directory):
        participant = filename[:4]
        print("Starting " + participant)
        start_date = date_dic[participant]['Week 0']
        days = split_into_days(directory+'/'+filename, left, start=start_date)

        percentages_per_day = per_day_percentages(days)
        percentages_per_day_by_all_participants[filename] = percentages_per_day
        print(participant + " has been finished")
        print()
    if left:
        chart_hours_per_day_average(percentages_per_day_by_all_participants, "Left Hand Compliance Data", left)
    else:
        chart_hours_per_day_average(percentages_per_day_by_all_participants, "Right Hand Compliance Data", left)


def get_average_hours(directory, left):
    all_percentages = []
    date_dic = get_date_dic('HAMD_final_scores.csv')
    participants_to_average = {}
    for filename in get_files(directory):
        participant = filename[:4]
        start_date = date_dic[participant]['Week 0']
        days = split_into_days(directory+'/'+filename, left, start=start_date)

        percentages_per_day = per_day_percentages(days)
        all_percentages.extend(percentages_per_day)
        participants_to_average[participant] = sum(percentages_per_day)*24/len(percentages_per_day)

    average = sum(all_percentages)*24/len(all_percentages)
    print("Average total number hours uploaded:", average)
    return (average, participants_to_average)


def get_hours_per_day_all_participants(directory):
    left_hand_percentage_per_day = {}
    right_hand_percentage_per_day = {}
    for filename in get_files(directory):
        left_days = split_into_days(directory+'/'+filename, left=True)
        right_days = split_into_days(directory+'/'+filename, left=False)
        left_percentages = per_day_percentages(left_days)
        right_percentages = per_day_percentages(right_days)
        left_hand_percentage_per_day[filename] = left_percentages
        right_hand_percentage_per_day[filename] = right_percentages

    return left_hand_percentage_per_day, right_hand_percentage_per_day


def save_chart_one_by_one(directory):
    date_dic = get_date_dic('HAMD_final_scores.csv')
    for filename in get_files(directory):
    # filename = 'M001_temp.h5'
        print("Trying to create "+filename)
        # try:
        participant = filename[:4]
        start_date = date_dic[participant]['Week 0']
        left_days = split_into_days(directory + '/' + filename, left=True, start=start_date)
        right_days = split_into_days(directory + '/' + filename, left=False, start=start_date)
        assessments = date_dic[participant]
        assessments.pop('Screen')
        day_of_study_assessments = {}
        for assessment_day in assessments:
            day_difference = assessments[assessment_day] - assessments['Week 0']
            day_difference = day_difference.days
            day_of_study_assessments[day_difference] = assessment_day
        # except:
        #     print('Error on ' + filename)
        left_percentages = per_day_percentages(left_days)
        right_percentages = per_day_percentages(right_days)
        save_chart_hours_per_day_per_person(participant, left_percentages, right_percentages, day_of_study_assessments, assessments)


def get_date_dic(dates_csv_filepath):
    dates = pd.read_csv(dates_csv_filepath, index_col=0, usecols=['ID', 'Name', 'date'])
    participants = sorted(list(set(dates.index.values)))
    date_dic = {}
    for participant in participants:
        if participant not in date_dic:
            date_dic[participant] = {}
        segment = dates.loc[participant]
        for index, row in segment.iterrows():
            name = row['Name']
            date = row['date']
            if isinstance(date, str):
                date_dic[participant][name] = dt.strptime(date, '%m/%d/%y').replace(tzinfo=tz.gettz('America/New_York'))
    return date_dic


def string_to_datetime(date_string):
    return dt.strptime(date_string, '%x %z')


def make_histogram_of_assessment_days():
    date_dic = get_date_dic('HAMD_final_scores.csv')
    hist_dic = {}
    for participant in date_dic:
        assessments = date_dic[participant]
        for assessment_day in assessments:
            if assessment_day != 'Screen' and assessment_day != 'Week 0':
                day_difference = assessments[assessment_day] - assessments['Week 0']
                day_difference = day_difference.days
                if day_difference <= 70:
                    if day_difference not in hist_dic:
                        hist_dic[day_difference] = 0
                    hist_dic[day_difference] += 1
    assessment_days = [0 for x in range(max(hist_dic.keys())+1)]
    for key in hist_dic:
        assessment_days[key] = hist_dic[key]
    plt.bar(np.arange(len(assessment_days)), assessment_days, color=GREEN)
    plt.title('Assessment days over days in study')
    plt.xlabel('Day in Study')
    plt.ylabel('Number of participants')
    plt.show()


def write_average_percentages_file():
    average_percentages_file = open("average_hours_compliance.txt", 'w')
    average_total_hours_left, participants_left_averages = get_average_hours(TEMP_DIRECTORY, True)
    average_total_hours_right, participants_right_averages = get_average_hours(TEMP_DIRECTORY, False)
    average_percentages_file.write("Participant     Left Hand     Right Hand\n")
    average_percentages_file.write("-----------     ---------     ----------\n")
    for participant in participants_left_averages:
        average_hours_left = "{0:.2f}".format(participants_left_averages[participant])
        len_left = len(average_hours_left)
        for i in range(5-len_left):
            average_hours_left += " "
        average_hours_right = "{0:.2f}".format(participants_right_averages[participant])
        len_right = len(average_hours_right)
        for i in range(5-len_right):
            average_hours_right += " "
        average_percentages_file.write(participant + "              " + average_hours_left + "         " + average_hours_right + "    \n")

    average_percentages_file.write("-----------     ---------     ----------\n")

    average_total_hours_left = "{0:.2f}".format(average_total_hours_left)
    len_total_left = len(average_total_hours_left)
    for i in range(5-len_total_left):
        average_total_hours_left += " "
    average_total_hours_right = "{0:.2f}".format(average_total_hours_right)
    len_total_right = len(average_total_hours_right)
    for i in range(5-len_total_right):
        average_total_hours_right += " "
    average_percentages_file.write("Total             " + average_total_hours_left + "         " + average_total_hours_right)


def plot_individuals_to_days_uploaded():
    date_dic = get_date_dic("HAMD_final_scores.csv")
    participants = []
    days_per_participant = []
    for participant in date_dic:
        if 'Week 8' in date_dic[participant]:
            number_of_days = date_dic[participant]['Week 8'] - date_dic[participant]['Week 0']
            number_of_days = number_of_days.days
        elif 'Week 6' in date_dic[participant]:
            number_of_days = date_dic[participant]['Week 6'] - date_dic[participant]['Week 0']
            number_of_days = number_of_days.days
        elif 'Week 4' in date_dic[participant]:
            number_of_days = date_dic[participant]['Week 4'] - date_dic[participant]['Week 0']
            number_of_days = number_of_days.days
        elif 'Week 2' in date_dic[participant]:
            number_of_days = date_dic[participant]['Week 2'] - date_dic[participant]['Week 0']
            number_of_days = number_of_days.days
        else:
            number_of_days = 0
        participants.append(participant)
        days_per_participant.append(number_of_days)

    plt.bar(np.arange(len(participants)), days_per_participant, color=GREEN)
    plt.xlabel("Participant")
    plt.ylabel("Number of days")
    plt.title("Days participated in study")
    plt.xticks(np.arange(len(participants)), participants, rotation='vertical')
    plt.yticks(np.arange(0, (max(days_per_participant)+10)//10*10 + 1, 10))
    plt.show()


plot_individuals_to_days_uploaded()

# make_average_hours_per_day_chart_both_hands(TEMP_DIRECTORY)

# make_minutes_per_hour_chart_both_hands(TEMP_DIRECTORY)

# make_minutes_per_hour_chart_from_data(TEMP_DIRECTORY, left=True)
# make_minutes_per_hour_chart_from_data(TEMP_DIRECTORY, left=False)

# make_histogram_of_assessment_days()

# save_chart_one_by_one(TEMP_DIRECTORY)

# make_hours_per_day_chart_from_data(TEMP_DIRECTORY, left=True)
# make_hours_per_day_chart_from_data(TEMP_DIRECTORY, left=False)

# get_average_hours(TEMP_DIRECTORY, left=False)

# write_average_percentages_file()

# print(pd.read_hdf(NEW_FILES+'/M042_temp.h5', 'TEMP_left'))

# participants_to_assessment_dates('HAMD_final_scores.csv')
