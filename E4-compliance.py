import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
from dateutil import tz
import matplotlib.pyplot as plt
import os
import statistics as stat

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
SECONDS_PER_DAY = 86400
COMPLIANCE_THRESHOLD = SECONDS_PER_DAY/2
SECONDS_PER_HOUR = 3600
TEMP_DIRECTORY = 'test-temp-files'
NEW_FILES = 'new-files'
BLUE = '#2C55CA'
RED = '#D63434'
GREEN = '#5AD381'
YELLOW = '#F7D158'
PURPLE = '#7D5AD3'


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
    pared = []
    for i in range(len(temp_col)):
        if temp_col[i] >= 31:
            pared.append(temp_col[i])
    return len(pared)*.25/SECONDS_PER_DAY


def percentage_of_one_day_2(segment):   # segment must be temp hdf segment for exactly one day
    temp_col = get_temp_col(segment)
    pared = []
    for i in range(len(temp_col)):
        if 31 <= temp_col[i] <= 40:
            pared.append(temp_col[i])
    return len(pared)*.25/SECONDS_PER_DAY


def per_day_percentages(days):
    percentages = []
    for day in days:
        percentages.append(percentage_of_one_day(day))
    return percentages


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
        if day <= 56:
            hours_for_day = [x*24 for x in days_to_percentages[day]]
            avg_hours_per_day.append(sum(hours_for_day)/len(hours_for_day))
            std_devs.append(stat.pstdev(hours_for_day))
            labels.append(day)
    if left:
        plt.bar(np.arange(len(labels)), avg_hours_per_day, color=[BLUE], yerr=std_devs)
    else:
        plt.bar(np.arange(len(labels)), avg_hours_per_day, color=[RED], yerr=std_devs)
    plt.title(title)
    plt.xlabel('Day in Study')
    plt.ylabel('Average # of Hours')
    plt.show()


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

    plt.savefig('individual-compliance-data/'+name+'.png')
    print(name+'.png has been created')
    plt.close()


def split_into_days(hdf_temp_filepath, left, start=None):

    acc_first_row_left = pd.read_hdf(hdf_temp_filepath, 'TEMP_left')
    acc_first_row_left.sort_index()
    acc_first_row_right = pd.read_hdf(hdf_temp_filepath, 'TEMP_right')
    acc_first_row_right.sort_index()

    if not start:
        start_date_left = np.min(acc_first_row_left.index.values)
        start_date_right = np.min(acc_first_row_right.index.values)
        start_date = min(start_date_left, start_date_right)
        start_date = dt.utcfromtimestamp(start_date.tolist() / 1e9)
        start_date = start_date.replace(tzinfo=None)
    else:
        start_date = start

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
            hdf_in_days.append(acc_first_row_right.loc[beginning:end])
    # print()
    return hdf_in_days


def split_into_hours(hdf_temp_filepath, left=True):
    acc_first_row_left = pd.read_hdf(hdf_temp_filepath, 'TEMP_left')
    start_date_left = np.min(acc_first_row_left.index.values)
    end_date_left = np.max(acc_first_row_left.index.values)

    acc_first_row_right = pd.read_hdf(hdf_temp_filepath, 'TEMP_right')
    start_date_right = np.min(acc_first_row_right.index.values)
    end_date_right = np.max(acc_first_row_right.index.values)

    start_date = min(start_date_left, start_date_right)
    # convert to datetime
    start_date = dt.utcfromtimestamp(start_date.tolist() / 1e9)

    start_date = start_date.replace(tzinfo=tz.gettz('UTC')).astimezone(
        tz.gettz('America/New_York'))
    start_date = start_date.replace(hour=0, minute=0, second=0)
    end_date = max(end_date_left, end_date_right)
    # convert to datetime
    end_date = dt.utcfromtimestamp(end_date.tolist() / 1e9)
    end_date = end_date.replace(tzinfo=tz.gettz('UTC')).astimezone(
        tz.gettz('America/New_York'))
    end_date = end_date.replace(hour=0, minute=0, second=0)
    rng = pd.date_range(start_date, end_date, freq='H')

    hdf_in_hours = []
    for idx, beginning in enumerate(rng):
        end = beginning + timedelta(hours=1)
        if left:
            hdf_in_hours.append(acc_first_row_left.loc[beginning:end])
        else:
            hdf_in_hours.append(acc_first_row_right.loc[beginning:end])

    return hdf_in_hours


def split_day_into_hours(day_segment):
    start_date = np.min(day_segment.index.values)
    end_date = np.max(day_segment.index.values)

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
    rng = pd.date_range(start_date, end_date, freq='H')

    hdf_in_hours = []
    for idx, beginning in enumerate(rng):
        end = beginning + timedelta(hours=1)
        hdf_in_hours.append(day_segment.loc[beginning:end])

    return hdf_in_hours


def make_hours_per_day_chart_from_data(directory, left):
    percentages_per_day_by_all_participants = {}    # maps a participant to a list of their percentage compliance
    date_dic = get_date_dic('HAMD_final_scores.csv')
    for filename in get_files(directory):
        participant = filename[:4]
        start_date = date_dic[participant]['Week 0']
        days = split_into_days(directory+'/'+filename, left, start=start_date)

        days = strip_first_week(date_dic, participant, days)
        percentages_per_day = per_day_percentages(days)
        percentages_per_day_by_all_participants[filename] = percentages_per_day
    if left:
        chart_hours_per_day_average(percentages_per_day_by_all_participants, "Left Hand Compliance Data", left)
    else:
        chart_hours_per_day_average(percentages_per_day_by_all_participants, "Right Hand Compliance Data", left)


def get_average_total_hours(directory, left):
    all_percentages = []
    date_dic = get_date_dic('HAMD_final_scores.csv')
    for filename in get_files(directory):
        participant = filename[:4]
        start_date = date_dic[participant]['Week 0']
        days = split_into_days(directory+'/'+filename, left, start=start_date)

        days = strip_first_week(date_dic, participant, days)
        percentages_per_day = per_day_percentages(days)
        all_percentages += percentages_per_day

    average = (sum(all_percentages)*24/len(all_percentages))
    print("Average total number hours uploaded:", average)


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
        try:
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
        except:
            print('Error on ' + filename)
        print()

    left_percentages = per_day_percentages(left_days)
    right_percentages = per_day_percentages(right_days)
    save_chart_hours_per_day_per_person(participant, left_percentages, right_percentages, day_of_study_assessments, assessments)


def strip_first_week(date_dic, participant, days):
    assessments = date_dic[participant]
    days_to_remove = assessments['Week 0'] - assessments['Screen']
    days_to_remove = days_to_remove.days
    return days[days_to_remove:]


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
                date_dic[participant][name] = string_to_datetime(date+' -0400')
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


# make_histogram_of_assessment_days()

# save_chart_one_by_one(NEW_FILES)

# make_hours_per_day_chart_from_data(TEMP_DIRECTORY, left=False)

# get_average_total_hours(TEMP_DIRECTORY, left=False)

# print(pd.read_hdf(NEW_FILES+'/M042_temp.h5', 'TEMP_left'))

# participants_to_assessment_dates('HAMD_final_scores.csv')
