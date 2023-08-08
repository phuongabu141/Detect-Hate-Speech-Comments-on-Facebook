
from email import header
import imp
import json
from operator import ne
from re import L

from application import app

from plotly.offline import plot
from plotly.graph_objs import Scatter
import flask
from flask import Markup, jsonify, render_template, request, redirect, url_for
import ast
import base64
import pickle
import requests
import pandas as pd
import datetime as dt


count_clean = 0
count_offensive = 0
count_hate = 0
count_user = 0

labels_doughnut = []
values_doughnut = []

labels_line = []
values_line_clean = []
values_line_offensive = []
values_line_hate = []

data_table = []
data_word_cloud = []

data_display = 10
dict_map_label = {0: 'Clean', 1: 'Offensive', 2: 'Hate'}


def convert_timestamp_dt(x):
    return dt.datetime.fromtimestamp(x).strftime('%H:%M %d-%m-%Y')


def convert_timestamp_stime(x):
    return dt.datetime.fromtimestamp(x).strftime('%H:%M:%S')


def convert_timestamp_sdate(x):
    return dt.datetime.fromtimestamp(x).strftime('%d-%m-%Y')


def convert_to_dataframe(pickled_json):
    df = pickle.loads(base64.b64decode(pickled_json.encode()))
    df = df.sort_values(by=['timestamp', 'user'])
    df['label'] = df['prediction'].replace(dict_map_label)
    df['date_time'] = df['timestamp'].apply(lambda x: convert_timestamp_dt(x))
    df['date'] = df['timestamp'].apply(lambda x: convert_timestamp_sdate(x))
    df['time'] = df['timestamp'].apply(lambda x: convert_timestamp_stime(x))
    return df


def convert_dataframe_to_html(df):
    table_html = df.to_html(index=False, formatters={
                            'Prediction': map_status_label}, escape=False)

    config_htmt = {'Date': "20%", 'Time': "5%",
                   'User': "20%", "Comment": "40%", "Prediction": "15%"}

    # Loại border
    table_html = table_html.replace('border="1"', 'border="0"')
    # Thay đổi class của bảng thành zebra và set chiều rộng của bảng
    table_html = table_html.replace(
        'class="dataframe"', 'class="zebra" width="100%"')
    # Loại bỏ style mặc định
    table_html = table_html.replace('style="text-align: right;"', '')
    # Căn chỉnh các cột và độ rộng của mỗi cột

    for key, value in config_htmt.items():
        if key == "Prediction":
            table_html = table_html.replace(f'<th>{key}</th>',
                                        f'<th width="{value}">{key}</th>')
            continue
        table_html = table_html.replace(f'<th>{key}</th>',
                                        f'<th style="text-align:center" width="{value}">{key}</th>')

    # Căn giữa giá trị mỗi cột trong bảng
    table_html = table_html.replace('<td>', '<td style="text-align:center">')

    return table_html


@app.route('/')
def home_page():
    # global labels_doughnut, values_doughnut
    # global count_clean, count_offensive, count_hate, count_user
    # global labels_line, values_line_clean, values_line_offensive, values_line_hate

    return render_template('index.html',
                           labels_doughnut=labels_doughnut,
                           values_doughnut=values_doughnut,
                           count_clean=count_clean,
                           count_offensive=count_offensive,
                           count_hate=count_hate,
                           count_user=count_user,
                           labels_line=labels_line,
                           values_line_clean=values_line_clean,
                           values_line_offensive=values_line_offensive,
                           values_line_hate=values_line_hate)


@app.route('/refreshData')
def refresh_graph_data():
    global labels_doughnut, values_doughnut
    global count_clean, count_offensive, count_hate, count_user
    global labels_line, values_line_clean, values_line_offensive, values_line_hate
    global data_table, data_word_cloud

    # print("labels now: " + str(labels_doughnut))
    # print("data now: " + str(values_doughnut))

    # print("count clean now: " + str(count_clean))
    # print("count offensive now: " + str(count_offensive))
    # print("count hate now: " + str(count_hate))

    # print("timeseries clean now: " + str(values_timeseries_clean))
    # print("timeseries offensive now: " + str(values_timeseries_offensive))
    # print("timeseries hate now: " + str(values_timeseries_hate))

    return jsonify(labels_doughnut=labels_doughnut,
                   values_doughnut=values_doughnut,
                   count_clean=str(count_clean),
                   count_hate=str(count_hate),
                   count_offensive=str(count_offensive),
                   count_user=str(count_user),
                   labels_line=labels_line,
                   values_line_clean=values_line_clean,
                   values_line_offensive=values_line_offensive,
                   values_line_hate=values_line_hate,
                   data_table=data_table,
                   data_word_cloud=data_word_cloud)


@app.route('/updateData', methods=["POST"])
def html_table():
    global labels_doughnut, values_doughnut
    global count_clean, count_offensive, count_hate, count_user
    global labels_line, values_line_clean, values_line_offensive, values_line_hate
    global data_table, data_word_cloud

    # Không nhận được dữ liệu
    if not request.form:
        return "error", 400

    # Lấy dữ liệu được gửi từ spark_streaming và chuyển thành dataframe
    pickled_json = request.form.get('data')
    df = convert_to_dataframe(pickled_json)
    data_doughnut = df['label'].value_counts()

    # Dữ liệu dùng để vẽ biểu đồ doughnut
    labels_doughnut = ast.literal_eval(str(list(data_doughnut.index.values)))
    values_doughnut = ast.literal_eval(str(list(data_doughnut.values)))

    # Cập nhật giá trị từng nhãn và tổng số user
    if 'Clean' in labels_doughnut:
        count_clean = data_doughnut['Clean']
    if 'Offensive' in labels_doughnut:
        count_offensive = data_doughnut['Offensive']
    if 'Hate' in labels_doughnut:
        count_hate = data_doughnut['Hate']
    count_user = len(df['user'].unique())

    # Số lượng từng nhãn theo thời gian
    df_line = pd.crosstab(df['date_time'], df['label'])

    # Dữ liệu dùng để vẽ biểu đồ line
    labels_line = ast.literal_eval(str(list(df_line.index.values)))
    values_line_clean = ast.literal_eval(str(list(df_line['Clean'].values)))
    values_line_offensive = ast.literal_eval(str(list(df_line['Offensive'].values)))
    values_line_hate = ast.literal_eval(str(list(df_line['Hate'].values)))

    df_display = df[['date', 'time', 'user', 'comment', 'label']].tail(data_display)
    df_display.rename(columns={'date': 'Date',
                               'time': 'Time',
                               'user': 'User',
                               'comment': 'Comment',
                               'label': 'Prediction'}, inplace=True)

    data_table = convert_dataframe_to_html(df_display)
    data_word_cloud = convert_to_data_word_cloud(df)

    # print("labels received: " + str(labels_doughnut))
    # print("values received: " + str(values_doughnut))

    # print("count clean received: " + str(count_clean))
    # print("count offensive received: " + str(count_offensive))
    # print("count hate received: " + str(count_hate))
    # print("count user received: " + str(count_user))

    # print("data table received: " + data_table)
    # print("data word cloud received: " + data_word_cloud)

    return "success", 201


def map_status_label(x):
    value_result = '<span class="status {}"></span>'
    if x == "Clean":
        return value_result.format('green') + str(x)
    if x == "Offensive":
        return value_result.format('orange') + str(x)
    if x == "Hate":
        return value_result.format('red') + str(x)


def convert_to_data_word_cloud(df):
    words = df['comment'].str.lower().str.replace('[^\w\s]','')
    new_df = words.str.split(expand=True).stack().value_counts().reset_index().head(20)
    new_df.columns = ['text', 'size']
    # a, b = 0, 35
    import numpy as np
    new_df['size'] = np.random.randint(5, 50, size=(20))
    return new_df.to_json(orient='records', force_ascii = False)
