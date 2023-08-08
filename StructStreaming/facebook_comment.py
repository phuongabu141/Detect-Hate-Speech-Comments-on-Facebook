from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from time import sleep
import os
import pathlib
from uuid import uuid4
from datetime import datetime as dt
import json
import time
import requests


def init_driver(headless=False):
    CHROMEDRIVER_PATH = 'chromedriver/chromedriver'
    WINDOW_SIZE = "1366,768"
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControllered")
    chrome_options.add_experimental_option('useAutomationExtension', False)
    prefs = {"profile.default_content_setting_values.notifications": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    # chrome_options.add_argument("--start-maximized")  # open Browser in maximized mode
    # overcome limited resource problems
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation"])
    chrome_options.add_argument('disable-infobars')
    driver = webdriver.Chrome(
        executable_path=CHROMEDRIVER_PATH, options=chrome_options)
    return driver


def convert_to_cookie(cookie):
    try:
        new_cookie = ["c_user=", "xs="]
        cookie_arr = cookie.split(";")
        for i in cookie_arr:
            if i.__contains__('c_user='):
                new_cookie[0] = new_cookie[0] + \
                    (i.strip() + ";").split("c_user=")[1]
            if i.__contains__('xs='):
                new_cookie[1] = new_cookie[1] + \
                    (i.strip() + ";").split("xs=")[1]
                if (len(new_cookie[1].split("|"))):
                    new_cookie[1] = new_cookie[1].split("|")[0]
                if (";" not in new_cookie[1]):
                    new_cookie[1] = new_cookie[1] + ";"

        conv = new_cookie[0] + " " + new_cookie[1]
        if conv.split(" ")[0] == "c_user=":
            return
        else:
            return conv
    except:
        print("Error convert cookie")


def login_facebook_by_cookie(driver, cookie):
    cookie = convert_to_cookie(cookie)
    print(cookie)
    if cookie != None:
        script = 'javascript:void(function(){ function setCookie(t) { var list = t.split("; "); console.log(list); for (var i = list.length - 1; i >= 0; i--) { var cname = list[i].split("=")[0]; var cvalue = list[i].split("=")[1]; var d = new Date(); d.setTime(d.getTime() + (7*24*60*60*1000)); var expires = ";domain=.facebook.com;expires="+ d.toUTCString(); document.cookie = cname + "=" + cvalue + "; " + expires; } } function hex2a(hex) { var str = ""; for (var i = 0; i < hex.length; i += 2) { var v = parseInt(hex.substr(i, 2), 16); if (v) str += String.fromCharCode(v); } return str; } setCookie("' + cookie + '"); location.href = "https://mbasic.facebook.com"; })();'
        driver.execute_script(script)
        sleep(5)


def check_live_cookie(driver, cookie):
    driver.get('https://mbasic.facebook.com/')
    sleep(2)
    login_facebook_by_cookie(driver, cookie)

    return check_live_clone(driver)


def check_live_clone(driver):
    driver.get("https://mbasic.facebook.com/")
    sleep(2)
    element_live = driver.find_elements(
        By.XPATH, '//a[contains(@href, "/messages/")]')
    if len(element_live) > 0:
        print("Live")
        return True

    print('Not live')
    return False


def check_live_cookie(driver, cookie):
    driver.get('https://mbasic.facebook.com/')
    sleep(2)
    login_facebook_by_cookie(driver, cookie)

    return check_live_clone(driver)


def search_by_key(driver, key_word):
    search_results = []
    driver.get(
        f"https://mbasic.facebook.com/search/pages/?q={'+'.join(key_word.split())}")
    page_element = driver.find_elements(
        By.XPATH, '//td[contains(@class,"t cd")]/a')
    for link in page_element:
        href = link.get_attribute('href')
        page_id = href.split('/')[3].split('?')[0]
        if page_id == "profile.php":
            continue
        search_results.append(page_id)
    return search_results


def read_data(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    data = []
    for i, line in enumerate(f):
        line = repr(line)
        line = line[1:len(line) - 3]
        data.append(line)

    return data


def write_file_txt(file_name, content):
    with open(file_name, 'a') as f1:
        f1.write(content + os.linesep)


def get_post_ids(driver, file_path=os.path.join('data', 'post.csv')):
    all_posts = read_data(file_path)
    sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    share_btn = driver.find_elements(
        By.XPATH, '//a[contains(@href, "/sharer.php")]')
    if len(share_btn):
        for link in share_btn:
            post_id = link.get_attribute('href').split('sid=')[1].split('&')[0]
            if post_id not in all_posts:
                write_file_txt(file_path, post_id)


def get_num_of_post_fanpage(driver, page_id, amount, file_path=os.path.join('data', 'post.csv')):
    open(file_path, 'a').close()
    driver.get(f"https://touch.facebook.com/{page_id}")
    while len(read_data(file_path)) < amount:
        get_post_ids(driver, file_path)


def get_content_comment(driver):
    links = driver.find_elements(
        By.XPATH, '//a[contains(@href, "comment/replies")]')
    json_datas = []
    ids = []
    if (len(links)):
        for link in links:
            data = {}
            take_link = link.get_attribute('href').split('ctoken=')[
                1].split('&')[0]
            text_comment_element = driver.find_element(
                By.XPATH, ('//*[@id="' + take_link.split('_')[1] + '"]/div/div[1]'))
            user_name_element = driver.find_element(
                By.XPATH, ('//*[@id="' + take_link.split('_')[1] + '"]/div/h3/a[1]'))
            if take_link not in ids:
                comment = text_comment_element.text.strip()
                user_name = user_name_element.text.strip()
                if user_name != "" and comment != "":
                    data["user"] = user_name
                    data["comment"] = comment
                    data["timestamp"] = str(dt.now().timestamp())
                    json_datas.append(data)
                    ids.append(take_link)
    return ids, json_datas


def save_data_as_json(data, output_path):
    path = os.path.join(output_path, f"{data['timestamp']}_{uuid4()}.json")
    with open(path, "w", encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False)


def delete_file_in_folder(path):
    count = 0
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
        count += 1
    print(f"Deleted {count} files")


class FacebookComment():
    def __init__(self, cookie, key_word, number_post_id, headless, number_commnt_take=100000):
        self.cookie = cookie
        self.key_word = key_word
        self.number_post_id = number_post_id
        self.number_commnt_take = number_commnt_take
        self.driver = init_driver(headless=headless)
        self.is_live = check_live_cookie(self.driver, cookie)
        page_ids = search_by_key(self.driver, key_word)
        get_num_of_post_fanpage(self.driver, page_ids[0] if len(
            page_ids) else "thinhseu.official", number_post_id)
        self.post_ids = read_data("data/post.csv")
        self.is_first_comment = True
        self.position_post_id = 0
        self.post_id_current = self.post_ids[self.position_post_id]
        self.number_taked = 0

    def get_comment(self):
        if self.number_taked < self.number_commnt_take:
            if self.is_first_comment:
                self.driver.get(
                    f"https://mbasic.facebook.com/{self.post_id_current}")
                ids, json_datas = get_content_comment(self.driver)
                self.number_taked += len(ids)
                self.is_first_comment = False
                print(json_datas)
                return json_datas
            else:
                next_btn = self.driver.find_elements(
                    By.XPATH, '//*[contains(@id,"see_next")]/a')
                if len(next_btn):
                    next_btn[0].click()
                    ids, json_datas = get_content_comment(self.driver)
                    self.number_taked += len(ids)
                    print(json_datas)
                    return json_datas
                else:
                    self.position_post_id += 1
                    self.post_id_current = self.post_ids[self.position_post_id]
                    self.is_first_comment = True
                    self.get_comment()
        else:
            print("The limit on the number of comments allowed has been exceeded")
            return None


def main():
    with open("config.json") as f:
        config = json.load(f)

    if config['delete_old_data']:
        delete_file_in_folder(config['output_path'])

    crawler = FacebookComment(config['cookie_facebook'], config['key_word'],
                              config['number_post_id'], config['headless'],
                              config['number_comment_take'])
    i = 0
    while True:
        json_datas = crawler.get_comment()
        if json_datas is None or not len(json_datas):
            continue
        for json_data in json_datas:
            save_data_as_json(json_data, config['output_path'])
            i += 1
            if i % 100 == 0:
                print(f"{i} comments have been retrieved.")
        time.sleep(1)

# python facebook_comment.py
if __name__ == "__main__":
    main()
