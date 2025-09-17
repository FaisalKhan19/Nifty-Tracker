import requests
import pandas as pd
from datetime import datetime, timedelta
import configparser
import json
import os
import time
from openai import OpenAI
import pdfplumber
import re
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()

class Tracker:
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.data = []
        self.fetched = set()
        self.processed_ids = set()
        self.processed_date = None
        self.data = []

    def _processed_path(self, date_str: str):
        return f"{date_str}_processed.json"

    def load_processed_for_date(self, date_str: str):
        self.processed_date = date_str
        path = self._processed_path(date_str)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    ids = json.load(f)
                if isinstance(ids, list):
                    self.processed_ids = set(ids)
                else:
                    self.processed_ids = set()
            except Exception:
                self.processed_ids = set()
        else:
            self.processed_ids = set()

    def save_processed(self):
        if not self.processed_date:
            return
        path = self._processed_path(self.processed_date)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self.processed_ids)), f)
        except Exception:
            pass

    def getAnnouncements(self, code: int, prevDate: str = None, toDate: str = None, test: bool = False):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.bseindia.com/",
            "Origin": "https://www.bseindia.com",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9"
        }
        session = requests.Session()
        session.headers.update(headers)
        
        if toDate is None:
            to_date_obj = datetime.now()
            strToDate_val = to_date_obj.strftime("%Y%m%d")
        else:
            try:
                to_date_obj = datetime.strptime(toDate, "%Y%m%d")
                strToDate_val = toDate
            except ValueError:
                print("Error parsing dates, defaulting to today")
                to_date_obj = datetime.now()
                strToDate_val = to_date_obj.strftime("%Y%m%d")

        if prevDate is None:
            prev_date_obj = to_date_obj - timedelta(days=1)
            strPrevDate_val = prev_date_obj.strftime("%Y%m%d")
        else:
            prevDate_str = str(prevDate)
            try:
                prev_date_obj = datetime.strptime(prevDate_str, "%Y%m%d")
                strPrevDate_val = prevDate_str
            except ValueError:
                prev_date_obj = to_date_obj - timedelta(days=1)
                strPrevDate_val = prev_date_obj.strftime("%Y%m%d")

        params = {
            "pageno": 1,
            "strCat": "Result",
            "strPrevDate": strPrevDate_val,
            "strScrip": str(code),
            "strSearch": "P",
            "strToDate": strToDate_val,
            "strType": "C",
            "subcategory": "Financial Results"
        }
        if test:
            params.pop('strScrip')
        session.params.update(params)
        response = session.get(url=self.config['URLs']['Feed'])
        try:
            self.data.extend(response.json()['Table'])
            return response.json()
        except Exception as e:
            print(e)
            return None
    
    def getProcessed(self):
        message = ""
        _data = []
        for r in self.data:
            news = r['NEWSSUB']
            fileurl = f"{self.config['URLs']['Doc']}?Pname={r['ATTACHMENTNAME']}"
            pub_time = r['DT_TM']
            id = r['NEWSID']
            name = r['SLONGNAME']
            _data.append({
                'id': id,
                'company': name,
                'sec_code': r['SCRIP_CD'],
                'headline': news,
                'file_url': fileurl,
                'time': pub_time
            })
            message += f"🏢 {name}\n📢 {news}\n⏰ {pub_time}\n🔗 {fileurl}\n\n"
        return message, pd.DataFrame(_data)
    
    def format_date(self, date: datetime):
        date_str = str(date)
        date_str = date_str.split(" ")[0].replace("-", "")
        return date_str
    
    def build_set(self, df: pd.DataFrame, prevDate: str = None, toDate: str = None):
        # Reset accumulated announcements before a new polling batch
        for i, row in tqdm(df.iterrows(), desc='Processing Announcements'):
            self.getAnnouncements(code=row['Security Code'], prevDate=prevDate, toDate=toDate)
            time.sleep(0.2)

class Bot:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.client = OpenAI()
        self.page_filter = re.compile(r"(financial\s+highlights|results|profit|revenue|PAT|Profit After Tax|EBITDA|EPS)", re.IGNORECASE)
        with open("subscribers.json", "r") as f:
            content = f.read().strip()
        self.mailing_list = set(json.loads(content))
        self.min_delay_between_calls = float(os.getenv("OPENAI_MIN_DELAY", "0.4"))
        self._last_openai_call_ts = 0.0

    def extract_pages(self, pdf_path):
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages_text.append((i, text))
        return pages_text

    def filter_financial_pages(self, pages_text):
        return [(i, text) for i, text in pages_text if self.page_filter.search(text)]

    def extract_with_ai(self, filtered_pages):
        if not filtered_pages:
            return None
        content = "\n\n".join([f"Page {i}:\n{text}" for i, text in filtered_pages])

        system_msg = (
            "You extract numeric KPIs from financial result excerpts and must return a single valid JSON object only. "
            "No prose, no markdown, no code fences."
        )
        user_msg = (
            "Extract the following values if present from the latest quarter (numbers only, no units):\n"
            "- Revenue\n- PAT (Profit After Tax)\n- EBITDA\n- EPS Basic\n- EPS Diluted\n- Operating Margin\n- Earnings\n\n"
            "Return STRICT JSON only with these keys. If missing, set value to empty string.\n\n"
            "Text:\n" + content
        )

        # Simple client-side rate limiter and retry with exponential backoff on 429s/timeouts
        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            # Respect minimum spacing between OpenAI calls
            now = time.time()
            delay = self.min_delay_between_calls - (now - self._last_openai_call_ts)
            if delay > 0:
                time.sleep(delay)
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout=60,
                )
                self._last_openai_call_ts = time.time()
                return resp.choices[0].message.content
            except Exception as e:
                # Detect likely rate limit or transient errors
                err_text = str(e).lower()
                is_rate_limited = ("rate" in err_text or "429" in err_text)
                is_retryable = is_rate_limited or "timeout" in err_text or "temporarily unavailable" in err_text or "service unavailable" in err_text
                if attempt < max_retries - 1 and is_retryable:
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 60.0)
                    continue
                return None

    def buildNumbers(self, data: pd.DataFrame):
        numbers = []
        for _, row in tqdm(data.iterrows(), desc='Processing PDFs'):
            pdf_path = os.path.join("PDFs", row['file_url'].split("=")[1])
            pages_text = self.extract_pages(pdf_path)
            filtered = self.filter_financial_pages(pages_text)
            if not filtered:
                continue
            result = self.extract_with_ai(filtered)
            try:
                if not result:
                    continue
                try:
                    result = json.loads(result)
                except Exception:
                    match = re.findall(r"\{[\s\S]*\}", result)
                    if not match:
                        raise
                    result = json.loads(match[-1])
                result.update({'company': row['company']})
                result.update({'date': row['time']})
                result.update({'sec_code': row['sec_code']})
                result.update({'file_url': row['file_url']})
                numbers.append(result)
            except Exception as e:
                print("Error occured while processing PDF:", e)
                print(result)
                continue
        if not numbers:
            return pd.DataFrame(columns=['company', 'date', 'sec_code', 'Revenue', 'PAT', 'EBITDA', 'EPS Basic', 'EPS Diluted', 'Operating Margin', 'Earnings', 'file_url'])
        df = pd.DataFrame(numbers)
        cols = [c for c in ['company', 'date', 'sec_code', 'Revenue', 'PAT', 'EBITDA', 'EPS Basic', 'EPS Diluted', 'Operating Margin', 'Earnings', 'file_url'] if c in df.columns]
        return df[cols]
    
    def send_messages(self, message, data_path):
        BOT_TOKEN = self.config['Bot']['Token']
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        msg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        with open(data_path, "rb") as f:
            for chat_id in self.mailing_list:
                # res = requests.post(msg_url, data={"chat_id": chat_id, "text": message})
                f.seek(0)
                res = requests.post(
                    url,
                    data={
                        "chat_id": chat_id,
                        "caption": "Extracted Numbers",
                        "parse_mode": "Markdown"
                    },
                    files={"document": (os.path.basename(data_path), f, "text/csv")}
                )

def getReport(url, quarter=None):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
    }
    os.makedirs(os.path.join("PDFs", quarter), exist_ok=True)
    res = requests.get(url, headers=headers, timeout=20)

    if res.status_code == 200:
        path = os.path.join("PDFs", quarter, url.split('=')[1])
        with open(path, "wb") as f:
            f.write(res.content)

def fetch_historical_quarterly():
    os.makedirs('QuarterlyResults')
    df = pd.read_csv("tracking_list.csv")
    df = df[:5]
    # Last 10 quarter end dates (most recent first, ending 30-Jun-2025)
    quarter_ends = [
        # "2025-06-30",
        "2025-03-31",
        "2024-12-31",
        "2024-09-30",
        "2024-06-30",
        "2024-03-31",
        "2023-12-31",
        "2023-09-30",
        # "2023-06-30",
        # "2023-03-31"
    ]

    # Convert to datetime
    quarter_ends = [datetime.strptime(d, "%Y-%m-%d") for d in quarter_ends]

    # Compute upload windows (45 days after quarter end)
    upload_windows = []
    for q_end in quarter_ends:
        tracker = Tracker(config_path='.ini')
        bot = Bot(config_path='.ini')
        start = q_end + timedelta(days=1)
        end = start + timedelta(days=30)
        start1 = end + timedelta(days=1)
        end1 = start1 + timedelta(days=14)   # 45 days inclusive
        
        prev = tracker.format_date(start)
        to = tracker.format_date(end)
        prev1 = tracker.format_date(start1)
        to1 = tracker.format_date(end1)

        tracker.build_set(df, prev, to)
        tracker.build_set(df, prev1, to1)

        _, fetched = tracker.getProcessed()
        fetched.to_csv(os.path.join('QuarterlyResults', f"{tracker.format_date(q_end)}_announcements"))
        if not fetched.empty:
            for _, row in fetched.iterrows():
                url = row['file_url']
                getReport(url, tracker.format_date(q_end))

            extracted = bot.buildNumbers(fetched)
            extracted.to_csv(os.path.join('QuarterlyResults', tracker.format_date(q_end)), index=False)


def main():
    tracker = Tracker(config_path='.ini')
    df = pd.read_csv("tracking_list.csv")
    bot = Bot(config_path='.ini')
    _today = tracker.format_date(datetime.today())
    tracker.load_processed_for_date(_today)
    while True:
        save_path = f"{_today}.csv"
        today = tracker.format_date(datetime.today())
        prev = tracker.format_date(datetime.today() - timedelta(days=0))
        tracker.build_set(df, prev, today) # For deployment

        _, data = tracker.getProcessed()

        # Filter out announcements already processed today
        if not data.empty:
            new_data = data[~data['id'].isin(tracker.processed_ids)].copy()
        else:
            new_data = data
        if not new_data.empty:
            # Download only new PDFs
            for _, row in new_data.iterrows():
                url = row['file_url']
                getReport(url)

            # Run extraction only on new announcements
            extracted_data = bot.buildNumbers(data=new_data)
            extracted_data.to_csv(save_path, index=False)

            # Build message only for new announcements
            new_message = ""
            for _, row in new_data.iterrows():
                new_message += f"🏢 {row['company']}\n📢 {row['headline']}\n⏰ {row['time']}\n🔗 {row['file_url']}\n\n"

            if new_message:
                bot.send_messages(new_message, save_path)

            # Mark as processed for today and persist
            tracker.processed_ids.update(new_data['id'].tolist())
            tracker.save_processed()

        time.sleep(60)

        if today != _today:
            _today = tracker.format_date(datetime.today())
            tracker.fetched = set()
            tracker.load_processed_for_date(_today)

        break
if __name__ == '__main__':
    main()